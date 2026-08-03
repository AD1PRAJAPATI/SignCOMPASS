"""
train_islr.py  —  ISLR training on ASL-Citizen (2,731 isolated signs).

TARGET METRICS (signer-independent test split):
    top-1 accuracy > 75%     (baseline: I3D 63%)
    recall@10      > 95%     (baseline: I3D 91%)

WHAT THIS SCRIPT DOES:
  1. Loads ASL-Citizen pose features (run extract_pose_islr.py first).
  2. Builds a 2731-class signer-independent split.
  3. Trains ISLRModel (Conformer + AttentionPool + ArcFace) with:
       - ArcFace loss (margin=0.3, scale=64)
       - Class-balanced WeightedRandomSampler
       - Cosine LR decay with linear warmup
       - Heavy temporal augmentation (speed_perturb, frame_dropout, mask)
  4. Evaluates top-1, top-5, recall@10 every epoch on val (signer-independent).
  5. Final evaluation on test split. Reports both raw and signer-independent metrics.

QUICK START:
    # Default: base model, full data
    python train_islr.py

    # Large model (needs more GPU memory):
    python train_islr.py --size large --batch_size 64

    # Reproduce baseline comparison with smaller model:
    python train_islr.py --size base --epochs 100 --lr 5e-4

GATE (Week 1-3):
    val top-1 > 63% (beats I3D baseline) → proceed to Week 3-5 (aux heads + fusion)
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_islr import (
    build_datasets,
    collate_fn_islr,
    make_class_balanced_sampler,
)
from models.islr_model import build_islr_model


# ─── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 5, 10)):
    """
    Compute top-k accuracy for multiple k values in a single pass.
    logits: (B, C), labels: (B,)
    Returns dict {k: accuracy_percent}
    """
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)   # (B, maxk)
    pred    = pred.t()                                                # (maxk, B)
    correct = pred.eq(labels.view(1, -1).expand_as(pred))           # (maxk, B)

    results = {}
    for k in ks:
        # A sample is correct if ANY of the top-k predictions matches
        correct_k = correct[:k].any(dim=0).float().sum()
        results[k] = 100.0 * correct_k.item() / max(labels.shape[0], 1)
    return results


@torch.no_grad()
def evaluate(model, loader, device, desc="val"):
    """
    Full evaluation pass: top-1, top-5, recall@10.
    Returns dict with accuracy values and per-signer breakdown if available.
    """
    model.eval()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        features = batch["features"].to(device)
        lengths  = batch["lengths"].to(device)
        labels   = batch["labels"].to(device)

        # Inference: no margin applied (labels=None)
        out = model(features, lengths, labels=None)
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)   # (N, C)
    all_labels = torch.cat(all_labels, dim=0)   # (N,)

    accs = topk_accuracy(all_logits, all_labels, ks=(1, 5, 10))
    return {
        "top1":    accs[1],
        "top5":    accs[5],
        "recall10": accs[10],
        "n":       all_labels.shape[0],
    }


# ─── LR schedule: cosine with linear warmup ───────────────────────────────────

def cosine_warmup_schedule(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    """
    Linear warmup for `warmup_epochs`, then cosine decay to `min_lr_ratio * base_lr`.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return LambdaLR(optimizer, lr_lambda)


# ─── Single epoch ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, device, blank_id=None):
    """
    One training epoch. Returns average loss.
    """
    model.train()
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc="train", leave=False):
        features = batch["features"].to(device)
        lengths  = batch["lengths"].to(device)
        labels   = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=scaler is not None):
            out  = model(features, lengths, labels=labels)
            loss = F.cross_entropy(out["logits"], labels, label_smoothing=0.1)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    # ── Seeds ──────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model size: {args.size} | seed: {args.seed}")

    # ── Data ───────────────────────────────────────────────────────────────────
    save_dir = os.path.join(
        args.save_dir,
        f"islr_{args.size}_seed{args.seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    train_ds, val_ds, test_ds, vocab = build_datasets(
        data_root    = args.data_root,
        metadata_csv = args.metadata_csv or None,
        features_dir = args.features_dir or None,
        max_frames   = args.max_frames,
        seed         = args.seed,
    )
    vocab.save(os.path.join(save_dir, "vocab.json"))
    print(f"Vocab: {len(vocab)} classes saved to {save_dir}/vocab.json")

    # Class-balanced sampler for train
    sampler = make_class_balanced_sampler(train_ds.df, vocab)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler,                # balanced, so shuffle=False
        num_workers=args.num_workers,
        collate_fn=collate_fn_islr,
        pin_memory=True,
        multiprocessing_context="forkserver",
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn_islr,
        multiprocessing_context="forkserver",
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn_islr,
        multiprocessing_context="forkserver",
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} clips")

    # ── Model ──────────────────────────────────────────────────────────────────
    # Infer input_dim from first batch (handles both 258-dim and 261-dim pose)
    sample = train_ds[0]
    input_dim = sample["features"].shape[-1]
    print(f"Inferred pose feature dim: {input_dim}")

    model = build_islr_model(
        num_classes = len(vocab),
        input_dim   = input_dim,
        size        = args.size,
        dropout     = args.dropout,
    ).to(device)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_warmup_schedule(
        optimizer,
        warmup_epochs = args.warmup_epochs,
        total_epochs  = args.epochs,
        min_lr_ratio  = 0.01,
    )
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop ──────────────────────────────────────────────────────────
    best_top1 = 0.0
    patience  = 0
    history   = []

    print(f"\n{'='*65}")
    print(f"ISLR Training | {len(vocab)} classes | {args.epochs} epochs")
    print(f"{'='*65}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        avg_loss = train_epoch(model, train_dl, optimizer, scaler, device)
        scheduler.step()

        val_metrics = evaluate(model, val_dl, device, desc="val")
        elapsed = time.time() - t0

        print(
            f"Ep {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | "
            f"val top-1={val_metrics['top1']:5.2f}% "
            f"top-5={val_metrics['top5']:5.2f}% "
            f"R@10={val_metrics['recall10']:5.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )

        row = {
            "epoch": epoch, "loss": avg_loss,
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "val_recall10": val_metrics["recall10"],
        }
        history.append(row)

        # Save checkpoint
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            patience  = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "val_top1":    val_metrics["top1"],
                "val_recall10": val_metrics["recall10"],
                "args":        vars(args),
            }, os.path.join(save_dir, "best.pt"))
            print(f"  ✓ Best val top-1: {best_top1:.2f}%")

            # Gate check
            if best_top1 > 63.0:
                print(f"  🎯 GATE PASSED: val top-1 {best_top1:.2f}% > 63% (I3D baseline)")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Save training history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL TEST EVALUATION  (signer-independent)")
    print(f"{'='*65}")

    ckpt = torch.load(os.path.join(save_dir, "best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_metrics = evaluate(model, test_dl, device, desc="test")
    val_metrics  = evaluate(model, val_dl,  device, desc="val_final")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model size: {args.size} | Params: {n_params:,}")
    print(f"Classes:    {len(vocab):,}")
    print(f"Input dim:  {input_dim}")
    print(f"")
    print(f"Val   top-1:   {val_metrics['top1']:6.2f}%  "
          f"top-5: {val_metrics['top5']:6.2f}%  "
          f"R@10: {val_metrics['recall10']:6.2f}%")
    print(f"Test  top-1:   {test_metrics['top1']:6.2f}%  "
          f"top-5: {test_metrics['top5']:6.2f}%  "
          f"R@10: {test_metrics['recall10']:6.2f}%")
    print(f"")
    print(f"Baseline (I3D):    top-1=63.0%  R@10=91.0%")

    def delta(val, base):
        sign = "+" if val >= base else ""
        return f"{sign}{val - base:.1f}%"

    print(f"vs. baseline:      top-1={delta(test_metrics['top1'], 63.0)}  "
          f"R@10={delta(test_metrics['recall10'], 91.0)}")
    print(f"")
    print(f"TARGET:            top-1 > 75%  R@10 > 95%")
    top1_pass   = test_metrics['top1']    > 75.0
    recall_pass = test_metrics['recall10'] > 95.0
    print(f"top-1 gate:  {'✅ PASS' if top1_pass   else '❌ not yet'}")
    print(f"R@10  gate:  {'✅ PASS' if recall_pass else '❌ not yet'}")
    print(f"{'='*65}")

    # Machine-readable result line for grep / logging
    print(
        f"ISLR_RESULT: size={args.size} seed={args.seed} "
        f"input_dim={input_dim} n_classes={len(vocab)} params={n_params} "
        f"val_top1={val_metrics['top1']:.2f} val_r10={val_metrics['recall10']:.2f} "
        f"test_top1={test_metrics['top1']:.2f} test_r10={test_metrics['recall10']:.2f}"
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train ISLR model on ASL-Citizen."
    )

    # Paths (auto-resolved from script location if not given)
    p.add_argument("--data_root",     default=None,
                   help="Project root (parent of data/). Defaults to parent of cslr/.")
    p.add_argument("--metadata_csv",  default=None,
                   help="Override path to metadata.csv.")
    p.add_argument("--features_dir",  default=None,
                   help="Override path to pose_features/.")
    p.add_argument("--save_dir",      default="checkpoints_islr",
                   help="Directory for checkpoints and vocab (default: checkpoints_islr/).")

    # Model
    p.add_argument("--size",     choices=["base", "large", "edge"], default="base",
                   help="Model size preset (default: base ~7M params).")
    p.add_argument("--dropout",  type=float, default=0.2)

    # Training
    p.add_argument("--epochs",        type=int,   default=120)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--wd",            type=float, default=1e-4)
    p.add_argument("--patience",      type=int,   default=20,
                   help="Early-stop if val top-1 doesn't improve for this many epochs.")
    p.add_argument("--max_frames",    type=int,   default=64,
                   help="Max frames per clip (clips longer than this are cropped).")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)

    args = p.parse_args()

    # Auto-resolve data_root relative to this script's location
    if args.data_root is None:
        script_dir    = os.path.dirname(os.path.abspath(__file__))
        args.data_root = os.path.dirname(script_dir)   # parent of cslr/

    main(args)
