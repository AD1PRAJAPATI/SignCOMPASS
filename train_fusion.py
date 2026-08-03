"""
train_fusion.py  —  Train the two-stream FusionISLRModel (pose + RGB) on ASL-Citizen.

RGB stream is source-agnostic: point --rgb_dir at VideoMAE features now, or at
converted SHuBERT features later. Same trainer either way.

Modes:
    --streams pose+rgb   full fusion (default, best)
    --streams pose       pose-only ablation (== train_islr baseline)
    --streams rgb        rgb-only ablation

Reuses metrics + LR schedule from train_islr.py so results are comparable.

Run:
    python train_fusion.py --streams pose+rgb --size base
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset_fusion import build_fusion_datasets, collate_fusion
from dataset_islr import make_class_balanced_sampler
from models.fusion_model import build_fusion_model
from train_islr import topk_accuracy, cosine_warmup_schedule


def _streams(s):
    return ("pose" in s, "rgb" in s)


@torch.no_grad()
def evaluate(model, loader, device, desc="val"):
    model.eval()
    logits_all, labels_all = [], []
    for b in loader:
        pose = b.get("pose_feats"); plen = b.get("pose_lengths")
        rgb  = b.get("rgb_feats");  rlen = b.get("rgb_lengths")
        out = model(
            pose_feats=pose.to(device) if pose is not None else None,
            pose_lengths=plen.to(device) if plen is not None else None,
            rgb_feats=rgb.to(device) if rgb is not None else None,
            rgb_lengths=rlen.to(device) if rlen is not None else None,
            labels=None,
        )
        logits_all.append(out["logits"].cpu())
        labels_all.append(b["labels"])
    if not logits_all: return {"top1":0.0,"top5":0.0,"recall10":0.0,"n":0}
    logits = torch.cat(logits_all); labels = torch.cat(labels_all)
    a = topk_accuracy(logits, labels, ks=(1, 5, 10))
    return {"top1": a[1], "top5": a[5], "recall10": a[10], "n": labels.shape[0]}


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pose, use_rgb = _streams(args.streams)
    print(f"Device {device} | streams={args.streams} (pose={use_pose}, rgb={use_rgb}) | size={args.size}")

    save_dir = os.path.join(args.save_dir, f"fusion_{args.streams.replace('+','_')}_{args.size}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    train_ds, val_ds, test_ds, vocab = build_fusion_datasets(
        data_root=args.data_root, use_pose=use_pose, use_rgb=use_rgb,
        rgb_dir=args.rgb_dir or None, max_frames=args.max_frames, seed=args.seed)
    vocab.save(os.path.join(save_dir, "vocab.json"))

    sampler = make_class_balanced_sampler(train_ds.df, vocab)
    dl = lambda ds, samp=None, bs=None: DataLoader(
        ds, batch_size=bs or args.batch_size, sampler=samp,
        shuffle=(samp is None and ds is train_ds),
        num_workers=args.num_workers, collate_fn=collate_fusion,
        pin_memory=True, multiprocessing_context="forkserver")
    train_dl = dl(train_ds, sampler)
    val_dl = dl(val_ds, bs=args.batch_size * 2)
    test_dl = dl(test_ds, bs=args.batch_size * 2)

    # infer dims from one sample
    s0 = train_ds[0]
    pose_dim = s0["pose"].shape[-1] if use_pose else 261
    rgb_dim = s0["rgb"].shape[-1] if use_rgb else 768
    print(f"pose_dim={pose_dim} rgb_dim={rgb_dim} | classes={len(vocab)}")

    model = build_fusion_model(
        num_classes=len(vocab), pose_dim=pose_dim, rgb_dim=rgb_dim,
        phon_vocab=None, size=args.size,
        use_pose=use_pose, use_rgb=use_rgb, dropout=args.dropout,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = cosine_warmup_schedule(opt, args.warmup_epochs, args.epochs)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    best, patience = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        t0 = time.time()
        for b in train_dl:
            pose = b.get("pose_feats"); plen = b.get("pose_lengths")
            rgb = b.get("rgb_feats"); rlen = b.get("rgb_lengths")
            labels = b["labels"].to(device)
            opt.zero_grad()
            with autocast("cuda", enabled=scaler is not None):
                out = model(
                    pose_feats=pose.to(device) if pose is not None else None,
                    pose_lengths=plen.to(device) if plen is not None else None,
                    rgb_feats=rgb.to(device) if rgb is not None else None,
                    rgb_lengths=rlen.to(device) if rlen is not None else None,
                    labels=labels)
                loss = F.cross_entropy(out["logits"], labels, label_smoothing=0.1)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); n += 1
        sched.step()
        m = evaluate(model, val_dl, device, "val")
        print(f"Ep {epoch:3d}/{args.epochs} | loss={tot/max(n,1):.4f} | "
              f"val top-1={m['top1']:.2f}% top-5={m['top5']:.2f}% R@10={m['recall10']:.2f}% "
              f"| {time.time()-t0:.0f}s")
        if m["top1"] > best:
            best, patience = m["top1"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_top1": m["top1"], "args": vars(args)},
                       os.path.join(save_dir, "best.pt"))
            print(f"  ✓ best val top-1 {best:.2f}%")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}."); break

    ckpt = torch.load(os.path.join(save_dir, "best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_dl, device, "test")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"FUSION RESULT — streams={args.streams} size={args.size} seed={args.seed}")
    print(f"  params={n_params:,} classes={len(vocab)}")
    print(f"  TEST  top-1={test_m['top1']:.2f}%  top-5={test_m['top5']:.2f}%  R@10={test_m['recall10']:.2f}%")
    print(f"  baseline I3D: top-1=63.0%  R@10=91.0%  | target: top-1>75% R@10>95%")
    print(f"FUSION_RESULT: streams={args.streams} size={args.size} seed={args.seed} "
          f"params={n_params} n_classes={len(vocab)} "
          f"test_top1={test_m['top1']:.2f} test_top5={test_m['top5']:.2f} test_r10={test_m['recall10']:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--streams", default="pose+rgb", choices=["pose+rgb", "pose", "rgb"])
    p.add_argument("--size", default="base", choices=["base", "large"])
    p.add_argument("--data_root", default=None)
    p.add_argument("--rgb_dir", default=None, help="override rgb features dir (videomae or shubert)")
    p.add_argument("--save_dir", default="checkpoints_fusion")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.data_root is None:
        args.data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(args)
