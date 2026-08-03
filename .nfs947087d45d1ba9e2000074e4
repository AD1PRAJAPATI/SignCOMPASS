"""
dataset_islr.py  —  ASL-Citizen DataLoader for Isolated Sign Language Recognition.

ASL-Citizen: 83k videos, 2,731 signs, multiple signers.
Source: https://huggingface.co/datasets/google/asl_citizen

Expected layout (relative to data_root):
    data/asl_citizen/
        metadata.csv              — video_id, gloss, participant_id, split
        pose_features/            — (T, 261) float16 tensors named <video_id>.pt

metadata.csv required columns:
    video_id        : unique video identifier  (matches pose feature filename stem)
    gloss           : sign label string
    participant_id  : signer ID (used for signer-independent evaluation)
    split           : 'train' | 'val' | 'test'  (pre-defined signer-independent)

__getitem__ returns:
    features    : FloatTensor (T, D_pose=261)
    label       : int
    uid         : str
    signer_id   : str

Augmentation (train only): speed_perturb, frame_dropout, temporal_mask.
NO horizontal flip — it reverses handedness and corrupts sign identity.
"""

import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from augmentations import VideoAugmentor


# ─── Vocabulary ───────────────────────────────────────────────────────────────

class GlossVocabISLR:
    """
    Label encoder/decoder for isolated signs (classification, not CTC).
    No blank token — this is not a sequence-to-sequence problem.
    """

    def __init__(self) -> None:
        self._gloss2id: Dict[str, int] = {}
        self._id2gloss: List[str] = []

    def build(self, glosses: List[str]) -> None:
        """Build vocab from a list of gloss strings (one per training sample)."""
        unique = sorted(set(glosses))
        self._id2gloss = unique
        self._gloss2id = {g: i for i, g in enumerate(unique)}
        print(f"[GlossVocabISLR] Built: {len(unique)} classes")

    def encode(self, gloss: str) -> int:
        return self._gloss2id[gloss]

    def decode(self, idx: int) -> str:
        return self._id2gloss[idx]

    def __len__(self) -> int:
        return len(self._id2gloss)

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self._id2gloss, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GlossVocabISLR":
        import json
        v = cls()
        with open(path) as f:
            v._id2gloss = json.load(f)
        v._gloss2id = {g: i for i, g in enumerate(v._id2gloss)}
        return v


# ─── CSV parsing & split logic ────────────────────────────────────────────────

def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load ASL-Citizen metadata CSV.
    Normalizes column names to: video_id, gloss, participant_id, split.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names to handle minor variations across download methods
    col_map = {}
    for col in df.columns:
        lc = col.lower().strip().replace(" ", "_").replace("-", "_")
        if lc in ("video_id", "id", "file_id", "filename", "clip_id"):
            col_map[col] = "video_id"
        elif lc in ("gloss", "label", "sign", "word", "class"):
            col_map[col] = "gloss"
        elif lc in ("participant_id", "signer_id", "signer", "participant", "subject"):
            col_map[col] = "participant_id"
        elif lc in ("split", "subset", "partition", "set"):
            col_map[col] = "split"
    df = df.rename(columns=col_map)

    required = {"video_id", "gloss", "participant_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"metadata.csv missing required columns: {missing}. "
            f"Found: {list(df.columns)}\n"
            f"Run: python download_asl_citizen.py --verify to inspect the CSV."
        )

    df["video_id"]       = df["video_id"].astype(str).str.strip()
    df["gloss"]          = df["gloss"].astype(str).str.strip().str.upper()
    df["participant_id"] = df["participant_id"].astype(str).str.strip()

    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()
    else:
        df["split"] = "unknown"

    print(f"[metadata] Loaded {len(df)} rows | "
          f"{df['gloss'].nunique()} glosses | "
          f"{df['participant_id'].nunique()} signers")
    return df


def make_signer_splits(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign entire signers to train/val/test (signer-independent split).
    Called when the pre-defined split column is missing or degenerate.
    CRITICAL: val and test sets must contain signers never seen in training.
    """
    signers = sorted(df["participant_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(signers)
    n         = len(signers)
    n_train   = int(n * train_frac)
    n_val     = int(n * val_frac)
    train_s   = set(signers[:n_train])
    val_s     = set(signers[n_train : n_train + n_val])

    def assign(pid):
        if pid in train_s: return "train"
        if pid in val_s:   return "val"
        return "test"

    df = df.copy()
    df["split"] = df["participant_id"].map(assign)
    print(f"[splits] Created signer-independent splits: "
          f"train={len(train_s)} | val={len(val_s)} | "
          f"test={n - n_train - n_val} signers")
    return df


def get_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Return df with a valid 'split' column covering train/val/test.
    Prefers pre-defined splits; falls back to signer-independent auto-split.
    """
    unique_splits = set(df["split"].unique())
    needed = {"train", "val", "test"}
    if needed.issubset(unique_splits):
        counts = df["split"].value_counts().to_dict()
        print(f"[splits] Using pre-defined splits: "
              f"train={counts.get('train',0)} | val={counts.get('val',0)} | "
              f"test={counts.get('test',0)}")
        return df
    else:
        print(f"[splits] Pre-defined splits not found ({unique_splits}); "
              f"creating signer-independent splits.")
        return make_signer_splits(df, seed=seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ASLCitizenDataset(Dataset):
    """
    Isolated sign dataset.
    Loads pre-extracted pose features (T, D_pose) from disk.

    augment=True applies: speed_perturb + frame_dropout + temporal_mask.
    No horizontal flip augmentation — ASL handedness is semantically meaningful.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: GlossVocabISLR,
        features_dir: str,
        max_frames: int = 64,
        augment: bool = False,
    ) -> None:
        self.vocab        = vocab
        self.features_dir = features_dir
        self.max_frames   = max_frames
        self.augment      = augment

        self.augmentor = VideoAugmentor(
            speed_perturb_prob=0.8, speed_min=0.7,  speed_max=1.3,
            frame_dropout_prob=0.3, frame_dropout_p=0.05,
            temporal_mask_prob=0.5, num_masks=2,    mask_ratio=0.1,
        ) if augment else None

        # Filter out samples with missing pose feature files
        present = [
            i for i, row in df.iterrows()
            if os.path.exists(self._feat_path(row["video_id"]))
        ]
        if len(present) < len(df):
            print(f"[Dataset] WARNING: {len(df) - len(present)} feature files missing "
                  f"out of {len(df)} — run extract_pose_islr.py first.")
        self.df = df.loc[present].reset_index(drop=True)

    def _feat_path(self, video_id: str) -> str:
        """Support both <id>.pt and <id> (bare filename) conventions."""
        with_ext = os.path.join(self.features_dir, video_id + ".pt")
        bare     = os.path.join(self.features_dir, video_id)
        return with_ext if os.path.exists(with_ext) else bare

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        uid   = row["video_id"]
        label = self.vocab.encode(row["gloss"])

        feat = torch.load(self._feat_path(uid),
                          map_location="cpu", weights_only=True).float()
        # feat: (T, D_pose)  — float16 on disk, loaded as float32

        # Temporal crop for very long clips
        if feat.shape[0] > self.max_frames:
            if self.augment:
                start = random.randint(0, feat.shape[0] - self.max_frames)
                feat  = feat[start : start + self.max_frames]
            else:
                # Center crop for deterministic eval
                mid   = feat.shape[0] // 2
                half  = self.max_frames // 2
                feat  = feat[max(0, mid - half) : mid - half + self.max_frames]

        if self.augmentor is not None:
            feat = self.augmentor(feat)

        return {
            "features":  feat,
            "label":     label,
            "uid":       uid,
            "signer_id": str(row.get("participant_id", "")),
        }


# ─── Collate ──────────────────────────────────────────────────────────────────

def collate_fn_islr(batch):
    """
    Pads variable-length feature sequences to the longest in the batch.

    Returns:
        features : (B, T_max, D)  — zero-padded
        labels   : (B,)           — class indices
        lengths  : (B,)           — true sequence lengths before padding
        uids     : list[str]      — video IDs
    """
    max_T = max(b["features"].shape[0] for b in batch)
    D     = batch[0]["features"].shape[1]

    features, labels, lengths, uids = [], [], [], []
    for b in batch:
        T   = b["features"].shape[0]
        pad = torch.zeros(max_T - T, D, dtype=b["features"].dtype)
        features.append(torch.cat([b["features"], pad], dim=0))
        labels.append(b["label"])
        lengths.append(T)
        uids.append(b["uid"])

    return {
        "features": torch.stack(features),                    # (B, T_max, D)
        "labels":   torch.tensor(labels,  dtype=torch.long),  # (B,)
        "lengths":  torch.tensor(lengths, dtype=torch.long),  # (B,)
        "uids":     uids,
    }


# ─── Class-balanced sampler ───────────────────────────────────────────────────

def make_class_balanced_sampler(
    df: pd.DataFrame, vocab: GlossVocabISLR
) -> WeightedRandomSampler:
    """
    WeightedRandomSampler that gives each class equal probability in expectation.
    Use this for the training DataLoader to handle any long-tail imbalance.
    ASL-Citizen is fairly balanced (~31 clips/sign avg), but balance is never perfect.
    """
    labels  = [vocab.encode(g) for g in df["gloss"]]
    counts  = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ─── Top-level builder ────────────────────────────────────────────────────────

def build_datasets(
    data_root: str,
    metadata_csv: Optional[str] = None,
    features_dir: Optional[str] = None,
    max_frames: int = 64,
    seed: int = 42,
) -> Tuple[ASLCitizenDataset, ASLCitizenDataset, ASLCitizenDataset, GlossVocabISLR]:
    """
    One-call helper: load metadata → build vocab → split → create datasets.

    Args:
        data_root    : project root (parent of data/)
        metadata_csv : override path to metadata.csv
        features_dir : override path to pose_features/
        max_frames   : max frames per clip (clips longer than this are cropped)
        seed         : random seed for auto-splits

    Returns: (train_ds, val_ds, test_ds, vocab)
    """
    asl_dir  = os.path.join(data_root, "data", "asl_citizen")
    csv_path = metadata_csv or os.path.join(asl_dir, "metadata.csv")
    feat_dir = features_dir or os.path.join(asl_dir, "pose_features")

    df = load_metadata(csv_path)
    df = get_splits(df, seed=seed)

    # Build vocab from training glosses only — val/test OOV signs are dropped
    train_glosses = df[df["split"] == "train"]["gloss"].tolist()
    vocab = GlossVocabISLR()
    vocab.build(train_glosses)

    # Drop val/test rows whose gloss never appeared in training
    known  = set(vocab._gloss2id.keys())
    before = len(df)
    df     = df[df["gloss"].isin(known)].reset_index(drop=True)
    if before - len(df):
        print(f"[build_datasets] Dropped {before - len(df)} OOV rows from val/test")

    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        print(f"  {s:5s}: {len(sub):6d} clips | "
              f"{sub['gloss'].nunique():5d} classes | "
              f"{sub['participant_id'].nunique():4d} signers")

    train_ds = ASLCitizenDataset(
        df[df["split"] == "train"], vocab, feat_dir, max_frames=max_frames, augment=True)
    val_ds   = ASLCitizenDataset(
        df[df["split"] == "val"],   vocab, feat_dir, max_frames=max_frames, augment=False)
    test_ds  = ASLCitizenDataset(
        df[df["split"] == "test"],  vocab, feat_dir, max_frames=max_frames, augment=False)

    return train_ds, val_ds, test_ds, vocab
