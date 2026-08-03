"""
dataset_fusion.py  —  Dual-stream (pose + RGB) dataset for FusionISLRModel.

Loads, per video_id:
    pose features : pose_features/<id>.pt   (T_p, 261)   from extract_pose_islr.py
    rgb  features : rgb_features/<id>.pt     (T_r, 768)   from extract_videomae.py
                                                          OR converted SHuBERT (T,768)

The two streams are pooled INDEPENDENTLY inside FusionISLRModel, so they need
NOT share a time axis — we just return both with their own lengths.

Reuses vocab / split / sampler logic from dataset_islr.py so behaviour matches
the pose-only baseline exactly (same classes, same signer splits).
"""
import os
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset_islr import (
    GlossVocabISLR, load_metadata, get_splits, make_class_balanced_sampler,
)
from augmentations import VideoAugmentor


class FusionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: GlossVocabISLR,
        pose_dir: str,
        rgb_dir: str,
        use_pose: bool = True,
        use_rgb: bool = True,
        max_frames: int = 64,
        augment: bool = False,
    ) -> None:
        assert use_pose or use_rgb
        self.vocab = vocab
        self.pose_dir = pose_dir
        self.rgb_dir = rgb_dir
        self.use_pose = use_pose
        self.use_rgb = use_rgb
        self.max_frames = max_frames
        self.augment = augment
        self.aug = VideoAugmentor() if augment else None

        # Keep only rows whose required feature files exist AND load cleanly
        # (truncated .pt files raise EOFError mid-epoch otherwise)
        bad_pose = 0

        def _readable_pt(path: str) -> bool:
            nonlocal bad_pose
            try:
                if os.path.getsize(path) < 64:
                    bad_pose += 1
                    return False
                torch.load(path, map_location="cpu", weights_only=True)
                return True
            except Exception:
                bad_pose += 1
                return False

        def ok(vid):
            if use_pose:
                pp = self._p(pose_dir, vid)
                if not os.path.exists(pp) or not _readable_pt(pp):
                    return False
            if use_rgb:
                rp = self._p(rgb_dir, vid)
                if not os.path.exists(rp) or not _readable_pt(rp):
                    return False
            return True

        keep = [pos for pos, (_, r) in enumerate(df.iterrows()) if ok(r["video_id"])]
        if len(keep) < len(df):
            print(f"[FusionDataset] {len(df)-len(keep)} rows dropped (missing/corrupt pose/rgb features)")
        if bad_pose:
            print(f"[FusionDataset] unreadable .pt files skipped: {bad_pose}")
        self.df = df.iloc[keep].reset_index(drop=True)

    @staticmethod
    def _p(d, vid):
        a = os.path.join(d, vid + ".pt")
        return a if os.path.exists(a) else os.path.join(d, vid)

    def __len__(self):
        return len(self.df)

    def _load(self, d, vid):
        x = torch.load(self._p(d, vid), map_location="cpu", weights_only=True).float()
        # temporal crop
        if x.shape[0] > self.max_frames:
            if self.augment:
                import random
                s = random.randint(0, x.shape[0] - self.max_frames)
                x = x[s:s + self.max_frames]
            else:
                mid = x.shape[0] // 2
                half = self.max_frames // 2
                x = x[max(0, mid - half): max(0, mid - half) + self.max_frames]
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = row["video_id"]
        item = {"label": self.vocab.encode(row["gloss"]), "uid": vid}

        if self.use_pose:
            p = self._load(self.pose_dir, vid)
            if self.aug is not None:
                p = self.aug(p)
            item["pose"] = p
        if self.use_rgb:
            r = self._load(self.rgb_dir, vid)
            # light augmentation on rgb too (temporal only)
            if self.aug is not None:
                r = self.aug(r)
            item["rgb"] = r
        return item


def collate_fusion(batch):
    """Pad pose and rgb streams independently; return per-stream lengths."""
    out = {"labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
           "uids": [b["uid"] for b in batch]}

    for key, feat_key, len_key in (("pose", "pose_feats", "pose_lengths"),
                                   ("rgb",  "rgb_feats",  "rgb_lengths")):
        if key in batch[0]:
            maxT = max(b[key].shape[0] for b in batch)
            D = batch[0][key].shape[1]
            feats, lens = [], []
            for b in batch:
                T = b[key].shape[0]
                pad = torch.zeros(maxT - T, D, dtype=b[key].dtype)
                feats.append(torch.cat([b[key], pad], dim=0))
                lens.append(T)
            out[feat_key] = torch.stack(feats)
            out[len_key] = torch.tensor(lens, dtype=torch.long)
    return out


def build_fusion_datasets(
    data_root: str,
    use_pose: bool = True,
    use_rgb: bool = True,
    pose_dir: Optional[str] = None,
    rgb_dir: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    max_frames: int = 64,
    seed: int = 42,
) -> Tuple[FusionDataset, FusionDataset, FusionDataset, GlossVocabISLR]:
    asl = os.path.join(data_root, "data", os.environ.get("ISLR_DATASET","asl_citizen"))
    csv_path = metadata_csv or os.path.join(asl, "metadata.csv")
    pose_dir = pose_dir or os.path.join(asl, "pose_features")
    rgb_dir = rgb_dir or os.path.join(asl, "rgb_features")

    df = load_metadata(csv_path)
    df = get_splits(df, seed=seed)
    vocab = GlossVocabISLR()
    vocab.build(df[df["split"] == "train"]["gloss"].tolist())

    known = set(vocab._gloss2id.keys())
    df = df[df["gloss"].isin(known)].reset_index(drop=True)

    def mk(split, aug):
        return FusionDataset(df[df["split"] == split], vocab, pose_dir, rgb_dir,
                             use_pose=use_pose, use_rgb=use_rgb,
                             max_frames=max_frames, augment=aug)

    train_ds, val_ds, test_ds = mk("train", True), mk("val", False), mk("test", False)
    for s, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        print(f"  {s:5s}: {len(ds):6d} clips")
    return train_ds, val_ds, test_ds, vocab
