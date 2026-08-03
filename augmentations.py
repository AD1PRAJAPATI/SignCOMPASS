"""
augmentations.py  —  Temporal augmentations for ISLR pose-feature sequences.

dataset_islr.py uses:
    self.augmentor = VideoAugmentor(
        speed_perturb_prob=0.8, speed_min=0.7,  speed_max=1.3,
        frame_dropout_prob=0.3, frame_dropout_p=0.05,
        temporal_mask_prob=0.5, num_masks=2,    mask_ratio=0.1,
    )
    feat = self.augmentor(feat)        # feat: (T, D) float tensor

All augmentations operate on a single (T, D) sequence and return (T', D).
No horizontal flip — handedness is semantically meaningful in ASL.
"""
import random
import torch
import torch.nn.functional as F


class VideoAugmentor:
    """Stochastic temporal augmentation for (T, D) pose-feature sequences."""

    def __init__(
        self,
        speed_perturb_prob: float = 0.8,
        speed_min: float = 0.7,
        speed_max: float = 1.3,
        frame_dropout_prob: float = 0.3,
        frame_dropout_p: float = 0.05,
        temporal_mask_prob: float = 0.5,
        num_masks: int = 2,
        mask_ratio: float = 0.1,
    ) -> None:
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.frame_dropout_prob = frame_dropout_prob
        self.frame_dropout_p = frame_dropout_p
        self.temporal_mask_prob = temporal_mask_prob
        self.num_masks = num_masks
        self.mask_ratio = mask_ratio

    # ── individual ops ────────────────────────────────────────────────────────

    def speed_perturb(self, x: torch.Tensor) -> torch.Tensor:
        """Resample along time to simulate signing faster/slower. x: (T, D)."""
        T = x.shape[0]
        if T < 2:
            return x
        rate = random.uniform(self.speed_min, self.speed_max)
        new_T = max(2, int(round(T / rate)))
        # (T, D) -> (1, D, T) -> interpolate -> (new_T, D)
        xi = x.transpose(0, 1).unsqueeze(0)                 # (1, D, T)
        xi = F.interpolate(xi, size=new_T, mode="linear", align_corners=False)
        return xi.squeeze(0).transpose(0, 1).contiguous()   # (new_T, D)

    def frame_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out individual frames at random (occlusion/motion-blur robustness)."""
        T = x.shape[0]
        keep = (torch.rand(T, device=x.device) > self.frame_dropout_p).float()
        return x * keep.unsqueeze(-1)

    def temporal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """SpecAugment-style: zero a few contiguous time spans."""
        T = x.shape[0]
        x = x.clone()
        span = max(1, int(T * self.mask_ratio))
        for _ in range(self.num_masks):
            if T - span <= 0:
                break
            start = random.randint(0, T - span)
            x[start:start + span] = 0.0
        return x

    # ── pipeline ──────────────────────────────────────────────────────────────

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.speed_perturb_prob:
            x = self.speed_perturb(x)
        if random.random() < self.frame_dropout_prob:
            x = self.frame_dropout(x)
        if random.random() < self.temporal_mask_prob:
            x = self.temporal_mask(x)
        return x
