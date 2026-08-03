"""
islr_model.py  —  ISLR model: Pose → Conformer → AttentionPool → ArcFace.

Architecture:
    (B, T, D_pose)
         │
    InputProj: Linear(D_pose → d_model) + LayerNorm + Dropout
         │
    ConformerEncoder: d_model, num_layers, num_heads, conv_kernel
         │
    AttentionPool: weighted sum over T → (B, d_model)
         │
    ArcFaceHead: cosine classifier with additive angular margin
         └── logits (B, num_classes)   — for cross-entropy during training
         └── embeddings (B, d_model)   — for recall@K retrieval at eval

ArcFace reference:
    Deng et al. "ArcFace: Additive Angular Margin Loss" CVPR 2019.
    Margin m=0.3, scale s=64 are the recommended defaults for large-class problems.

Why ArcFace over plain cross-entropy for 2731 classes:
    - Enforces a minimum angular margin between class centres in embedding space.
    - Produces more discriminative embeddings → better recall@K at inference.
    - +1–3% top-1 accuracy on large-class classification benchmarks.

Usage:
    model = ISLRModel(input_dim=261, d_model=256, num_layers=4,
                      num_classes=2731, dropout=0.2)
    out = model(features, lengths, labels=labels)  # training
    out = model(features, lengths)                  # inference (no margin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ConformerEncoder lives in models/conformer.py (already in your codebase).
# Import relative to the models package.
try:
    from models.conformer import ConformerEncoder
except ImportError:
    from conformer import ConformerEncoder


# ─── Attention pooling ────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    Soft attention pooling over the time axis.
    Learns a scalar "importance" score per frame and returns the weighted sum.

    Better than global mean pooling for isolated signs because:
    - The sign's core handshape often spans only a subset of frames.
    - Attention weights can focus on the most discriminative frames.

    Input : (B, T, D)
    Output: (B, D)
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn  = nn.Linear(d_model, 1)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, D)
        lengths: Optional[torch.Tensor] = None,   # (B,)
    ) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)          # (B, T)
        if lengths is not None:
            B, T, _ = x.shape
            mask = (
                torch.arange(T, device=x.device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )
            scores = scores.masked_fill(mask, -1e4)
        weights = torch.softmax(scores, dim=-1)    # (B, T)
        weights = self.drop(weights)
        return (weights.unsqueeze(-1) * x).sum(dim=1)   # (B, D)


# ─── ArcFace head ─────────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Loss head (ArcFace, Deng et al. CVPR 2019).

    During training  (labels provided):
        For each sample, adds angular margin `m` to the angle of the ground-truth
        class before re-scaling by `s`. This forces the network to produce
        embeddings that are at least m radians apart from wrong classes.

    During inference (labels=None):
        Returns raw cosine similarity logits × scale (no margin applied).
        Use argmax for top-1; use argsort for recall@K.

    Parameters (recommended defaults for 2731 classes):
        margin = 0.3  radians  (~17°)
        scale  = 64.0
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.3,
        scale: float  = 64.0,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.scale  = scale

        self.weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        features: torch.Tensor,            # (B, D)  — should be L2-normalised before
        labels: Optional[torch.Tensor] = None,  # (B,) — int class indices
    ) -> torch.Tensor:
        # L2-normalise both features and weight vectors
        normed_feat   = F.normalize(features, dim=-1)        # (B, D)
        normed_weight = F.normalize(self.weight, dim=-1)     # (C, D)
        cosine = F.linear(normed_feat, normed_weight)         # (B, C)

        if labels is None:
            # Inference: plain cosine similarity (no margin)
            return cosine * self.scale

        # Training: apply additive angular margin to ground-truth class only
        # cos(θ + m)  =  cos θ · cos m  −  sin θ · sin m
        theta       = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        target_cos  = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Replace GT cosine with margined value; keep others unchanged
        logits = self.scale * (one_hot * target_cos + (1.0 - one_hot) * cosine)
        return logits


# ─── Full ISLR model ──────────────────────────────────────────────────────────

class ISLRModel(nn.Module):
    """
    Pose-stream isolated sign language recognition model.

    Forward returns a dict:
        "logits"     : (B, num_classes)  — for cross-entropy loss (training)
        "embeddings" : (B, d_model)      — L2-normalised, for recall@K (eval)

    Typical call pattern:
        # Training
        out = model(features, lengths, labels=labels)
        loss = F.cross_entropy(out["logits"], labels)

        # Eval / inference
        out  = model(features, lengths)       # labels=None → no margin
        top1 = out["logits"].argmax(dim=-1)
        topK = out["logits"].argsort(dim=-1, descending=True)[:, :K]
    """

    def __init__(
        self,
        input_dim:    int   = 261,      # D_pose from extract_pose_islr.py
        d_model:      int   = 256,      # Conformer width
        num_heads:    int   = 8,        # attention heads (d_model must be divisible)
        num_layers:   int   = 4,        # Conformer depth
        conv_kernel:  int   = 15,       # depthwise conv kernel (odd)
        ff_expansion: int   = 4,        # feed-forward expansion ratio
        drop_path:    float = 0.1,      # stochastic depth
        dropout:      float = 0.2,      # general dropout
        num_classes:  int   = 2731,     # ASL-Citizen sign classes
        arc_margin:   float = 0.3,
        arc_scale:    float = 64.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        # Input projection: D_pose → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Temporal encoder
        self.conformer = ConformerEncoder(
            d_in        = d_model,
            d_model     = d_model,
            num_heads   = num_heads,
            num_layers  = num_layers,
            conv_kernel = conv_kernel,
            dropout     = dropout,
            ff_expansion= ff_expansion,
            drop_path   = drop_path,
        )

        # Temporal aggregation: (B, T, D) → (B, D)
        self.pool = AttentionPool(d_model, dropout=dropout)

        # Embedding normalisation before ArcFace
        self.embed_norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = ArcFaceHead(
            embedding_dim = d_model,
            num_classes   = num_classes,
            margin        = arc_margin,
            scale         = arc_scale,
        )

    def encode(
        self,
        features: torch.Tensor,             # (B, T, D_pose)
        lengths:  torch.Tensor,             # (B,)
    ) -> torch.Tensor:
        """
        Encode a batch of pose sequences to L2-normalised d_model embeddings.
        Useful for offline embedding extraction and retrieval.
        """
        x = self.input_proj(features)           # (B, T, d_model)
        x = self.conformer(x, lengths)          # (B, T, d_model)
        x = self.pool(x, lengths)               # (B, d_model)
        x = self.embed_norm(x)
        return F.normalize(x, dim=-1)           # (B, d_model)  L2-unit sphere

    def forward(
        self,
        features: torch.Tensor,             # (B, T, D_pose)
        lengths:  torch.Tensor,             # (B,)
        labels:   Optional[torch.Tensor] = None,  # (B,) int, or None for inference
    ) -> dict:
        embeddings = self.encode(features, lengths)        # (B, d_model)
        logits     = self.head(embeddings, labels)         # (B, num_classes)
        return {
            "logits":     logits,
            "embeddings": embeddings,
        }


# ─── Edge-distilled variant ───────────────────────────────────────────────────

class ISLRModelEdge(nn.Module):
    """
    Lightweight ISLR model for on-device inference (smart glasses / companion chip).
    ~1/4 the size of ISLRModel. No ArcFace at inference — just cosine logits.

    Target: <5 MB model size, real-time on CPU/NPU.
    """

    def __init__(
        self,
        input_dim:   int   = 261,
        d_model:     int   = 128,
        num_layers:  int   = 2,
        num_classes: int   = 2731,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.conformer = ConformerEncoder(
            d_in=d_model, d_model=d_model, num_heads=4, num_layers=num_layers,
            conv_kernel=15, dropout=dropout, ff_expansion=2, drop_path=0.0,
        )
        self.pool   = AttentionPool(d_model, dropout=0.0)
        self.head   = ArcFaceHead(d_model, num_classes, margin=0.3, scale=64.0)

    def forward(self, features, lengths, labels=None):
        x = self.input_proj(features)
        x = self.conformer(x, lengths)
        x = self.pool(x, lengths)
        x = F.normalize(x, dim=-1)
        return {"logits": self.head(x, labels), "embeddings": x}


# ─── Model factory ────────────────────────────────────────────────────────────

def build_islr_model(
    num_classes: int,
    input_dim:   int   = 261,
    size:        str   = "base",  # "base" | "large" | "edge"
    dropout:     float = 0.2,
) -> nn.Module:
    """
    Factory function — use this in train_islr.py.

    size="base"  : d_model=256, layers=4, ~7M params   (default, start here)
    size="large" : d_model=512, layers=6, ~28M params  (after base converges)
    size="edge"  : d_model=128, layers=2, ~1M params   (on-device)
    """
    if size == "edge":
        model = ISLRModelEdge(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    elif size == "large":
        model = ISLRModel(
            input_dim=input_dim, d_model=512, num_heads=8, num_layers=6,
            conv_kernel=15, dropout=dropout, num_classes=num_classes,
        )
    else:  # "base"
        model = ISLRModel(
            input_dim=input_dim, d_model=256, num_heads=8, num_layers=4,
            conv_kernel=15, dropout=dropout, num_classes=num_classes,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ISLRModel] size={size} | params={n_params:,} | "
          f"input_dim={input_dim} | num_classes={num_classes}")
    return model
