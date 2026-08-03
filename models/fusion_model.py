"""
fusion_model.py  —  Two-stream ISLR fusion model: Pose + SHuBERT → ArcFace.

Architecture:
                    ┌─────────────────────────┐
    Pose (T,261) ──►│ Conformer → AttnPool(A) │──►(A,)
                    └─────────────────────────┘       │
                                                       ▼ concat
                    ┌─────────────────────────┐       │
 SHuBERT (T,768) ──►│ Linear → AttnPool(A)   │──►(A,)
                    └─────────────────────────┘       │
                                                       ▼
                              LayerNorm(2A) → ArcFace(2731)
                                                       │
                              Phonological aux heads:
                                Linear → handshape logits
                                Linear → location  logits
                                Linear → movement  logits

Training loss:
    L_total = L_arcface  +  w_aux * (L_hs + L_loc + L_mov)

At inference: argmax(ArcFace logits) for top-1; argsort for recall@K.

The fusion head can also run with only one stream present (stream_pose_only=True
or stream_rgb_only=True) — useful for ablation and fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from models.conformer import ConformerEncoder
    from models.islr_model import AttentionPool, ArcFaceHead
except ImportError:
    from conformer import ConformerEncoder
    from islr_model import AttentionPool, ArcFaceHead


# ─── Per-stream encoders ──────────────────────────────────────────────────────

class PoseStream(nn.Module):
    """
    Pose keypoints (T, D_pose) → Conformer → AttentionPool → (embed_dim,)
    """
    def __init__(self, input_dim=261, d_model=256, num_heads=8,
                 num_layers=4, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.conformer = ConformerEncoder(
            d_in=d_model, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, conv_kernel=15,
            dropout=dropout, ff_expansion=4, drop_path=0.1,
        )
        self.pool = AttentionPool(d_model, dropout=dropout)
        self.out_dim = d_model

    def forward(self, x, lengths):
        # x: (B, T, D_pose)
        x = self.proj(x)
        x = self.conformer(x, lengths)
        return self.pool(x, lengths)   # (B, d_model)


class RGBStream(nn.Module):
    """SHuBERT embeddings (T,768) -> proj -> Conformer -> AttentionPool -> (d_model,)."""
    def __init__(self, input_dim=768, d_model=256, num_layers=2, num_heads=8, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.conformer = ConformerEncoder(d_in=d_model, d_model=d_model, num_heads=num_heads,
                                          num_layers=num_layers, conv_kernel=15, dropout=dropout,
                                          ff_expansion=4, drop_path=0.1)
        self.pool = AttentionPool(d_model, dropout=dropout)
        self.out_dim = d_model
    def forward(self, x, lengths):
        x = self.proj(x)
        x = self.conformer(x, lengths)
        return self.pool(x, lengths)


# ─── Phonological auxiliary heads ────────────────────────────────────────────

class PhonologicalHeads(nn.Module):
    """
    Three lightweight heads that predict phonological features from the
    fused embedding. Used as auxiliary losses during training only.

    n_handshapes, n_locations, n_movements come from phonological_vocab.json.
    """
    def __init__(self, input_dim: int, n_handshapes: int,
                 n_locations: int, n_movements: int, dropout: float = 0.1):
        super().__init__()
        self.hs  = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, n_handshapes))
        self.loc = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, n_locations))
        self.mov = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, n_movements))

    def forward(self, x):
        # x: (B, input_dim)
        return {
            "handshape": self.hs(x),   # (B, n_handshapes)
            "location":  self.loc(x),  # (B, n_locations)
            "movement":  self.mov(x),  # (B, n_movements)
        }


# ─── Fusion model ─────────────────────────────────────────────────────────────

class FusionISLRModel(nn.Module):
    """
    Two-stream fusion ISLR model.

    Can operate in three modes controlled at construction:
        use_pose=True,  use_rgb=True   → full fusion   (default)
        use_pose=True,  use_rgb=False  → pose-only     (baseline / ablation)
        use_pose=False, use_rgb=True   → rgb-only      (SHuBERT-only ablation)

    forward() signature:
        pose_feats  : (B, T, 261) or None
        pose_lengths: (B,) or None
        rgb_feats   : (B, T, 768) or None
        rgb_lengths : (B,) or None
        labels      : (B,) int, or None for inference

    Returns dict:
        "logits"     : (B, num_classes)
        "embeddings" : (B, fusion_dim)  L2-normalised
        "phon_logits": dict with "handshape", "location", "movement"  (or None)
    """

    def __init__(
        self,
        num_classes:   int   = 2731,
        pose_dim:      int   = 261,
        rgb_dim:       int   = 768,
        d_model:       int   = 256,    # per-stream embedding dim; fusion = 2 × d_model
        pose_layers:   int   = 4,
        pose_heads:    int   = 8,
        dropout:       float = 0.2,
        arc_margin:    float = 0.3,
        arc_scale:     float = 64.0,
        use_pose:      bool  = True,
        use_rgb:       bool  = True,
        # Phonological aux heads (set to None to disable)
        n_handshapes:  Optional[int] = None,
        n_locations:   Optional[int] = None,
        n_movements:   Optional[int] = None,
    ):
        super().__init__()
        assert use_pose or use_rgb, "At least one stream must be enabled."
        self.use_pose = use_pose
        self.use_rgb  = use_rgb

        fusion_dim = 0
        if use_pose:
            self.pose_stream = PoseStream(
                input_dim=pose_dim, d_model=d_model,
                num_heads=pose_heads, num_layers=pose_layers, dropout=dropout,
            )
            fusion_dim += d_model

        if use_rgb:
            self.rgb_stream = RGBStream(
                input_dim=rgb_dim, d_model=d_model, dropout=dropout,
            )
            fusion_dim += d_model

        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.head = ArcFaceHead(
            embedding_dim=fusion_dim,
            num_classes=num_classes,
            margin=arc_margin,
            scale=arc_scale,
        )

        # Phonological aux heads (optional)
        self.phon_heads = None
        if n_handshapes and n_locations and n_movements:
            self.phon_heads = PhonologicalHeads(
                input_dim=fusion_dim,
                n_handshapes=n_handshapes,
                n_locations=n_locations,
                n_movements=n_movements,
                dropout=dropout,
            )

        self.fusion_dim = fusion_dim
        self.aux_pose = nn.Linear(d_model, num_classes) if (use_pose and use_rgb) else None
        self.aux_rgb  = nn.Linear(d_model, num_classes) if (use_pose and use_rgb) else None

    def encode(self, pose_feats=None, pose_lengths=None,
               rgb_feats=None, rgb_lengths=None):
        """
        Encode both streams to a single L2-normalised fusion embedding.
        Returns (B, fusion_dim).
        """
        parts = []
        if self.use_pose and pose_feats is not None:
            parts.append(self.pose_stream(pose_feats, pose_lengths))
        if self.use_rgb and rgb_feats is not None:
            parts.append(self.rgb_stream(rgb_feats, rgb_lengths))

        if not parts:
            raise ValueError("No stream features provided to FusionISLRModel.encode()")

        fused = torch.cat(parts, dim=-1)   # (B, fusion_dim)
        fused = self.fusion_norm(fused)
        return F.normalize(fused, dim=-1)  # (B, fusion_dim)  unit sphere

    def forward(self, pose_feats=None, pose_lengths=None,
                rgb_feats=None, rgb_lengths=None, labels=None):
        embeddings = self.encode(pose_feats, pose_lengths, rgb_feats, rgb_lengths)
        logits     = self.head(embeddings, labels)

        phon_logits = None
        if self.phon_heads is not None:
            phon_logits = self.phon_heads(embeddings)

        return {
            "logits":      logits,
            "embeddings":  embeddings,
            "phon_logits": phon_logits,
        }

    def forward_cotrain(self, pose_feats=None, pose_lengths=None,
                        rgb_feats=None, rgb_lengths=None, labels=None):
        p = self.pose_stream(pose_feats, pose_lengths)
        r = self.rgb_stream(rgb_feats, rgb_lengths)
        fused = F.normalize(self.fusion_norm(torch.cat([p, r], dim=-1)), dim=-1)
        return {"logits": self.head(fused, labels),
                "aux_pose": self.aux_pose(p), "aux_rgb": self.aux_rgb(r),
                "embeddings": fused}


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_fusion_model(
    num_classes:  int,
    pose_dim:     int  = 261,
    rgb_dim:      int  = 768,
    phon_vocab:   Optional[dict] = None,   # from phonological_vocab.json
    size:         str  = "base",           # "base" | "large"
    use_pose:     bool = True,
    use_rgb:      bool = True,
    dropout:      float = 0.2,
) -> FusionISLRModel:
    """
    Factory function used by train_fusion.py.

    phon_vocab: dict loaded from phonological_vocab.json, e.g.
        { "handshape": {"n_classes": 18}, "location": {"n_classes": 8}, ... }
    """
    d_model     = 256 if size == "base" else 512
    pose_layers = 4   if size == "base" else 6
    pose_heads  = 8   if size == "base" else 16

    n_hs  = phon_vocab["handshape"]["n_classes"] if phon_vocab else None
    n_loc = phon_vocab["location"]["n_classes"]  if phon_vocab else None
    n_mov = phon_vocab["movement"]["n_classes"]  if phon_vocab else None

    model = FusionISLRModel(
        num_classes=num_classes,
        pose_dim=pose_dim,   rgb_dim=rgb_dim,
        d_model=d_model,     pose_layers=pose_layers, pose_heads=pose_heads,
        dropout=dropout,
        use_pose=use_pose,   use_rgb=use_rgb,
        n_handshapes=n_hs,   n_locations=n_loc,  n_movements=n_mov,
    )

    streams = []
    if use_pose: streams.append("pose")
    if use_rgb:  streams.append("rgb")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FusionISLRModel] streams={'+'.join(streams)} | size={size} | "
          f"fusion_dim={model.fusion_dim} | params={n_params:,} | "
          f"phon_aux={'yes' if phon_vocab else 'no'}")
    return model
