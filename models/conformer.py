"""
conformer.py — Conformer encoder block.

Architecture per block (Gulati et al. 2020):
    x = FF1(x)           half-step feed-forward
    x = MHSA(x)          multi-head self-attention
    x = Conv(x)          depthwise convolution module
    x = FF2(x)           half-step feed-forward
    x = LayerNorm(x)
"""
import torch
import torch.nn as nn
from typing import Optional


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(self.norm(x))


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd"
        self.norm = nn.LayerNorm(d_model)
        self.pw1   = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu   = nn.GLU(dim=1)
        self.dw    = nn.Conv1d(d_model, d_model, kernel,
                               padding=kernel // 2, groups=d_model)
        self.bn    = nn.BatchNorm1d(d_model)
        self.act   = nn.SiLU()
        self.pw2   = nn.Conv1d(d_model, d_model, 1)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        res = x
        x = self.norm(x).transpose(1, 2)   # (B, D, T)
        x = self.glu(self.pw1(x))          # (B, D, T)
        x = self.act(self.bn(self.dw(x)))  # (B, D, T)
        x = self.drop(self.pw2(x))         # (B, D, T)
        return res + x.transpose(1, 2)     # (B, T, D)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        return res + self.drop(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        ff_expansion: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.ff1  = FeedForward(d_model, ff_expansion, dropout)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2  = FeedForward(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop_path_prob = drop_path

    def _drop_path(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.drop_path_prob > 0.0 and self.training:
            keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.drop_path_prob
            return residual + x * keep.float()
        return residual + x

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x, key_padding_mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    """
    Stack of ConformerBlocks.
    Input:  (B, T, d_in)
    Output: (B, T, d_model)
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        ff_expansion: int = 4,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        # Stochastic depth: linearly increase drop_path per layer
        dpr = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel, dropout, ff_expansion, dpr[i])
            for i in range(num_layers)
        ])
        self.out_dim = d_model

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d_in)
        lengths: Optional[torch.Tensor] = None,   # (B,)
    ) -> torch.Tensor:
        x = self.proj(x)                          # (B, T, d_model)

        # Build key_padding_mask from lengths
        mask = None
        if lengths is not None:
            B, T, _ = x.shape
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, mask)
        return x                                  # (B, T, d_model)
