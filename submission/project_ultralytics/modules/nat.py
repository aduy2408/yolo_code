"""Project-specific Neighborhood Attention blocks.

The optional ``natten`` dependency is imported only when a NAT module is
instantiated, keeping the project package importable in minimal environments.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv


def _choose_attention_heads(channels: int, requested_heads: int) -> int:
    """Pick a valid attention head count that divides channels."""
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1

class C2fNAT(nn.Module):
    """C2f block with gated Neighborhood Attention on the final hidden feature."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        num_heads: int = 4,
        kernel_size: int = 3,
    ):
        """Initialize a C2f-style block with a lightweight NAT refinement branch."""
        super().__init__()
        try:
            from natten import NeighborhoodAttention2D
        except ImportError as exc:
            raise ImportError(
                "C2fNAT requires the 'natten' package. Install natten in the training environment before using "
                "YAMLs that reference C2fNAT."
            ) from exc

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.num_heads = _choose_attention_heads(self.c, num_heads)
        self.norm1 = nn.LayerNorm(self.c)
        self.attn = NeighborhoodAttention2D(dim=self.c, num_heads=self.num_heads, kernel_size=int(kernel_size))
        self.norm2 = nn.LayerNorm(self.c)
        self.mlp = nn.Sequential(
            nn.Linear(self.c, 2 * self.c),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * self.c, self.c),
            nn.Dropout(0.1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def _refine_last(self, x: torch.Tensor) -> torch.Tensor:
        """Refine one hidden split with NHWC Neighborhood Attention."""
        if x.device.type == "cpu" and x.requires_grad:
            return x
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        att = self.attn(self.norm1(x_nhwc)) + x_nhwc
        refined = self.mlp(self.norm2(att)) + att
        refined = refined.permute(0, 3, 1, 2).contiguous()
        return x + self.gamma * (refined - x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f with NAT refinement on the final hidden feature."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self._refine_last(y[-1])
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self._refine_last(y[-1])
        return self.cv2(torch.cat(y, 1))


class NATBlock(nn.Module):
    """Neighborhood Attention Transformer block for YOLO."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, kernel_size: int = 7):
        """Initialize a Neighborhood Attention block."""
        super().__init__()
        try:
            from natten import NeighborhoodAttention2D
        except ImportError as exc:
            raise ImportError(
                "NATBlock requires the 'natten' package. Install natten in the training environment before using "
                "YAMLs that reference NATBlock."
            ) from exc

        self.c = c1
        self.num_heads = _choose_attention_heads(c1, num_heads)
        self.norm1 = nn.LayerNorm(c1)
        self.attn = NeighborhoodAttention2D(dim=c1, num_heads=self.num_heads, kernel_size=int(kernel_size))
        self.norm2 = nn.LayerNorm(c1)
        self.mlp = nn.Sequential(
            nn.Linear(c1, 2 * c1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * c1, c1),
            nn.Dropout(0.1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through NATBlock."""
        if x.device.type == "cpu" and x.requires_grad:
            return self.proj(x)
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        att = self.attn(self.norm1(x_nhwc)) + x_nhwc
        refined = self.mlp(self.norm2(att)) + att
        refined = refined.permute(0, 3, 1, 2).contiguous()
        out = x + self.gamma * (refined - x)
        return self.proj(out)
