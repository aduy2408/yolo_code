# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Partial Convolution (PConv) modules for efficient representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class PConv(nn.Module):
    """Partial Convolution (PConv) layer for reducing redundant computations."""

    def __init__(self, dim: int, n_div: int = 4):
        """Initialize PConv with channel dimension and division ratio."""
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of PConv slicing and applying conv on subset of channels."""
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        return torch.cat((x1, x2), dim=1)


class FasterNetBlock(nn.Module):
    """FasterNet block using PConv followed by pointwise convolutions."""

    def __init__(self, c: int, n_div: int = 4, shortcut: bool = True):
        """Initialize block with PConv, channel expansion and projection Conv layers."""
        super().__init__()
        self.pconv = PConv(c, n_div=n_div)
        self.conv1 = Conv(c, 2 * c, 1, 1)  # Pointwise Conv (expansion)
        self.conv2 = Conv(2 * c, c, 1, 1, act=False)  # Pointwise Conv (projection without activation)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FasterNet block logic with residual connection if enabled."""
        if self.add:
            return x + self.conv2(self.conv1(self.pconv(x)))
        return self.conv2(self.conv1(self.pconv(x)))


class C2f_PConv(nn.Module):
    """C2f module using FasterNet Blocks (PConv-based Bottlenecks)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, n_div: int = 4):
        """Initialize C2f_PConv layer."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(FasterNetBlock(self.c, n_div=n_div, shortcut=shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f_PConv layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Split-based forward pass for C2f_PConv."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
