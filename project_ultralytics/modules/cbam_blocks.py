"""Project-specific Ultralytics blocks using CBAM refinement."""

from __future__ import annotations

import torch

from ultralytics.nn.modules.block import C2f, C3

from .cbam import CBAM

class C2fCBAM(C2f):
    """C2f block followed by CBAM channel and spatial attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        kernel_size: int = 7,
    ):
        """Initialize C2f with CBAM refinement on the output feature map."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attn = CBAM(c2, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C2f and refine its output with CBAM attention."""
        return self.attn(super().forward(x))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Apply split-based C2f and refine its output with CBAM attention."""
        return self.attn(super().forward_split(x))


class C3CBAM(C3):
    """C3 block followed by CBAM channel and spatial attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        kernel_size: int = 7,
    ):
        """Initialize C3 with CBAM refinement on the output feature map."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attn = CBAM(c2, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3 and refine its output with CBAM attention."""
        return self.attn(super().forward(x))
