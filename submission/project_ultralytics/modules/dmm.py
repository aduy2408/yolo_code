"""DMM-inspired directional refinement blocks for small-object P2 features."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class DMMRefine(nn.Module):
    """Directional 3x5/5x3 residual refinement, not the full DMM paper block."""

    def __init__(self, c1: int, reduction: int = 2, alpha_init: float = 0.1):
        super().__init__()
        c_hidden = max(int(c1) // int(reduction), 16)
        self.reduce = Conv(c1, c_hidden, k=1, s=1)
        self.conv_h = Conv(c_hidden, c_hidden, k=(3, 5), s=1, p=(1, 2), g=1)
        self.conv_v = Conv(c_hidden, c_hidden, k=(5, 3), s=1, p=(2, 1), g=1)
        self.fuse = Conv(2 * c_hidden, c1, k=1, s=1, act=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        return self.fuse(torch.cat((self.conv_h(z), self.conv_v(z)), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self._residual(x)


class DMMGatedRefine(DMMRefine):
    """DMMRefine followed by a lightweight GAP/MLP/SiLU channel recalibration."""

    def __init__(
        self,
        c1: int,
        reduction: int = 2,
        alpha_init: float = 0.1,
        beta_init: float = 0.1,
    ):
        super().__init__(c1, reduction, alpha_init)
        c_gate = max(int(c1) // int(reduction), 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c_gate, 1),
            nn.SiLU(),
            nn.Conv2d(c_gate, c1, 1),
        )
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = super().forward(x)
        gate = torch.sigmoid(self.gate(refined))
        return refined * (1.0 + self.beta * gate)
