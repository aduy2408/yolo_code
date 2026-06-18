"""Example PyTorch blocks for editable Ultralytics custom models.

Copy these classes into an editable Ultralytics checkout, or import/expose them
there, before referencing them from a model YAML.

YAML usage after registration:
    - [-1, 1, VarroaConvBlock, [256, 3, True]]
    - [-1, 1, VarroaSEBlock, [256, 8]]

For both blocks, add a parse_model() case in Ultralytics so the YAML's first
arg is c2 and Ultralytics injects c1 from the previous layer:
    c1, c2 = ch[f], args[0]
    args = [c1, c2, *args[1:]]
"""

from __future__ import annotations

from torch import nn


class VarroaConvBlock(nn.Module):
    """Small Conv-BN-SiLU residual-friendly block with c1/c2 constructor."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 3, shortcut: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        y = self.act(self.bn(self.conv(x)))
        return x + y if self.shortcut else y


class VarroaSEBlock(nn.Module):
    """Lightweight squeeze-excitation block useful after C2f/SPPF stages."""

    def __init__(self, c1: int, c2: int, reduction: int = 8):
        super().__init__()
        hidden = max(c2 // reduction, 8)
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c2, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.proj(x)
        return x * self.fc(self.pool(x))
