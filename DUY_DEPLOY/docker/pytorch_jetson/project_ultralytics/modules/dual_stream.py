"""Project-owned dual-stream feature formation modules."""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules import Bottleneck, Conv


class DualChannelFormationBackbone(nn.Module):
    """Persistent Dual-Stream Channel-Formation Block.
    Maintains and propagates a tuple (M, I) of mixed and isolated representations.
    If the input is a single tensor X, it splits X using stem projections.
    Supports Progressive Cross-conditioned formation (Option C) and Late Concat (Option B, progressive=False).
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.67, progressive: bool = True) -> None:
        super().__init__()
        self.c2_m = int(round(c2 * e))
        self.c2_i = c2 - self.c2_m
        self.progressive = progressive

        # Stem transitions (used only if input is a single tensor)
        self.stem_m = Conv(c1, self.c2_m, 1) if c1 != self.c2_m else nn.Identity()
        self.stem_i = Conv(c1, self.c2_i, 1) if c1 != self.c2_i else nn.Identity()

        # Mixed Stream Blocks: Φ_M (takes [M, selected_I] which has c2_m + c2_i channels and outputs c2_m)
        # We will use Bottleneck with input dimension c2_m + c2_i if progressive, else c2_m.
        # To keep it neat, we define:
        if self.progressive:
            self.m_blocks = nn.ModuleList(
                nn.Sequential(
                    Conv(self.c2_m + self.c2_i, self.c2_m, 1),
                    Bottleneck(self.c2_m, self.c2_m, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                ) for _ in range(n)
            )
        else:
            self.m_blocks = nn.ModuleList(Bottleneck(self.c2_m, self.c2_m, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        # Isolated Stream Blocks: Φ_I (DWConv3x3 only, no pointwise mix)
        self.i_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.c2_i, self.c2_i, 3, padding=1, groups=self.c2_i, bias=False),
                nn.BatchNorm2d(self.c2_i),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.c2_i, self.c2_i, 3, padding=1, groups=self.c2_i, bias=False),
                nn.BatchNorm2d(self.c2_i),
                nn.SiLU(inplace=True)
            ) for _ in range(n)
        )

        if self.progressive:
            # Interaction cues: M, I, M_proj*I, abs(M_proj - I) -> c2_m + 3 * c2_i channels
            self.proj_m_to_i = nn.Conv2d(self.c2_m, self.c2_i, 1, bias=False)
            self.proj_i = nn.Conv2d(self.c2_i, self.c2_i, 1, bias=False) # P_I(I)
            
            fusion_channels = self.c2_m + 3 * self.c2_i
            # Z -> H (interaction bottleneck)
            self.fusion_phi = nn.Sequential(
                nn.Conv2d(fusion_channels, self.c2_m + self.c2_i, 1, bias=False),
                nn.BatchNorm2d(self.c2_m + self.c2_i),
                nn.SiLU(inplace=True)
            )
            
            # Asymmetric Gates
            # G_I(Z) for selecting I info -> c2_i channels
            self.gate_i = nn.Sequential(
                nn.Conv2d(self.c2_m + self.c2_i, self.c2_i, 1, bias=False),
                nn.Sigmoid()
            )
            # G_M(Z) for modulating I channel weights -> c2_i channels
            self.gate_m = nn.Sequential(
                nn.Conv2d(self.c2_m + self.c2_i, self.c2_i, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            m, i = x
        else:
            m = self.stem_m(x)
            i = self.stem_i(x)

        for m_blk, i_blk in zip(self.m_blocks, self.i_blocks):
            if self.progressive:
                # 1. Compute interaction Z
                m_proj = self.proj_m_to_i(m)
                prod = m_proj * i
                diff = torch.abs(m_proj - i)
                cues = torch.cat([m, i, prod, diff], dim=1)
                z = self.fusion_phi(cues)

                # 2. Select I -> inject into mixed stream formation
                g_i = self.gate_i(z)
                i_sel = g_i * self.proj_i(i)
                m = m_blk(torch.cat([m, i_sel], dim=1))

                # 3. Modulate isolated stream formation
                g_m = self.gate_m(z)
                i = i_blk(i) * g_m
            else:
                m = m_blk(m)
                i = i_blk(i)

        return m, i


class DualDownsample(nn.Module):
    """Downsampling module for persistent dual-stream.
    Independently downsamples M (via standard Conv stride=2) and I (via DWConv stride=2).
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, e: float = 0.67) -> None:
        super().__init__()
        c1_m = int(round(c1 * e))
        c1_i = c1 - c1_m
        c2_m = int(round(c2 * e))
        c2_i = c2 - c2_m

        self.down_m = Conv(c1_m, c2_m, k, s)
        self.down_i = nn.Sequential(
            nn.Conv2d(c1_i, c1_i, k, s, padding=k // 2, groups=c1_i, bias=False),
            nn.BatchNorm2d(c1_i),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1_i, c2_i, 1, bias=False),
            nn.BatchNorm2d(c2_i)
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        m, i = x
        return self.down_m(m), self.down_i(i)


class DualCollapse(nn.Module):
    """Collapses persistent dual-stream (M, I) back to a single tensor.
    Utilizes Concat + Conv 1x1.
    """

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        # c1 is total channel capacity of input streams (M + I)
        self.conv = Conv(c1, c2, 1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        m, i = x
        return self.conv(torch.cat([m, i], dim=1))


