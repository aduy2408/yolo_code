"""Small project-owned compatibility block for historical RepC2f YAMLs."""

from __future__ import annotations

from torch import nn

from ultralytics.nn.modules import C2f, Conv, RepConv


class RepC2f(C2f):
    """C2f with RepConv bottlenecks, matching the historical module contract."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                RepConv(self.c, self.c, 3, 1, g=g),
                Conv(self.c, self.c, 3, 1, g=g),
            )
            for _ in range(n)
        )


__all__ = ("RepC2f",)
