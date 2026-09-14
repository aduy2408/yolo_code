"""Paired fixed-Haar frequency-domain resampling modules.

The Haar basis is deliberately fixed.  All learnable capacity lives in feature
formation, per-band refinement, and within-channel cross-band mixing.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules import Conv


def haar_analysis(x: torch.Tensor) -> torch.Tensor:
    """Apply one-level orthonormal Haar analysis, returning ``[B, C, 4, h, w]``."""
    if x.ndim != 4:
        raise ValueError(f"haar_analysis expects [B,C,H,W], got shape {tuple(x.shape)}")
    pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    return torch.stack(((a + b + c + d) / 2, (a - b + c - d) / 2,
                        (a + b - c - d) / 2, (a - b - c + d) / 2), dim=2)


def haar_synthesis(bands: torch.Tensor) -> torch.Tensor:
    """Apply the exact inverse of :func:`haar_analysis`."""
    if bands.ndim != 5 or bands.shape[2] != 4:
        raise ValueError(f"haar_synthesis expects [B,C,4,H,W], got shape {tuple(bands.shape)}")
    ll, lh, hl, hh = bands.unbind(dim=2)
    a = (ll + lh + hl + hh) / 2
    b = (ll - lh + hl - hh) / 2
    c = (ll + lh - hl - hh) / 2
    d = (ll - lh - hl + hh) / 2
    out = bands.new_empty((*bands.shape[:2], bands.shape[-2] * 2, bands.shape[-1] * 2))
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


class BandRefine(nn.Module):
    """Independent signed-coefficient residual refinement for one band."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, band: torch.Tensor) -> torch.Tensor:
        return band + self.beta.to(dtype=band.dtype) * self.bn(self.dw(band))


def _make_divisible(value: int, divisor: int = 8) -> int:
    return max(divisor, int((value + divisor - 1) // divisor) * divisor)


class _Formation(nn.Module):
    def __init__(self, c1: int, c_mid: int, c2: int, k: int = 3) -> None:
        super().__init__()
        self.pre = Conv(c1, c_mid, k, 1)
        self.dw = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, k, padding=k // 2, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
        )
        self.compress = Conv(c_mid, c2, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        return self.compress(h + self.dw(h))


class _CrossBandMix(nn.Module):
    """Grouped 1x1 mixer with one independent 4x4 matrix per latent channel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.mix = nn.Conv2d(4 * channels, 4 * channels, 1, groups=channels, bias=False)
        with torch.no_grad():
            self.mix.weight.zero_()
            for i in range(channels):
                self.mix.weight[4 * i:4 * i + 4, :, 0, 0] = torch.eye(4)

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        b, c, _, h, w = bands.shape
        packed = bands.permute(0, 2, 1, 3, 4).reshape(b, 4 * c, h, w)
        mixed = self.mix(packed)
        return mixed.reshape(b, 4, c, h, w).permute(0, 2, 1, 3, 4)

    def stats(self) -> Dict[str, float]:
        weight = self.mix.weight.detach().reshape(self.channels, 4, 4)
        eye = torch.eye(4, device=weight.device, dtype=weight.dtype).expand_as(weight)
        distance = (weight - eye).norm().div(eye.norm()).item()
        offdiag = (weight - torch.diag_embed(torch.diagonal(weight, dim1=1, dim2=2))).abs().mean().item()
        return {"cross_band_identity_distance": distance, "cross_band_offdiag_mean": offdiag}


class _FrequencyBase(nn.Module):
    def __init__(self, band_refine: bool, cross_band: bool, record_stats: bool) -> None:
        super().__init__()
        self.record_stats = bool(record_stats)
        self.last_stats: Dict[str, float] = {}
        self.band_refine_enabled = bool(band_refine)
        self.cross_band_enabled = bool(cross_band)

    def _process_bands(self, bands: torch.Tensor) -> torch.Tensor:
        if self.band_refine_enabled:
            bands = torch.stack([refine(bands[:, :, i]) for i, refine in enumerate(self.band_refine)], dim=2)
        if self.cross_band_enabled:
            bands = self.cross_band(bands)
        return bands


class FreqDown(_FrequencyBase):
    """Feature formation followed by fixed Haar analysis and band processing."""

    def __init__(self, c1: int, c2: int, k: int = 3, expansion: float = 2.0,
                 band_refine: bool = True, cross_band: bool = True,
                 record_stats: bool = False, basis: str = "haar") -> None:
        super().__init__(band_refine, cross_band, record_stats)
        if c2 % 4 or basis != "haar":
            raise ValueError(f"FreqDown requires c2 divisible by 4 and basis='haar', got c2={c2}, basis={basis!r}")
        self.c_band = c2 // 4
        self.c_mid = _make_divisible(max(self.c_band, int(self.c_band * expansion)))
        self.formation = _Formation(c1, self.c_mid, self.c_band, k)
        self.band_refine = nn.ModuleList(BandRefine(self.c_band) for _ in range(4))
        self.cross_band = _CrossBandMix(self.c_band)
        self.out = nn.Sequential(nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = haar_analysis(self.formation(x))
        raw = bands.detach()
        bands = self._process_bands(bands)
        out = self.out(bands.flatten(1, 2))
        if self.record_stats:
            rms = raw.float().square().mean(dim=(0, 1, 3, 4)).sqrt()
            ll = rms[0].clamp_min(torch.finfo(rms.dtype).eps)
            self.last_stats = {f"rms_{name}": rms[i].item() for i, name in enumerate(("LL", "LH", "HL", "HH"))}
            self.last_stats.update({f"{name}/LL": (rms[i] / ll).item() for i, name in enumerate(("LL", "LH", "HL", "HH")) if i})
            self.last_stats.update(self.cross_band.stats())
        return out


class FreqUp(_FrequencyBase):
    """Learned synthesis-band formation followed by inverse Haar reconstruction."""

    def __init__(self, c1: int, c2: int | None = None, expansion: float = 2.0,
                 band_refine: bool = True, cross_band: bool = True,
                 record_stats: bool = False, basis: str = "haar") -> None:
        super().__init__(band_refine, cross_band, record_stats)
        c2 = c1 if c2 is None else c2
        if c2 % 4 or basis != "haar":
            raise ValueError(f"FreqUp requires c2 divisible by 4 and basis='haar', got c2={c2}, basis={basis!r}")
        self.c_band = c2 // 4
        self.c_mid = _make_divisible(max(self.c_band, int(self.c_band * expansion)))
        self.formation = _Formation(c1, self.c_mid, c2, 3)
        self.band_refine = nn.ModuleList(BandRefine(self.c_band) for _ in range(4))
        self.cross_band = _CrossBandMix(self.c_band)
        self.reconstruct = _Formation(self.c_band, self.c_mid, c2, 3)
        self.out = nn.Sequential(nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.formation(x).reshape(x.shape[0], 4, self.c_band, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        raw = bands.detach()
        bands = self._process_bands(bands)
        out = self.out(self.reconstruct(haar_synthesis(bands)))
        if self.record_stats:
            rms = raw.float().square().mean(dim=(0, 1, 3, 4)).sqrt()
            self.last_stats = {f"pred_{name}": rms[i].item() for i, name in enumerate(("LL", "LH", "HL", "HH"))}
            self.last_stats.update(self.cross_band.stats())
        return out


__all__ = ("haar_analysis", "haar_synthesis", "BandRefine", "FreqDown", "FreqUp")
