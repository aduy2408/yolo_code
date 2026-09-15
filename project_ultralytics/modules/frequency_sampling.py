"""Paired fixed-Haar frequency-domain resampling modules.

The Haar basis is deliberately fixed.  All learnable capacity lives in feature
formation, per-band refinement, and (for V1) within-channel cross-band mixing.

V2 keeps the fixed Haar representation but removes the learned 4x4 band
transform.  Its refinement blocks are zero-initialized residuals, so every
subband starts as an identity path.
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


_BAND_NAMES = ("LL", "LH", "HL", "HH")


def _band_stats(bands: torch.Tensor, prefix: str) -> Dict[str, float]:
    rms = bands.float().square().mean(dim=(0, 1, 3, 4)).sqrt()
    return {f"{prefix}_{name}": rms[i].item() for i, name in enumerate(_BAND_NAMES)}


def _band_ratio_stats(bands: torch.Tensor, prefix: str) -> Dict[str, float]:
    rms = bands.float().square().mean(dim=(0, 1, 3, 4)).sqrt()
    ll = rms[0].clamp_min(torch.finfo(rms.dtype).eps)
    return {f"{prefix}_{name}/LL": (rms[i] / ll).item() for i, name in enumerate(_BAND_NAMES[1:], 1)}


def _beta_stats(refiners: nn.ModuleList) -> Dict[str, float]:
    return {f"beta_{name}": refiners[i].beta.detach().item() for i, name in enumerate(_BAND_NAMES)}


def _refine_ratio_stats(pre: torch.Tensor, post: torch.Tensor) -> Dict[str, float]:
    delta = (post.float() - pre.float()).square().mean(dim=(0, 1, 3, 4)).sqrt()
    baseline = pre.float().square().mean(dim=(0, 1, 3, 4)).sqrt().clamp_min(torch.finfo(delta.dtype).eps)
    return {f"refine_{name}_ratio": (delta[i] / baseline[i]).item() for i, name in enumerate(_BAND_NAMES)}


class _Formation(nn.Module):
    def __init__(self, c1: int, c_mid: int, c2: int, k: int = 3, compress_bn: bool = True) -> None:
        super().__init__()
        self.pre = Conv(c1, c_mid, k, 1)
        self.dw = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, k, padding=k // 2, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
        )
        self.compress = Conv(c_mid, c2, 1, 1, act=False) if compress_bn else nn.Conv2d(c_mid, c2, 1, bias=False)

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
        # Pack each latent channel as LL_i,LH_i,HL_i,HH_i for grouped 4x4 mixing.
        packed = bands.reshape(b, 4 * c, h, w)
        mixed = self.mix(packed)
        return mixed.reshape(b, c, 4, h, w)

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
        pre = bands.detach()
        bands = self._process_bands(bands)
        post = bands.detach()
        out = self.out(bands.flatten(1, 2))
        if self.record_stats:
            self.last_stats = _band_stats(pre, "pre")
            self.last_stats.update(_band_stats(post, "post"))
            self.last_stats.update(_band_ratio_stats(pre, "pre"))
            self.last_stats.update(_band_ratio_stats(post, "post"))
            self.last_stats.update(_beta_stats(self.band_refine))
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
        self.reconstruct = _Formation(self.c_band, self.c_mid, c2, 3, compress_bn=False)
        self.out = nn.Sequential(nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.formation(x).reshape(x.shape[0], 4, self.c_band, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        pre = bands.detach()
        bands = self._process_bands(bands)
        post = bands.detach()
        out = self.out(self.reconstruct(haar_synthesis(bands)))
        if self.record_stats:
            self.last_stats = _band_stats(pre, "pre")
            self.last_stats.update(_band_stats(post, "post"))
            self.last_stats.update(_band_ratio_stats(pre, "pre"))
            self.last_stats.update(_band_ratio_stats(post, "post"))
            self.last_stats.update(_beta_stats(self.band_refine))
            self.last_stats.update(self.cross_band.stats())
        return out


class BandRefineV2(nn.Module):
    """Identity-initialized independent residual refinement for one band."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, band: torch.Tensor) -> torch.Tensor:
        return band + self.bn(self.dw(band))


class _FrequencyBaseV2(nn.Module):
    def __init__(self, band_refine: bool, record_stats: bool) -> None:
        super().__init__()
        self.record_stats = bool(record_stats)
        self.last_stats: Dict[str, float] = {}
        self.band_refine_enabled = bool(band_refine)

    def _process_bands(self, bands: torch.Tensor) -> torch.Tensor:
        if self.band_refine_enabled:
            return torch.stack([refine(bands[:, :, i]) for i, refine in enumerate(self.band_refine)], dim=2)
        return bands

    def _record_stats(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        if not self.record_stats:
            return
        self.last_stats = _band_stats(pre, "pre")
        self.last_stats.update(_band_stats(post, "post"))
        self.last_stats.update(_band_ratio_stats(pre, "pre"))
        self.last_stats.update(_band_ratio_stats(post, "post"))
        self.last_stats.update(_refine_ratio_stats(pre, post))


class FreqDownV2(_FrequencyBaseV2):
    """V2 feature formation, fixed Haar analysis, and independent refinement."""

    def __init__(self, c1: int, c2: int, k: int = 3, expansion: float = 2.0,
                 band_refine: bool = True, record_stats: bool = False, basis: str = "haar") -> None:
        super().__init__(band_refine, record_stats)
        if c2 % 4 or basis != "haar":
            raise ValueError(f"FreqDownV2 requires c2 divisible by 4 and basis='haar', got c2={c2}, basis={basis!r}")
        self.c_band = c2 // 4
        self.c_mid = _make_divisible(max(self.c_band, int(self.c_band * expansion)))
        self.formation = _Formation(c1, self.c_mid, self.c_band, k)
        self.band_refine = nn.ModuleList(BandRefineV2(self.c_band) for _ in range(4))
        self.out = nn.Sequential(nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = haar_analysis(self.formation(x))
        pre = bands.detach()
        bands = self._process_bands(bands)
        post = bands.detach()
        out = self.out(bands.flatten(1, 2))
        self._record_stats(pre, post)
        return out


class FreqUpV2(_FrequencyBaseV2):
    """V2 learned synthesis bands, independent refinement, and fixed inverse Haar."""

    def __init__(self, c1: int, c2: int | None = None, expansion: float = 2.0,
                 band_refine: bool = True, record_stats: bool = False, basis: str = "haar") -> None:
        super().__init__(band_refine, record_stats)
        c2 = c1 if c2 is None else c2
        if c2 % 4 or basis != "haar":
            raise ValueError(f"FreqUpV2 requires c2 divisible by 4 and basis='haar', got c2={c2}, basis={basis!r}")
        self.c_band = c2 // 4
        self.c_mid = _make_divisible(max(self.c_band, int(self.c_band * expansion)))
        self.formation = _Formation(c1, self.c_mid, c2, 3)
        self.band_refine = nn.ModuleList(BandRefineV2(self.c_band) for _ in range(4))
        self.reconstruct = _Formation(self.c_band, self.c_mid, c2, 3, compress_bn=False)
        self.out = nn.Sequential(nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.formation(x).reshape(x.shape[0], 4, self.c_band, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        pre = bands.detach()
        bands = self._process_bands(bands)
        post = bands.detach()
        out = self.out(self.reconstruct(haar_synthesis(bands)))
        self._record_stats(pre, post)
        return out


class IBSDown(nn.Module):
    """IBS-style learned formation followed by 2x2 pixel unshuffle.

    Unlike the frequency samplers, this operator does not decompose or mix
    bands.  The formation width is defined from the requested output width:
    ``c_mid = expansion * c2`` with the default ``expansion=2``.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, expansion: float = 2.0,
                 record_stats: bool = False) -> None:
        super().__init__()
        if c2 % 4:
            raise ValueError(f"IBSDown requires c2 divisible by 4, got c2={c2}")
        self.c_mid = _make_divisible(int(c2 * expansion))
        self.c_fold = c2 // 4
        self.record_stats = bool(record_stats)
        self.last_stats: Dict[str, float] = {}
        self.formation = _Formation(c1, self.c_mid, self.c_fold, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        out = F.pixel_unshuffle(self.formation(x), 2)
        if self.record_stats:
            self.last_stats = {
                "input_rms": x.detach().float().square().mean().sqrt().item(),
                "output_rms": out.detach().float().square().mean().sqrt().item(),
                "c_mid": float(self.c_mid),
                "c_fold": float(self.c_fold),
            }
        return out


class IBSUp(nn.Module):
    """IBS-style 2x2 pixel shuffle followed by learned feature formation."""

    def __init__(self, c1: int, c2: int | None = None, expansion: float = 2.0,
                 record_stats: bool = False) -> None:
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 % 4:
            raise ValueError(f"IBSUp requires c1 divisible by 4, got c1={c1}")
        self.c_mid = _make_divisible(int(c2 * expansion))
        self.c_fold = c1 // 4
        self.record_stats = bool(record_stats)
        self.last_stats: Dict[str, float] = {}
        self.formation = _Formation(self.c_fold, self.c_mid, c2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        folded = F.pixel_shuffle(x, 2)
        out = self.formation(folded)
        if self.record_stats:
            self.last_stats = {
                "input_rms": x.detach().float().square().mean().sqrt().item(),
                "output_rms": out.detach().float().square().mean().sqrt().item(),
                "c_mid": float(self.c_mid),
                "c_fold": float(self.c_fold),
            }
        return out


__all__ = (
    "haar_analysis", "haar_synthesis", "BandRefine", "FreqDown", "FreqUp",
    "BandRefineV2", "FreqDownV2", "FreqUpV2", "IBSDown", "IBSUp",
)
