"""Fixed low-level cue channels computed inside the YOLO model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


VARIANTS = (
    "sobel_xy",
    "laplacian_split",
    "log",
    "haar",
    "lab_ab",
    "ycbcr_cbcr",
    "chromatic_edge",
    "local_zscore",
    "structure_coherence",
    "top_hat",
)

_CUE_CHANNELS = {
    "sobel_xy": 2,
    "laplacian_split": 2,
    "log": 1,
    "haar": 3,
    "lab_ab": 2,
    "ycbcr_cbcr": 2,
    "chromatic_edge": 2,
    "local_zscore": 1,
    "structure_coherence": 1,
    "top_hat": 1,
}


def _kernel(kernel: list[list[float]], channels: int, device, dtype):
    value = torch.tensor(kernel, device=device, dtype=dtype)
    return value.view(1, 1, *value.shape).repeat(channels, 1, 1, 1)


class InputCueBank(nn.Module):
    """Compute one deterministic, full-resolution cue from normalized RGB."""

    def __init__(self, cue_type: str, eps: float = 1e-6):
        super().__init__()
        if cue_type not in VARIANTS:
            raise ValueError(f"Unknown input cue {cue_type!r}. Expected one of {VARIANTS}.")
        self.cue_type = cue_type
        self.eps = eps

    @staticmethod
    def gray(rgb):
        return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    def sobel(self, image):
        gx = _kernel([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], image.shape[1], image.device, image.dtype) / 4
        gy = _kernel([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], image.shape[1], image.device, image.dtype) / 4
        return (
            F.conv2d(image, gx, padding=1, groups=image.shape[1]),
            F.conv2d(image, gy, padding=1, groups=image.shape[1]),
        )

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"InputCueBank expects BCHW RGB input, got {tuple(rgb.shape)}")
        y = self.gray(rgb)
        if self.cue_type == "sobel_xy":
            gx, gy = self.sobel(y)
            return torch.cat((gx.clamp(-1, 1), gy.clamp(-1, 1)), dim=1)
        if self.cue_type == "laplacian_split":
            lap = F.conv2d(y, _kernel([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 1, y.device, y.dtype), padding=1)
            return torch.cat((lap.relu().clamp(0, 1), (-lap).relu().clamp(0, 1)), dim=1)
        if self.cue_type == "log":
            coords = torch.arange(-2, 3, device=y.device, dtype=y.dtype)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            gaussian = torch.exp(-(xx.square() + yy.square()) / 2)
            gaussian = gaussian / gaussian.sum()
            blur = F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), gaussian.view(1, 1, 5, 5))
            laplacian = _kernel([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 1, y.device, y.dtype)
            log = F.conv2d(F.pad(blur, (1, 1, 1, 1), mode="reflect"), laplacian)
            return log.clamp(-1, 1)
        if self.cue_type == "haar":
            scale = 2**-0.5
            filters = [
                [[scale, -scale], [scale, -scale]],
                [[scale, scale], [-scale, -scale]],
                [[scale, -scale], [-scale, scale]],
            ]
            return torch.cat(
                [F.conv2d(F.pad(y, (0, 1, 0, 1), mode="reflect"), _kernel(f, 1, y.device, y.dtype)).clamp(-1, 1) for f in filters],
                1,
            )
        if self.cue_type == "lab_ab":
            rgb_lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
            r, g, b = rgb_lin[:, 0:1], rgb_lin[:, 1:2], rgb_lin[:, 2:3]
            x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
            yy = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b)
            z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
            delta = 6 / 29
            f = lambda value: torch.where(value > delta**3, value.clamp_min(self.eps).pow(1 / 3), value / (3 * delta**2) + 4 / 29)
            fx, fy, fz = f(x), f(yy), f(z)
            return torch.cat(((500 * (fx - fy) / 128).clamp(-1, 1), (200 * (fy - fz) / 128).clamp(-1, 1)), 1)
        if self.cue_type == "ycbcr_cbcr":
            cb = -0.168736 * rgb[:, 0:1] - 0.331264 * rgb[:, 1:2] + 0.5 * rgb[:, 2:3]
            cr = 0.5 * rgb[:, 0:1] - 0.418688 * rgb[:, 1:2] - 0.081312 * rgb[:, 2:3]
            return torch.cat(((2 * cb).clamp(-1, 1), (2 * cr).clamp(-1, 1)), 1)
        if self.cue_type == "chromatic_edge":
            rg = rgb[:, 0:1] - rgb[:, 1:2]
            by = rgb[:, 2:3] - (rgb[:, 0:1] + rgb[:, 1:2]) / 2
            rgx, rgy = self.sobel(rg)
            byx, byy = self.sobel(by)
            return torch.cat(((rgx.square() + rgy.square() + self.eps).sqrt().clamp(0, 1), (byx.square() + byy.square() + self.eps).sqrt().clamp(0, 1)), 1)
        if self.cue_type == "local_zscore":
            padded = F.pad(y, (3, 3, 3, 3), mode="reflect")
            mean = F.avg_pool2d(padded, 7, stride=1)
            variance = (F.avg_pool2d(padded.square(), 7, stride=1) - mean.square()).clamp_min(0)
            return ((y - mean) / (variance + self.eps).sqrt()).clamp(-3, 3) / 3
        if self.cue_type == "structure_coherence":
            gx, gy = self.sobel(y)
            mean = lambda value: F.avg_pool2d(F.pad(value, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
            jxx, jyy, jxy = mean(gx.square()), mean(gy.square()), mean(gx * gy)
            return ((jxx - jyy).square() + 4 * jxy.square()).sqrt() / (jxx + jyy + self.eps)
        if self.cue_type == "top_hat":
            eroded = -F.max_pool2d(-y, 5, stride=1, padding=2)
            opened = F.max_pool2d(eroded, 5, stride=1, padding=2)
            return (y - opened).clamp(0, 1)
        raise AssertionError(self.cue_type)


class InputCueConv(Conv):
    """YOLO stem convolution that concatenates a fixed cue after augmentation."""

    def __init__(self, c1, c2, k=3, s=2, cue_type="sobel_xy", p=None, g=1, d=1, act=True):
        if c1 != 3:
            raise ValueError(f"InputCueConv currently requires RGB input (c1=3), got c1={c1}")
        self.cue_type = cue_type
        self.cue_channels = _CUE_CHANNELS[cue_type]
        super().__init__(c1 + self.cue_channels, c2, k, s, p=p, g=g, d=d, act=act)
        self.cue_bank = InputCueBank(cue_type)

    def forward(self, rgb):
        return super().forward(self._with_cue(rgb))

    def forward_fuse(self, rgb):
        """Apply the fused stem while preserving inside-model cue computation."""
        return self.act(self.conv(self._with_cue(rgb)))

    def _with_cue(self, rgb):
        return torch.cat((rgb, self.cue_bank(rgb)), dim=1)


def cue_channels(cue_type: str) -> int:
    """Return the number of channels produced by a registered cue."""
    return _CUE_CHANNELS[cue_type]


def copy_rgb_stem_weights(cue_model, rgb_model):
    """Copy the RGB stem and zero cue channels, preserving exact pretrained identity."""
    cue_stem, rgb_stem = cue_model.model[0], rgb_model.model[0]
    if not isinstance(cue_stem, InputCueConv) or not isinstance(rgb_stem, Conv):
        raise TypeError("Expected InputCueConv at cue_model.model[0] and Conv at rgb_model.model[0]")
    with torch.no_grad():
        cue_stem.conv.weight.zero_()
        cue_stem.conv.weight[:, :3].copy_(rgb_stem.conv.weight)
        cue_stem.bn.load_state_dict(rgb_stem.bn.state_dict())
        if hasattr(rgb_stem, "act"):
            cue_stem.act = rgb_stem.act
