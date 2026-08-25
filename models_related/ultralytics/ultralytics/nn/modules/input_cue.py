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
    "robust_ring_contrast",
    "lbp_stats",
    "multiscale_tophat",
    "local_rank",
    "phase_coherence",
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
    "robust_ring_contrast": 2,
    "lbp_stats": 2,
    "multiscale_tophat": 2,
    "local_rank": 1,
    "phase_coherence": 1,
}


def _kernel(kernel: list[list[float]], channels: int, device, dtype):
    value = torch.tensor(kernel, device=device, dtype=dtype)
    return value.view(1, 1, *value.shape).repeat(channels, 1, 1, 1)


def _shift_samples(image, offsets):
    radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    _, _, height, width = image.shape
    padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    return torch.cat(
        [padded[:, :, radius + dy:radius + dy + height, radius + dx:radius + dx + width] for dy, dx in offsets],
        dim=1,
    )


def _dilation(image, kernel_size):
    radius = kernel_size // 2
    return F.max_pool2d(F.pad(image, (radius, radius, radius, radius), mode="reflect"), kernel_size, stride=1)


def _erosion(image, kernel_size):
    return -_dilation(-image, kernel_size)


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
            scale = 0.5
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
            opened = _dilation(_erosion(y, 5), 5)
            return (y - opened).clamp(0, 1)
        if self.cue_type == "robust_ring_contrast":
            center = _shift_samples(y, [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 0), (0, 1),
                (1, -1), (1, 0), (1, 1),
            ])
            ring = _shift_samples(y, [
                (-4, -4), (-4, -2), (-4, 0), (-4, 2), (-4, 4),
                (-2, -4), (-2, 4), (0, -4), (0, 4),
                (2, -4), (2, 4), (4, -4), (4, -2), (4, 0), (4, 2), (4, 4),
            ])
            center_median = center.median(dim=1, keepdim=True).values
            ring_median = ring.median(dim=1, keepdim=True).values
            mad = (ring - ring_median).abs().median(dim=1, keepdim=True).values
            contrast = ((center_median - ring_median) / (1.4826 * mad + 1e-3)).clamp(-3, 3) / 3
            return torch.cat((contrast.relu(), (-contrast).relu()), dim=1)
        if self.cue_type == "lbp_stats":
            neighbors = _shift_samples(y, [
                (-1, -1), (-1, 0), (-1, 1), (0, 1),
                (1, 1), (1, 0), (1, -1), (0, -1),
            ])
            bits = (neighbors >= y).to(y.dtype)
            transitions = (bits != torch.roll(bits, shifts=-1, dims=1)).to(y.dtype).mean(dim=1, keepdim=True)
            return torch.cat((bits.mean(dim=1, keepdim=True), transitions), dim=1)
        if self.cue_type == "multiscale_tophat":
            whites, blacks = [], []
            for kernel_size in (5, 11, 21):
                opened = _dilation(_erosion(y, kernel_size), kernel_size)
                closed = _erosion(_dilation(y, kernel_size), kernel_size)
                whites.append((y - opened).clamp_min(0))
                blacks.append((closed - y).clamp_min(0))
            white = torch.stack(whites, dim=1).amax(dim=1)
            black = torch.stack(blacks, dim=1).amax(dim=1)
            return torch.cat((white, black), dim=1).clamp(0, 1)
        if self.cue_type == "local_rank":
            kernel_size = 5
            radius = kernel_size // 2
            patches = F.unfold(F.pad(y, (radius, radius, radius, radius), mode="reflect"), kernel_size)
            center = y.flatten(2)
            mask = torch.ones(kernel_size * kernel_size, dtype=torch.bool, device=y.device)
            mask[kernel_size * kernel_size // 2] = False
            rank = (patches[:, mask] < center).to(y.dtype).mean(dim=1)
            return rank.view(y.shape[0], 1, y.shape[2], y.shape[3])
        if self.cue_type == "phase_coherence":
            coordinates = torch.arange(-4, 5, device=y.device, dtype=y.dtype)
            yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
            filters = []
            for wavelength in (3.0, 5.0):
                sigma = 0.5 * wavelength
                for angle in (0.0, torch.pi / 4, torch.pi / 2, 3 * torch.pi / 4):
                    x_theta = xx * torch.cos(torch.as_tensor(angle, device=y.device, dtype=y.dtype)) + yy * torch.sin(torch.as_tensor(angle, device=y.device, dtype=y.dtype))
                    y_theta = -xx * torch.sin(torch.as_tensor(angle, device=y.device, dtype=y.dtype)) + yy * torch.cos(torch.as_tensor(angle, device=y.device, dtype=y.dtype))
                    envelope = torch.exp(-(x_theta.square() + 0.5**2 * y_theta.square()) / (2 * sigma**2))
                    even = envelope * torch.cos(2 * torch.pi * x_theta / wavelength)
                    odd = envelope * torch.sin(2 * torch.pi * x_theta / wavelength)
                    even = even - even.mean()
                    odd = odd - odd.mean()
                    filters.extend((even / (even.abs().sum() + self.eps), odd / (odd.abs().sum() + self.eps)))
            bank = torch.stack(filters).unsqueeze(1)
            response = F.conv2d(F.pad(y, (4, 4, 4, 4), mode="reflect"), bank)
            response = response.view(y.shape[0], 2, 4, 2, y.shape[2], y.shape[3]).permute(0, 2, 1, 3, 4, 5)
            even, odd = response[:, :, :, 0], response[:, :, :, 1]
            amplitude = (even.square() + odd.square()).sqrt()
            amplitude_sum = amplitude.sum(dim=2)
            numerator = (even.sum(dim=2).square() + odd.sum(dim=2).square()).sqrt()
            phase = torch.where(amplitude_sum > self.eps, numerator / (amplitude_sum + self.eps), torch.zeros_like(numerator))
            return phase.max(dim=1, keepdim=True).values.clamp(0, 1)
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
        with torch.no_grad():
            cue = self.cue_bank(rgb)
        return torch.cat((rgb, cue), dim=1)


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
