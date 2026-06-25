from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class HBSContext:
    batch_idx: torch.Tensor
    bboxes: torch.Tensor
    image_shape: tuple[int, int, int, int]


_HBS_CONTEXT: HBSContext | None = None
_HBS_ENABLED = True


def set_hbs_enabled(enabled: bool) -> None:
    global _HBS_ENABLED
    _HBS_ENABLED = enabled


def set_hbs_context(
    batch_idx: torch.Tensor | None,
    bboxes: torch.Tensor | None,
    image_shape: tuple[int, int, int, int],
) -> None:
    """Store batch labels for train-time HBS foreground/background masks."""

    global _HBS_CONTEXT
    if batch_idx is None or bboxes is None:
        _HBS_CONTEXT = None
        return

    _HBS_CONTEXT = HBSContext(
        batch_idx=batch_idx.detach(),
        bboxes=bboxes.detach(),
        image_shape=image_shape,
    )


def clear_hbs_context() -> None:
    global _HBS_CONTEXT
    _HBS_CONTEXT = None


class HBSBlock(nn.Module):
    """Hierarchical Background Smoothing for one FPN level.

    The block is intentionally train-only. During validation, testing, export,
    or inference it returns the input unchanged.
    """

    def __init__(self, c1: int, reduction: int = 4, min_mask_value: float = 0.0) -> None:
        super().__init__()
        hidden_channels = max(c1 // reduction, 8)
        self.min_mask_value = float(min_mask_value)
        self.smoother = nn.Sequential(
            nn.Conv2d(c1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _HBS_ENABLED or not self.training or _HBS_CONTEXT is None:
            return x

        mask = _build_foreground_mask(
            batch_idx=_HBS_CONTEXT.batch_idx,
            bboxes=_HBS_CONTEXT.bboxes,
            batch_size=x.shape[0],
            feature_height=x.shape[2],
            feature_width=x.shape[3],
            device=x.device,
            dtype=x.dtype,
            min_mask_value=self.min_mask_value,
        )
        fg = x * mask
        bg = x * (1.0 - mask)
        return fg + self.smoother(bg)


def _build_foreground_mask(
    batch_idx: torch.Tensor,
    bboxes: torch.Tensor,
    batch_size: int,
    feature_height: int,
    feature_width: int,
    device: torch.device,
    dtype: torch.dtype,
    min_mask_value: float,
) -> torch.Tensor:
    mask = torch.full(
        (batch_size, 1, feature_height, feature_width),
        fill_value=min_mask_value,
        device=device,
        dtype=dtype,
    )
    if bboxes.numel() == 0:
        return mask

    batch_idx = batch_idx.to(device=device, dtype=torch.long).view(-1)
    bboxes = bboxes.to(device=device, dtype=dtype)

    for idx, box in zip(batch_idx.tolist(), bboxes, strict=False):
        if idx < 0 or idx >= batch_size:
            continue

        x_center, y_center, width, height = box.tolist()
        x1 = int((x_center - width / 2.0) * feature_width)
        y1 = int((y_center - height / 2.0) * feature_height)
        x2 = int((x_center + width / 2.0) * feature_width)
        y2 = int((y_center + height / 2.0) * feature_height)

        x1 = max(0, min(feature_width - 1, x1))
        y1 = max(0, min(feature_height - 1, y1))
        x2 = max(x1 + 1, min(feature_width, x2 + 1))
        y2 = max(y1 + 1, min(feature_height, y2 + 1))

        mask[idx, :, y1:y2, x1:x2] = 1.0

    return mask
