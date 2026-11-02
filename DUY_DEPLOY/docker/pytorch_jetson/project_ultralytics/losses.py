"""Project-specific standalone losses.

These objectives are extracted from the legacy fork and intentionally do not
modify or import Ultralytics' upstream loss implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class BoundaryContrastiveLossConfig:
    """YOLO.train kwargs for the boundary-aware contrastive localization loss."""

    gain: float = 0.05
    levels: int = 2
    ring: float = 1.0
    samples: int = 16
    tau: float = 0.2
    shrinkage: float = 0.25

    def as_train_kwargs(self) -> dict[str, float | int]:
        """Return kwargs accepted by YOLO.train(...)."""

        return {
            "boundary_contrast": float(self.gain),
            "boundary_levels": int(self.levels),
            "boundary_ring": float(self.ring),
            "boundary_samples": int(self.samples),
            "boundary_tau": float(self.tau),
            "boundary_shrinkage": float(self.shrinkage),
        }


def boundary_contrastive_loss_kwargs(
    gain: float = 0.05,
    levels: int = 2,
    ring: float = 1.0,
    samples: int = 16,
    tau: float = 0.2,
    shrinkage: float = 0.25,
) -> dict[str, float | int]:
    """Build YOLO.train kwargs that enable the boundary contrastive loss."""

    return BoundaryContrastiveLossConfig(
        gain=gain,
        levels=levels,
        ring=ring,
        samples=samples,
        tau=tau,
        shrinkage=shrinkage,
    ).as_train_kwargs()


def add_boundary_contrastive_loss(
    train_kwargs: dict | None = None,
    *,
    gain: float = 0.05,
    levels: int = 2,
    ring: float = 1.0,
    samples: int = 16,
    tau: float = 0.2,
    shrinkage: float = 0.25,
) -> dict:
    """Return train kwargs with boundary contrastive localization loss enabled."""

    kwargs = dict(train_kwargs or {})
    kwargs.update(
        boundary_contrastive_loss_kwargs(
            gain=gain,
            levels=levels,
            ring=ring,
            samples=samples,
            tau=tau,
            shrinkage=shrinkage,
        )
    )
    return kwargs

class WiseIouLoss(nn.Module):
    """Wise-IoU family losses for xyxy boxes with optional dynamic focusing."""

    momentum = 1e-2
    alpha = 1.7
    delta = 2.7

    def __init__(self, ltype: str = "WIoU", monotonous: bool | None = False, eps: float = 1e-7):
        """Initialize Wise-IoU loss.

        Args:
            ltype: IoU loss type, e.g. ``WIoU``, ``IoU``, ``GIoU``, ``DIoU``, ``CIoU``, ``EIoU`` or ``SIoU``.
            monotonous: ``True`` for monotonic focusing, ``False`` for non-monotonic focusing, ``None`` to disable it.
            eps: Small value for numerical stability.
        """
        super().__init__()
        self.ltype = ltype.upper()
        if self.ltype == "WIOU":
            self.ltype = "WIoU"
        elif self.ltype.endswith("IOU"):
            self.ltype = self.ltype[:-3] + "IoU"
        if getattr(self, f"_{self.ltype}", None) is None:
            raise ValueError(f"Unsupported bbox_iou_loss='{ltype}'.")
        self.monotonous = monotonous
        self.eps = eps
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def __getitem__(self, item: str) -> torch.Tensor:
        """Evaluate cached geometry terms lazily."""
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-box IoU-family loss and plain IoU for xyxy boxes."""
        self._fget = {
            "pred": pred,
            "target": target,
            "pred_xy": lambda: (self["pred"][..., :2] + self["pred"][..., 2:4]) / 2,
            "pred_wh": lambda: (self["pred"][..., 2:4] - self["pred"][..., :2]).clamp(min=self.eps),
            "target_xy": lambda: (self["target"][..., :2] + self["target"][..., 2:4]) / 2,
            "target_wh": lambda: (self["target"][..., 2:4] - self["target"][..., :2]).clamp(min=self.eps),
            "min_coord": lambda: torch.minimum(self["pred"][..., :4], self["target"][..., :4]),
            "max_coord": lambda: torch.maximum(self["pred"][..., :4], self["target"][..., :4]),
            "wh_inter": lambda: torch.relu(self["min_coord"][..., 2:4] - self["max_coord"][..., :2]),
            "s_inter": lambda: torch.prod(self["wh_inter"], dim=-1),
            "s_union": lambda: torch.prod(self["pred_wh"], dim=-1)
            + torch.prod(self["target_wh"], dim=-1)
            - self["s_inter"]
            + self.eps,
            "wh_box": lambda: (self["max_coord"][..., 2:4] - self["min_coord"][..., :2]).clamp(min=self.eps),
            "s_box": lambda: torch.prod(self["wh_box"], dim=-1) + self.eps,
            "l2_box": lambda: torch.square(self["wh_box"]).sum(dim=-1) + self.eps,
            "d_center": lambda: self["pred_xy"] - self["target_xy"],
            "l2_center": lambda: torch.square(self["d_center"]).sum(dim=-1),
            "iou": lambda: self["s_inter"] / self["s_union"],
            "iou_loss": lambda: 1.0 - self["iou"],
        }

        if self.training:
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * self["iou_loss"].detach().mean())

        loss = self._scaled_loss(getattr(self, f"_{self.ltype}")())
        iou = self["iou"]
        delattr(self, "_fget")
        return loss, iou

    def _scaled_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply Wise-IoU dynamic focusing when configured."""
        if isinstance(self.monotonous, bool):
            beta = self["iou_loss"].detach() / self.iou_mean.clamp(min=self.eps)
            if self.monotonous:
                loss = loss * beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss = loss * beta / divisor
        return loss

    def _IoU(self) -> torch.Tensor:
        return self["iou_loss"]

    def _WIoU(self) -> torch.Tensor:
        dist = torch.exp(self["l2_center"] / self["l2_box"].detach())
        return dist * self["iou_loss"]

    def _EIoU(self) -> torch.Tensor:
        penalty = self["l2_center"] / self["l2_box"] + torch.square(self["d_center"] / self["wh_box"]).sum(dim=-1)
        return self["iou_loss"] + penalty

    def _GIoU(self) -> torch.Tensor:
        return self["iou_loss"] + (self["s_box"] - self["s_union"]) / self["s_box"]

    def _DIoU(self) -> torch.Tensor:
        return self["iou_loss"] + self["l2_center"] / self["l2_box"]

    def _CIoU(self) -> torch.Tensor:
        v = (
            4
            / math.pi**2
            * (
                torch.atan(self["pred_wh"][..., 0] / self["pred_wh"][..., 1])
                - torch.atan(self["target_wh"][..., 0] / self["target_wh"][..., 1])
            ).pow(2)
        )
        alpha = v / (self["iou_loss"] + v + self.eps)
        return self["iou_loss"] + self["l2_center"] / self["l2_box"] + alpha.detach() * v

    def _SIoU(self, theta: int = 4) -> torch.Tensor:
        angle = torch.arcsin(torch.abs(self["d_center"]).min(dim=-1)[0] / (self["l2_center"].sqrt() + self.eps))
        angle = torch.sin(2 * angle) - 2
        dist = angle[..., None] * torch.square(self["d_center"] / self["wh_box"])
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        d_shape = torch.abs(self["pred_wh"] - self["target_wh"])
        big_shape = torch.maximum(self["pred_wh"], self["target_wh"])
        w_shape = 1 - torch.exp(-d_shape[..., 0] / big_shape[..., 0])
        h_shape = 1 - torch.exp(-d_shape[..., 1] / big_shape[..., 1])
        shape = w_shape**theta + h_shape**theta
        return self["iou_loss"] + (dist + shape) / 2


class BoundaryContrastiveLoss(nn.Module):
    """Boundary-aware InfoNCE loss over Detect feature maps."""

    def __init__(self, levels: int = 2, ring: float = 1.0, samples: int = 16, tau: float = 0.2, shrinkage: float = 0.25):
        """Initialize boundary contrast settings."""
        super().__init__()
        self.levels = max(int(levels), 0)
        self.ring = max(float(ring), 0.0)
        self.samples = max(int(samples), 1)
        self.tau = max(float(tau), 1e-6)
        self.shrinkage = max(float(shrinkage), 0.0)

    @staticmethod
    def _sample_indices(mask: torch.Tensor, limit: int) -> torch.Tensor:
        """Sample up to limit flattened indices from a boolean mask."""
        idx = mask.flatten().nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > limit:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:limit]]
        return idx

    def forward(
        self,
        feats: list[torch.Tensor],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary-aware contrastive loss from feature maps and image-space GT boxes."""
        losses = []
        num_levels = min(self.levels, len(feats), len(strides))
        for level, feat in enumerate(feats[:num_levels]):
            bs, _, h, w = feat.shape
            stride = strides[level].to(device=feat.device, dtype=feat.dtype).clamp(min=1)
            y, x = torch.meshgrid(
                torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5,
                torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5,
                indexing="ij",
            )

            for bi in range(bs):
                boxes = gt_bboxes[bi][mask_gt[bi, :, 0].bool()]
                if boxes.numel() == 0:
                    continue

                fmap = feat[bi].flatten(1).transpose(0, 1)
                fmap = F.normalize(fmap, dim=1)

                for box in boxes:
                    x1, y1, x2, y2 = box / stride
                    x1 = x1.clamp(0, w)
                    x2 = x2.clamp(0, w)
                    y1 = y1.clamp(0, h)
                    y2 = y2.clamp(0, h)
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue

                    # EXPERIMENTAL: prefer a tighter interior region when it
                    # exists, so positives represent object texture, not edges.
                    pad_x = torch.minimum((x2 - x1) * self.shrinkage, torch.tensor(0.5, device=feat.device, dtype=feat.dtype))
                    pad_y = torch.minimum((y2 - y1) * self.shrinkage, torch.tensor(0.5, device=feat.device, dtype=feat.dtype))
                    inner = (x >= x1 + pad_x) & (x < x2 - pad_x) & (y >= y1 + pad_y) & (y < y2 - pad_y)
                    obj = inner if inner.any() else ((x >= x1) & (x < x2) & (y >= y1) & (y < y2))

                    dx1 = (x1 - self.ring).clamp(0, w)
                    dy1 = (y1 - self.ring).clamp(0, h)
                    dx2 = (x2 + self.ring).clamp(0, w)
                    dy2 = (y2 + self.ring).clamp(0, h)
                    dilated = (x >= dx1) & (x < dx2) & (y >= dy1) & (y < dy2)
                    original = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                    boundary = dilated & ~original

                    # EXPERIMENTAL: nearby background is a cheap hard-negative
                    # proxy without using prediction confidence.
                    near_x1 = (x1 - self.ring * 3).clamp(0, w)
                    near_y1 = (y1 - self.ring * 3).clamp(0, h)
                    near_x2 = (x2 + self.ring * 3).clamp(0, w)
                    near_y2 = (y2 + self.ring * 3).clamp(0, h)
                    near = (x >= near_x1) & (x < near_x2) & (y >= near_y1) & (y < near_y2)
                    background = near & ~dilated
                    if not background.any():
                        background = ~dilated

                    obj_idx = self._sample_indices(obj, self.samples)
                    bnd_idx = self._sample_indices(boundary, self.samples)
                    bg_idx = self._sample_indices(background, self.samples)
                    if obj_idx.numel() < 2 or bnd_idx.numel() == 0 or bg_idx.numel() == 0:
                        continue

                    obj_feat = fmap[obj_idx]
                    pos = obj_feat.roll(1, dims=0)
                    neg = fmap[torch.cat((bnd_idx, bg_idx), 0)]

                    pos_logits = (obj_feat * pos).sum(1, keepdim=True) / self.tau
                    neg_logits = obj_feat @ neg.T / self.tau
                    logits = torch.cat((pos_logits, neg_logits), 1)
                    labels = torch.zeros(logits.shape[0], device=feat.device, dtype=torch.long)
                    losses.append(F.cross_entropy(logits, labels))

        return torch.stack(losses).mean() if losses else gt_bboxes.sum() * 0.0


# EXPERIMENTAL: Localization Quality Map loss for tiny-object localization.
# This auxiliary objective teaches train-only 1x1 heads to predict a smooth
# center-high localization target for each GT box. It has no inference path.
class LocalizationQualityLoss(nn.Module):
    """Gaussian localization quality map supervision over Detect feature maps."""

    def __init__(self, levels: int = 2, sigma: float = 0.45, loss_type: str = "mse"):
        """Initialize LQM settings."""
        super().__init__()
        self.levels = max(int(levels), 0)
        self.sigma = max(float(sigma), 1e-3)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"mse", "smoothl1"}:
            raise ValueError("loc_quality_loss must be 'mse' or 'smoothl1'.")

    def forward(
        self,
        loc_maps: list[torch.Tensor],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LQM loss from predicted maps and image-space GT boxes."""
        losses = []
        num_levels = min(self.levels, len(loc_maps), len(strides))
        for level, loc_map in enumerate(loc_maps[:num_levels]):
            bs, _, h, w = loc_map.shape
            stride = strides[level].to(device=loc_map.device, dtype=loc_map.dtype).clamp(min=1)
            y, x = torch.meshgrid(
                torch.arange(h, device=loc_map.device, dtype=loc_map.dtype) + 0.5,
                torch.arange(w, device=loc_map.device, dtype=loc_map.dtype) + 0.5,
                indexing="ij",
            )
            target = torch.zeros((bs, 1, h, w), device=loc_map.device, dtype=loc_map.dtype)

            for bi in range(bs):
                boxes = gt_bboxes[bi][mask_gt[bi, :, 0].bool()]
                if boxes.numel() == 0:
                    continue
                image_target = target[bi, 0]
                for box in boxes:
                    x1, y1, x2, y2 = box / stride
                    x1 = x1.clamp(0, w)
                    x2 = x2.clamp(0, w)
                    y1 = y1.clamp(0, h)
                    y2 = y2.clamp(0, h)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < 1 or bh < 1:
                        continue

                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    sigma_x = (bw * self.sigma).clamp(min=1e-3)
                    sigma_y = (bh * self.sigma).clamp(min=1e-3)
                    inside = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                    quality = torch.exp(
                        -0.5 * (((x - cx) / sigma_x).pow(2) + ((y - cy) / sigma_y).pow(2))
                    ) * inside.to(loc_map.dtype)
                    image_target.copy_(torch.maximum(image_target, quality))

            pred = loc_map.sigmoid()
            if self.loss_type == "smoothl1":
                losses.append(F.smooth_l1_loss(pred, target))
            else:
                losses.append(F.mse_loss(pred, target))

        return torch.stack(losses).mean() if losses else gt_bboxes.sum() * 0.0
