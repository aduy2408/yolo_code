"""Project-owned target transforms for custom detection loss experiments.

This module contains the tensor-only pieces of the legacy detection-loss fork.
It intentionally does not subclass or patch Ultralytics' ``v8DetectionLoss``.
A trainer adapter can call these transforms around upstream TAL assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FactorizedTALConfig:
    """Configuration for small-object Factorized TAL target shaping."""

    tau: float = 0.75
    kappa: float = 1.5
    lambda_: float = 0.5
    small_object_max_size: float = 32.0
    warmup_start: int = 5
    warmup_end: int = 15
    p2_only: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.kappa <= 0 or not 0 <= self.lambda_ <= 1:
            raise ValueError("kappa must be positive and lambda_ must be in [0, 1]")
        if self.small_object_max_size <= 0:
            raise ValueError("small_object_max_size must be positive")
        if self.warmup_end < self.warmup_start:
            raise ValueError("warmup_end must be >= warmup_start")

    def gain_at(self, epoch: int) -> float:
        """Return linearly warmed-up target shaping gain."""
        if self.lambda_ <= 0 or epoch < self.warmup_start:
            return 0.0
        if self.warmup_end <= self.warmup_start:
            return self.lambda_
        ramp = (epoch - self.warmup_start) / (self.warmup_end - self.warmup_start)
        return self.lambda_ * min(max(float(ramp), 0.0), 1.0)


@dataclass(frozen=True)
class ScaleTemperedTALConfig:
    """Configuration for the earlier scale-tempered TAL target transform."""

    s1: float = 16.0
    s2: float = 32.0
    tau_min: float = 0.5
    lambda_: float = 0.5
    warmup_start: int = 5
    warmup_end: int = 15
    p2_only: bool = True

    def __post_init__(self) -> None:
        if self.s2 <= self.s1:
            raise ValueError("s2 must be greater than s1")
        if not 0 < self.tau_min <= 1:
            raise ValueError("tau_min must be in (0, 1]")
        if not 0 <= self.lambda_ <= 1:
            raise ValueError("lambda_ must be in [0, 1]")

    def gain_at(self, epoch: int) -> float:
        """Return linearly warmed-up target shaping gain."""
        if self.lambda_ <= 0 or epoch < self.warmup_start:
            return 0.0
        if self.warmup_end <= self.warmup_start:
            return self.lambda_
        ramp = (epoch - self.warmup_start) / (self.warmup_end - self.warmup_start)
        return self.lambda_ * min(max(float(ramp), 0.0), 1.0)


def factorize_tal_targets(
    q: torch.Tensor,
    u: torch.Tensor,
    *,
    tau: float = 0.75,
    kappa: float = 1.5,
    gain: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the historical Factorized TAL transform to one GT's targets.

    ``q`` is the assigned TAL classification target and ``u`` is the detached
    aligned IoU quality. The transform preserves non-positive targets.
    """
    eps = 1e-12
    q_max = q.max().clamp_min(eps)
    q_new = q_max.pow(tau) * (q / q_max).clamp(0, 1).pow(kappa)
    return q + gain * (torch.where(q > 0, q_new, q) - q), {}


def _aligned_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute aligned IoU for two ``[..., 4]`` xyxy tensors."""
    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(-1)
    return intersection / (area1 + area2 - intersection).clamp_min(1e-12)


def factorized_tal_cls_targets(
    target_scores: torch.Tensor,
    gt_bboxes: torch.Tensor,
    target_gt_idx: torch.Tensor,
    fg_mask: torch.Tensor,
    n_p2: int,
    pred_bboxes: torch.Tensor,
    stride_tensor: torch.Tensor,
    *,
    config: FactorizedTALConfig = FactorizedTALConfig(),
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Shape TAL targets for small GTs and return diagnostic metrics."""
    gain = config.gain_at(epoch)
    if gain <= 0 or not fg_mask.any():
        return target_scores, {"n_small_gt": 0.0}

    pos_mask = fg_mask.clone()
    if config.p2_only:
        pos_mask[:, n_p2:] = False
    out = target_scores.clone()
    metric_count = 0

    for batch_idx in range(target_scores.shape[0]):
        for gt_idx in target_gt_idx[batch_idx, pos_mask[batch_idx]].unique():
            group = pos_mask[batch_idx] & (target_gt_idx[batch_idx] == gt_idx)
            if not group.any():
                continue
            box = gt_bboxes[batch_idx, gt_idx]
            size = (box[2:] - box[:2]).clamp_min(1e-6).prod().sqrt()
            if size >= config.small_object_max_size:
                continue
            q = target_scores[batch_idx, group]
            gt_box = (box / stride_tensor[group].reshape(-1, 1)).detach()
            aligned_iou = _aligned_iou_xyxy(pred_bboxes[batch_idx, group].detach(), gt_box)
            q_new, metrics = factorize_tal_targets(q, aligned_iou, tau=config.tau, kappa=config.kappa, gain=gain)
            out[batch_idx, group] = torch.where(q > 0, q_new, q)
            metric_count += 1

    metrics = {"n_small_gt": float(metric_count)}
    return out, metrics


def scale_tempered_cls_targets(
    target_scores: torch.Tensor,
    gt_bboxes: torch.Tensor,
    target_gt_idx: torch.Tensor,
    fg_mask: torch.Tensor,
    n_p2: int,
    *,
    config: ScaleTemperedTALConfig = ScaleTemperedTALConfig(),
    epoch: int = 0,
) -> torch.Tensor:
    """Raise low positive TAL targets according to GT image-space size."""
    gain = config.gain_at(epoch)
    if gain <= 0 or not fg_mask.any():
        return target_scores
    out = target_scores.clone()
    pos_mask = fg_mask.clone()
    if config.p2_only:
        pos_mask[:, n_p2:] = False

    for batch_idx in range(target_scores.shape[0]):
        mask = pos_mask[batch_idx]
        if not mask.any():
            continue
        boxes = gt_bboxes[batch_idx, target_gt_idx[batch_idx, mask]]
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
        size = wh.prod(-1).sqrt()
        tau = config.tau_min + (1.0 - config.tau_min) * ((size - config.s1) / (config.s2 - config.s1)).clamp(0, 1)
        q = target_scores[batch_idx, mask]
        positive = q > 0
        out[batch_idx, mask] = torch.where(positive, q + gain * (q.clamp_min(1e-12).pow(tau[:, None]) - q), q)
    return out


def positive_confidence_rescue_loss(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    fg_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rescue loss for low-confidence TAL positives."""
    positive_targets, positive_classes = target_scores.max(dim=-1)
    positive_logits = pred_scores.gather(-1, positive_classes.unsqueeze(-1)).squeeze(-1)
    target = positive_targets[fg_mask].detach().to(pred_scores.dtype)
    logits = positive_logits[fg_mask]
    raw = ((1.0 - target).pow(gamma) * F.softplus(-logits)).sum()
    return raw / fg_mask.sum().clamp_min(1), target, logits
