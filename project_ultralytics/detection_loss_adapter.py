"""Project-owned adapters for custom detection loss on clean Ultralytics."""

from __future__ import annotations

from typing import Any

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from .detection_loss import (
    FactorizedTALConfig,
    ScaleTemperedTALConfig,
    factorized_tal_cls_targets,
    positive_confidence_rescue_loss,
    scale_tempered_cls_targets,
)


class FactorizedTALDetectionLoss(v8DetectionLoss):
    """Upstream v8 detection loss with project-owned TAL target transforms.

    The upstream class and vendor package remain untouched. This adapter copies
    only the small assignment seam needed to transform assigned classification
    targets before the standard upstream BCE/box/DFL computation.
    """

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        h = model.args
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss" if self.use_dfl else "l1_loss")
        self.factorized_tal_enabled = bool(getattr(h, "factorized_tal_target", False))
        self.factorized_tal_config = FactorizedTALConfig(
            tau=float(getattr(h, "factorized_tal_tau", 0.75)),
            kappa=float(getattr(h, "factorized_tal_kappa", 1.5)),
            lambda_=float(getattr(h, "factorized_tal_lambda", 0.5)),
            small_object_max_size=float(getattr(h, "factorized_tal_s_max", 32.0)),
            warmup_start=int(getattr(h, "factorized_tal_warmup_start", 5)),
            warmup_end=int(getattr(h, "factorized_tal_warmup_end", 15)),
            p2_only=bool(getattr(h, "factorized_tal_p2_only", True)),
        )
        self.scale_temper_enabled = bool(getattr(h, "scale_temper_target", False))
        self.scale_temper_config = ScaleTemperedTALConfig(
            s1=float(getattr(h, "scale_temper_s1", 16.0)),
            s2=float(getattr(h, "scale_temper_s2", 32.0)),
            tau_min=float(getattr(h, "scale_temper_tau_min", 0.5)),
            lambda_=float(getattr(h, "scale_temper_lambda", 0.5)),
            warmup_start=int(getattr(h, "scale_temper_warmup_start", 5)),
            warmup_end=int(getattr(h, "scale_temper_warmup_end", 15)),
            p2_only=bool(getattr(h, "scale_temper_p2_only", True)),
        )
        self.positive_rescue_gain = float(getattr(h, "positive_confidence_rescue_gain", 0.0))
        self.positive_rescue_gamma = float(getattr(h, "positive_confidence_rescue_gamma", 1.0))
        self.custom_detection_metrics: dict[str, float] = {}

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Run upstream assignment/loss with project target shaping inserted."""
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        self.custom_detection_metrics = {}
        if self.factorized_tal_enabled:
            target_scores, metrics = factorized_tal_cls_targets(
                target_scores,
                gt_bboxes,
                target_gt_idx,
                fg_mask,
                int((stride_tensor == stride_tensor.min()).sum().item()),
                pred_bboxes,
                stride_tensor,
                config=self.factorized_tal_config,
                epoch=int(getattr(self, "epoch", 0)),
            )
            self.custom_detection_metrics.update({f"factorized_{k}": v for k, v in metrics.items()})
        if self.scale_temper_enabled:
            target_scores = scale_tempered_cls_targets(
                target_scores,
                gt_bboxes,
                target_gt_idx,
                fg_mask,
                int((stride_tensor == stride_tensor.min()).sum().item()),
                config=self.scale_temper_config,
                epoch=int(getattr(self, "epoch", 0)),
            )

        target_scores_sum = max(target_scores.sum(), 1)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        if self.positive_rescue_gain > 0:
            rescue, _, _ = positive_confidence_rescue_loss(
                pred_scores, target_scores, fg_mask, gamma=self.positive_rescue_gamma
            )
            loss[1] += self.positive_rescue_gain * rescue
            self.custom_detection_metrics["positive_rescue_loss"] = float(rescue.detach())

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            {**dict(zip(self.loss_names, loss.detach())), **self.custom_detection_metrics},
        )
