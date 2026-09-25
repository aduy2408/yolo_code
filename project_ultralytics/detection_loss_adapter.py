"""Project-owned adapters for custom detection loss on clean Ultralytics."""

from __future__ import annotations

from typing import Any


def _arg(obj: Any, name: str, default: Any) -> Any:
    """Read an attribute from namespace-style or dict-style model args."""
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


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
from .assignment import CollisionPreservingTaskAlignedAssigner
from .hardness import foreground_assignment_hardness
from .online_negative_bank import deduplicate_candidate_indices


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
        self.factorized_tal_enabled = bool(_arg(h, "factorized_tal_target", False))
        self.factorized_tal_config = FactorizedTALConfig(
            tau=float(_arg(h, "factorized_tal_tau", 0.75)),
            kappa=float(_arg(h, "factorized_tal_kappa", 1.5)),
            lambda_=float(_arg(h, "factorized_tal_lambda", 0.5)),
            small_object_max_size=float(_arg(h, "factorized_tal_s_max", 32.0)),
            warmup_start=int(_arg(h, "factorized_tal_warmup_start", 5)),
            warmup_end=int(_arg(h, "factorized_tal_warmup_end", 15)),
            p2_only=bool(_arg(h, "factorized_tal_p2_only", True)),
        )
        self.scale_temper_enabled = bool(_arg(h, "scale_temper_target", False))
        self.scale_temper_config = ScaleTemperedTALConfig(
            s1=float(_arg(h, "scale_temper_s1", 16.0)),
            s2=float(_arg(h, "scale_temper_s2", 32.0)),
            tau_min=float(_arg(h, "scale_temper_tau_min", 0.5)),
            lambda_=float(_arg(h, "scale_temper_lambda", 0.5)),
            warmup_start=int(_arg(h, "scale_temper_warmup_start", 5)),
            warmup_end=int(_arg(h, "scale_temper_warmup_end", 15)),
            p2_only=bool(_arg(h, "scale_temper_p2_only", True)),
        )
        self.positive_rescue_gain = float(_arg(h, "positive_confidence_rescue_gain", 0.0))
        self.positive_rescue_gamma = float(_arg(h, "positive_confidence_rescue_gamma", 1.0))
        self.custom_detection_metrics: dict[str, float] = {}

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Run upstream assignment/loss with project target shaping inserted."""
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        p2_slot_count = int(preds.get("p2_slot_count", 1))
        p2_base_count = int(preds.get("p2_base_count", 0))
        if "anchor_points" in preds and "stride_tensor" in preds:
            anchor_points = preds["anchor_points"]
            stride_tensor = preds["stride_tensor"]
        else:
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
        if p2_slot_count > 1:
            p2_fg = fg_mask[:, : p2_slot_count * p2_base_count].view(
                fg_mask.shape[0], p2_slot_count, p2_base_count
            )
            positive_locations = p2_fg.any(dim=1)
            dual_locations = p2_fg.sum(dim=1) >= 2
            dual_denominator = positive_locations.sum().clamp_min(1)
            dual_numerator = dual_locations.sum()
            different_gt = torch.zeros((), device=fg_mask.device, dtype=torch.float32)
            different_count = torch.zeros((), device=fg_mask.device, dtype=torch.float32)
            slot_gt = target_gt_idx[:, : p2_slot_count * p2_base_count].view(
                target_gt_idx.shape[0], p2_slot_count, p2_base_count
            )
            for batch_index in range(fg_mask.shape[0]):
                for location in torch.where(dual_locations[batch_index])[0]:
                    assigned = slot_gt[batch_index, :, location][p2_fg[batch_index, :, location]]
                    different_gt += float(assigned.unique().numel() > 1)
                    different_count += 1.0
            self.custom_detection_metrics.update(
                {
                    "p2_dual_occupancy": (dual_numerator / dual_denominator).detach(),
                    "p2_different_gt_occupancy": (
                        different_gt / different_count.clamp_min(1)
                    ).detach(),
                }
            )
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
        if p2_slot_count > 1:
            candidate_weight = torch.ones(
                pred_scores.shape[:2], device=pred_scores.device, dtype=pred_scores.dtype
            )
            p2_negative = ~fg_mask[:, : p2_slot_count * p2_base_count].bool()
            candidate_weight[:, : p2_slot_count * p2_base_count] = torch.where(
                p2_negative,
                torch.full_like(candidate_weight[:, : p2_slot_count * p2_base_count], 1.0 / p2_slot_count),
                torch.ones_like(candidate_weight[:, : p2_slot_count * p2_base_count]),
            )
            bce_loss = bce_loss * candidate_weight.unsqueeze(-1)
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

        loss[0] *= _arg(self.hyp, "box", 1.0)
        loss[1] *= _arg(self.hyp, "cls", 1.0)
        loss[2] *= _arg(self.hyp, "dfl", 1.0)
        # Keep the optional OACP feedback path available when FTAL overrides
        # the upstream assignment/loss seam. This statistic is detached and
        # never contributes to the training objective.
        with torch.no_grad():
            self.last_per_image_hardness = foreground_assignment_hardness(
                pred_scores, target_scores, fg_mask
            )
            self.last_hard_negative_candidates = []
            threshold = float(_arg(self.hyp, "online_negcp_conf_threshold", 0.25))
            limit = int(_arg(self.hyp, "online_negcp_max_candidates", 3))
            scores = pred_scores.detach().sigmoid().amax(dim=-1)
            boxes = (pred_bboxes.detach() * stride_tensor).detach()
            for batch_idx in range(pred_scores.shape[0]):
                if mask_gt[batch_idx].any():
                    self.last_hard_negative_candidates.append([])
                    continue
                indices = deduplicate_candidate_indices(
                    boxes[batch_idx], scores[batch_idx], (~fg_mask[batch_idx].bool()) & (scores[batch_idx] >= threshold), limit
                )
                rows = []
                for index in indices:
                    confidence = scores[batch_idx, index].clamp_max(1 - 1e-6)
                    rows.append([
                        *boxes[batch_idx, index].float().cpu().tolist(),
                        float(confidence.item()),
                        float(-torch.log1p(-confidence).item()),
                    ])
                self.last_hard_negative_candidates.append(rows)
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class P2SlotsDetectionLoss(FactorizedTALDetectionLoss):
    """Project loss adapter for P2SlotsDetect with matched negative weighting."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.assigner = CollisionPreservingTaskAlignedAssigner(
            topk=int(getattr(self.hyp, "tal_topk", tal_topk)),
            num_classes=self.nc,
            alpha=float(getattr(self.hyp, "tal_alpha", 0.5)),
            beta=float(getattr(self.hyp, "tal_beta", 6.0)),
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        self.assigner.p2_base_count = int(preds.get("p2_base_count", 0))
        self.assigner.p2_slot_count = int(preds.get("p2_slot_count", 1))
        return super().get_assigned_targets_and_loss(preds, batch)
