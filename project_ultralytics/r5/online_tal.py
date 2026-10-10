"""Online TAL difficulty feedback for AS-NCCP R5.

The collector is intentionally detached from gradients and from the assigner
implementation. It consumes the decoded predictions and the assignment result
that the detector loss already has, then emits per-GT difficulty observations
for the epoch-level controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .scale_bins import DEFAULT_SCALE_BINS, ScaleBinSpec


def _aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute aligned IoU for tensors shaped ``[..., 4]`` in xyxy coordinates."""
    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area1 + area2 - intersection).clamp_min(eps)


@dataclass(frozen=True)
class TALDifficultyBatch:
    quality: torch.Tensor
    difficulty: torch.Tensor
    valid_gt: torch.Tensor
    bin_ids: torch.Tensor
    values: torch.Tensor


class TALDifficultyCollector:
    """Collect per-GT TAL IoU or classification-times-IoU difficulty."""

    def __init__(
        self,
        bin_spec: ScaleBinSpec = DEFAULT_SCALE_BINS,
        mode: str = "iou",
        scores_are_logits: bool = False,
    ) -> None:
        if mode not in {"iou", "alignment"}:
            raise ValueError("mode must be 'iou' or 'alignment'")
        self.bin_spec = bin_spec
        self.mode = mode
        self.scores_are_logits = bool(scores_are_logits)

    @torch.no_grad()
    def collect(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        valid_gt_mask: torch.Tensor,
        original_gt_mask: torch.Tensor | None = None,
    ) -> TALDifficultyBatch:
        """Return detached per-GT quality and flattened bin observations.

        All boxes must be in the same xyxy coordinate space. ``pred_scores``
        must be sigmoid probabilities unless ``scores_are_logits=True``.
        GTs with no assigned positive receive quality 0 and difficulty 1.
        Invalid or synthetic GTs are excluded from ``values``.
        """
        if pred_bboxes.ndim != 3 or pred_bboxes.shape[-1] != 4:
            raise ValueError("pred_bboxes must have shape [B, A, 4]")
        if gt_bboxes.ndim != 3 or gt_bboxes.shape[-1] != 4:
            raise ValueError("gt_bboxes must have shape [B, G, 4]")
        batch_size, anchors = pred_bboxes.shape[:2]
        if gt_bboxes.shape[0] != batch_size or pred_scores.shape[:2] != (batch_size, anchors):
            raise ValueError("prediction and GT batch dimensions do not match")
        gt_count = gt_bboxes.shape[1]
        if gt_count == 0:
            empty = gt_bboxes.new_zeros((batch_size, 0))
            return TALDifficultyBatch(empty, empty, empty.bool(), empty.long(), empty)
        if fg_mask.shape != (batch_size, anchors) or target_gt_idx.shape != (batch_size, anchors):
            raise ValueError("fg_mask and target_gt_idx must have shape [B, A]")
        valid = valid_gt_mask.squeeze(-1) if valid_gt_mask.ndim == 3 else valid_gt_mask
        if valid.shape != (batch_size, gt_count):
            raise ValueError("valid_gt_mask must have shape [B, G] or [B, G, 1]")
        valid = valid.bool()
        if original_gt_mask is not None:
            original = original_gt_mask.squeeze(-1) if original_gt_mask.ndim == 3 else original_gt_mask
            if original.shape != valid.shape:
                raise ValueError("original_gt_mask must match valid_gt_mask")
            valid = valid & original.bool()

        safe_gt_idx = target_gt_idx.long().clamp(0, gt_count - 1)
        assigned_gt_boxes = torch.gather(
            gt_bboxes,
            1,
            safe_gt_idx.unsqueeze(-1).expand(-1, -1, 4),
        )
        iou = _aligned_iou(pred_bboxes, assigned_gt_boxes)
        positive = fg_mask.bool() & valid.gather(1, safe_gt_idx)
        values = iou
        if self.mode == "alignment":
            labels = gt_labels.squeeze(-1) if gt_labels.ndim == 3 else gt_labels
            if labels.shape != (batch_size, gt_count):
                raise ValueError("gt_labels must have shape [B, G] or [B, G, 1]")
            class_ids = labels.long().clamp(0, pred_scores.shape[-1] - 1)
            assigned_classes = torch.gather(class_ids, 1, safe_gt_idx)
            class_scores = torch.gather(pred_scores, 2, assigned_classes.unsqueeze(-1)).squeeze(-1)
            if self.scores_are_logits:
                class_scores = class_scores.sigmoid()
            values = iou * class_scores.clamp(0.0, 1.0)

        group_ids = torch.arange(batch_size, device=pred_bboxes.device).unsqueeze(1) * gt_count + safe_gt_idx
        group_ids = group_ids.reshape(-1)
        positive_flat = positive.reshape(-1)
        quality_flat = values.reshape(-1)
        quality = pred_bboxes.new_zeros(batch_size * gt_count)
        if positive_flat.any():
            quality.scatter_reduce_(0, group_ids[positive_flat], quality_flat[positive_flat], reduce="amax", include_self=True)
        quality = quality.reshape(batch_size, gt_count).clamp(0.0, 1.0)
        difficulty = (1.0 - quality).masked_fill(~valid, 0.0)

        widths = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_min(0.0)
        heights = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_min(0.0)
        sizes = (widths * heights).sqrt()
        edges = gt_bboxes.new_tensor(self.bin_spec.edges[1:])
        bin_ids = torch.bucketize(sizes, edges, right=True).clamp_max(self.bin_spec.num_bins - 1)
        in_range = (sizes >= self.bin_spec.edges[0]) & (sizes <= self.bin_spec.edges[-1])
        observed = valid & in_range
        return TALDifficultyBatch(
            quality=quality,
            difficulty=difficulty,
            valid_gt=valid,
            bin_ids=bin_ids[observed].long(),
            values=difficulty[observed].to(dtype=torch.float32),
        )


class R5EpochFeedbackAdapter:
    """Bridge loss-time TAL observations to the epoch-level R5 sampler update."""

    def __init__(
        self,
        transform,
        controller,
        collector: TALDifficultyCollector | None = None,
        frequency=None,
        hybrid_ratio: float = 1.0,
        shared_state=None,
        warmup_epochs: int = 0,
        log_path: str | Path | None = None,
        feasible_provider=None,
    ) -> None:
        self.transform = transform
        self.controller = controller
        self.collector = collector or TALDifficultyCollector()
        self.frequency = frequency
        self.hybrid_ratio = float(hybrid_ratio)
        self.shared_state = shared_state
        self.warmup_epochs = int(warmup_epochs)
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        self.log_path = Path(log_path) if log_path else None
        self.feasible_provider = feasible_provider
        self.observations = 0
        self.last_probabilities: list[float] | None = None

    @torch.no_grad()
    def observe(self, **kwargs: Any) -> TALDifficultyBatch:
        batch = self.collector.collect(**kwargs)
        if batch.values.numel():
            self.controller.accumulate(
                batch.bin_ids.detach().cpu().tolist(),
                batch.values.detach().cpu().tolist(),
            )
            self.observations += int(batch.values.numel())
        return batch

    def _resolve_transform(self, trainer=None):
        if self.transform is not None:
            return self.transform
        dataset = getattr(getattr(trainer, "train_loader", None), "dataset", None)
        stack = [getattr(dataset, "transforms", None)]
        seen = set()
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            if hasattr(current, "end_epoch_from_controller"):
                self.transform = current
                return current
            children = getattr(current, "transforms", None)
            if children:
                stack.extend(children)
        return None

    def _feasible(self, transform):
        if callable(self.feasible_provider):
            return self.feasible_provider(transform)
        values = getattr(transform, "donor_feasible_bin_values", None)
        sampler = getattr(transform, "scale_sampler", None)
        if values is not None and sampler is not None:
            return sampler.feasible_mask(values)
        return None

    def _log_epoch(self, probabilities: list[float], *, epoch: int, transform) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "epoch": int(epoch),
            "observations": int(self.observations),
            "difficulty": self.controller.difficulty.tolist(),
            "probabilities": probabilities,
            "warmup": bool(epoch <= self.warmup_epochs),
            "transform": type(transform).__name__ if transform is not None else None,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def end_epoch(self, trainer=None) -> list[float]:
        transform = self._resolve_transform(trainer)
        epoch = int(getattr(self.controller, "epoch", 0)) + 1
        if transform is not None and self.warmup_epochs == 0 and hasattr(transform, "end_epoch_from_controller"):
            probabilities = transform.end_epoch_from_controller(
                self.controller,
                frequency=self.frequency,
                hybrid_ratio=self.hybrid_ratio,
            )
        else:
            self.controller.end_epoch()
            feasible = self._feasible(transform)
            if feasible is None:
                feasible = [True] * self.controller.num_bins
            if epoch <= self.warmup_epochs:
                import numpy as np

                mask = np.asarray(list(feasible), dtype=bool)
                probabilities_array = np.zeros(self.controller.num_bins, dtype=np.float64)
                probabilities_array[mask] = 1.0 / max(int(mask.sum()), 1)
            else:
                probabilities_array = self.controller.probabilities(
                    frequency=self.frequency,
                    hybrid_ratio=self.hybrid_ratio,
                    feasible=feasible,
                )
            probabilities = probabilities_array.tolist()
            if transform is not None and hasattr(transform, "update_probabilities"):
                transform.update_probabilities(probabilities)
            elif self.shared_state is not None:
                self.shared_state.update(probabilities)
        self.last_probabilities = list(probabilities)
        if self.shared_state is not None:
            self.shared_state.update(self.last_probabilities)
        self._log_epoch(self.last_probabilities, epoch=epoch, transform=transform)
        return self.last_probabilities

    def state_dict(self) -> dict[str, object]:
        return {
            "controller": self.controller.state_dict(),
            "observations": self.observations,
            "last_probabilities": self.last_probabilities,
            "warmup_epochs": self.warmup_epochs,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.controller.load_state_dict(state["controller"])
        self.observations = int(state.get("observations", 0))
        values = state.get("last_probabilities")
        self.last_probabilities = None if values is None else [float(value) for value in values]

    def on_train_epoch_end(self, trainer=None) -> list[float]:
        return self.end_epoch(trainer=trainer)


def register_r5_epoch_callback(owner, adapter: R5EpochFeedbackAdapter) -> bool:
    """Register one adapter callback on a model, trainer, or callback owner."""
    add_callback = getattr(owner, "add_callback", None)
    if not callable(add_callback):
        return False
    callbacks = getattr(owner, "callbacks", {})
    event_callbacks = callbacks.get("on_train_epoch_end", []) if isinstance(callbacks, dict) else []
    callback = adapter.on_train_epoch_end
    if callback not in event_callbacks:
        add_callback("on_train_epoch_end", callback)
    return True


def attach_r5_feedback(model, adapter: R5EpochFeedbackAdapter):
    """Attach an adapter to a model and register its epoch-end callback."""
    model.r5_feedback_adapter = adapter
    register_r5_epoch_callback(model, adapter)
    return model


__all__ = [
    "R5EpochFeedbackAdapter",
    "TALDifficultyBatch",
    "TALDifficultyCollector",
    "attach_r5_feedback",
    "register_r5_epoch_callback",
]
