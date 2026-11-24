"""Scale-binned recall evaluation shared by R5-A and R5-E.

The evaluator is deliberately independent from Ultralytics internals. Callers pass
post-NMS predictions and ground truth boxes in one coordinate system. Matching is
class-aware, confidence-ordered, and one-to-one.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .scale_bins import DEFAULT_SCALE_BINS, ScaleBinSpec


def _as_boxes(value: Any) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    boxes = boxes.reshape(-1, 4)
    if not np.all(np.isfinite(boxes)):
        raise ValueError("boxes must contain only finite values")
    return boxes


def _as_classes(value: Any, count: int) -> np.ndarray:
    classes = np.asarray(value, dtype=np.int64).reshape(-1)
    if classes.shape != (count,):
        raise ValueError("class labels must match the number of boxes")
    return classes


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-12)


def match_predictions(
    pred_boxes: Any,
    pred_scores: Any,
    pred_classes: Any,
    gt_boxes: Any,
    gt_classes: Any,
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched GT indices and TP flags for each prediction.

    Predictions are sorted by confidence descending. A prediction can match only
    one still-unmatched GT of the same class.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    predictions = _as_boxes(pred_boxes)
    scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    classes = _as_classes(pred_classes, len(predictions))
    if scores.shape != (len(predictions),) or not np.all(np.isfinite(scores)):
        raise ValueError("prediction scores must match boxes and be finite")
    gt = _as_boxes(gt_boxes)
    gt_labels = _as_classes(gt_classes, len(gt))
    keep = scores >= confidence_threshold
    order = np.argsort(-scores[keep], kind="stable")
    kept_indices = np.flatnonzero(keep)[order]
    matched = np.full(len(predictions), -1, dtype=np.int64)
    true_positive = np.zeros(len(predictions), dtype=bool)
    used_gt: set[int] = set()
    for prediction_index in kept_indices:
        candidates = [
            gt_index
            for gt_index, label in enumerate(gt_labels)
            if gt_index not in used_gt and int(label) == int(classes[prediction_index])
        ]
        if not candidates:
            continue
        best_gt = max(candidates, key=lambda index: box_iou(predictions[prediction_index], gt[index]))
        if box_iou(predictions[prediction_index], gt[best_gt]) >= iou_threshold:
            used_gt.add(best_gt)
            matched[prediction_index] = best_gt
            true_positive[prediction_index] = True
    return matched, true_positive


@dataclass(frozen=True)
class RecallProfile:
    """Serializable per-scale recall counts and protocol metadata."""

    method: str
    edges: tuple[float, ...]
    gt_count: tuple[int, ...]
    tp_count: tuple[int, ...]
    iou_threshold: float
    confidence_threshold: float
    source_split: str
    image_count: int

    def __post_init__(self) -> None:
        if len(self.gt_count) != len(self.edges) - 1 or len(self.tp_count) != len(self.gt_count):
            raise ValueError("recall counts must match the scale-bin specification")
        if any(count < 0 for count in (*self.gt_count, *self.tp_count)):
            raise ValueError("recall counts must be non-negative")
        if any(tp > gt for tp, gt in zip(self.tp_count, self.gt_count)):
            raise ValueError("true positives cannot exceed ground-truth counts")

    @property
    def global_recall(self) -> float:
        return sum(self.tp_count) / max(sum(self.gt_count), 1)

    def difficulty(self, *, kappa: float = 10.0, gamma: float = 1.0) -> np.ndarray:
        if kappa < 0.0 or not np.isfinite(kappa):
            raise ValueError("kappa must be finite and non-negative")
        if gamma < 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and non-negative")
        numerator = np.asarray(self.tp_count, dtype=np.float64) + kappa * self.global_recall
        denominator = np.asarray(self.gt_count, dtype=np.float64) + kappa
        # A bin with no GT and kappa=0 has undefined recall. Treat it as
        # neutral/easy rather than emitting NaN. Such bins are still subject
        # to the caller's feasible-bin mask.
        recall = np.ones_like(numerator)
        np.divide(numerator, denominator, out=recall, where=denominator > 0.0)
        return np.power(np.clip(1.0 - recall, 0.0, 1.0), gamma)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "bins": list(self.edges),
            "gt_count": list(self.gt_count),
            "tp_count": list(self.tp_count),
            "iou_threshold": self.iou_threshold,
            "confidence_threshold": self.confidence_threshold,
            "source_split": self.source_split,
            "image_count": self.image_count,
            "global_recall": self.global_recall,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RecallProfile":
        return cls(
            method=str(payload["method"]),
            edges=tuple(float(value) for value in payload["bins"]),
            gt_count=tuple(int(value) for value in payload["gt_count"]),
            tp_count=tuple(int(value) for value in payload["tp_count"]),
            iou_threshold=float(payload["iou_threshold"]),
            confidence_threshold=float(payload["confidence_threshold"]),
            source_split=str(payload.get("source_split", "unknown")),
            image_count=int(payload.get("image_count", 0)),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RecallProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class RecallEvaluator:
    """Evaluate class-aware recall and aggregate it into shared scale bins."""

    def __init__(
        self,
        bin_spec: ScaleBinSpec = DEFAULT_SCALE_BINS,
        *,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.01,
    ) -> None:
        self.bin_spec = bin_spec
        self.iou_threshold = float(iou_threshold)
        self.confidence_threshold = float(confidence_threshold)
        if not 0.0 <= self.iou_threshold <= 1.0 or not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("evaluation thresholds must be in [0, 1]")

    def evaluate_records(self, records: Iterable[Mapping[str, Any]], *, method: str, source_split: str) -> RecallProfile:
        gt_count = np.zeros(self.bin_spec.num_bins, dtype=np.int64)
        tp_count = np.zeros(self.bin_spec.num_bins, dtype=np.int64)
        image_count = 0
        for record in records:
            image_count += 1
            gt_boxes = _as_boxes(record.get("gt_boxes", []))
            gt_classes = _as_classes(record.get("gt_classes", []), len(gt_boxes))
            matched, _ = match_predictions(
                record.get("pred_boxes", []),
                record.get("pred_scores", []),
                record.get("pred_classes", []),
                gt_boxes,
                gt_classes,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold,
            )
            widths = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 0.0)
            heights = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1], 0.0)
            sizes = np.sqrt(widths * heights)
            for size in sizes:
                bin_id = self.bin_spec.index(float(size))
                if bin_id is not None:
                    gt_count[int(bin_id)] += 1
            for gt_index in matched[matched >= 0]:
                size = float(sizes[int(gt_index)])
                bin_id = self.bin_spec.index(size)
                if bin_id is not None:
                    tp_count[int(bin_id)] += 1
        return RecallProfile(
            method=method,
            edges=tuple(self.bin_spec.edges),
            gt_count=tuple(int(value) for value in gt_count),
            tp_count=tuple(int(value) for value in tp_count),
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold,
            source_split=source_split,
            image_count=image_count,
        )


__all__ = ["RecallEvaluator", "RecallProfile", "box_iou", "match_predictions"]
