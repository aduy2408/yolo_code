"""Dataset-derived soft gating for scene-compatible Mosaic augmentation.

The wrapper intentionally does not change Mosaic geometry. It only scores the
candidate produced by an existing Mosaic transform and probabilistically keeps
or rejects that candidate.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np


FEATURES = ("relative_size", "spacing", "occupancy")
_EPS = 1e-9


def _as_xyxy(boxes: Any, shape: tuple[int, int], normalized: bool = False, fmt: str = "xywh") -> np.ndarray:
    """Return finite absolute xyxy boxes from common Ultralytics label formats."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return boxes
    h, w = shape
    if fmt == "xywh":
        x, y, bw, bh = boxes.T
        boxes = np.stack((x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2), axis=1)
    elif fmt != "xyxy":
        raise ValueError(f"Unsupported bbox format: {fmt}")
    if normalized:
        boxes = boxes * np.array([w, h, w, h], dtype=np.float32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def _descriptor_from_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> dict[str, float | None]:
    """Compute [N, R, S, O] from absolute xyxy boxes."""
    h, w = shape
    image_area = max(float(h * w), _EPS)
    if boxes is None or len(boxes) == 0:
        return {"count": 0.0, "relative_size": None, "spacing": None, "occupancy": 0.0}

    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    areas = widths * heights
    valid = areas > 0
    areas = areas[valid]
    if not len(areas):
        return {"count": 0.0, "relative_size": None, "spacing": None, "occupancy": 0.0}

    centers = np.column_stack(((boxes[valid, 0] + boxes[valid, 2]) / 2, (boxes[valid, 1] + boxes[valid, 3]) / 2))
    relative_sizes = np.sqrt(areas) / np.sqrt(image_area)
    spacing = None
    if len(areas) >= 2:
        deltas = centers[:, None, :] - centers[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        distances[distances == 0] = np.inf
        nearest = distances.min(axis=1)
        spacing = float(np.median(nearest / (np.sqrt(areas) + _EPS)))
    return {
        "count": float(len(areas)),
        "relative_size": float(np.median(relative_sizes)),
        "spacing": spacing,
        "occupancy": float(np.sum(areas) / image_area),
    }


def scene_descriptor(labels: dict[str, Any]) -> dict[str, float | None]:
    """Compute the scene descriptor from an Ultralytics label dictionary."""
    if "instances" in labels:
        instances = labels["instances"]
        shape = labels.get("resized_shape") or labels.get("img", np.zeros((1, 1))).shape[:2]
        boxes = getattr(instances, "bboxes", instances)
        normalized = bool(getattr(instances, "normalized", False))
        fmt = str(getattr(instances, "format", "xyxy"))
        return _descriptor_from_boxes(_as_xyxy(boxes, shape, normalized, fmt), shape)

    shape = tuple(labels.get("shape") or labels.get("ori_shape") or (1, 1))
    boxes = _as_xyxy(
        labels.get("bboxes", np.empty((0, 4))),
        shape,
        normalized=bool(labels.get("normalized", False)),
        fmt=str(labels.get("bbox_format", "xywh")),
    )
    return _descriptor_from_boxes(boxes, shape)


def build_reference_stats(dataset_labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Build positive-scene q05/q50/q95 support statistics from real labels."""
    values = {feature: [] for feature in FEATURES}
    empty_count = 0
    positive_count = 0
    for labels in dataset_labels:
        descriptor = scene_descriptor(labels)
        if descriptor["count"] == 0:
            empty_count += 1
            continue
        positive_count += 1
        for feature in FEATURES:
            value = descriptor[feature]
            if value is not None and np.isfinite(value):
                values[feature].append(float(value))

    stats: dict[str, Any] = {
        "count": len(dataset_labels),
        "empty_count": empty_count,
        "positive_count": positive_count,
        "p_empty": empty_count / len(dataset_labels) if dataset_labels else 0.0,
        "features": {},
    }
    for feature in FEATURES:
        array = np.asarray(values[feature], dtype=np.float64)
        if not len(array):
            stats["features"][feature] = {"q05": None, "q50": None, "q95": None, "valid_count": 0}
            continue
        q05, q50, q95 = np.quantile(array, [0.05, 0.50, 0.95])
        stats["features"][feature] = {
            "q05": float(q05), "q50": float(q50), "q95": float(q95), "valid_count": int(len(array))
        }
    return stats


def feature_drift(value: float | None, stats: dict[str, Any]) -> float | None:
    """Penalize only values outside the symmetric q05-q95 positive support."""
    if value is None or stats.get("q05") is None:
        return None
    q05, q95 = (float(stats[key]) for key in ("q05", "q95"))
    scale = max(q95 - q05, _EPS)
    if value < q05:
        return max(0.0, (q05 - value) / scale)
    if value > q95:
        return max(0.0, (value - q95) / scale)
    return 0.0


def compatibility_drift(descriptor: dict[str, float | None], reference: dict[str, Any]) -> tuple[float, dict[str, float | None]]:
    """Return mean valid positive-scene feature drift and per-feature diagnostics."""
    if float(descriptor.get("count") or 0.0) == 0.0:
        return 0.0, {feature: None for feature in FEATURES}
    drifts = {feature: feature_drift(descriptor.get(feature), reference["features"][feature]) for feature in FEATURES}
    valid = [value for value in drifts.values() if value is not None]
    return (float(np.mean(valid)) if valid else 0.0), drifts


def _clone_labels(labels: dict[str, Any]) -> dict[str, Any]:
    """Clone mutable annotation state while sharing the immutable image array."""
    cloned = dict(labels)
    for key in ("instances", "semantic_mask", "cls", "bboxes", "segments", "keypoints", "mix_labels"):
        if key in cloned:
            cloned[key] = copy.deepcopy(cloned[key])
    return cloned


class SceneCompatibleMosaic:
    """Soft-gate an existing Mosaic transform using dataset-derived scene support."""

    def __init__(self, dataset, mosaic: Callable, p: float = 1.0) -> None:
        self.dataset = dataset
        self.mosaic = mosaic
        self.p = float(p)
        self.reference = build_reference_stats(getattr(dataset, "labels", []))
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> dict[str, Any]:
        return {
            "candidates": 0,
            "accepted": 0,
            "rejected": 0,
            "drift_sum": 0.0,
            "drift_by_feature": {feature: {"sum": 0.0, "count": 0} for feature in FEATURES},
            "accepted_gt_sum": 0.0,
            "accepted_gt_count": 0,
            "rejected_gt_sum": 0.0,
            "rejected_gt_count": 0,
        }

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.p <= 0.0 or random.random() > self.p:
            return labels

        original = _clone_labels(labels)
        candidate = self.mosaic(_clone_labels(labels))
        descriptor = scene_descriptor(candidate)
        drift, feature_drifts = compatibility_drift(descriptor, self.reference)
        p_accept = float(np.exp(-drift))
        accepted = random.random() < p_accept

        self._stats["candidates"] += 1
        self._stats["drift_sum"] += drift
        for feature, value in feature_drifts.items():
            if value is not None:
                item = self._stats["drift_by_feature"][feature]
                item["sum"] += value
                item["count"] += 1
        count = float(descriptor["count"] or 0.0)
        key = "accepted" if accepted else "rejected"
        self._stats[key] += 1
        self._stats[f"{key}_gt_sum"] += count
        self._stats[f"{key}_gt_count"] += 1
        return candidate if accepted else original

    def diagnostics(self) -> dict[str, Any]:
        stats = self._stats
        candidates = stats["candidates"]
        accepted = stats["accepted"]
        output = {
            "scm/candidates": candidates,
            "scm/accepted": accepted,
            "scm/rejected": stats["rejected"],
            "scm/accept_rate": accepted / candidates if candidates else 0.0,
            "scm/drift_mean": stats["drift_sum"] / candidates if candidates else 0.0,
            "scm/accepted_gt_mean": stats["accepted_gt_sum"] / stats["accepted_gt_count"] if stats["accepted_gt_count"] else 0.0,
            "scm/rejected_gt_mean": stats["rejected_gt_sum"] / stats["rejected_gt_count"] if stats["rejected_gt_count"] else 0.0,
            "scm/reference_images": self.reference["count"],
            "scm/reference_positive_scenes": self.reference["positive_count"],
            "scm/reference_empty_scenes": self.reference["empty_count"],
            "scm/reference_p_empty": self.reference["p_empty"],
        }
        for feature in FEATURES:
            item = stats["drift_by_feature"][feature]
            diagnostic_name = "size" if feature == "relative_size" else feature
            output[f"scm/drift_{diagnostic_name}"] = item["sum"] / item["count"] if item["count"] else 0.0
            ref = self.reference["features"][feature]
            output[f"scm/reference_{diagnostic_name}_q05"] = ref["q05"]
            output[f"scm/reference_{diagnostic_name}_q50"] = ref["q50"]
            output[f"scm/reference_{diagnostic_name}_q95"] = ref["q95"]
        return {"config": {"p": self.p, "reference": self.reference}, "stats": output}

    def save_diagnostics(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.diagnostics(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "SceneCompatibleMosaic",
    "build_reference_stats",
    "compatibility_drift",
    "feature_drift",
    "scene_descriptor",
]
