"""Adaptive Copy-Paste policies built on the legacy placement primitives.

The module deliberately keeps policy selection separate from image placement.
``AdaptivePasteBudget`` allocates added *objects*; positive policies then choose
the source/scale representation used to spend that budget.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

import cv2
import numpy as np

from .copy_paste import ObjectRecord, SmallObjectCopyPaste, _box_area


@dataclass(frozen=True)
class AdaptivePasteBudget:
    """Allocate an object budget from an empirical scene-count distribution."""

    target_counts: tuple[int, ...]
    max_objects: int = 4

    def __init__(self, target_counts: Iterable[int], max_objects: int = 4) -> None:
        counts = tuple(max(0, int(x)) for x in target_counts)
        if not counts:
            raise ValueError("target_counts must not be empty")
        if max_objects < 0:
            raise ValueError("max_objects must be non-negative")
        object.__setattr__(self, "target_counts", counts)
        object.__setattr__(self, "max_objects", int(max_objects))

    @classmethod
    def from_dataset(cls, dataset, max_objects: int = 4) -> "AdaptivePasteBudget":
        counts = [len(label.get("bboxes", ())) for label in getattr(dataset, "labels", ())]
        return cls(counts or (0,), max_objects=max_objects)

    def target(self, current_count: int, rng=random) -> int | None:
        valid = [count for count in self.target_counts if count > int(current_count)]
        return int(rng.choice(valid)) if valid else None

    def __call__(self, current_count: int, rng=random) -> int:
        target = self.target(current_count, rng)
        return 0 if target is None else min(target - int(current_count), self.max_objects)


class AdaptiveCopyPaste(SmallObjectCopyPaste):
    """Spend an adaptive budget using scale, deficit, or natural-cluster policy."""

    def __init__(
        self,
        dataset,
        budget: AdaptivePasteBudget | None = None,
        policy: str = "scale_conditioned",
        p: float = 1.0,
        max_objects: int = 4,
        factor_min: float = 0.4,
        factor_max: float = 1.0,
        deficit_gamma: float = 0.5,
        max_weight_ratio: float = 3.0,
        **kwargs,
    ) -> None:
        if policy not in {"scale_conditioned", "scale_deficit", "cluster"}:
            raise ValueError("policy must be scale_conditioned, scale_deficit, or cluster")
        if not 0 < factor_min <= factor_max or deficit_gamma < 0 or max_weight_ratio < 1:
            raise ValueError("invalid scale or deficit parameters")
        super().__init__(dataset, p=p, unit="single", copies=1, placement="collision_aware", **kwargs)
        self.budget = budget or AdaptivePasteBudget.from_dataset(dataset, max_objects=max_objects)
        self.policy = policy
        self.factor_min = float(factor_min)
        self.factor_max = float(factor_max)
        self.deficit_gamma = float(deficit_gamma)
        self.max_weight_ratio = float(max_weight_ratio)
        self._train_scales: list[float] = []
        self._scale_bins: Counter[int] = Counter()
        self._scale_bin_values: dict[int, list[float]] = {}
        self._build_pool()
        for record in self.object_pool:
            size = math.sqrt(max(_box_area(np.asarray(record.bbox_xyxy)), 1e-8))
            self._train_scales.append(size)
            bin_id = self._bin(size)
            self._scale_bins[bin_id] += 1
            self._scale_bin_values.setdefault(bin_id, []).append(size)

    @staticmethod
    def _bin(size: float) -> int:
        return int(round(math.log2(max(float(size), 1.0))))

    def _scene_scales(self, boxes: np.ndarray) -> list[float]:
        return [math.sqrt(max(_box_area(box), 1e-8)) for box in boxes]

    def _target_scale(self, scene_scales: Sequence[float]) -> float:
        if self.policy in {"scale_conditioned", "cluster"} and scene_scales:
            return float(self.rng.choice(scene_scales))
        if self.policy == "scale_deficit" and self._scale_bins:
            bins = sorted(self._scale_bins)
            counts = np.asarray([self._scale_bins[b] for b in bins], dtype=np.float64)
            weights = np.power(counts + 1e-6, -self.deficit_gamma)
            weights = np.maximum(weights, weights.max() / self.max_weight_ratio)
            weights /= weights.sum()
            selected = self.rng.choices(bins, weights=weights.tolist(), k=1)[0]
            return float(self.rng.choice(self._scale_bin_values[selected]))
        return float(self.rng.choice(self._train_scales))

    def _record_for_scale(self, target: float, target_index: int | None) -> ObjectRecord | None:
        candidates = []
        for record in self.object_pool:
            if not self.allow_same_source and record.image_index == target_index:
                continue
            source = math.sqrt(max(_box_area(np.asarray(record.bbox_xyxy)), 1e-8))
            factor = target / max(source, 1e-8)
            if self.factor_min <= factor <= self.factor_max:
                candidates.append(record)
        return self.rng.choice(candidates) if candidates else None

    def _paste_single(self, labels, record: ObjectRecord, target_size: float, existing: np.ndarray):
        source = self._load_raw(record.image_index)
        prepared = self._crop_single(record, source) if source is not None else None
        if prepared is None:
            return None
        crop, source_box = prepared
        native = math.sqrt(max(_box_area(source_box), 1e-8))
        factor = target_size / max(native, 1e-8)
        interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=interpolation)
        destination = self._choose_destination(crop.shape[:2], labels["img"].shape[:2], existing)
        if destination is None:
            return None
        x, y, box = destination
        labels["img"][y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        return box.reshape(1, 4), np.array([record.class_id], dtype=np.int64)

    def _paste_cluster(self, labels, record: ObjectRecord, existing: np.ndarray):
        source = self._load_raw(record.image_index)
        cluster = self._cluster(record.image_index, record.bbox_xyxy, source) if source is not None else None
        if cluster is None:
            return None
        crop, relative, classes, union = cluster
        member_scales = [math.sqrt(max(_box_area(box), 1e-8)) for box in relative]
        target = self._target_scale(self._scene_scales(existing))
        native = float(np.median(member_scales))
        factor = target / max(native, 1e-8)
        if not self.factor_min <= factor <= self.factor_max:
            return None
        interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=interpolation)
        destination = self._choose_destination(crop.shape[:2], labels["img"].shape[:2], existing)
        if destination is None:
            return None
        x, y, _ = destination
        labels["img"][y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        translated = relative * factor
        translated[:, [0, 2]] += x
        translated[:, [1, 3]] += y
        return translated, classes

    def __call__(self, labels):
        if self.p <= 0 or self.rng.random() >= self.p:
            return labels
        self._build_pool()
        if not self.object_pool:
            return labels
        existing = self._target_boxes(labels)
        if not existing.size and not self.allow_empty_target:
            return labels
        target_index = labels.get("image_index")
        budget = self.budget(len(existing), self.rng)
        if budget <= 0:
            return labels
        boxes, classes = [], []
        scene_scales = self._scene_scales(existing)
        remaining = budget
        attempts = 0
        max_attempts = max(3, remaining * 4)
        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            record = self.rng.choice(self.object_pool)
            if self.policy == "cluster":
                result = self._paste_cluster(labels, record, existing)
                added = len(result[0]) if result is not None else 0
                if added > remaining:
                    target_size = self._target_scale(scene_scales)
                    selected = self._record_for_scale(target_size, target_index)
                    result = self._paste_single(labels, selected, target_size, existing) if selected is not None else None
                    added = 1 if result is not None else 0
            else:
                target_size = self._target_scale(scene_scales)
                selected = self._record_for_scale(target_size, target_index)
                result = self._paste_single(labels, selected, target_size, existing) if selected is not None else None
                added = 1 if result is not None else 0
            if result is None:
                continue
            new_boxes, new_classes = result
            boxes.append(new_boxes)
            classes.append(new_classes)
            existing = np.concatenate([existing, new_boxes], axis=0)
            scene_scales.extend(self._scene_scales(new_boxes))
            remaining -= added
        if boxes:
            self._append_instances(labels, boxes, classes)
            self.stats["applied_images"] += 1
            self.stats["pasted_instances"] += sum(len(x) for x in boxes)
        return labels


__all__ = ["AdaptivePasteBudget", "AdaptiveCopyPaste"]
