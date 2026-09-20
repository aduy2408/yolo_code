"""Negative-canvas small-object Copy-Paste study.

This transform is deliberately isolated from the legacy Copy-Paste strategies. It
turns only empty training images into synthetic-positive scenes, with target-scale
sampling, disjoint donor policies, and an optional fixed weak blur ablation.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .copy_paste import ObjectRecord, SmallObjectCopyPaste, _box_area


class NegativeCanvasCopyPaste(SmallObjectCopyPaste):
    """Paste one resized donor object onto a truly empty image.

    The transform intentionally locks the study protocol: one hard-pasted object,
    collision-aware placement, zero allowed overlap, and no fallback between donor
    policies. ``p`` is conditional on the image having no existing instances.
    """

    def __init__(
        self,
        dataset,
        p: float = 0.30,
        target_policy: str = "empirical",
        donor_policy: str = "matched",
        target_max_size: float = 20.0,
        deficit_gamma: float = 0.5,
        max_weight_ratio: float = 3.0,
        matched_ratio_max: float = 1.5,
        large_ratio_min: float = 1.5,
        large_ratio_max: float = 2.5,
        degradation: str = "none",
        blur_sigma: float = 0.5,
        max_trials: int = 30,
        rng=None,
        debug_dir: str | Path | None = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if target_policy not in {"empirical", "deficit"}:
            raise ValueError("target_policy must be 'empirical' or 'deficit'")
        if donor_policy not in {"matched", "larger"}:
            raise ValueError("donor_policy must be 'matched' or 'larger'")
        if target_max_size <= 0 or deficit_gamma < 0 or max_weight_ratio < 1:
            raise ValueError("invalid target or deficit parameters")
        if not 1.0 < matched_ratio_max <= 2.5:
            raise ValueError("matched_ratio_max must be in (1, 2.5]")
        if not 1.0 <= large_ratio_min <= large_ratio_max:
            raise ValueError("invalid larger donor ratio range")
        if degradation not in {"none", "weak_blur"}:
            raise ValueError("degradation must be 'none' or 'weak_blur'")
        if blur_sigma <= 0:
            raise ValueError("blur_sigma must be positive")

        super().__init__(
            dataset=dataset,
            p=p,
            unit="single",
            copies=1,
            placement="collision_aware",
            max_overlap=0.0,
            padding=0.0,
            scale=1.0,
            blend="hard",
            max_trials=max_trials,
            allow_empty_target=True,
            allow_same_source=True,
            rng=rng,
            debug_dir=debug_dir,
        )
        self.target_policy = target_policy
        self.donor_policy = donor_policy
        self.target_max_size = float(target_max_size)
        self.deficit_gamma = float(deficit_gamma)
        self.max_weight_ratio = float(max_weight_ratio)
        self.matched_ratio_max = float(matched_ratio_max)
        self.large_ratio_min = float(large_ratio_min)
        self.large_ratio_max = float(large_ratio_max)
        self.degradation = degradation
        self.blur_sigma = float(blur_sigma)
        self.target_sizes: list[float] = []
        self.scale_bins: Counter[int] = Counter()
        self.scale_bin_values: dict[int, list[float]] = {}
        self._original_negative_indices: set[int] = set()
        self._original_negative_paths: set[str] = set()
        self.stats = self._study_stats()
        self._study_pool_built = False

    @staticmethod
    def _study_stats() -> dict[str, float]:
        stats = SmallObjectCopyPaste._new_stats()
        stats.update({
            "negative_seen": 0,
            "negative_selected": 0,
            "applied_images": 0,
            "pasted_instances": 0,
            "target_size_sum": 0.0,
            "source_size_sum": 0.0,
            "source_target_ratio_sum": 0.0,
            "resize_factor_sum": 0.0,
            "donor_failed": 0,
            "placement_failed": 0,
            "degradation_applied": 0,
            "debug_dump_count": 0,
        })
        return stats

    def reset_stats(self) -> None:
        self.stats = self._study_stats()

    @staticmethod
    def _bin(size: float) -> int:
        return int(round(math.log2(max(float(size), 1.0))))

    def _build_pool(self) -> None:
        if self._study_pool_built:
            return
        super()._build_pool()
        for image_index, label in enumerate(getattr(self.dataset, "labels", ())):
            if len(label.get("bboxes", ())) == 0:
                self._original_negative_indices.add(image_index)
                paths = getattr(self.dataset, "im_files", ())
                if image_index < len(paths):
                    self._original_negative_paths.add(str(Path(paths[image_index]).resolve()))
        for record in self.object_pool:
            size = math.sqrt(max(_box_area(np.asarray(record.bbox_xyxy)), 1e-8))
            if size <= self.target_max_size:
                self.target_sizes.append(size)
                bin_id = self._bin(size)
                self.scale_bins[bin_id] += 1
                self.scale_bin_values.setdefault(bin_id, []).append(size)
        self._study_pool_built = True

    def _is_original_negative(self, labels: dict[str, Any]) -> bool:
        image_index = labels.get("image_index")
        if image_index is not None:
            try:
                if int(image_index) in self._original_negative_indices:
                    return True
            except (TypeError, ValueError):
                pass
        image_path = labels.get("im_file")
        if image_path is None:
            return False
        return str(Path(image_path).resolve()) in self._original_negative_paths

    def _sample_target_size(self) -> float:
        if not self.scale_bins or not self.target_sizes:
            raise ValueError("negative-canvas study requires at least one small target object")
        if self.target_policy == "empirical":
            return float(self.rng.choice(self.target_sizes))
        bins = sorted(self.scale_bins)
        counts = np.asarray([self.scale_bins[b] for b in bins], dtype=np.float64)
        weights = np.power(counts + 1e-6, -self.deficit_gamma)
        weights = np.maximum(weights, weights.max() / self.max_weight_ratio)
        weights /= weights.sum()
        selected_bin = self.rng.choices(bins, weights=weights.tolist(), k=1)[0]
        return float(self.rng.choice(self.scale_bin_values[selected_bin]))

    def _donor_valid(self, source_size: float, target_size: float) -> bool:
        ratio = source_size / max(target_size, 1e-8)
        if self.donor_policy == "matched":
            return 1.0 <= ratio < self.matched_ratio_max
        if self.donor_policy == "larger":
            return self.large_ratio_min <= ratio <= self.large_ratio_max
        raise ValueError(f"unknown donor policy: {self.donor_policy}")

    @staticmethod
    def _effective_source_size(record: ObjectRecord) -> float:
        x1, y1, x2, y2 = map(round, record.bbox_xyxy)
        width = max(x2 - x1, 0)
        height = max(y2 - y1, 0)
        return math.sqrt(max(width * height, 1e-8))

    def _choose_donor(self, target_size: float):
        candidates = []
        for record in self.object_pool:
            source_size = self._effective_source_size(record)
            if self._donor_valid(source_size, target_size):
                candidates.append((record, source_size))
        return self.rng.choice(candidates) if candidates else None

    def _apply_degradation(self, crop: np.ndarray) -> np.ndarray:
        if self.degradation == "none":
            return crop
        if self.degradation == "weak_blur":
            return cv2.GaussianBlur(crop, (3, 3), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma)
        raise ValueError(f"unknown degradation: {self.degradation}")

    def _dump_study_debug(self, labels: dict[str, Any], metadata: dict[str, Any]) -> None:
        if self.debug_dir is None or self.stats["debug_dump_count"] >= 32:
            return
        out_dir = self.debug_dir / "negative_canvas"
        out_dir.mkdir(parents=True, exist_ok=True)
        index = int(self.stats["debug_dump_count"])
        image_path = out_dir / f"{index:04d}.jpg"
        cv2.imwrite(str(image_path), labels["img"])
        image_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.stats["debug_dump_count"] += 1

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        self._build_pool()
        if not self._is_original_negative(labels):
            return labels
        existing = self._target_boxes(labels)
        if len(existing) != 0:
            return labels

        self.stats["negative_seen"] += 1
        if self.p <= 0 or self.rng.random() >= self.p:
            return labels
        self.stats["negative_selected"] += 1

        if not self.target_sizes:
            self.stats["donor_failed"] += 1
            return labels

        target_size = self._sample_target_size()
        selected = self._choose_donor(target_size)
        if selected is None:
            self.stats["donor_failed"] += 1
            return labels
        record, source_size = selected
        source_image = self._load_raw(record.image_index)
        prepared = self._crop_single(record, source_image) if source_image is not None else None
        if prepared is None:
            self.stats["donor_failed"] += 1
            return labels
        crop, source_box = prepared
        source_size = math.sqrt(max(_box_area(source_box), 1e-8))
        factor = target_size / max(source_size, 1e-8)
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        crop = self._apply_degradation(crop)
        destination = self._choose_destination(crop.shape[:2], labels["img"].shape[:2], existing)
        if destination is None:
            self.stats["placement_failed"] += 1
            return labels

        x, y, box = destination
        labels["img"][y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        self._append_instances(
            labels,
            [box.reshape(1, 4)],
            [np.array([record.class_id], dtype=np.int64)],
        )
        self.stats["applied_images"] += 1
        self.stats["pasted_instances"] += 1
        self.stats["target_size_sum"] += target_size
        self.stats["source_size_sum"] += source_size
        self.stats["source_target_ratio_sum"] += source_size / max(target_size, 1e-8)
        self.stats["resize_factor_sum"] += factor
        if self.degradation == "weak_blur":
            self.stats["degradation_applied"] += 1
        self._dump_study_debug(labels, {
            "target_policy": self.target_policy,
            "donor_policy": self.donor_policy,
            "target_size": target_size,
            "source_size": source_size,
            "source_target_ratio": source_size / max(target_size, 1e-8),
            "resize_factor": factor,
            "degradation": self.degradation,
            "blur_sigma": self.blur_sigma,
        })
        return labels

    def diagnostics(self) -> dict[str, float]:
        out = dict(self.stats)
        applied = max(int(out["applied_images"]), 1)
        selected = max(int(out["negative_selected"]), 1)
        out["effective_rate"] = out["applied_images"] / max(out["negative_seen"], 1)
        out["selection_success_rate"] = out["applied_images"] / selected
        out["mean_target_size"] = out["target_size_sum"] / applied
        out["mean_source_size"] = out["source_size_sum"] / applied
        out["mean_resize_factor"] = out["resize_factor_sum"] / applied
        out["mean_source_target_ratio"] = out["source_target_ratio_sum"] / applied
        return {f"negcanvas/{key}": value for key, value in out.items()}


__all__ = ["NegativeCanvasCopyPaste"]
