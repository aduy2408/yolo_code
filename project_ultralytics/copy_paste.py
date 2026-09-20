"""Vanilla small-object Copy-Paste augmentation.

This module intentionally keeps the baseline narrow: raw training-image crops,
hard paste, native scale, and explicit single-object or natural-cluster modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import random

import cv2
import numpy as np

try:
    from ultralytics.utils.instance import Instances
except ImportError:  # pragma: no cover - allows documentation/import inspection
    Instances = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class ObjectRecord:
    image_index: int
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int


class SmallObjectCopyPaste:
    """Paste raw training objects into the current final spatial canvas.

    The dataset is expected to be the training dataset. Its label cache is used
    only for metadata, while source pixels are loaded on demand from the raw
    image path and are never cached by this transform.
    """

    def __init__(
        self,
        dataset,
        p: float = 0.5,
        unit: str = "single",
        copies: int = 1,
        placement: str = "random",
        max_overlap: float = 0.0,
        padding: float = 0.0,
        scale: float = 1.0,
        blend: str = "hard",
        max_trials: int = 30,
        allow_empty_target: bool = True,
        allow_same_source: bool = True,
        cluster_expand: float = 3.0,
        cluster_min_objects: int = 2,
        policy: str = "fixed",
        scene_stats: Mapping[str, float] | None = None,
        rng=None,
        debug_dir: str | Path | None = None,
        debug_limit: int = 32,
    ) -> None:
        if unit not in {"single", "cluster"}:
            raise ValueError("unit must be 'single' or 'cluster'")
        if placement not in {"random", "collision_aware"}:
            raise ValueError("placement must be 'random' or 'collision_aware'")
        if copies not in {1, 2}:
            raise ValueError("copies must be 1 or 2")
        if blend != "hard":
            raise ValueError("only hard blending is supported by the baseline")
        if scale != 1.0:
            raise ValueError("the baseline locks scale=1.0; use a separate scale ablation")
        if padding != 0.0:
            raise ValueError("the baseline locks padding=0.0; use a separate padding ablation")
        if policy not in {"fixed", "load_adaptive", "layout_adaptive"}:
            raise ValueError("policy must be 'fixed', 'load_adaptive', or 'layout_adaptive'")
        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        self.dataset = dataset
        self.p = float(p)
        self.unit = unit
        self.copies = int(copies)
        self.placement = placement
        self.max_overlap = float(max_overlap)
        self.padding = float(padding)
        self.scale = float(scale)
        self.blend = blend
        self.max_trials = int(max_trials)
        self.allow_empty_target = bool(allow_empty_target)
        self.allow_same_source = bool(allow_same_source)
        self.cluster_expand = float(cluster_expand)
        self.cluster_min_objects = int(cluster_min_objects)
        self.policy = policy
        self.scene_stats = dict(scene_stats or {})
        if self.policy != "fixed":
            required = {"count_q33", "count_q67"}
            missing = sorted(required - self.scene_stats.keys())
            if missing:
                raise ValueError(f"scene_stats missing required fields: {', '.join(missing)}")
            if self.policy == "layout_adaptive" and "spacing_q50" not in self.scene_stats:
                raise ValueError("scene_stats missing required field: spacing_q50")
        self.rng = rng or random
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_limit = int(debug_limit)
        self.object_pool: list[ObjectRecord] = []
        self._source_boxes: dict[int, np.ndarray] = {}
        self._source_classes: dict[int, np.ndarray] = {}
        self._pool_built = False
        self.stats = self._new_stats()
        self.stats["debug_dump_count"] = 0

    @staticmethod
    def _new_stats() -> dict[str, float]:
        return {
            "applied_images": 0,
            "pasted_instances": 0,
            "pasted_clusters": 0,
            "source_single_count": 0,
            "source_cluster_size_sum": 0,
            "source_cluster_count": 0,
            "source_cluster_size_max": 0,
            "rejected_boundary": 0,
            "rejected_collision": 0,
            "failed_trials": 0,
            "empty_target_count": 0,
            "empty_target_seen": 0,
            "empty_target_augmented": 0,
            "pasted_width_sum": 0,
            "pasted_height_sum": 0,
            "pasted_area_sum": 0,
            "cluster_crop_area_sum": 0,
            "scene_count_sum": 0,
            "scene_count_max": 0,
            "effective_p_sum": 0.0,
            "effective_p_count": 0,
            "load_low_seen": 0,
            "load_mid_seen": 0,
            "load_high_seen": 0,
            "policy_skipped_load": 0,
            "policy_selected_single": 0,
            "policy_selected_cluster": 0,
            "spacing_sum": 0.0,
            "spacing_count": 0,
        }

    def reset_stats(self) -> None:
        self.stats = self._new_stats()
        self.stats["debug_dump_count"] = 0

    def _dump_debug(self, labels: dict[str, Any], mode: str = "copy_paste") -> None:
        if self.debug_dir is None or self.stats.get("debug_dump_count", 0) >= self.debug_limit:
            return
        canvas = labels["img"].copy()
        instances = labels.get("instances")
        if instances is not None and len(instances):
            boxes = self._target_boxes(labels)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 1)
        out_dir = self.debug_dir / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        index = int(self.stats["debug_dump_count"])
        cv2.imwrite(str(out_dir / f"{index:04d}.jpg"), canvas)
        self.stats["debug_dump_count"] += 1

    def diagnostics(self) -> dict[str, float]:
        out = dict(self.stats)
        n = max(out["source_cluster_count"], 1)
        pasted = max(out["pasted_instances"], 1)
        out["source_cluster_size_mean"] = out["source_cluster_size_sum"] / n
        out["cluster_objects_mean"] = out["source_cluster_size_sum"] / n
        out["cluster_objects_max"] = out["source_cluster_size_max"]
        out["empty_target_fraction"] = out["empty_target_augmented"] / max(out["applied_images"], 1)
        out["pasted_width_mean"] = out["pasted_width_sum"] / pasted
        out["pasted_height_mean"] = out["pasted_height_sum"] / pasted
        out["pasted_area_mean"] = out["pasted_area_sum"] / pasted
        out["cluster_crop_area_mean"] = out["cluster_crop_area_sum"] / n
        out["effective_p_mean"] = out["effective_p_sum"] / max(out["effective_p_count"], 1)
        out["scene_count_mean"] = out["scene_count_sum"] / max(out["effective_p_count"], 1)
        out["spacing_mean"] = out["spacing_sum"] / max(out["spacing_count"], 1)
        # Also expose the manifest-friendly names used by experiment logs.
        out.update({f"cp/{key}": value for key, value in out.items() if not key.startswith("cp/")})
        return out

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        xc, yc, w, h = boxes.T
        return np.stack((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), axis=1)

    def _build_pool(self) -> None:
        if self._pool_built:
            return
        labels = getattr(self.dataset, "labels", [])
        for image_index, label in enumerate(labels):
            boxes = np.asarray(label.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
            if label.get("bbox_format", "xywh") == "xywh":
                boxes = self._xywh_to_xyxy(boxes)
            if label.get("normalized", False):
                h, w = label["shape"][:2]
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h
            classes = np.asarray(label.get("cls", []), dtype=np.int64).reshape(-1)
            self._source_boxes[image_index] = boxes
            self._source_classes[image_index] = classes
            for box, cls in zip(boxes, classes):
                self.object_pool.append(ObjectRecord(image_index, tuple(map(float, box)), int(cls)))
        self._pool_built = True

    def _load_raw(self, image_index: int) -> np.ndarray | None:
        paths = getattr(self.dataset, "im_files", [])
        if image_index >= len(paths):
            return None
        image = cv2.imread(str(Path(paths[image_index])), cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def _intersection_over_area(candidate: np.ndarray, existing: np.ndarray) -> float:
        if len(existing) == 0:
            return 0.0
        x1 = np.maximum(candidate[0], existing[:, 0])
        y1 = np.maximum(candidate[1], existing[:, 1])
        x2 = np.minimum(candidate[2], existing[:, 2])
        y2 = np.minimum(candidate[3], existing[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area = max(float((candidate[2] - candidate[0]) * (candidate[3] - candidate[1])), 1e-8)
        return float(np.max(intersection / area, initial=0.0))

    def _target_boxes(self, labels: dict[str, Any]) -> np.ndarray:
        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return np.empty((0, 4), dtype=np.float32)
        instances.convert_bbox("xyxy")
        if instances.normalized:
            h, w = labels["img"].shape[:2]
            instances.denormalize(w, h)
        return np.asarray(instances.bboxes, dtype=np.float32).copy()

    @staticmethod
    def _normalized_nearest_spacing(boxes: np.ndarray) -> float | None:
        """Return median nearest edge gap normalized by object area scale."""
        if len(boxes) < 2:
            return None
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 1e-6)
        gaps = []
        for i, box in enumerate(boxes):
            dx = np.maximum(np.maximum(box[0] - boxes[:, 2], boxes[:, 0] - box[2]), 0.0)
            dy = np.maximum(np.maximum(box[1] - boxes[:, 3], boxes[:, 1] - box[3]), 0.0)
            distance = np.sqrt(dx * dx + dy * dy)
            distance[i] = np.inf
            nearest = int(np.argmin(distance))
            scale = float(np.sqrt(sizes[i, 0] * sizes[i, 1]))
            gaps.append(float(distance[nearest]) / max(scale, 1e-6))
        return float(np.median(gaps))

    def _effective_probability(self, existing: np.ndarray) -> float:
        """Calculate load-conditioned probability and record policy diagnostics."""
        if self.policy == "fixed":
            return self.p
        count = len(existing)
        low = float(self.scene_stats["count_q33"])
        high = float(self.scene_stats["count_q67"])
        load = float(np.clip((count - low) / max(high - low, 1.0), 0.0, 1.0))
        self.stats["scene_count_sum"] += count
        self.stats["scene_count_max"] = max(self.stats["scene_count_max"], count)
        self.stats["effective_p_sum"] += self.p * (1.0 - load)
        self.stats["effective_p_count"] += 1
        if load <= 0.0:
            self.stats["load_low_seen"] += 1
        elif load >= 1.0:
            self.stats["load_high_seen"] += 1
        else:
            self.stats["load_mid_seen"] += 1
        return float(self.p * (1.0 - load))

    def _choose_unit(self, existing: np.ndarray) -> str:
        if self.policy != "layout_adaptive":
            return self.unit
        spacing = self._normalized_nearest_spacing(existing)
        if spacing is None:
            self.stats["policy_selected_single"] += 1
            return "single"
        self.stats["spacing_sum"] += spacing
        self.stats["spacing_count"] += 1
        if spacing <= float(self.scene_stats["spacing_q50"]):
            self.stats["policy_selected_cluster"] += 1
            return "cluster"
        self.stats["policy_selected_single"] += 1
        return "single"

    def _choose_destination(self, patch_shape: tuple[int, int], canvas_shape: tuple[int, int], existing: np.ndarray):
        ph, pw = patch_shape
        h, w = canvas_shape
        if ph <= 0 or pw <= 0 or ph > h or pw > w:
            self.stats["rejected_boundary"] += 1
            return None
        for _ in range(self.max_trials):
            x = self.rng.randint(0, w - pw)
            y = self.rng.randint(0, h - ph)
            box = np.array([x, y, x + pw, y + ph], dtype=np.float32)
            if self.placement == "collision_aware" and self._intersection_over_area(box, existing) > self.max_overlap:
                self.stats["rejected_collision"] += 1
                continue
            return x, y, box
        self.stats["failed_trials"] += 1
        return None

    def _crop_single(self, record: ObjectRecord, source_image: np.ndarray):
        x1, y1, x2, y2 = map(round, record.bbox_xyxy)
        h, w = source_image.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = source_image[y1:y2, x1:x2].copy()
        if self.padding:
            px, py = round((x2 - x1) * self.padding), round((y2 - y1) * self.padding)
            crop = cv2.copyMakeBorder(crop, py, py, px, px, cv2.BORDER_REPLICATE)
        if self.scale != 1.0:
            crop = cv2.resize(crop, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        return crop, np.array([x1, y1, x2, y2], dtype=np.float32)

    def _cluster(self, image_index: int, seed_box: tuple[float, float, float, float], source_image: np.ndarray):
        boxes = self._source_boxes.get(image_index, np.empty((0, 4), dtype=np.float32))
        classes = self._source_classes.get(image_index, np.empty((0,), dtype=np.int64))
        if len(boxes) < self.cluster_min_objects:
            return None
        seed_distances = np.max(np.abs(boxes - np.asarray(seed_box, dtype=np.float32)), axis=1)
        seed = int(np.argmin(seed_distances))
        sx1, sy1, sx2, sy2 = boxes[seed]
        cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        rw, rh = (sx2 - sx1) * self.cluster_expand, (sy2 - sy1) * self.cluster_expand
        selected = np.where((boxes[:, 0] + boxes[:, 2]) / 2 >= cx - rw / 2)[0]
        selected = selected[((boxes[selected, 0] + boxes[selected, 2]) / 2 <= cx + rw / 2)]
        selected = selected[((boxes[selected, 1] + boxes[selected, 3]) / 2 >= cy - rh / 2)]
        selected = selected[((boxes[selected, 1] + boxes[selected, 3]) / 2 <= cy + rh / 2)]
        if len(selected) < self.cluster_min_objects:
            return None
        union = np.array([boxes[selected, 0].min(), boxes[selected, 1].min(), boxes[selected, 2].max(), boxes[selected, 3].max()])
        x1, y1, x2, y2 = map(round, union)
        crop = source_image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None
        relative = boxes[selected].copy()
        relative[:, [0, 2]] -= x1
        relative[:, [1, 3]] -= y1
        return crop, relative, classes[selected], union

    def _append_instances(self, labels: dict[str, Any], boxes: list[np.ndarray], classes: list[np.ndarray]) -> None:
        if not boxes:
            return
        instances = labels["instances"]
        instances.convert_bbox("xyxy")
        h, w = labels["img"].shape[:2]
        if instances.normalized:
            instances.denormalize(w, h)
        new_boxes = np.concatenate(boxes, axis=0).astype(np.float32)
        segment_shape = instances.segments.shape[1:] if getattr(instances.segments, "ndim", 0) == 3 else (0, 2)
        new_segments = np.zeros((len(new_boxes), *segment_shape), dtype=np.float32)
        new_keypoints = None
        if instances.keypoints is not None:
            new_keypoints = np.zeros((len(new_boxes), *instances.keypoints.shape[1:]), dtype=np.float32)
        new_instances = Instances(
            new_boxes, new_segments, new_keypoints, bbox_format="xyxy", normalized=False
        )
        labels["instances"] = Instances.concatenate([instances, new_instances], axis=0)
        labels["cls"] = np.concatenate(
            [
                np.asarray(labels.get("cls", []), dtype=np.float32).reshape(-1, 1),
                np.concatenate(classes).astype(np.float32, copy=False).reshape(-1, 1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        # Preserve the fixed baseline's RNG and setup order exactly. Adaptive
        # modes must inspect final-canvas boxes before drawing their decision.
        if self.policy == "fixed" and (self.p <= 0 or self.rng.random() >= self.p):
            return labels
        self._build_pool()
        if not self.object_pool:
            return labels
        target_image = labels["img"]
        target_index = labels.get("image_index")
        if target_index is None:
            target_path = labels.get("im_file")
            target_index = next(
                (i for i, path in enumerate(getattr(self.dataset, "im_files", [])) if str(path) == str(target_path)),
                None,
            )
        existing = self._target_boxes(labels)
        p_eff = self.p if self.policy == "fixed" else self._effective_probability(existing)
        if self.policy != "fixed" and (p_eff <= 0 or self.rng.random() >= p_eff):
            self.stats["policy_skipped_load"] += 1
            return labels
        if len(existing) == 0:
            self.stats["empty_target_seen"] += 1
            self.stats["empty_target_count"] += 1
            if not self.allow_empty_target:
                return labels
        pasted_boxes: list[np.ndarray] = []
        pasted_classes: list[np.ndarray] = []
        source_record = self.rng.choice(self.object_pool)
        if not self.allow_same_source and target_index == source_record.image_index:
            alternatives = [r for r in self.object_pool if r.image_index != target_index]
            if not alternatives:
                return labels
            source_record = self.rng.choice(alternatives)
        source_image = self._load_raw(source_record.image_index)
        if source_image is None:
            return labels
        unit = self._choose_unit(existing)
        if unit == "cluster":
            cluster = self._cluster(source_record.image_index, source_record.bbox_xyxy, source_image)
            if cluster is None:
                self.stats["failed_trials"] += 1
                return labels
            crop, relative, classes, union = cluster
            for _ in range(self.copies):
                destination = self._choose_destination(crop.shape[:2], target_image.shape[:2], existing)
                if destination is None:
                    continue
                x, y, box = destination
                ph, pw = crop.shape[:2]
                target_image[y:y + ph, x:x + pw] = crop
                translated = relative.copy()
                translated[:, [0, 2]] += x
                translated[:, [1, 3]] += y
                pasted_boxes.append(translated)
                pasted_classes.append(classes)
                existing = np.concatenate([existing, translated], axis=0)
                self.stats["pasted_clusters"] += 1
                self.stats["source_cluster_count"] += 1
                self.stats["source_cluster_size_sum"] += len(classes)
                self.stats["source_cluster_size_max"] = max(self.stats["source_cluster_size_max"], len(classes))
                self.stats["cluster_crop_area_sum"] += float(pw * ph)
        else:
            prepared = self._crop_single(source_record, source_image)
            if prepared is None:
                return labels
            crop, _ = prepared
            for _ in range(self.copies):
                destination = self._choose_destination(crop.shape[:2], target_image.shape[:2], existing)
                if destination is None:
                    continue
                x, y, box = destination
                ph, pw = crop.shape[:2]
                target_image[y:y + ph, x:x + pw] = crop
                pasted_boxes.append(box.reshape(1, 4))
                pasted_classes.append(np.array([source_record.class_id], dtype=np.int64))
                existing = np.concatenate([existing, box.reshape(1, 4)], axis=0)
                self.stats["source_single_count"] += 1
                self.stats["pasted_width_sum"] += pw
                self.stats["pasted_height_sum"] += ph
                self.stats["pasted_area_sum"] += pw * ph
        if pasted_boxes:
            self._append_instances(labels, pasted_boxes, pasted_classes)
            self.stats["applied_images"] += 1
            self.stats["pasted_instances"] += sum(len(b) for b in pasted_boxes)
            if len(self._target_boxes(labels)) == sum(len(b) for b in pasted_boxes):
                self.stats["empty_target_augmented"] += 1
            self._dump_debug(labels, self.unit)
        return labels


def _box_area(box: np.ndarray) -> float:
    box = np.asarray(box, dtype=np.float32)
    return max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)


def _intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


class CrowdedCopyPaste(SmallObjectCopyPaste):
    """Paste one similarly-sized object over an existing target object.

    This intentionally uses rectangular, hard pasted bbox crops.  It isolates
    controlled overlap from mask quality, scaling, and depth/consensus losses.
    """

    def __init__(
        self,
        dataset,
        p: float = 0.3,
        copies: int = 1,
        overlap_min: float = 0.10,
        overlap_max: float = 0.30,
        min_visibility: float = 0.60,
        size_ratio_min: float = 0.75,
        size_ratio_max: float = 1.33,
        trials: int = 30,
        rng=None,
        debug_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            dataset, p=p, unit="single", copies=copies, placement="random",
            max_overlap=0.0, padding=0.0, scale=1.0, blend="hard",
            max_trials=trials, allow_empty_target=False, allow_same_source=True,
            rng=rng,
            debug_dir=debug_dir,
        )
        if not 0.0 <= overlap_min <= overlap_max:
            raise ValueError("crowd overlap range must satisfy 0 <= min <= max")
        if not 0.0 < min_visibility <= 1.0:
            raise ValueError("crowd min_visibility must be in (0, 1]")
        if not 0.0 < size_ratio_min <= size_ratio_max:
            raise ValueError("crowd size ratio range must be positive and ordered")
        self.overlap_min = float(overlap_min)
        self.overlap_max = float(overlap_max)
        self.min_visibility = float(min_visibility)
        self.size_ratio_min = float(size_ratio_min)
        self.size_ratio_max = float(size_ratio_max)
        self.stats.update({
            "crowd_attempted": 0,
            "crowd_applied": 0,
            "crowd_overlap_sum": 0.0,
            "crowd_visibility_sum": 0.0,
            "crowd_size_ratio_sum": 0.0,
            "crowd_rejected_overlap": 0,
        })

    def _crowd_destination(self, anchor: np.ndarray, patch_shape: tuple[int, int], canvas_shape: tuple[int, int]):
        ph, pw = patch_shape
        h, w = canvas_shape
        if ph > h or pw > w:
            self.stats["rejected_boundary"] += 1
            return None
        anchor_w = max(float(anchor[2] - anchor[0]), 1.0)
        anchor_h = max(float(anchor[3] - anchor[1]), 1.0)
        for _ in range(self.max_trials):
            # Sampling around the anchor keeps the constructed relation local.
            cx = self.rng.uniform(float(anchor[0] - pw * 0.35), float(anchor[2] + pw * 0.35))
            cy = self.rng.uniform(float(anchor[1] - ph * 0.35), float(anchor[3] + ph * 0.35))
            x = int(round(np.clip(cx - pw / 2, 0, w - pw)))
            y = int(round(np.clip(cy - ph / 2, 0, h - ph)))
            box = np.array([x, y, x + pw, y + ph], dtype=np.float32)
            overlap = _intersection_area(anchor, box) / max(min(_box_area(anchor), _box_area(box)), 1e-8)
            visibility = 1.0 - _intersection_area(anchor, box) / max(_box_area(anchor), 1e-8)
            if self.overlap_min <= overlap <= self.overlap_max and visibility >= self.min_visibility:
                return x, y, box, overlap, visibility
        self.stats["crowd_rejected_overlap"] += 1
        self.stats["failed_trials"] += 1
        return None

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.p <= 0 or self.rng.random() >= self.p:
            return labels
        self._build_pool()
        existing = self._target_boxes(labels)
        if len(existing) == 0 or not self.object_pool:
            return labels
        self.stats["crowd_attempted"] += 1
        target = existing[int(self.rng.randrange(len(existing)))]
        target_size = np.sqrt(max(_box_area(target), 1e-8))
        candidates = [
            record for record in self.object_pool
            if self.size_ratio_min <= np.sqrt(max(_box_area(np.asarray(record.bbox_xyxy)), 1e-8)) / target_size <= self.size_ratio_max
        ]
        if not candidates:
            return labels
        pasted_boxes, pasted_classes = [], []
        for _ in range(self.copies):
            record = self.rng.choice(candidates)
            source = self._load_raw(record.image_index)
            prepared = None if source is None else self._crop_single(record, source)
            if prepared is None:
                continue
            crop, _ = prepared
            destination = self._crowd_destination(target, crop.shape[:2], labels["img"].shape[:2])
            if destination is None:
                continue
            x, y, box, overlap, visibility = destination
            ph, pw = crop.shape[:2]
            labels["img"][y:y + ph, x:x + pw] = crop
            pasted_boxes.append(box.reshape(1, 4))
            pasted_classes.append(np.array([record.class_id], dtype=np.int64))
            existing = np.concatenate([existing, box.reshape(1, 4)], axis=0)
            self.stats["crowd_overlap_sum"] += overlap
            self.stats["crowd_visibility_sum"] += visibility
            self.stats["crowd_size_ratio_sum"] += np.sqrt(max(_box_area(box), 1e-8)) / target_size
        if pasted_boxes:
            self._append_instances(labels, pasted_boxes, pasted_classes)
            self.stats["crowd_applied"] += 1
            self.stats["applied_images"] += 1
            self.stats["pasted_instances"] += sum(len(b) for b in pasted_boxes)
            self._dump_debug(labels, "crowded")
        return labels

    def diagnostics(self) -> dict[str, float]:
        out = super().diagnostics()
        n = max(int(out.get("crowd_applied", 0)), 1)
        out["crowd/mean_overlap"] = out.get("crowd_overlap_sum", 0.0) / n
        out["crowd/mean_visibility"] = out.get("crowd_visibility_sum", 0.0) / n
        out["crowd/mean_size_ratio"] = out.get("crowd_size_ratio_sum", 0.0) / n
        return out


class ScaleMatchedCopyPaste(SmallObjectCopyPaste):
    """Downsample source objects to sizes sampled from real GT statistics."""

    def __init__(
        self,
        dataset,
        p: float = 0.5,
        copies: int = 1,
        target_max_size: float = 20.0,
        source_min_ratio: float = 1.25,
        factor_min: float = 0.50,
        factor_max: float = 0.90,
        max_overlap: float = 0.0,
        trials: int = 30,
        rng=None,
        debug_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            dataset, p=p, unit="single", copies=copies, placement="collision_aware",
            max_overlap=max_overlap, padding=0.0, scale=1.0, blend="hard",
            max_trials=trials, rng=rng,
            debug_dir=debug_dir,
        )
        if factor_min <= 0 or factor_min > factor_max or factor_max > 1.0:
            raise ValueError("scale factors must satisfy 0 < min <= max <= 1")
        self.target_max_size = float(target_max_size)
        self.source_min_ratio = float(source_min_ratio)
        self.factor_min = float(factor_min)
        self.factor_max = float(factor_max)
        self.target_size_pool: list[float] = []
        self.stats.update({
            "scale_attempted": 0,
            "scale_applied": 0,
            "scale_source_sum": 0.0,
            "scale_target_sum": 0.0,
            "scale_factor_sum": 0.0,
            "scale_result_sum": 0.0,
        })

    def _build_pool(self) -> None:
        super()._build_pool()
        if self.target_size_pool:
            return
        for record in self.object_pool:
            size = np.sqrt(max(_box_area(np.asarray(record.bbox_xyxy)), 1e-8))
            if size <= self.target_max_size:
                self.target_size_pool.append(float(size))

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.p <= 0 or self.rng.random() >= self.p:
            return labels
        self._build_pool()
        if not self.object_pool or not self.target_size_pool:
            return labels
        self.stats["scale_attempted"] += 1
        target_size = float(self.rng.choice(self.target_size_pool))
        candidates = []
        for record in self.object_pool:
            source_size = np.sqrt(max(_box_area(np.asarray(record.bbox_xyxy)), 1e-8))
            ratio = source_size / max(target_size, 1e-8)
            if ratio >= self.source_min_ratio:
                candidates.append((record, source_size))
        if not candidates:
            return labels
        existing = self._target_boxes(labels)
        pasted_boxes, pasted_classes = [], []
        for _ in range(self.copies):
            record, source_size = self.rng.choice(candidates)
            source = self._load_raw(record.image_index)
            if source is None:
                continue
            x1, y1, x2, y2 = map(round, record.bbox_xyxy)
            crop = source[max(0, y1):min(source.shape[0], y2), max(0, x1):min(source.shape[1], x2)].copy()
            if crop.size == 0:
                continue
            factor = float(np.clip(target_size / max(source_size, 1e-8), self.factor_min, self.factor_max))
            crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            destination = self._choose_destination(crop.shape[:2], labels["img"].shape[:2], existing)
            if destination is None:
                continue
            x, y, box = destination
            ph, pw = crop.shape[:2]
            labels["img"][y:y + ph, x:x + pw] = crop
            pasted_boxes.append(box.reshape(1, 4))
            pasted_classes.append(np.array([record.class_id], dtype=np.int64))
            existing = np.concatenate([existing, box.reshape(1, 4)], axis=0)
            result_size = np.sqrt(max(_box_area(box), 1e-8))
            self.stats["scale_source_sum"] += source_size
            self.stats["scale_target_sum"] += target_size
            self.stats["scale_factor_sum"] += factor
            self.stats["scale_result_sum"] += result_size
        if pasted_boxes:
            self._append_instances(labels, pasted_boxes, pasted_classes)
            self.stats["scale_applied"] += 1
            self.stats["applied_images"] += 1
            self.stats["pasted_instances"] += sum(len(b) for b in pasted_boxes)
            self._dump_debug(labels, "scale_matched")
        return labels

    def diagnostics(self) -> dict[str, float]:
        out = super().diagnostics()
        n = max(int(out.get("scale_applied", 0)), 1)
        out["scale/mean_source_size"] = out.get("scale_source_sum", 0.0) / n
        out["scale/mean_target_size"] = out.get("scale_target_sum", 0.0) / n
        out["scale/mean_factor"] = out.get("scale_factor_sum", 0.0) / n
        out["scale/mean_result_size"] = out.get("scale_result_sum", 0.0) / n
        return out


__all__ = [
    "ObjectRecord", "SmallObjectCopyPaste", "CrowdedCopyPaste", "ScaleMatchedCopyPaste",
    "build_small_object_copy_paste", "copy_paste_config",
]


def build_small_object_copy_paste(dataset, hyp):
    """Build the transform from explicit experiment hyperparameters.

    Keeping the switch separate from Ultralytics' legacy ``copy_paste`` knob
    makes CP0 and the new CP1-CP4 matrix unambiguous.
    """
    if not bool(getattr(hyp, "copy_paste_enabled", False)):
        return None
    mode = str(getattr(hyp, "copy_paste_mode", "single")).lower()
    if mode in {"double", "cp2"}:
        mode = "single"
        copies = 2
    else:
        copies = int(getattr(hyp, "copy_paste_copies", 1))
    if mode == "cluster":
        mode = "single"
        unit = "cluster"
    else:
        unit = str(getattr(hyp, "copy_paste_unit", "single"))
    if mode == "negative":
        from .negative_copy_paste import HardNegativeBank, NegativeCopyPaste

        bank_path = str(
            getattr(hyp, "negcp_bank_path", "")
            or getattr(hyp, "copy_paste_negative_bank", "")
            or ""
        )
        if not bank_path:
            raise ValueError("copy_paste_mode='negative' requires negcp_bank_path")
        bank = HardNegativeBank.load(bank_path)
        return NegativeCopyPaste(
            bank=bank,
            p=float(getattr(hyp, "negcp", getattr(hyp, "copy_paste_p", 0.30))),
            num=int(getattr(hyp, "negcp_num", 1)),
            scale=float(getattr(hyp, "negcp_scale", 1.0)),
            max_gt_ioa=float(getattr(hyp, "negcp_max_gt_ioa", 0.05)),
            max_trials=int(getattr(hyp, "copy_paste_max_trials", 30)),
            same_source=bool(getattr(hyp, "negcp_same_source", False)),
            debug_dir=getattr(hyp, "copy_paste_debug_dir", None),
        )
    if mode in {"adaptive", "adaptive_scale_conditioned", "adaptive_scale_deficit", "adaptive_cluster"}:
        from .adaptive_copy_paste import AdaptiveCopyPaste, AdaptivePasteBudget

        policy = {
            "adaptive": str(getattr(hyp, "adaptive_cp_policy", "scale_conditioned")),
            "adaptive_scale_conditioned": "scale_conditioned",
            "adaptive_scale_deficit": "scale_deficit",
            "adaptive_cluster": "cluster",
        }[mode]
        target_counts = getattr(hyp, "adaptive_cp_target_counts", None)
        if float(getattr(hyp, "mosaic", 0.0)) > 0 and target_counts is None:
            raise ValueError(
                "adaptive Copy-Paste after Mosaic requires adaptive_cp_target_counts from post-Mosaic scenes; "
                "disable mosaic for the no-Mosaic study or provide matched statistics"
            )
        budget = AdaptivePasteBudget(
            target_counts if target_counts is not None else [len(label.get("bboxes", ())) for label in getattr(dataset, "labels", ())] or [0],
            max_objects=int(getattr(hyp, "adaptive_cp_max_objects", 4)),
        )
        return AdaptiveCopyPaste(
            dataset=dataset,
            budget=budget,
            policy=policy,
            p=float(getattr(hyp, "copy_paste_p", 1.0)),
            factor_min=float(getattr(hyp, "adaptive_cp_factor_min", 0.4)),
            factor_max=float(getattr(hyp, "adaptive_cp_factor_max", 1.0)),
            deficit_gamma=float(getattr(hyp, "adaptive_cp_deficit_gamma", 0.5)),
            max_weight_ratio=float(getattr(hyp, "adaptive_cp_max_weight_ratio", 3.0)),
            max_trials=int(getattr(hyp, "copy_paste_max_trials", 30)),
            allow_empty_target=bool(getattr(hyp, "copy_paste_allow_empty_target", True)),
            allow_same_source=bool(getattr(hyp, "copy_paste_allow_same_source", True)),
            debug_dir=getattr(hyp, "copy_paste_debug_dir", None),
            rng=getattr(hyp, "copy_paste_rng", None),
        )
    if mode == "negative_canvas":
        from .negative_canvas_copy_paste import NegativeCanvasCopyPaste

        return NegativeCanvasCopyPaste(
            dataset=dataset,
            p=float(getattr(hyp, "copy_paste_p", 0.30)),
            target_policy=str(getattr(hyp, "negative_cp_target_policy", "empirical")),
            donor_policy=str(getattr(hyp, "negative_cp_donor_policy", "matched")),
            target_max_size=float(getattr(hyp, "negative_cp_target_max_size", 20.0)),
            deficit_gamma=float(getattr(hyp, "negative_cp_deficit_gamma", 0.5)),
            max_weight_ratio=float(getattr(hyp, "negative_cp_max_weight_ratio", 3.0)),
            matched_ratio_max=float(getattr(hyp, "negative_cp_matched_ratio_max", 1.5)),
            large_ratio_min=float(getattr(hyp, "negative_cp_large_ratio_min", 1.5)),
            large_ratio_max=float(getattr(hyp, "negative_cp_large_ratio_max", 2.5)),
            degradation=str(getattr(hyp, "negative_cp_degradation", "none")),
            blur_sigma=float(getattr(hyp, "negative_cp_blur_sigma", 0.5)),
            max_trials=int(getattr(hyp, "copy_paste_max_trials", 30)),
            debug_dir=getattr(hyp, "copy_paste_debug_dir", None),
            rng=getattr(hyp, "copy_paste_rng", None),
        )
    if mode in {"online_negative", "online_negative_scale_matched"}:
        from .online_negative_bank import OnlineHardNegativeBank, OnlineNegativeCopyPaste
        from .adaptive_copy_paste import AdaptivePasteBudget

        bank_path = str(getattr(hyp, "online_negcp_bank_path", "") or "")
        if not bank_path:
            raise ValueError("online negative modes require online_negcp_bank_path")
        target_counts = getattr(hyp, "adaptive_cp_target_counts", None)
        if float(getattr(hyp, "mosaic", 0.0)) > 0 and target_counts is None:
            raise ValueError(
                "online NegCP after Mosaic requires adaptive_cp_target_counts from post-Mosaic scenes; "
                "disable mosaic for the no-Mosaic study or provide matched statistics"
            )
        if target_counts is None:
            target_counts = [len(label.get("bboxes", ())) for label in getattr(dataset, "labels", ())] or [0]
        budget = AdaptivePasteBudget(target_counts, max_objects=int(getattr(hyp, "online_negcp_max_objects", 3)))
        return OnlineNegativeCopyPaste(
            bank=OnlineHardNegativeBank.load(bank_path),
            p=float(getattr(hyp, "negcp", getattr(hyp, "copy_paste_p", 0.30))),
            max_objects=int(getattr(hyp, "online_negcp_max_objects", 3)),
            scale_matched=mode == "online_negative_scale_matched",
            max_gt_ioa=float(getattr(hyp, "negcp_max_gt_ioa", 0.05)),
            max_trials=int(getattr(hyp, "copy_paste_max_trials", 30)),
            budget=budget,
        )
    if mode == "crowded":
        return CrowdedCopyPaste(
            dataset=dataset,
            p=float(getattr(hyp, "copy_paste_p", 0.30)),
            copies=int(getattr(hyp, "crowd_num", copies)),
            overlap_min=float(getattr(hyp, "crowd_overlap_min", 0.10)),
            overlap_max=float(getattr(hyp, "crowd_overlap_max", 0.30)),
            min_visibility=float(getattr(hyp, "crowd_min_visibility", 0.60)),
            size_ratio_min=float(getattr(hyp, "crowd_size_ratio_min", 0.75)),
            size_ratio_max=float(getattr(hyp, "crowd_size_ratio_max", 1.33)),
            trials=int(getattr(hyp, "crowd_trials", 30)),
            debug_dir=getattr(hyp, "copy_paste_debug_dir", None),
        )
    if mode == "scale_matched":
        return ScaleMatchedCopyPaste(
            dataset=dataset,
            p=float(getattr(hyp, "copy_paste_p", 0.50)),
            copies=int(getattr(hyp, "scale_cp_num", copies)),
            target_max_size=float(getattr(hyp, "scale_cp_target_max_size", 20.0)),
            source_min_ratio=float(getattr(hyp, "scale_cp_source_min_ratio", 1.25)),
            factor_min=float(getattr(hyp, "scale_cp_factor_min", 0.50)),
            factor_max=float(getattr(hyp, "scale_cp_factor_max", 0.90)),
            max_overlap=float(getattr(hyp, "scale_cp_max_overlap", 0.0)),
            trials=int(getattr(hyp, "scale_cp_trials", 30)),
            debug_dir=getattr(hyp, "copy_paste_debug_dir", None),
        )
    stats_path = str(getattr(hyp, "copy_paste_stats_path", "") or "")
    scene_stats = getattr(hyp, "copy_paste_scene_stats", None)
    if scene_stats is None and stats_path:
        scene_stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    return SmallObjectCopyPaste(
        dataset=dataset,
        p=float(getattr(hyp, "copy_paste_p", 0.5)),
        unit=unit,
        copies=copies,
        placement=str(getattr(hyp, "copy_paste_placement", "random")),
        max_overlap=float(getattr(hyp, "copy_paste_max_overlap", 0.0)),
        padding=float(getattr(hyp, "copy_paste_padding", 0.0)),
        scale=float(getattr(hyp, "copy_paste_scale", 1.0)),
        blend=str(getattr(hyp, "copy_paste_blend", "hard")),
        max_trials=int(getattr(hyp, "copy_paste_max_trials", 30)),
        allow_empty_target=bool(getattr(hyp, "copy_paste_allow_empty_target", True)),
        allow_same_source=bool(getattr(hyp, "copy_paste_allow_same_source", True)),
        cluster_expand=float(getattr(hyp, "copy_paste_cluster_expand", 3.0)),
        cluster_min_objects=int(getattr(hyp, "copy_paste_cluster_min_objects", 2)),
        policy=str(getattr(hyp, "copy_paste_policy", "fixed")),
        scene_stats=scene_stats,
        debug_dir=getattr(hyp, "copy_paste_debug_dir", None),
    )


def copy_paste_config(hyp) -> dict[str, Any]:
    """Return the explicit Copy-Paste settings for a run manifest."""
    return {
        "enabled": bool(getattr(hyp, "copy_paste_enabled", False)),
        "mode": str(getattr(hyp, "copy_paste_mode", "single")),
        "p": float(getattr(hyp, "copy_paste_p", 0.5)),
        "unit": str(getattr(hyp, "copy_paste_unit", "single")),
        "copies": int(getattr(hyp, "copy_paste_copies", 1)),
        "placement": str(getattr(hyp, "copy_paste_placement", "random")),
        "max_overlap": float(getattr(hyp, "copy_paste_max_overlap", 0.0)),
        "scale": float(getattr(hyp, "copy_paste_scale", 1.0)),
        "padding": float(getattr(hyp, "copy_paste_padding", 0.0)),
        "blend": str(getattr(hyp, "copy_paste_blend", "hard")),
        "allow_empty_target": bool(getattr(hyp, "copy_paste_allow_empty_target", True)),
        "allow_same_source": bool(getattr(hyp, "copy_paste_allow_same_source", True)),
        "max_trials": int(getattr(hyp, "copy_paste_max_trials", 30)),
        "cluster_expand": float(getattr(hyp, "copy_paste_cluster_expand", 3.0)),
        "cluster_min_objects": int(getattr(hyp, "copy_paste_cluster_min_objects", 2)),
        "policy": str(getattr(hyp, "copy_paste_policy", "fixed")),
        "adaptive_policy": str(getattr(hyp, "adaptive_cp_policy", "scale_conditioned")),
        "adaptive_target_counts": getattr(hyp, "adaptive_cp_target_counts", None),
        "adaptive_max_objects": int(getattr(hyp, "adaptive_cp_max_objects", 4)),
        "adaptive_factor_min": float(getattr(hyp, "adaptive_cp_factor_min", 0.4)),
        "adaptive_factor_max": float(getattr(hyp, "adaptive_cp_factor_max", 1.0)),
        "adaptive_deficit_gamma": float(getattr(hyp, "adaptive_cp_deficit_gamma", 0.5)),
        "online_negcp_bank_path": str(getattr(hyp, "online_negcp_bank_path", "") or ""),
        "online_negcp_max_objects": int(getattr(hyp, "online_negcp_max_objects", 3)),
        "online_negcp_conf_threshold": float(getattr(hyp, "online_negcp_conf_threshold", 0.25)),
        "online_negcp_max_candidates": int(getattr(hyp, "online_negcp_max_candidates", 3)),
        "stats_path": str(getattr(hyp, "copy_paste_stats_path", "") or ""),
        "debug_dir": str(getattr(hyp, "copy_paste_debug_dir", "") or ""),
        "negcp": float(getattr(hyp, "negcp", 0.30)),
        "negcp_num": int(getattr(hyp, "negcp_num", 1)),
        "negcp_bank_path": str(getattr(hyp, "negcp_bank_path", "") or ""),
        "negcp_max_gt_ioa": float(getattr(hyp, "negcp_max_gt_ioa", 0.05)),
        "crowd_overlap_min": float(getattr(hyp, "crowd_overlap_min", 0.10)),
        "crowd_overlap_max": float(getattr(hyp, "crowd_overlap_max", 0.30)),
        "crowd_min_visibility": float(getattr(hyp, "crowd_min_visibility", 0.60)),
        "crowd_size_ratio_min": float(getattr(hyp, "crowd_size_ratio_min", 0.75)),
        "crowd_size_ratio_max": float(getattr(hyp, "crowd_size_ratio_max", 1.33)),
        "scale_cp_target_max_size": float(getattr(hyp, "scale_cp_target_max_size", 20.0)),
        "scale_cp_source_min_ratio": float(getattr(hyp, "scale_cp_source_min_ratio", 1.25)),
        "scale_cp_factor_min": float(getattr(hyp, "scale_cp_factor_min", 0.50)),
        "scale_cp_factor_max": float(getattr(hyp, "scale_cp_factor_max", 0.90)),
        "negative_cp_target_policy": str(getattr(hyp, "negative_cp_target_policy", "empirical")),
        "negative_cp_donor_policy": str(getattr(hyp, "negative_cp_donor_policy", "matched")),
        "negative_cp_target_max_size": float(getattr(hyp, "negative_cp_target_max_size", 20.0)),
        "negative_cp_deficit_gamma": float(getattr(hyp, "negative_cp_deficit_gamma", 0.5)),
        "negative_cp_max_weight_ratio": float(getattr(hyp, "negative_cp_max_weight_ratio", 3.0)),
        "negative_cp_matched_ratio_max": float(getattr(hyp, "negative_cp_matched_ratio_max", 1.5)),
        "negative_cp_large_ratio_min": float(getattr(hyp, "negative_cp_large_ratio_min", 1.5)),
        "negative_cp_large_ratio_max": float(getattr(hyp, "negative_cp_large_ratio_max", 2.5)),
        "negative_cp_degradation": str(getattr(hyp, "negative_cp_degradation", "none")),
        "negative_cp_blur_sigma": float(getattr(hyp, "negative_cp_blur_sigma", 0.5)),
    }
