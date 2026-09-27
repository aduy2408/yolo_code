"""Scale-Transfer Copy-Paste augmentation.

STCP is intentionally a final-canvas transform.  Mosaic and geometric
transforms must already have run before this class is called, so the sampled
target scale is the scale visible to the detector rather than a pre-Mosaic
scale.  Donor metadata is built once from the raw training set and pixels are
loaded only when a donor is selected.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .copy_paste import SmallObjectCopyPaste, _box_area


@dataclass(frozen=True)
class STCPDonorRecord:
    image_index: int
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    raw_w: int
    raw_h: int
    raw_scale: float
    final_scale: float
    aspect_ratio: float
    ignore: bool = False


class ScaleTransferCopyPaste(SmallObjectCopyPaste):
    """Paste one or two same-class, larger real donors at tiny final scales."""

    def __init__(
        self,
        dataset,
        p: float = 0.30,
        min_pastes: int = 1,
        max_pastes: int = 2,
        target_max_scale: float = 20.0,
        ratio_min: float = 1.5,
        ratio_max: float = 2.5,
        interpolation: int = cv2.INTER_AREA,
        blur_sigma: float = 0.5,
        blur_min_side: int = 5,
        max_trials: int = 30,
        max_ioa: float = 0.10,
        min_donor_w: int = 1,
        min_donor_h: int = 1,
        rng=None,
        debug_dir: str | Path | None = None,
        debug_limit: int = 50,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if min_pastes < 1 or max_pastes < min_pastes:
            raise ValueError("paste budget must satisfy 1 <= min_pastes <= max_pastes")
        if target_max_scale <= 0 or not 0 < ratio_min <= ratio_max:
            raise ValueError("invalid STCP scale parameters")
        if blur_sigma <= 0 or blur_min_side < 1:
            raise ValueError("invalid STCP blur parameters")
        if max_trials < 1 or not 0 <= max_ioa <= 1:
            raise ValueError("invalid STCP placement parameters")

        super().__init__(
            dataset=dataset, p=p, unit="single", copies=1,
            placement="collision_aware", max_overlap=max_ioa,
            padding=0.0, scale=1.0, blend="hard", max_trials=max_trials,
            allow_empty_target=True, allow_same_source=True, rng=rng,
            debug_dir=debug_dir, debug_limit=debug_limit,
        )
        self.min_pastes = int(min_pastes)
        self.max_pastes = int(max_pastes)
        self.target_max_scale = float(target_max_scale)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.interpolation = int(interpolation)
        self.blur_sigma = float(blur_sigma)
        self.blur_min_side = int(blur_min_side)
        self.min_donor_w = int(min_donor_w)
        self.min_donor_h = int(min_donor_h)
        self.stcp_records: list[STCPDonorRecord] = []
        self.records_by_class: dict[int, list[STCPDonorRecord]] = defaultdict(list)
        self.target_scale_pool: dict[int, list[float]] = defaultdict(list)
        self.class_weights: list[tuple[int, float]] = []
        self.class_histogram: Counter[int] = Counter()
        self._stcp_pool_built = False
        self.stats = self._stcp_stats()

    @staticmethod
    def _stcp_stats() -> dict[str, float]:
        return {
            "applications": 0, "requested_pastes": 0,
            "successful_pastes": 0, "no_donor": 0,
            "placement_fail": 0, "source_scale_sum": 0.0,
            "target_scale_sum": 0.0, "source_target_ratio_sum": 0.0,
            "target_w_sum": 0.0, "target_h_sum": 0.0,
            "degradation_applied": 0, "debug_dump_count": 0,
        }

    def reset_stats(self) -> None:
        self.stats = self._stcp_stats()
        self.class_histogram.clear()

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)
        xc, yc, w, h = boxes.T
        return np.stack((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), axis=1)

    @staticmethod
    def _label_boxes(label: dict[str, Any]) -> np.ndarray:
        boxes = np.asarray(label.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
        if label.get("bbox_format", "xywh") == "xywh":
            boxes = ScaleTransferCopyPaste._xywh_to_xyxy(boxes)
        if label.get("normalized", False) and len(boxes):
            h, w = label["shape"][:2]
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
        return boxes

    @staticmethod
    def _ignore_flags(label: dict[str, Any], count: int) -> np.ndarray:
        for key in ("ignore", "is_ignore", "uncertain"):
            if key in label:
                values = np.asarray(label[key]).reshape(-1).astype(bool)
                if len(values) == count:
                    return values
        return np.zeros(count, dtype=bool)

    def _build_pool(self) -> None:
        if self._stcp_pool_built:
            return
        imgsz = float(getattr(self.dataset, "imgsz", 0) or 0)
        if imgsz <= 0:
            imgsz = float(getattr(self.dataset, "stride", 640) or 640)
        labels = getattr(self.dataset, "labels", ())
        for image_index, label in enumerate(labels):
            boxes = self._label_boxes(label)
            classes = np.asarray(label.get("cls", []), dtype=np.int64).reshape(-1)
            if len(classes) != len(boxes):
                continue
            raw_h, raw_w = map(int, label.get("shape", (0, 0))[:2])
            if raw_w <= 0 or raw_h <= 0:
                paths = getattr(self.dataset, "im_files", ())
                if image_index < len(paths):
                    probe = cv2.imread(str(paths[image_index]), cv2.IMREAD_UNCHANGED)
                    if probe is not None:
                        raw_h, raw_w = probe.shape[:2]
            if raw_w <= 0 or raw_h <= 0:
                continue
            raw_scale = min(imgsz / raw_w, imgsz / raw_h)
            ignored = self._ignore_flags(label, len(boxes))
            for box, cls, is_ignored in zip(boxes, classes, ignored):
                x1, y1, x2, y2 = map(float, box)
                width, height = x2 - x1, y2 - y1
                if width < self.min_donor_w or height < self.min_donor_h or is_ignored:
                    continue
                record = STCPDonorRecord(
                    image_index=image_index, bbox_xyxy=(x1, y1, x2, y2),
                    class_id=int(cls), raw_w=raw_w, raw_h=raw_h,
                    raw_scale=math.sqrt(max(width * height, 1e-8)),
                    final_scale=math.sqrt(max(width * height, 1e-8)) * raw_scale,
                    aspect_ratio=width / max(height, 1e-8), ignore=bool(is_ignored),
                )
                self.stcp_records.append(record)
                self.records_by_class[record.class_id].append(record)
                target_scale = record.final_scale
                if target_scale <= self.target_max_scale:
                    self.target_scale_pool[record.class_id].append(target_scale)
        counts = Counter(record.class_id for record in self.stcp_records)
        total = sum(counts.values())
        if total:
            self.class_weights = [(cls, count / total) for cls, count in sorted(counts.items())]
        self._stcp_pool_built = True

    def _sample_class(self) -> int | None:
        if not self.class_weights:
            return None
        classes, weights = zip(*self.class_weights)
        return int(self.rng.choices(classes, weights=weights, k=1)[0])

    def _sample_target_scale(self, class_id: int) -> float | None:
        pool = self.target_scale_pool.get(class_id, ())
        return float(self.rng.choice(pool)) if pool else None

    def _sample_donor(self, class_id: int, target_scale: float) -> STCPDonorRecord | None:
        lo, hi = self.ratio_min * target_scale, self.ratio_max * target_scale
        candidates = [
            record for record in self.records_by_class.get(class_id, ())
            if lo <= record.final_scale <= hi
        ]
        return self.rng.choice(candidates) if candidates else None

    def _choose_position(self, patch_shape: tuple[int, int], labels: dict[str, Any], existing: np.ndarray):
        ph, pw = patch_shape
        h, w = labels["img"].shape[:2]
        ignore = np.asarray(labels.get("ignore_boxes", []), dtype=np.float32).reshape(-1, 4)
        if ph <= 0 or pw <= 0 or ph > h or pw > w:
            return None
        for _ in range(self.max_trials):
            x = self.rng.randint(0, w - pw)
            y = self.rng.randint(0, h - ph)
            box = np.array([x, y, x + pw, y + ph], dtype=np.float32)
            if self._intersection_over_area(box, existing) > self.max_overlap:
                continue
            if self._intersection_over_area(box, ignore) > self.max_overlap:
                continue
            return x, y, box
        return None

    def _debug_dump(self, labels: dict[str, Any], metadata: list[dict[str, Any]]) -> None:
        if self.debug_dir is None or self.stats["debug_dump_count"] >= self.debug_limit:
            return
        out_dir = self.debug_dir / "stcp"
        out_dir.mkdir(parents=True, exist_ok=True)
        index = int(self.stats["debug_dump_count"])
        image = labels["img"].copy()
        for item in metadata:
            x1, y1, x2, y2 = map(int, item["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(image, f"src={item['source_scale']:.1f}->tgt={item['target_scale']:.1f}",
                        (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        path = out_dir / f"{index:04d}.jpg"
        cv2.imwrite(str(path), image)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.stats["debug_dump_count"] += 1

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.p <= 0 or self.rng.random() >= self.p:
            return labels
        self._build_pool()
        if not self.stcp_records:
            return labels
        self.stats["applications"] += 1
        count = self.rng.randint(self.min_pastes, self.max_pastes)
        self.stats["requested_pastes"] += count
        existing = self._target_boxes(labels)
        pasted_boxes, pasted_classes, metadata = [], [], []
        for _ in range(count):
            class_id = self._sample_class()
            target_scale = None if class_id is None else self._sample_target_scale(class_id)
            donor = None if class_id is None or target_scale is None else self._sample_donor(class_id, target_scale)
            if donor is None:
                self.stats["no_donor"] += 1
                continue
            source = self._load_raw(donor.image_index)
            prepared = self._crop_single(
                type("Record", (), {"bbox_xyxy": donor.bbox_xyxy})(), source
            ) if source is not None else None
            if prepared is None:
                self.stats["no_donor"] += 1
                continue
            crop, _source_box = prepared
            source_scale = donor.final_scale
            factor = target_scale / max(source_scale, 1e-8)
            target_w = max(2, round(crop.shape[1] * factor))
            target_h = max(2, round(crop.shape[0] * factor))
            crop = cv2.resize(crop, (target_w, target_h), interpolation=self.interpolation)
            if min(target_w, target_h) >= self.blur_min_side:
                crop = cv2.GaussianBlur(crop, (3, 3), self.blur_sigma)
                self.stats["degradation_applied"] += 1
            destination = self._choose_position(crop.shape[:2], labels, existing)
            if destination is None:
                self.stats["placement_fail"] += 1
                continue
            x, y, box = destination
            labels["img"][y:y + target_h, x:x + target_w] = crop
            pasted_boxes.append(box.reshape(1, 4))
            pasted_classes.append(np.array([class_id], dtype=np.int64))
            existing = np.concatenate([existing, box.reshape(1, 4)], axis=0)
            ratio = source_scale / max(target_scale, 1e-8)
            self.stats["successful_pastes"] += 1
            self.class_histogram[class_id] += 1
            self.stats["source_scale_sum"] += source_scale
            self.stats["target_scale_sum"] += target_scale
            self.stats["source_target_ratio_sum"] += ratio
            self.stats["target_w_sum"] += target_w
            self.stats["target_h_sum"] += target_h
            metadata.append({"bbox": box.tolist(), "class_id": class_id,
                             "source_scale": source_scale, "target_scale": target_scale,
                             "ratio": ratio})
        if pasted_boxes:
            self._append_instances(labels, pasted_boxes, pasted_classes)
            self._debug_dump(labels, metadata)
        return labels

    def diagnostics(self) -> dict[str, float]:
        out = dict(self.stats)
        n = max(int(out["successful_pastes"]), 1)
        out["source_scale_mean"] = out["source_scale_sum"] / n
        out["target_scale_mean"] = out["target_scale_sum"] / n
        out["source_target_ratio_mean"] = out["source_target_ratio_sum"] / n
        out["target_w_mean"] = out["target_w_sum"] / n
        out["target_h_mean"] = out["target_h_sum"] / n
        out["class_histogram"] = dict(self.class_histogram)
        out.update({f"stcp/{key}": value for key, value in out.items() if not key.startswith("stcp/")})
        return out


__all__ = ["STCPDonorRecord", "ScaleTransferCopyPaste"]
