"""Offline hard-negative mining and negative Copy-Paste augmentation.

The miner consumes post-NMS detector predictions.  Negative patches never add
annotations, so this transform is safe to ablate independently from positive
Copy-Paste strategies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HardNegativeRecord:
    image_path: str
    crop_xyxy: tuple[float, float, float, float]
    pred_xyxy: tuple[float, float, float, float]
    conf: float
    pred_size: float
    source_image_id: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HardNegativeRecord":
        return cls(
            image_path=str(value["image_path"]),
            crop_xyxy=tuple(float(x) for x in value["crop_xyxy"]),
            pred_xyxy=tuple(float(x) for x in value["pred_xyxy"]),
            conf=float(value["conf"]),
            pred_size=float(value.get("pred_size", 0.0)),
            source_image_id=str(value.get("source_image_id", value["image_path"])),
        )


class HardNegativeBank:
    """Bounded metadata-only bank. Source crops are read on demand."""

    def __init__(self, records: Iterable[HardNegativeRecord] = (), max_size: int = 10000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        self.records = sorted(list(records), key=lambda record: record.conf, reverse=True)[: self.max_size]

    def __len__(self) -> int:
        return len(self.records)

    def append(self, record: HardNegativeRecord) -> None:
        self.records.append(record)
        self.records.sort(key=lambda item: item.conf, reverse=True)
        del self.records[self.max_size :]

    def sample(self, rng=random):
        if not self.records:
            return None
        return rng.choice(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {"max_size": self.max_size, "records": [asdict(record) for record in self.records]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HardNegativeBank":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = [HardNegativeRecord.from_dict(item) for item in payload.get("records", [])]
        return cls(records, max_size=int(payload.get("max_size", 10000)))


def _area(box: Sequence[float]) -> float:
    return max(float(box[2] - box[0]) * float(box[3] - box[1]), 0.0)


def _intersection(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    return max(min(float(box_a[2]), float(box_b[2])) - max(float(box_a[0]), float(box_b[0])), 0.0) * max(
        min(float(box_a[3]), float(box_b[3])) - max(float(box_a[1]), float(box_b[1])), 0.0
    )


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    inter = _intersection(box_a, box_b)
    return inter / max(_area(box_a) + _area(box_b) - inter, 1e-8)


def _ioa(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    return _intersection(box_a, box_b) / max(_area(box_a), 1e-8)


class HardNegativeMiner:
    """Build an offline hard-FP bank from post-NMS predictions and labels."""

    def __init__(
        self,
        conf_threshold: float = 0.25,
        max_iou_with_valid_gt: float = 0.10,
        max_ioa_with_ignore: float = 0.30,
        crop_expand: float = 1.5,
        bank_per_image: int = 3,
        bank_max: int = 10000,
    ) -> None:
        if crop_expand < 1.0 or bank_per_image < 1 or bank_max < 1:
            raise ValueError("crop_expand >= 1 and positive bank limits are required")
        self.conf_threshold = float(conf_threshold)
        self.max_iou_with_valid_gt = float(max_iou_with_valid_gt)
        self.max_ioa_with_ignore = float(max_ioa_with_ignore)
        self.crop_expand = float(crop_expand)
        self.bank_per_image = int(bank_per_image)
        self.bank_max = int(bank_max)
        self.stats = {"candidates": 0, "accepted": 0, "rejected_gt": 0, "rejected_ignore": 0}

    @staticmethod
    def _prediction_rows(predictions: Any) -> list[tuple[np.ndarray, float]]:
        if isinstance(predictions, dict):
            boxes = predictions.get("boxes", predictions.get("bboxes", []))
            conf = predictions.get("conf", predictions.get("confidence", []))
            return [(np.asarray(box, dtype=np.float32), float(score)) for box, score in zip(boxes, conf)]
        array = np.asarray(predictions, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        rows = array.reshape(-1, array.shape[-1]) if array.ndim else np.empty((0, 0), dtype=np.float32)
        return [(row[:4], float(row[4])) for row in rows if len(row) >= 5]

    @staticmethod
    def _box_array(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0, 4), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return array.reshape(-1, 4)

    def mine(
        self,
        image_paths: Sequence[str | Path],
        predictions: Sequence[Any],
        valid_gts: Sequence[Any],
        ignore_regions: Sequence[Any] | None = None,
    ) -> HardNegativeBank:
        bank = HardNegativeBank(max_size=self.bank_max)
        if ignore_regions is None:
            ignore_regions = [[] for _ in image_paths]
        for image_path, image_predictions, gts, ignores in zip(image_paths, predictions, valid_gts, ignore_regions):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            rows = []
            for box, conf in self._prediction_rows(image_predictions):
                if conf < self.conf_threshold:
                    continue
                self.stats["candidates"] += 1
                if any(_iou(box, gt) >= self.max_iou_with_valid_gt for gt in self._box_array(gts)):
                    self.stats["rejected_gt"] += 1
                    continue
                if any(_ioa(box, ignore) >= self.max_ioa_with_ignore for ignore in self._box_array(ignores)):
                    self.stats["rejected_ignore"] += 1
                    continue
                rows.append((conf, box))
            rows.sort(key=lambda item: item[0], reverse=True)
            for conf, box in rows[: self.bank_per_image]:
                cx, cy = (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)
                hw, hh = (float(box[2] - box[0]) * self.crop_expand / 2, float(box[3] - box[1]) * self.crop_expand / 2)
                crop = (max(0.0, cx - hw), max(0.0, cy - hh), min(float(w), cx + hw), min(float(h), cy + hh))
                bank.append(HardNegativeRecord(
                    image_path=str(image_path), crop_xyxy=crop,
                    pred_xyxy=tuple(float(x) for x in box), conf=float(conf),
                    pred_size=float(np.sqrt(max(_area(box), 0.0))), source_image_id=str(image_path),
                ))
                self.stats["accepted"] += 1
        return bank


class NegativeCopyPaste:
    """Paste a hard-negative patch while leaving labels byte-for-byte unchanged."""

    def __init__(
        self,
        bank: HardNegativeBank,
        p: float = 0.30,
        num: int = 1,
        scale: float = 1.0,
        max_gt_ioa: float = 0.05,
        max_trials: int = 30,
        same_source: bool = False,
        rng=None,
        debug_dir: str | Path | None = None,
        debug_limit: int = 32,
    ) -> None:
        if scale != 1.0:
            raise ValueError("NegCP V1 locks scale=1.0")
        if num < 1 or max_trials < 1:
            raise ValueError("num and max_trials must be positive")
        self.bank = bank
        self.p = float(p)
        self.num = int(num)
        self.scale = float(scale)
        self.max_gt_ioa = float(max_gt_ioa)
        self.max_trials = int(max_trials)
        self.same_source = bool(same_source)
        self.rng = rng or random
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_limit = int(debug_limit)
        self.stats = {"attempted_images": 0, "applied_images": 0, "pasted_patches": 0, "sampled_conf_sum": 0.0, "rejected_gt": 0, "rejected_boundary": 0, "failed_trials": 0}

    def _dump_debug(self, labels: dict[str, Any]) -> None:
        count = self.stats.get("debug_dump_count", 0)
        if self.debug_dir is None or count >= self.debug_limit:
            return
        canvas = labels["img"].copy()
        instances = labels.get("instances")
        if instances is not None and len(instances):
            for x1, y1, x2, y2 in self._boxes(labels):
                cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 1)
        out_dir = self.debug_dir / "negative"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{count:04d}.jpg"), canvas)
        self.stats["debug_dump_count"] = count + 1

    @staticmethod
    def _boxes(labels: dict[str, Any]) -> np.ndarray:
        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return np.empty((0, 4), dtype=np.float32)
        boxes = np.asarray(instances.bboxes, dtype=np.float32).copy()
        bbox_format = getattr(getattr(instances, "_bboxes", None), "format", "xywh")
        if bbox_format == "xywh":
            xc, yc, width, height = boxes.T
            boxes = np.stack((xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2), axis=1)
        if getattr(instances, "normalized", False):
            h, w = labels["img"].shape[:2]
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
        return boxes

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if not self.bank.records or self.p <= 0 or self.rng.random() >= self.p:
            return labels
        self.stats["attempted_images"] += 1
        before = self._boxes(labels).copy()
        h, w = labels["img"].shape[:2]
        source_id = str(labels.get("im_file", ""))
        applied = 0
        for _ in range(self.num):
            choices = self.bank.records
            if not self.same_source and source_id:
                choices = [record for record in choices if record.source_image_id != source_id] or choices
            record = self.rng.choice(choices)
            image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            x1, y1, x2, y2 = map(round, record.crop_xyxy)
            patch = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)].copy()
            if patch.size == 0 or patch.shape[0] > h or patch.shape[1] > w:
                self.stats["rejected_boundary"] += 1
                continue
            placed = False
            for _ in range(self.max_trials):
                x = self.rng.randint(0, w - patch.shape[1])
                y = self.rng.randint(0, h - patch.shape[0])
                candidate = np.array([x, y, x + patch.shape[1], y + patch.shape[0]], dtype=np.float32)
                if any(_ioa(candidate, gt) >= self.max_gt_ioa for gt in before):
                    self.stats["rejected_gt"] += 1
                    continue
                labels["img"][y:y + patch.shape[0], x:x + patch.shape[1]] = patch
                self.stats["sampled_conf_sum"] += record.conf
                self.stats["pasted_patches"] += 1
                applied += 1
                placed = True
                break
            if not placed:
                self.stats["failed_trials"] += 1
        # The key invariant: no annotation operation is performed in NegCP.
        if applied:
            self.stats["applied_images"] += 1
            self._dump_debug(labels)
        return labels

    def diagnostics(self) -> dict[str, float]:
        out = dict(self.stats)
        out.setdefault("debug_dump_count", 0)
        out["bank_size"] = len(self.bank)
        out["sampled_fp_confidence_mean"] = out["sampled_conf_sum"] / max(out["pasted_patches"], 1)
        out.update({f"negcp/{key}": value for key, value in out.items()})
        return out


__all__ = ["HardNegativeRecord", "HardNegativeBank", "HardNegativeMiner", "NegativeCopyPaste"]
