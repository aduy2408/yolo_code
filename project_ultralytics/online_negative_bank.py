"""Worker-safe, previous-epoch hard-negative patch bank."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import os
import random

import cv2
import numpy as np


@dataclass
class OnlineHardNegativeRecord:
    patch_path: str
    pred_size: float
    hardness: float
    ema_hardness: float
    source_image_id: str
    epoch_seen: int = 0
    times_seen: int = 1

    def update(self, hardness: float, beta: float = 0.9, epoch: int = 0) -> None:
        self.ema_hardness = beta * self.ema_hardness + (1.0 - beta) * float(hardness)
        self.hardness = float(hardness)
        self.epoch_seen = int(epoch)
        self.times_seen += 1


class OnlineHardNegativeBank:
    """A bounded bank committed atomically so persistent workers see one version."""

    def __init__(self, root: str | Path, max_size: int = 1000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.root = Path(root)
        self.max_size = int(max_size)
        self.records: list[OnlineHardNegativeRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add_patch(self, image: np.ndarray, record: OnlineHardNegativeRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        patch_name = Path(record.patch_path).name
        path = self.root / patch_name
        if not cv2.imwrite(str(path), image):
            raise IOError(f"could not write hard-negative patch: {path}")
        record.patch_path = str(path)
        self.records.append(record)
        self.records.sort(key=lambda item: item.ema_hardness, reverse=True)
        self.records = self.records[: self.max_size]

    def commit(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "bank.json.tmp"
        final = self.root / "bank.json"
        tmp.write_text(json.dumps({"max_size": self.max_size, "records": [asdict(r) for r in self.records]}, indent=2), encoding="utf-8")
        os.replace(tmp, final)
        return final

    @classmethod
    def load(cls, root: str | Path) -> "OnlineHardNegativeBank":
        root = Path(root)
        bank = cls(root)
        path = root / "bank.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            bank.max_size = int(payload.get("max_size", 1000))
            bank.records = [OnlineHardNegativeRecord(**item) for item in payload.get("records", [])]
        return bank

    def ranked(self, target_size: float | None = None, hardness_power: float = 1.0, scale_weight: float = 1.0):
        def score(record):
            hard = max(record.ema_hardness, 1e-8) ** hardness_power
            scale = 0.0 if target_size is None else abs(np.log(max(record.pred_size, 1e-8) / max(target_size, 1e-8)))
            return hard - scale_weight * scale
        return sorted(self.records, key=score, reverse=True)


class OnlineHardNegativeCollector:
    """Collect detached false-positive crops for the next epoch."""

    def __init__(self, bank: OnlineHardNegativeBank, crop_expand: float = 1.5, beta: float = 0.9) -> None:
        if crop_expand < 1.0 or not 0.0 <= beta < 1.0:
            raise ValueError("crop_expand must be >= 1 and beta must be in [0, 1)")
        self.bank = bank
        self.crop_expand = float(crop_expand)
        self.beta = float(beta)

    def collect(self, image: np.ndarray, predictions, source_image_id: str, epoch: int = 0) -> int:
        """Collect ``(x1,y1,x2,y2,confidence)`` predictions as negative patches.

        Callers are responsible for filtering foreground/ignored predictions at
        the criterion seam. This keeps the collector independent of tensor types.
        """
        h, w = image.shape[:2]
        added = 0
        for index, row in enumerate(np.asarray(predictions, dtype=np.float32).reshape(-1, 5)):
            x1, y1, x2, y2, confidence = map(float, row)
            size = max(np.sqrt(max((x2 - x1) * (y2 - y1), 0.0)), 1e-6)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = (x2 - x1) * self.crop_expand / 2, (y2 - y1) * self.crop_expand / 2
            ax1, ay1 = max(0, round(cx - hw)), max(0, round(cy - hh))
            ax2, ay2 = min(w, round(cx + hw)), min(h, round(cy + hh))
            patch = image[ay1:ay2, ax1:ax2]
            if patch.size == 0:
                continue
            name = f"epoch{epoch:04d}_{index:05d}.jpg"
            record = OnlineHardNegativeRecord(name, size, confidence, confidence, str(source_image_id), epoch)
            self.bank.add_patch(patch, record)
            added += 1
        return added


class OnlineNegativeCopyPaste:
    """Paste previous-epoch hard negatives without modifying annotations."""

    def __init__(self, bank: OnlineHardNegativeBank, p: float = 0.3, max_objects: int = 3,
                 scale_matched: bool = False, max_gt_ioa: float = 0.05, max_trials: int = 30, rng=None) -> None:
        if max_objects < 1 or max_trials < 1:
            raise ValueError("max_objects and max_trials must be positive")
        self.bank, self.p, self.max_objects = bank, float(p), int(max_objects)
        self.scale_matched, self.max_gt_ioa, self.max_trials = bool(scale_matched), float(max_gt_ioa), int(max_trials)
        self.rng = rng or random

    @staticmethod
    def _boxes(labels):
        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return np.empty((0, 4), dtype=np.float32)
        instances.convert_bbox("xyxy")
        if instances.normalized:
            h, w = labels["img"].shape[:2]
            instances.denormalize(w, h)
        return np.asarray(instances.bboxes, dtype=np.float32).copy()

    @staticmethod
    def _ioa(candidate, gt):
        inter = max(0.0, min(candidate[2], gt[2]) - max(candidate[0], gt[0])) * max(0.0, min(candidate[3], gt[3]) - max(candidate[1], gt[1]))
        area = max((candidate[2] - candidate[0]) * (candidate[3] - candidate[1]), 1e-8)
        return inter / area

    def __call__(self, labels):
        if not self.bank.records or self.p <= 0 or self.rng.random() >= self.p:
            return labels
        image = labels["img"]
        h, w = image.shape[:2]
        boxes = self._boxes(labels)
        target = float(np.median([np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1e-8)) for b in boxes])) if len(boxes) else 0.0
        choices = self.bank.ranked(target if self.scale_matched and target else None)
        for record in choices[: self.max_objects]:
            patch = cv2.imread(record.patch_path, cv2.IMREAD_COLOR)
            if patch is None or patch.shape[0] > h or patch.shape[1] > w:
                continue
            for _ in range(self.max_trials):
                x, y = self.rng.randint(0, w - patch.shape[1]), self.rng.randint(0, h - patch.shape[0])
                candidate = np.array([x, y, x + patch.shape[1], y + patch.shape[0]], dtype=np.float32)
                if any(self._ioa(candidate, gt) >= self.max_gt_ioa for gt in boxes):
                    continue
                image[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
                break
        return labels


__all__ = ["OnlineHardNegativeRecord", "OnlineHardNegativeBank", "OnlineHardNegativeCollector", "OnlineNegativeCopyPaste"]
