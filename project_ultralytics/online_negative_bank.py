"""Worker-safe, previous-epoch hard-negative patch bank."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from multiprocessing import Value
from pathlib import Path
import random
from typing import Iterable

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
    region_key: str = ""

    def update(self, hardness: float, beta: float = 0.9, epoch: int = 0) -> None:
        self.ema_hardness = beta * self.ema_hardness + (1.0 - beta) * float(hardness)
        self.hardness = float(hardness)
        self.epoch_seen = int(epoch)
        self.times_seen += 1


class OnlineHardNegativeBank:
    """Bounded metadata bank with atomic manifests and monotonic versions."""

    def __init__(self, root: str | Path, max_size: int = 1000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.root = Path(root)
        self.max_size = int(max_size)
        self.records: list[OnlineHardNegativeRecord] = []
        self.version = 0

    def __len__(self) -> int:
        return len(self.records)

    def append_or_update(self, record: OnlineHardNegativeRecord, image: np.ndarray, beta: float = 0.9) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / Path(record.patch_path).name
        if not cv2.imwrite(str(path), image):
            raise IOError(f"could not write hard-negative patch: {path}")
        existing = next((item for item in self.records if record.region_key and item.region_key == record.region_key), None)
        if existing is not None:
            existing.patch_path = str(path)
            existing.pred_size = record.pred_size
            existing.update(record.hardness, beta=beta, epoch=record.epoch_seen)
        else:
            record.patch_path = str(path)
            self.records.append(record)
        self.records.sort(key=lambda item: item.ema_hardness, reverse=True)
        self.records = self.records[: self.max_size]

    def add_patch(self, image: np.ndarray, record: OnlineHardNegativeRecord) -> None:
        self.append_or_update(record, image)

    def commit(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        next_version = self.version + 1
        payload = {"max_size": self.max_size, "version": next_version, "records": [asdict(r) for r in self.records]}
        tmp = self.root / "bank.json.tmp"
        final = self.root / "bank.json"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, final)
        self.version = next_version
        return final

    @classmethod
    def load(cls, root: str | Path) -> "OnlineHardNegativeBank":
        root = Path(root)
        path = root / "bank.json"
        if not path.exists():
            return cls(root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        bank = cls(root, max_size=int(payload.get("max_size", 1000)))
        bank.version = int(payload.get("version", 0))
        bank.records = [OnlineHardNegativeRecord(**item) for item in payload.get("records", [])]
        return bank

    def ranked(self, target_size: float | None = None, hardness_power: float = 1.0, scale_weight: float = 1.0):
        def score(record):
            hard = max(record.ema_hardness, 1e-8) ** hardness_power
            scale_penalty = 0.0 if target_size is None else abs(np.log(max(record.pred_size, 1e-8) / max(target_size, 1e-8)))
            return hard - scale_weight * scale_penalty
        return sorted(self.records, key=score, reverse=True)


class OnlineHardNegativeCollector:
    """Collect detached false-positive crops for the next epoch."""

    def __init__(self, bank: OnlineHardNegativeBank, crop_expand: float = 1.5, beta: float = 0.9) -> None:
        if crop_expand < 1.0 or not 0.0 <= beta < 1.0:
            raise ValueError("crop_expand must be >= 1 and beta must be in [0, 1)")
        self.bank = bank
        self.crop_expand = float(crop_expand)
        self.beta = float(beta)

    @staticmethod
    def bce_hardness(confidence: float) -> float:
        probability = float(np.clip(confidence, 1e-6, 1.0 - 1e-6))
        return float(-np.log1p(-probability))

    def collect(self, image: np.ndarray, predictions: Iterable, source_image_id: str, epoch: int = 0) -> int:
        """Collect rows ``x1,y1,x2,y2,confidence[,hardness]`` as patches."""
        h, w = image.shape[:2]
        added = 0
        for index, row in enumerate(predictions):
            row = np.asarray(row, dtype=np.float32).reshape(-1)
            if row.size < 5:
                continue
            x1, y1, x2, y2, confidence = map(float, row[:5])
            hardness = float(row[5]) if row.size > 5 else self.bce_hardness(confidence)
            size = max(np.sqrt(max((x2 - x1) * (y2 - y1), 0.0)), 1e-6)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = (x2 - x1) * self.crop_expand / 2, (y2 - y1) * self.crop_expand / 2
            ax1, ay1 = max(0, round(cx - hw)), max(0, round(cy - hh))
            ax2, ay2 = min(w, round(cx + hw)), min(h, round(cy + hh))
            patch = image[ay1:ay2, ax1:ax2]
            if patch.size == 0:
                continue
            region_key = f"{source_image_id}:{round(cx / 4)}:{round(cy / 4)}:{round(size / 2)}"
            digest = hashlib.sha1(region_key.encode("utf-8")).hexdigest()[:16]
            name = f"epoch{epoch:04d}_{digest}_{index:05d}.jpg"
            record = OnlineHardNegativeRecord(name, size, hardness, hardness, str(source_image_id), epoch, 1, region_key)
            self.bank.append_or_update(record, patch, beta=self.beta)
            added += 1
        return added

    def collect_batch(self, images, candidates: Iterable[Iterable], source_image_ids: Iterable[str], epoch: int = 0) -> int:
        total = 0
        for image, rows, source_id in zip(images, candidates, source_image_ids):
            total += self.collect(image, rows, str(source_id), epoch=epoch)
        return total


class OnlineNegativeCopyPaste:
    """Paste previous-epoch hard negatives without modifying annotations."""

    def __init__(self, bank: OnlineHardNegativeBank, p: float = 0.3, max_objects: int = 3,
                 scale_matched: bool = False, max_gt_ioa: float = 0.05, max_trials: int = 30,
                 budget=None, rng=None) -> None:
        if max_objects < 1 or max_trials < 1:
            raise ValueError("max_objects and max_trials must be positive")
        self.bank, self.p, self.max_objects = bank, float(p), int(max_objects)
        self.scale_matched, self.max_gt_ioa, self.max_trials = bool(scale_matched), float(max_gt_ioa), int(max_trials)
        self.budget = budget
        self.rng = rng or random
        self._bank_version = bank.version
        self._refresh_version = Value("i", bank.version, lock=True)

    def refresh(self) -> None:
        """Reload a committed manifest and publish a worker-visible version."""
        self.bank = OnlineHardNegativeBank.load(self.bank.root)
        with self._refresh_version.get_lock():
            self._refresh_version.value = self.bank.version
        self._bank_version = self._refresh_version.value

    def _ensure_fresh(self) -> None:
        version = self._refresh_version.value
        if version != self._bank_version:
            self.bank = OnlineHardNegativeBank.load(self.bank.root)
            self._bank_version = version

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
        self._ensure_fresh()
        if not self.bank.records or self.p <= 0 or self.rng.random() >= self.p:
            return labels
        image = labels["img"]
        h, w = image.shape[:2]
        boxes = self._boxes(labels)
        target = float(np.median([np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1e-8)) for b in boxes])) if len(boxes) else 0.0
        count = self.budget(len(boxes), self.rng) if self.budget is not None else self.max_objects
        choices = self.bank.ranked(target if self.scale_matched and target else None)
        for record in choices[: min(count, self.max_objects)]:
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


def iter_online_negative_transforms(dataset):
    """Yield online transforms from nested Ultralytics Compose objects."""
    root = getattr(dataset, "transforms", None)
    stack = [root] if root is not None else []
    while stack:
        current = stack.pop()
        if isinstance(current, OnlineNegativeCopyPaste):
            yield current
        for child in getattr(current, "transforms", ()) or ():
            stack.append(child)


def configure_online_hard_negative_training(trainer) -> bool:
    """Attach one epoch collector to a trainer when online NegCP is enabled."""
    mode = str(getattr(trainer.args, "copy_paste_mode", "")).lower()
    if mode not in {"online_negative", "online_negative_scale_matched"}:
        return False
    transforms = list(iter_online_negative_transforms(getattr(trainer.train_loader, "dataset", None)))
    if not transforms:
        raise RuntimeError("online NegCP is enabled but no OnlineNegativeCopyPaste transform was built")
    transform = transforms[0]
    trainer._online_hn_transform = transform
    trainer._online_hn_bank = transform.bank
    trainer._online_hn_collector = OnlineHardNegativeCollector(
        trainer._online_hn_bank,
        crop_expand=float(getattr(trainer.args, "online_negcp_crop_expand", 1.5)),
        beta=float(getattr(trainer.args, "online_negcp_beta", 0.9)),
    )
    return True


def collect_online_hard_negatives(trainer, batch, criterion, epoch: int) -> int:
    """Consume criterion candidates from one forward batch and persist patches."""
    collector = getattr(trainer, "_online_hn_collector", None)
    candidates = getattr(criterion, "last_hard_negative_candidates", None)
    if collector is None or candidates is None:
        return 0
    images = batch["img"].detach().float().cpu().numpy()
    images = np.transpose(np.clip(images * 255.0, 0, 255).astype(np.uint8), (0, 2, 3, 1))
    dataset_indices = batch.get("dataset_idx")
    if hasattr(dataset_indices, "detach"):
        dataset_indices = dataset_indices.detach().cpu().tolist()
    dataset_indices = dataset_indices or list(range(len(images)))
    source_ids = [str(index) for index in dataset_indices]
    return collector.collect_batch(images, candidates, source_ids, epoch=epoch)


def finish_online_hard_negative_epoch(trainer) -> int:
    """Commit the epoch bank then publish a worker-visible refresh version."""
    bank = getattr(trainer, "_online_hn_bank", None)
    transform = getattr(trainer, "_online_hn_transform", None)
    if bank is None or transform is None:
        return 0
    before = bank.version
    bank.commit()
    transform.refresh()
    return int(bank.version - before)


__all__ = [
    "OnlineHardNegativeRecord", "OnlineHardNegativeBank", "OnlineHardNegativeCollector",
    "OnlineNegativeCopyPaste", "iter_online_negative_transforms", "configure_online_hard_negative_training",
    "collect_online_hard_negatives", "finish_online_hard_negative_epoch",
]
