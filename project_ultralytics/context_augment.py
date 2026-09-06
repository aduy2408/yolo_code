"""Object/context augmentations for the LEVIR YOLO ablation matrix.

The transforms operate on raw ``labels['img']`` before Mosaic and do not alter
instances.  Selection is controlled by ``YOLO_CONTEXT_AUG`` so the canonical
pipeline remains unchanged when the variable is unset or ``none``.
"""
from __future__ import annotations

import os
import random
from typing import Any

import cv2
import numpy as np


def _boxes(labels: dict[str, Any], h: int, w: int) -> np.ndarray:
    inst = labels.get("instances")
    if inst is None and labels.get("bboxes") is not None:
        raw = np.asarray(labels["bboxes"], dtype=np.float32)
        if raw.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        raw = raw.reshape(-1, 4)
        xc, yc, bw, bh = raw.T
        return np.stack(((xc - bw / 2) * w, (yc - bh / 2) * h,
                         (xc + bw / 2) * w, (yc + bh / 2) * h), axis=1)
    if inst is None or len(inst.bboxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    b = np.asarray(inst.bboxes, dtype=np.float32).copy()
    if getattr(inst, "bbox_format", "xywh") == "xywh":
        xc, yc, bw, bh = b.T
        b = np.stack((xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2), axis=1)
    if getattr(inst, "normalized", False):
        b[:, [0, 2]] *= w
        b[:, [1, 3]] *= h
    return np.clip(b, [0, 0, 0, 0], [w, h, w, h])


def _mask_from_boxes(boxes: np.ndarray, h: int, w: int, expand: float) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = (x2 - x1) * expand / 2, (y2 - y1) * expand / 2
        xa, xb = max(0, int(cx - hw)), min(w, int(cx + hw + 1))
        ya, yb = max(0, int(cy - hh)), min(h, int(cy + hh + 1))
        mask[ya:yb, xa:xb] = 1
    return mask


def _soft_far_mask(boxes: np.ndarray, h: int, w: int, expand: float, sigma: float) -> np.ndarray:
    protected = _mask_from_boxes(boxes, h, w, expand)
    if not protected.any():
        return np.ones((h, w), np.float32)
    dist = cv2.distanceTransform((1 - protected).astype(np.uint8), cv2.DIST_L2, 3)
    return (1.0 - np.exp(-(dist * dist) / (2.0 * sigma * sigma))).astype(np.float32)


def _resize_degrade(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class OACP:
    """Object-Anchored Context Perturbation."""

    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return labels
        img = labels.get("img")
        if img is None or img.ndim != 3:
            return labels
        h, w = img.shape[:2]
        boxes = _boxes(labels, h, w)
        mask = _soft_far_mask(boxes, h, w, 3.0, max(3.0, min(h, w) * 0.06))
        if boxes.size:
            mask *= 1.0 - _mask_from_boxes(boxes, h, w, 1.2)
        strength = random.uniform(0.2, 0.4) * max(0.0, 1.0 - float(mask.mean() < 0.02) * 0.0)
        degraded = _resize_degrade(img, random.uniform(0.65, 0.85)).astype(np.float32)
        out = img.astype(np.float32) * (1.0 - strength * mask[..., None]) + degraded * (strength * mask[..., None])
        labels["img"] = np.clip(out, 0, 255).astype(img.dtype)
        return labels


class CEA:
    """Context Exchange Augmentation with donor-GT exclusion."""

    def __init__(self, dataset, p: float = 0.15) -> None:
        self.dataset, self.p = dataset, p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p or len(self.dataset) < 2:
            return labels
        img = labels.get("img")
        if img is None or img.ndim != 3:
            return labels
        h, w = img.shape[:2]
        source_boxes = _boxes(labels, h, w)
        exchange = _soft_far_mask(source_boxes, h, w, 3.0, max(3.0, min(h, w) * 0.06))
        if exchange.mean() < 0.25:
            return labels
        for _ in range(10):
            idx = random.randrange(len(self.dataset))
            donor, *_ = self.dataset.load_image(idx)
            if donor.shape[:2] != (h, w):
                donor = cv2.resize(donor, (w, h), interpolation=cv2.INTER_LINEAR)
            donor_boxes = _boxes({"bboxes": self.dataset.labels[idx].get("bboxes", [])}, h, w)
            donor_exclusion = _mask_from_boxes(donor_boxes, h, w, 1.2)
            use = (exchange > 0.55) & ~donor_exclusion.astype(bool)
            if use.mean() >= 0.15:
                src = img.astype(np.float32)
                d = donor.astype(np.float32)
                # Match luminance statistics only in exchanged pixels.
                ys = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
                yd = cv2.cvtColor(donor.astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
                mu_s, sd_s = ys[use].mean(), ys[use].std() + 1e-6
                mu_d, sd_d = yd[use].mean(), yd[use].std() + 1e-6
                d = np.clip((d - mu_d) * (sd_s / sd_d) + mu_s, 0, 255)
                alpha = cv2.GaussianBlur(use.astype(np.float32), (0, 0), 3)[..., None]
                labels["img"] = np.clip(src * (1 - alpha) + d * alpha, 0, 255).astype(img.dtype)
                return labels
        return labels


class LEA:
    """Local Evidence Attenuation preserving high-frequency object detail."""

    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return labels
        img = labels.get("img")
        if img is None or img.ndim != 3:
            return labels
        h, w = img.shape[:2]
        boxes = _boxes(labels, h, w)
        if not len(boxes):
            return labels
        out = img.astype(np.float32).copy()
        for x1, y1, x2, y2 in boxes[:2]:
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < 5 or max(bw, bh) > 64:
                continue
            xa, ya, xb, yb = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            if xb - xa < 3 or yb - ya < 3:
                continue
            crop = out[ya:yb, xa:xb]
            k = max(3, min(7, int(min(crop.shape[:2]) // 3) * 2 + 1))
            low = cv2.GaussianBlur(crop, (k, k), 0)
            retention = random.uniform(0.65, 0.85)
            softened = low + retention * (crop - low)
            yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
            cy, cx = (crop.shape[0] - 1) / 2, (crop.shape[1] - 1) / 2
            radius = ((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2
            mask = np.clip(1.0 - radius, 0, 1).astype(np.float32)[..., None]
            out[ya:yb, xa:xb] = crop + mask * (softened - crop)
        labels["img"] = np.clip(out, 0, 255).astype(img.dtype)
        return labels


def build_context_augment(dataset):
    """Build configured transforms from ``YOLO_CONTEXT_AUG``."""
    mode = os.environ.get("YOLO_CONTEXT_AUG", "none").lower()
    transforms = []
    if mode in {"oacp", "all"}:
        transforms.append(OACP())
    if mode in {"cea", "all"}:
        transforms.append(CEA(dataset))
    if mode in {"lea", "all"}:
        transforms.append(LEA())
    return transforms
