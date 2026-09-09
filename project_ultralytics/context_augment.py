"""Safe object/context augmentations for the LEVIR YOLO ablation matrix.

These transforms run on a raw sample before Mosaic.  ``YOLO_CONTEXT_AUG``
selects the method and ``LEA_STATS_PATH`` can point at dataset-specific JSON
quantiles produced by an offline statistics pass.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

import cv2
import numpy as np

AUG_CONFIG = {
    "object_max_size": 32.0,
    "protected_expand": 3.0,
    "safety_expand": 1.2,
    "transition_sigma_ratio": 0.06,
    "oacp_strength": [0.20, 0.40],
    "oacp_resolution_scale": [0.65, 0.85],
    "cea_probability": 0.15,
    "cea_min_exchange_ratio": 0.25,
    "cea_donor_overlap_max": 0.05,
    "lea_probability": 0.20,
    "lea_ring_expand": 1.75,
    "lea_min_ring_ratio": 0.40,
    "lea_contrast_retention": [0.65, 0.85],
    "lea_alpha_max": 0.35,
    "lea_max_objects": 2,
}


def augmentation_config() -> dict[str, Any]:
    """Return the complete immutable run configuration for manifests."""
    cfg = dict(AUG_CONFIG)
    cfg["protected_expand"] = float(os.environ.get("OACP_PROTECTED_EXPAND", cfg["protected_expand"]))
    cfg["oacp_probability"] = float(os.environ.get("OACP_P", "0.20"))
    cfg["oacp_strength"] = [
        float(os.environ.get("OACP_STRENGTH_MIN", cfg["oacp_strength"][0])),
        float(os.environ.get("OACP_STRENGTH_MAX", cfg["oacp_strength"][1])),
    ]
    cfg["oacp_resolution_scale"] = [
        float(os.environ.get("OACP_SCALE_MIN", cfg["oacp_resolution_scale"][0])),
        float(os.environ.get("OACP_SCALE_MAX", cfg["oacp_resolution_scale"][1])),
    ]
    cfg["mode"] = os.environ.get("YOLO_CONTEXT_AUG", "none").lower()
    cfg["legacy_double_oacp"] = os.environ.get("YOLO_LEGACY_DOUBLE_OACP", "0").lower() in {
        "1", "true", "yes", "on"
    }
    cfg["lea_stats_path"] = os.environ.get("LEA_STATS_PATH", "")
    cfg["sweep_label"] = os.environ.get("OACP_SWEEP_LABEL", "")
    return cfg


def _oacp_config() -> dict[str, Any]:
    cfg = augmentation_config()
    return {
        "p": cfg["oacp_probability"],
        "protected_expand": cfg["protected_expand"],
        "strength": cfg["oacp_strength"],
        "resolution_scale": cfg["oacp_resolution_scale"],
    }


def _boxes(labels: dict[str, Any], h: int, w: int) -> np.ndarray:
    inst = labels.get("instances")
    if inst is None and labels.get("bboxes") is not None:
        raw = np.asarray(labels["bboxes"], dtype=np.float32).reshape(-1, 4)
        if not len(raw):
            return np.empty((0, 4), dtype=np.float32)
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


def _protection(boxes: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.sqrt(np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])) if len(boxes) else np.empty(0)
    tiny = boxes[sizes < AUG_CONFIG["object_max_size"]]
    protected = _mask_from_boxes(tiny, h, w, _oacp_config()["protected_expand"])
    protected |= _mask_from_boxes(boxes, h, w, AUG_CONFIG["safety_expand"]).astype(bool)
    return protected.astype(np.uint8), tiny


def _far_mask(protected: np.ndarray, h: int, w: int) -> np.ndarray:
    if not protected.any():
        return np.zeros((h, w), np.float32)
    sigma = max(3.0, min(h, w) * AUG_CONFIG["transition_sigma_ratio"])
    dist = cv2.distanceTransform((1 - protected).astype(np.uint8), cv2.DIST_L2, 3)
    return (1.0 - np.exp(-(dist * dist) / (2.0 * sigma * sigma))).astype(np.float32)


def _resize_degrade(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class OACP:
    """Degrade only far context around eligible tiny objects."""

    def __init__(self, p: float = 0.20) -> None:
        self.p = float(p)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        cfg = _oacp_config()
        if random.random() >= cfg["p"]:
            return labels
        img = labels.get("img")
        if img is None or img.ndim != 3:
            return labels
        h, w = img.shape[:2]
        boxes = _boxes(labels, h, w)
        protected, tiny = _protection(boxes, h, w)
        if not len(tiny) or float(protected.mean()) > 0.55:
            return labels
        mask = _far_mask(protected, h, w)
        strength = random.uniform(*cfg["strength"]) * (1.0 - float(protected.mean()))
        degraded = _resize_degrade(img, random.uniform(*cfg["resolution_scale"]))
        out = img.astype(np.float32) * (1 - strength * mask[..., None]) + degraded.astype(np.float32) * (strength * mask[..., None])
        labels["img"] = np.clip(out, 0, 255).astype(img.dtype)
        return labels


class CEA:
    """Exchange far context with a donor while hard-excluding donor objects."""

    def __init__(self, dataset, p: float = 0.15) -> None:
        self.dataset, self.p = dataset, float(p)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p or len(self.dataset) < 2:
            return labels
        img = labels.get("img")
        if img is None or img.ndim != 3:
            return labels
        h, w = img.shape[:2]
        source_boxes = _boxes(labels, h, w)
        protected, tiny = _protection(source_boxes, h, w)
        if not len(tiny) or float(protected.mean()) > 0.50:
            return labels
        exchange = _far_mask(protected, h, w)
        if float((exchange > 0.55).mean()) < AUG_CONFIG["cea_min_exchange_ratio"]:
            return labels
        for _ in range(10):
            idx = random.randrange(len(self.dataset))
            donor, *_ = self.dataset.load_image(idx)
            if donor.shape[:2] != (h, w):
                donor = cv2.resize(donor, (w, h), interpolation=cv2.INTER_LINEAR)
            donor_boxes = _boxes({"bboxes": self.dataset.labels[idx].get("bboxes", [])}, h, w)
            donor_exclusion = _mask_from_boxes(donor_boxes, h, w, AUG_CONFIG["safety_expand"]).astype(bool)
            hard_use = (exchange > 0.55) & ~donor_exclusion
            if float(hard_use.mean()) < AUG_CONFIG["cea_min_exchange_ratio"]:
                continue
            # Match LAB L only, preserving donor chroma channels exactly.
            src_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            donor_lab = cv2.cvtColor(donor, cv2.COLOR_BGR2LAB).astype(np.float32)
            mu_s, sd_s = src_lab[..., 0][hard_use].mean(), src_lab[..., 0][hard_use].std() + 1e-6
            mu_d, sd_d = donor_lab[..., 0][hard_use].mean(), donor_lab[..., 0][hard_use].std() + 1e-6
            donor_lab[..., 0] = np.clip((donor_lab[..., 0] - mu_d) * (sd_s / sd_d) + mu_s, 0, 255)
            matched = cv2.cvtColor(donor_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
            alpha = cv2.GaussianBlur((exchange * (~donor_exclusion)).astype(np.float32), (0, 0), 3)
            alpha[donor_exclusion] = 0.0  # hard zero after blur prevents donor leakage
            alpha[protected.astype(bool)] = 0.0  # source object/local context is never replaced
            src = img.astype(np.float32)
            labels["img"] = np.clip(src * (1 - alpha[..., None]) + matched * alpha[..., None], 0, 255).astype(img.dtype)
            return labels
        return labels


def _lea_stats() -> dict[str, float]:
    path = os.environ.get("LEA_STATS_PATH", "")
    if path and os.path.isfile(path):
        try:
            return {k: float(v) for k, v in json.loads(open(path, encoding="utf-8").read()).items()}
        except (OSError, ValueError, TypeError):
            pass
    return {"contrast_q10": 0.50, "contrast_q20": 0.60, "contrast_q40": 0.80}


class LEA:
    """Reduce foreground/background low-frequency contrast while preserving H."""

    def __init__(self, p: float = 0.20) -> None:
        self.p = float(p)

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
        stats = _lea_stats()
        out = img.astype(np.float32).copy()
        all_mask = _mask_from_boxes(boxes, h, w, 1.0).astype(bool)
        candidates = list(range(len(boxes))); random.shuffle(candidates)
        applied = 0
        for i in candidates:
            if applied >= int(AUG_CONFIG["lea_max_objects"]):
                break
            x1, y1, x2, y2 = boxes[i]
            bw, bh = x2 - x1, y2 - y1
            if not (5.0 <= np.sqrt(max(0.0, bw * bh)) < AUG_CONFIG["object_max_size"]):
                continue
            ox1, oy1, ox2, oy2 = max(0, int((x1+x2)/2 - bw*AUG_CONFIG["lea_ring_expand"]/2)), max(0, int((y1+y2)/2 - bh*AUG_CONFIG["lea_ring_expand"]/2)), min(w, int((x1+x2)/2 + bw*AUG_CONFIG["lea_ring_expand"]/2)), min(h, int((y1+y2)/2 + bh*AUG_CONFIG["lea_ring_expand"]/2))
            ring = np.zeros((h, w), bool); ring[oy1:oy2, ox1:ox2] = True; ring &= ~all_mask
            obj = np.zeros((h, w), bool); obj[max(0,int(y1)):min(h,int(y2)), max(0,int(x1)):min(w,int(x2))] = True
            if ring.sum() < AUG_CONFIG["lea_min_ring_ratio"] * max(1, (ox2-ox1)*(oy2-oy1)):
                continue
            lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            contrast = abs(float(lum[obj].mean() - lum[ring].mean())) / (float(lum[ring].std()) + 1e-6)
            if contrast < stats["contrast_q40"]:
                continue
            xa, ya, xb, yb = max(0,int(x1)), max(0,int(y1)), min(w,int(x2)), min(h,int(y2))
            crop = out[ya:yb, xa:xb]
            if min(crop.shape[:2]) < 3:
                continue
            k = max(3, min(7, int(min(crop.shape[:2]) // 3) * 2 + 1))
            low = cv2.GaussianBlur(crop, (k, k), 0)
            ring_l = float(lum[ring].mean()); crop_l = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2LAB)[...,0].astype(np.float32)
            alpha = min(AUG_CONFIG["lea_alpha_max"], max(0.0, 1.0 - random.uniform(*AUG_CONFIG["lea_contrast_retention"])))
            low_lab = cv2.cvtColor(low.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            low_lab[..., 0] += alpha * (ring_l - crop_l)
            low_target = cv2.cvtColor(np.clip(low_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
            yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]; cy, cx = (crop.shape[0]-1)/2, (crop.shape[1]-1)/2
            mask = np.clip(1 - ((xx-cx)/max(cx,1))**2 - ((yy-cy)/max(cy,1))**2, 0, 1).astype(np.float32)[...,None]
            out[ya:yb, xa:xb] = crop + mask * (low_target + (crop-low) - crop)
            applied += 1
        labels["img"] = np.clip(out, 0, 255).astype(img.dtype)
        return labels


def build_context_augment(dataset):
    mode = os.environ.get("YOLO_CONTEXT_AUG", "none").lower()
    if mode == "oacp": return [OACP()]
    if mode == "cea": return [CEA(dataset)]
    if mode == "lea": return [LEA()]
    return []
