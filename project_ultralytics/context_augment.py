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
    "oacp_budget": [0.30, 0.60],
    "oacp_density_expand_min": 1.0,
    "oacp_density_expand_max": 3.0,
    "oacp_density_target_min": 0.08,
    "oacp_density_target_max": 0.22,
    "oacp_mass_target_min": 0.25,
    "oacp_mass_target_max": 0.40,
    "oacp_adaptive_budget_min": 0.20,
    "oacp_adaptive_budget_max": 0.70,
    "oacp_load_saturation_count": 10,
    "oacp_spacing_near": 1.0,
    "oacp_spacing_far": 6.0,
    "oacp_spacing_expand_min": 1.2,
    "oacp_spacing_expand_max": 3.0,
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
    cfg["oacp_budget"] = [
        float(os.environ.get("OACP_BUDGET_MIN", cfg["oacp_budget"][0])),
        float(os.environ.get("OACP_BUDGET_MAX", cfg["oacp_budget"][1])),
    ]
    cfg["oacp_mass_target"] = [
        float(os.environ.get("OACP_MASS_TARGET_MIN", cfg["oacp_mass_target_min"])),
        float(os.environ.get("OACP_MASS_TARGET_MAX", cfg["oacp_mass_target_max"])),
    ]
    cfg["oacp_adaptive_budget"] = [
        float(os.environ.get("OACP_ADAPTIVE_BUDGET_MIN", cfg["oacp_adaptive_budget_min"])),
        float(os.environ.get("OACP_ADAPTIVE_BUDGET_MAX", cfg["oacp_adaptive_budget_max"])),
    ]
    cfg["oacp_load_saturation_count"] = int(os.environ.get(
        "OACP_LOAD_SATURATION_COUNT", cfg["oacp_load_saturation_count"]
    ))
    cfg["oacp_spacing"] = {
        "near": float(os.environ.get("OACP_SPACING_NEAR", cfg["oacp_spacing_near"])),
        "far": float(os.environ.get("OACP_SPACING_FAR", cfg["oacp_spacing_far"])),
        "expand_min": float(os.environ.get(
            "OACP_SPACING_EXPAND_MIN", cfg["oacp_spacing_expand_min"]
        )),
        "expand_max": float(os.environ.get(
            "OACP_SPACING_EXPAND_MAX", cfg["oacp_spacing_expand_max"]
        )),
    }
    cfg["oacp_variant"] = os.environ.get("OACP_VARIANT", "current").lower()
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
        "budget": cfg["oacp_budget"],
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


def _mask_from_boxes_per_expand(
    boxes: np.ndarray, expands: np.ndarray, h: int, w: int
) -> np.ndarray:
    """Build a union mask where each box has its own context expansion."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    expands = np.asarray(expands, dtype=np.float32).reshape(-1)
    if len(boxes) != len(expands):
        raise ValueError("boxes and expands must have the same length")
    mask = np.zeros((h, w), np.uint8)
    for box, expand in zip(boxes, expands):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = (x2 - x1) * float(expand) / 2, (y2 - y1) * float(expand) / 2
        xa, xb = max(0, int(cx - hw)), min(w, int(cx + hw + 1))
        ya, yb = max(0, int(cy - hh)), min(h, int(cy + hh + 1))
        if xa < xb and ya < yb:
            mask[ya:yb, xa:xb] = 1
    return mask


def _bbox_edge_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Return Euclidean edge-to-edge distance between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(float(ax1 - bx2), float(bx1 - ax2), 0.0)
    dy = max(float(ay1 - by2), float(by1 - ay2), 0.0)
    return float(np.hypot(dx, dy))


def _spacing_adaptive_expands(
    boxes: np.ndarray,
    eligible_indices: np.ndarray,
    *,
    near_spacing: float,
    far_spacing: float,
    expand_min: float,
    expand_max: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Choose per-object context retention from normalized nearest spacing."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    eligible_indices = np.asarray(eligible_indices, dtype=np.int64).reshape(-1)
    records: list[dict[str, float]] = []
    expands = np.empty(len(eligible_indices), dtype=np.float32)
    denominator = max(float(far_spacing - near_spacing), 1e-8)

    for out_index, object_index in enumerate(eligible_indices):
        box = boxes[object_index]
        width = max(float(box[2] - box[0]), 0.0)
        height = max(float(box[3] - box[1]), 0.0)
        scale = float(np.sqrt(width * height))
        other_indices = np.arange(len(boxes), dtype=np.int64)
        other_indices = other_indices[other_indices != object_index]
        if len(other_indices):
            nearest_gap = min(_bbox_edge_distance(box, boxes[j]) for j in other_indices)
            normalized_spacing = nearest_gap / (scale + 1e-8)
        else:
            # An isolated single object is explicitly treated as far away.
            nearest_gap = float("inf")
            normalized_spacing = float("inf")
        retention = 1.0 - float(np.clip(
            (normalized_spacing - near_spacing) / denominator, 0.0, 1.0
        ))
        expand = float(expand_min + retention * (expand_max - expand_min))
        expands[out_index] = expand
        records.append({
            "object_size": scale,
            "nearest_gap": nearest_gap,
            "normalized_spacing": normalized_spacing,
            "retention": retention,
            "expand": expand,
        })
    return expands, records


def _mass_adaptive_budget(
    valid_bg_ratio: float,
    target_mass: float,
    budget_min: float,
    budget_max: float,
    eps: float = 1e-8,
) -> tuple[float, bool]:
    """Convert an image-area target into a valid-background budget."""
    raw_budget = float(target_mass) / max(float(valid_bg_ratio), eps)
    budget = float(np.clip(raw_budget, budget_min, budget_max))
    return budget, bool(raw_budget < budget_min or raw_budget > budget_max)


def _load_adaptive_target_mass(
    num_eligible: int,
    mass_min: float,
    mass_max: float,
    saturation_count: int,
) -> tuple[float, float]:
    """Return continuous object load and its monotonically decreasing mass target."""
    if num_eligible <= 1:
        load = 0.0
    else:
        load = (num_eligible - 1) / max(saturation_count - 1, 1)
    load = float(np.clip(load, 0.0, 1.0))
    target_mass = float(mass_max - load * (mass_max - mass_min))
    return load, target_mass


def _protection(boxes: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.sqrt(np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])) if len(boxes) else np.empty(0)
    tiny = boxes[sizes < AUG_CONFIG["object_max_size"]]
    protected = _mask_from_boxes(tiny, h, w, _oacp_config()["protected_expand"])
    protected |= _mask_from_boxes(boxes, h, w, AUG_CONFIG["safety_expand"]).astype(bool)
    return protected.astype(np.uint8), tiny


def _density_adaptive_expand(boxes: np.ndarray, h: int, w: int) -> tuple[float, float, float]:
    """Choose expansion by searching for a controlled union-mask coverage."""
    cfg = augmentation_config()
    if not len(boxes):
        return cfg["protected_expand"], 0.0, 0.0
    sizes = np.sqrt(np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1]))
    tiny = boxes[sizes < AUG_CONFIG["object_max_size"]]
    gt_mask = _mask_from_boxes(boxes, h, w, 1.0)
    safety = _mask_from_boxes(boxes, h, w, AUG_CONFIG["safety_expand"]).astype(bool)
    occupancy = float(gt_mask.mean())
    # Dense scenes receive a smaller context target. Search the actual union
    # mask rather than scaling each box independently, so overlapping GTs are
    # handled by scene geometry instead of object count.
    target = float(np.clip(
        cfg["oacp_density_target_max"] - 0.5 * occupancy,
        cfg["oacp_density_target_min"], cfg["oacp_density_target_max"],
    ))
    candidates = np.linspace(
        cfg["oacp_density_expand_min"], cfg["oacp_density_expand_max"], 25
    )
    coverages = np.asarray([
        (safety | _mask_from_boxes(tiny, h, w, float(exp)).astype(bool)).mean()
        for exp in candidates
    ])
    index = int(np.argmin(np.abs(coverages - target)))
    return float(candidates[index]), occupancy, target


def _protection_for_variant(boxes: np.ndarray, h: int, w: int, variant: str) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    sizes = np.sqrt(np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])) if len(boxes) else np.empty(0)
    eligible_indices = np.flatnonzero(sizes < AUG_CONFIG["object_max_size"])
    tiny = boxes[eligible_indices]
    expand = _oacp_config()["protected_expand"]
    occupancy = float(_mask_from_boxes(boxes, h, w, 1.0).mean()) if len(boxes) else 0.0
    target = 0.0
    if variant == "density":
        expand, occupancy, target = _density_adaptive_expand(boxes, h, w)
        protected = _mask_from_boxes(boxes, h, w, AUG_CONFIG["safety_expand"]).astype(bool)
        protected |= _mask_from_boxes(tiny, h, w, expand).astype(bool)
    elif variant == "spacing_adaptive":
        spacing = augmentation_config()["oacp_spacing"]
        expands, _ = _spacing_adaptive_expands(
            boxes,
            eligible_indices,
            near_spacing=spacing["near"],
            far_spacing=spacing["far"],
            expand_min=spacing["expand_min"],
            expand_max=spacing["expand_max"],
        )
        protected = _mask_from_boxes(
            boxes, h, w, AUG_CONFIG["safety_expand"]
        ).astype(bool)
        protected |= _mask_from_boxes_per_expand(tiny, expands, h, w).astype(bool)
        if len(expands):
            expand = float(np.mean(expands))
    else:
        # Preserve the historical OACP control exactly: tiny objects get the
        # configured context expansion, while every GT receives safety cover.
        protected, tiny = _protection(boxes, h, w)
    return protected.astype(np.uint8), tiny, float(expand), occupancy, target


def oacp_diagnostics(shape: tuple[int, int] | tuple[int, int, int], boxes: np.ndarray,
                     variant: str = "current", budget: float | None = None,
                     target_mass: float | None = None) -> dict[str, Any]:
    """Return geometry/severity diagnostics without modifying an image.

    ``boxes`` are absolute ``xyxy`` coordinates.  The budget is a fraction of
    valid background, not of the complete image, so the reported actual area
    makes the effective severity comparable across sparse and crowded scenes.
    """
    h, w = shape[:2]
    variant = variant.lower()
    if variant not in {"current", "budget", "density", "mass_adaptive", "load_adaptive", "spacing_adaptive"}:
        raise ValueError(f"unknown OACP variant: {variant}")
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    protected, tiny, expand, occupancy, density_target = _protection_for_variant(boxes, h, w, variant)
    protected_ratio = float(protected.mean())
    gt_mask = _mask_from_boxes(boxes, h, w, 1.0).astype(bool)
    available = (protected == 0) & ~gt_mask
    far = _far_mask(protected, h, w)
    cfg = augmentation_config()
    valid_bg_ratio = float(available.mean())
    num_eligible = int(len(tiny))
    object_load = 0.0
    budget_clipped = False
    if variant == "load_adaptive":
        object_load, target_mass = _load_adaptive_target_mass(
            num_eligible,
            cfg["oacp_mass_target"][0],
            cfg["oacp_mass_target"][1],
            cfg["oacp_load_saturation_count"],
        )
    elif variant == "mass_adaptive" and target_mass is None:
        target_mass = float(sum(cfg["oacp_mass_target"]) / 2.0)
    if variant in {"mass_adaptive", "load_adaptive"}:
        budget, budget_clipped = _mass_adaptive_budget(
            valid_bg_ratio,
            float(target_mass),
            cfg["oacp_adaptive_budget"][0],
            cfg["oacp_adaptive_budget"][1],
        )
    elif budget is None:
        budget = float(sum(cfg["oacp_budget"]) / 2.0)
    budget = float(np.clip(budget, 0.0, 1.0))
    eligible = bool(len(tiny))
    would_apply = eligible and protected_ratio <= 0.55
    if not eligible:
        skip_reason = "no_eligible_tiny"
    elif protected_ratio > 0.55:
        skip_reason = "protected_coverage_gt_0.55"
    else:
        skip_reason = ""
    if not would_apply:
        target_perturb = 0.0
        perturb = np.zeros((h, w), dtype=np.uint8)
    elif variant == "current":
        target_perturb = float(available.mean())
        perturb = (far > 0).astype(np.uint8)
    else:
        # Select the farthest valid pixels first. This gives a stable mask for
        # diagnostics and reserves near-object context even in dense scenes.
        budget = float(np.clip(budget, 0.0, 1.0))
        target_perturb = float(budget * available.mean())
        count = int(round(budget * float(available.sum())))
        flat = np.flatnonzero(available)
        if count >= len(flat):
            chosen = flat
        elif count:
            order = np.argsort(far.flat[flat])[::-1]
            chosen = flat[order[:count]]
        else:
            chosen = np.empty(0, dtype=np.int64)
        perturb = np.zeros(h * w, dtype=np.uint8)
        perturb[chosen] = 1
        perturb = perturb.reshape(h, w)
    sizes = np.sqrt(
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    ) if len(boxes) else np.empty(0)
    spacing_records: list[dict[str, float]] = []
    if variant == "spacing_adaptive" and len(tiny):
        spacing = cfg["oacp_spacing"]
        _, spacing_records = _spacing_adaptive_expands(
            boxes,
            np.flatnonzero(sizes < AUG_CONFIG["object_max_size"]),
            near_spacing=spacing["near"],
            far_spacing=spacing["far"],
            expand_min=spacing["expand_min"],
            expand_max=spacing["expand_max"],
        )
    normalized_spacings = [r["normalized_spacing"] for r in spacing_records]
    adaptive_expands = [r["expand"] for r in spacing_records]
    return {
        "variant": variant,
        "num_gt": int(len(boxes)),
        "num_tiny": int(len(tiny)),
        "protected_area_ratio": protected_ratio,
        "perturbable_area_ratio": float(available.mean()),
        "actual_perturbed_area_ratio": float(perturb.mean()),
        "actual_perturbed_area_ratio_image": float(perturb.mean()),
        "target_perturbed_area_ratio": target_perturb,
        "target_perturbed_area_ratio_image": target_perturb,
        "budget_fraction_of_valid_bg": budget if variant != "current" else 1.0,
        "valid_background_ratio": valid_bg_ratio,
        "target_image_mass": float(target_mass) if target_mass is not None else 0.0,
        "adaptive_budget": budget if variant in {"mass_adaptive", "load_adaptive"} else 0.0,
        "budget_clipped": budget_clipped,
        "num_eligible": num_eligible,
        "object_load": object_load,
        "mean_nearest_distance_norm": float(np.mean(normalized_spacings)) if normalized_spacings else 0.0,
        "median_nearest_distance_norm": float(np.median(normalized_spacings)) if normalized_spacings else 0.0,
        "mean_adaptive_expand": float(np.mean(adaptive_expands)) if adaptive_expands else 0.0,
        "min_adaptive_expand": float(np.min(adaptive_expands)) if adaptive_expands else 0.0,
        "max_adaptive_expand": float(np.max(adaptive_expands)) if adaptive_expands else 0.0,
        "spacing_objects": spacing_records,
        "gt_area_ratio": float(gt_mask.mean()),
        "perturb_gt_overlap_ratio": float((perturb.astype(bool) & gt_mask).mean()),
        "eligible": eligible,
        "would_apply": would_apply,
        "skip_reason": skip_reason,
        "density_occupancy": occupancy,
        "density_target_protected_ratio": density_target,
        "mean_object_size": float(np.mean(np.sqrt(np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])))) if len(boxes) else 0.0,
        "protected_expand": expand,
        "perturb_budget_valid_background": budget if variant != "current" else 1.0,
    }


def _record_oacp_diagnostics(labels: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    """Optionally append one JSON record per transformed sample."""
    path = os.environ.get("OACP_DIAGNOSTICS_PATH", "")
    if not path:
        return
    record = dict(diagnostics)
    record["image"] = labels.get("im_file", "")
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


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
        probability = cfg["p"] if self.p == 0.20 else self.p
        if random.random() >= probability:
            return labels
        img = labels.get("img")
        if img is None or img.ndim != 3:
            return labels
        h, w = img.shape[:2]
        boxes = _boxes(labels, h, w)
        variant = augmentation_config()["oacp_variant"]
        valid_variants = {
            "current", "budget", "density", "mass_adaptive", "load_adaptive", "spacing_adaptive"
        }
        if variant not in valid_variants:
            raise ValueError(f"unknown OACP_VARIANT: {variant}")
        target_mass = None
        if variant == "mass_adaptive":
            target_mass = random.uniform(*augmentation_config()["oacp_mass_target"])
            budget = None
        elif variant == "load_adaptive":
            budget = None
        else:
            budget = random.uniform(*cfg["budget"]) if variant != "current" else None
        protected, tiny, _, _, _ = _protection_for_variant(boxes, h, w, variant)
        diagnostics = oacp_diagnostics(
            (h, w), boxes, variant=variant, budget=budget, target_mass=target_mass
        )
        if variant in {"mass_adaptive", "load_adaptive"}:
            budget = float(diagnostics["adaptive_budget"])
        if not diagnostics["would_apply"]:
            _record_oacp_diagnostics(labels, diagnostics)
            return labels
        mask = _far_mask(protected, h, w)
        if variant in {"budget", "density", "mass_adaptive", "load_adaptive"}:
            selected = np.zeros((h, w), np.uint8)
            available = protected == 0
            count = int(round(float(budget) * float(available.sum())))
            flat = np.flatnonzero(available)
            if count >= len(flat):
                chosen = flat
            else:
                chosen = flat[np.argsort(mask.flat[flat])[::-1][:count]] if count else np.empty(0, dtype=np.int64)
            selected.flat[chosen] = 1
            mask *= selected
        diagnostics["actual_perturbed_area_ratio"] = float((mask > 0).mean())
        diagnostics["actual_perturbed_area_ratio_image"] = diagnostics["actual_perturbed_area_ratio"]
        diagnostics["target_perturbed_area_ratio"] = diagnostics["target_perturbed_area_ratio_image"]
        diagnostics["perturb_gt_overlap_ratio"] = 0.0
        _record_oacp_diagnostics(labels, diagnostics)
        strength = random.uniform(*cfg["strength"])
        if variant == "current":
            strength *= 1.0 - float(protected.mean())
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
