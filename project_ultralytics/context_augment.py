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

from .oacp_state import OACPSharedState

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
    "oacp_probability_policy": "fixed",
    "oacp_probability_min": 0.20,
    "oacp_probability_max": 0.40,
    "oacp_effect_policy": "fixed",
    "oacp_target_effect": 0.0,
    "oacp_strength_policy": "fixed",
    "oacp_load_strength_min": 0.05,
    "oacp_load_strength_max": 0.15,
    "oacp_protection_policy": "fixed",
    "oacp_size_expand_min": 2.0,
    "oacp_size_expand_max": 4.0,
    "oacp_size_expand_smax": 32.0,
    "oacp_curriculum_warmup_fraction": 0.15,
    "oacp_curriculum_cooldown_fraction": 0.10,
    "oacp_curriculum_strength": [0.05, 0.15],
    "oacp_hardness_warmup_epochs": 5,
    "oacp_hardness_beta": 0.90,
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
    "oacp_spatial_load_min": 0.0,
    "oacp_spatial_load_max": 0.04,
    "oacp_spacing_near": 1.0,
    "oacp_spacing_far": 6.0,
    "oacp_spacing_expand_min": 1.2,
    "oacp_spacing_expand_max": 3.0,
}


def augmentation_config() -> dict[str, Any]:
    """Return the complete immutable run configuration for manifests."""
    cfg = dict(AUG_CONFIG)
    profile = os.environ.get("OACP_PROFILE", "").lower()
    if profile not in {"", "r2"}:
        raise ValueError(f"unknown OACP_PROFILE: {profile}")
    profile_defaults = {
        "p": "0.40" if profile == "r2" else "0.20",
        "strength": ("0.10", "0.25") if profile == "r2" else ("0.20", "0.40"),
        "scale": ("0.80", "0.95") if profile == "r2" else ("0.65", "0.85"),
        "variant": "current" if profile == "r2" else "current",
        "placement": "pre_transform" if profile == "r2" else "post_mosaic",
    }
    cfg["profile"] = profile or "default"
    cfg["protected_expand"] = float(os.environ.get("OACP_PROTECTED_EXPAND", cfg["protected_expand"]))
    cfg["oacp_probability"] = float(os.environ.get("OACP_P", profile_defaults["p"]))
    cfg["oacp_probability_policy"] = os.environ.get(
        "OACP_PROB_POLICY", cfg["oacp_probability_policy"]
    ).lower()
    cfg["oacp_probability_min"] = float(os.environ.get(
        "OACP_P_MIN", cfg["oacp_probability_min"]
    ))
    cfg["oacp_probability_max"] = float(os.environ.get(
        "OACP_P_MAX", cfg["oacp_probability_max"]
    ))
    cfg["oacp_effect_policy"] = os.environ.get(
        "OACP_EFFECT_POLICY", cfg["oacp_effect_policy"]
    ).lower()
    cfg["oacp_target_effect"] = float(os.environ.get(
        "OACP_TARGET_EFFECT", cfg["oacp_target_effect"]
    ))
    cfg["oacp_strength"] = [
        float(os.environ.get("OACP_STRENGTH_MIN", profile_defaults["strength"][0])),
        float(os.environ.get("OACP_STRENGTH_MAX", profile_defaults["strength"][1])),
    ]
    explicit_strength_policy = os.environ.get("OACP_STRENGTH_POLICY", "").lower()
    if explicit_strength_policy:
        cfg["oacp_strength_policy"] = explicit_strength_policy
    elif cfg["oacp_effect_policy"] == "adaptive":
        # Backward-compatible alias for the original effect policy flag.
        cfg["oacp_strength_policy"] = "effect_adaptive"
    cfg["oacp_load_strength"] = [
        float(os.environ.get("OACP_LOAD_STRENGTH_MIN", cfg["oacp_load_strength_min"])),
        float(os.environ.get("OACP_LOAD_STRENGTH_MAX", cfg["oacp_load_strength_max"])),
    ]
    cfg["oacp_protection_policy"] = os.environ.get(
        "OACP_PROTECTION_POLICY", cfg["oacp_protection_policy"]
    ).lower()
    cfg["oacp_size_expand"] = [
        float(os.environ.get("OACP_SIZE_EXPAND_MIN", cfg["oacp_size_expand_min"])),
        float(os.environ.get("OACP_SIZE_EXPAND_MAX", cfg["oacp_size_expand_max"])),
    ]
    cfg["oacp_size_expand_smax"] = float(os.environ.get(
        "OACP_SIZE_EXPAND_SMAX", cfg["oacp_size_expand_smax"]
    ))
    cfg["oacp_curriculum_warmup_fraction"] = float(os.environ.get(
        "OACP_CURRICULUM_WARMUP_FRACTION", cfg["oacp_curriculum_warmup_fraction"]
    ))
    cfg["oacp_curriculum_cooldown_fraction"] = float(os.environ.get(
        "OACP_CURRICULUM_COOLDOWN_FRACTION", cfg["oacp_curriculum_cooldown_fraction"]
    ))
    cfg["oacp_curriculum_strength"] = [
        float(os.environ.get("OACP_CURRICULUM_STRENGTH_MIN", cfg["oacp_curriculum_strength"][0])),
        float(os.environ.get("OACP_CURRICULUM_STRENGTH_MAX", cfg["oacp_curriculum_strength"][1])),
    ]
    cfg["oacp_hardness_warmup_epochs"] = int(os.environ.get(
        "OACP_HARDNESS_WARMUP_EPOCHS", cfg["oacp_hardness_warmup_epochs"]
    ))
    cfg["oacp_hardness_beta"] = float(os.environ.get(
        "OACP_HARDNESS_BETA", cfg["oacp_hardness_beta"]
    ))
    cfg["oacp_resolution_scale"] = [
        float(os.environ.get("OACP_SCALE_MIN", profile_defaults["scale"][0])),
        float(os.environ.get("OACP_SCALE_MAX", profile_defaults["scale"][1])),
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
    cfg["oacp_spatial_load_min"] = float(os.environ.get(
        "OACP_SPATIAL_LOAD_MIN", cfg["oacp_spatial_load_min"]
    ))
    cfg["oacp_spatial_load_max"] = float(os.environ.get(
        "OACP_SPATIAL_LOAD_MAX", cfg["oacp_spatial_load_max"]
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
    cfg["oacp_variant"] = os.environ.get("OACP_VARIANT", profile_defaults["variant"]).lower()
    cfg["oacp_placement"] = os.environ.get("OACP_PLACEMENT", profile_defaults["placement"]).lower()
    cfg["mode"] = os.environ.get("YOLO_CONTEXT_AUG", "none").lower()
    cfg["legacy_double_oacp"] = os.environ.get("YOLO_LEGACY_DOUBLE_OACP", "0").lower() in {
        "1", "true", "yes", "on"
    }
    cfg["lea_stats_path"] = os.environ.get("LEA_STATS_PATH", "")
    cfg["sweep_label"] = os.environ.get("OACP_SWEEP_LABEL", "")
    valid_strength = {"fixed", "effect_adaptive", "load_adaptive", "curriculum", "hardness_adaptive"}
    if cfg["oacp_strength_policy"] not in valid_strength:
        raise ValueError(f"unknown OACP_STRENGTH_POLICY: {cfg['oacp_strength_policy']}")
    if cfg["oacp_effect_policy"] not in {"fixed", "adaptive"}:
        raise ValueError(f"unknown OACP_EFFECT_POLICY: {cfg['oacp_effect_policy']}")
    if cfg["oacp_protection_policy"] not in {"fixed", "size_adaptive"}:
        raise ValueError(f"unknown OACP_PROTECTION_POLICY: {cfg['oacp_protection_policy']}")
    if explicit_strength_policy and cfg["oacp_effect_policy"] == "adaptive" and cfg["oacp_strength_policy"] != "effect_adaptive":
        raise ValueError("OACP_EFFECT_POLICY=adaptive conflicts with the selected OACP_STRENGTH_POLICY")
    if cfg["oacp_strength_policy"] == "effect_adaptive" and cfg["oacp_effect_policy"] == "fixed":
        raise ValueError("OACP_STRENGTH_POLICY=effect_adaptive requires OACP_EFFECT_POLICY=adaptive")
    if cfg["oacp_strength_policy"] == "effect_adaptive" and cfg["oacp_target_effect"] <= 0.0:
        raise ValueError("OACP_TARGET_EFFECT must be > 0 for effect_adaptive strength")
    if cfg["oacp_protection_policy"] == "size_adaptive" and cfg["oacp_variant"] in {"density", "spacing_adaptive"}:
        raise ValueError("size_adaptive protection cannot be combined with density or spacing_adaptive")
    if profile == "r2":
        if cfg["oacp_variant"] != "current" or cfg["oacp_placement"] != "pre_transform":
            raise ValueError("OACP_PROFILE=r2 requires current variant and pre_transform placement")
        if cfg["oacp_probability_policy"] != "fixed" or not np.isclose(cfg["oacp_probability"], 0.40):
            raise ValueError("OACP_PROFILE=r2 requires fixed p=0.40")
        if not np.allclose(cfg["oacp_resolution_scale"], [0.80, 0.95]):
            raise ValueError("OACP_PROFILE=r2 requires resolution scale 0.80-0.95")
        if not np.allclose(cfg["oacp_strength"], [0.10, 0.25]):
            raise ValueError("OACP_PROFILE=r2 requires base strength range 0.10-0.25")
        if not np.isclose(cfg["protected_expand"], 3.0):
            raise ValueError("OACP_PROFILE=r2 requires protected_expand=3.0")
    if not 0.0 <= cfg["oacp_curriculum_warmup_fraction"] < 1.0:
        raise ValueError("OACP_CURRICULUM_WARMUP_FRACTION must be in [0, 1)")
    if not 0.0 <= cfg["oacp_curriculum_cooldown_fraction"] < 1.0:
        raise ValueError("OACP_CURRICULUM_COOLDOWN_FRACTION must be in [0, 1)")
    if cfg["oacp_curriculum_warmup_fraction"] + cfg["oacp_curriculum_cooldown_fraction"] >= 1.0:
        raise ValueError("curriculum warmup and cooldown fractions must sum to < 1")
    return cfg


def _oacp_config() -> dict[str, Any]:
    cfg = augmentation_config()
    return {
        "p": cfg["oacp_probability"],
        "protected_expand": cfg["protected_expand"],
        "strength": cfg["oacp_strength"],
        "resolution_scale": cfg["oacp_resolution_scale"],
        "budget": cfg["oacp_budget"],
        "oacp_load_strength": [
            float(cfg["oacp_load_strength_min"]),
            float(cfg["oacp_load_strength_max"]),
        ],
        "oacp_load_saturation_count": cfg["oacp_load_saturation_count"],
        "oacp_curriculum_strength": cfg["oacp_curriculum_strength"],
        "oacp_curriculum_warmup_fraction": cfg["oacp_curriculum_warmup_fraction"],
        "oacp_curriculum_cooldown_fraction": cfg["oacp_curriculum_cooldown_fraction"],
    }


def _load_adaptive_probability(
    num_eligible: int,
    p_min: float,
    p_max: float,
    saturation_count: int,
) -> tuple[float, float]:
    """Map eligible-object load to application probability only."""
    if num_eligible <= 0:
        return 0.0, 0.0
    load = (num_eligible - 1) / max(int(saturation_count) - 1, 1)
    load = float(np.clip(load, 0.0, 1.0))
    probability = float(p_max - load * (p_max - p_min))
    return load, float(np.clip(probability, 0.0, 1.0))


def _effect_adaptive_strength(
    raw_effect: float,
    target_effect: float,
    strength_min: float,
    strength_max: float,
    eps: float = 1e-8,
) -> float:
    """Choose blend strength to target a measured raw pixel effect."""
    strength = float(target_effect) / max(float(raw_effect), eps)
    return float(np.clip(strength, strength_min, strength_max))


def _load_adaptive_strength_range(
    num_eligible: int,
    base_range: tuple[float, float] | list[float],
    dense_range: tuple[float, float] | list[float],
    saturation_count: int,
) -> tuple[float, float, float]:
    """Interpolate severity range from sparse R2 to a milder dense range."""
    if num_eligible <= 1:
        load = 0.0
    else:
        load = (num_eligible - 1) / max(int(saturation_count) - 1, 1)
    load = float(np.clip(load, 0.0, 1.0))
    low = float(base_range[0] + load * (dense_range[0] - base_range[0]))
    high = float(base_range[1] + load * (dense_range[1] - base_range[1]))
    return load, min(low, high), max(low, high)


def _size_adaptive_expands(
    boxes: np.ndarray,
    *,
    expand_min: float,
    expand_max: float,
    size_smax: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Protect smaller eligible objects with a larger local context region."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    sizes = np.sqrt(
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    ) if len(boxes) else np.empty(0, dtype=np.float32)
    t = np.clip(sizes / max(float(size_smax), 1e-8), 0.0, 1.0)
    expands = float(expand_max) - t * (float(expand_max) - float(expand_min))
    records = [
        {"object_size": float(size), "expand": float(expand), "size_fraction": float(frac)}
        for size, expand, frac in zip(sizes, expands, t, strict=False)
    ]
    return expands.astype(np.float32), records


def _curriculum_strength_range(
    epoch: int,
    total_epochs: int,
    r2_range: tuple[float, float] | list[float],
    mild_range: tuple[float, float] | list[float],
    warmup_fraction: float,
    cooldown_fraction: float,
) -> tuple[str, float, float]:
    """Piecewise mild -> R2 -> mild strength schedule."""
    fraction = float(epoch) / max(int(total_epochs) - 1, 1)
    if fraction < warmup_fraction:
        phase = "warmup"
        current = mild_range
    elif fraction >= 1.0 - cooldown_fraction:
        phase = "cooldown"
        current = mild_range
    else:
        phase = "r2"
        current = r2_range
    return phase, float(current[0]), float(current[1])


def _hardness_strength_range(
    hardness: float,
    r2_range: tuple[float, float] | list[float],
    mild_range: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Map easy samples to R2 and hard samples to a milder severity range."""
    hardness = float(np.clip(hardness, 0.0, 1.0))
    low = float(r2_range[0] + hardness * (mild_range[0] - r2_range[0]))
    high = float(r2_range[1] + hardness * (mild_range[1] - r2_range[1]))
    return min(low, high), max(low, high)


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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Choose per-object context retention from normalized nearest spacing."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    eligible_indices = np.asarray(eligible_indices, dtype=np.int64).reshape(-1)
    records: list[dict[str, Any]] = []
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
            isolated = False
        else:
            # An isolated single object is explicitly treated as far away.
            nearest_gap = float("inf")
            normalized_spacing = float("inf")
            isolated = True
        retention = 1.0 - float(np.clip(
            (normalized_spacing - near_spacing) / denominator, 0.0, 1.0
        ))
        expand = float(expand_min + retention * (expand_max - expand_min))
        expands[out_index] = expand
        records.append({
            "object_size": scale,
            "nearest_gap": None if isolated else nearest_gap,
            "normalized_spacing": None if isolated else normalized_spacing,
            "isolated": isolated,
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


def _spatial_load_target_mass(
    protected_ratio: float,
    load_min: float,
    load_max: float,
    mass_min: float,
    mass_max: float,
) -> tuple[float, float]:
    """Map normalized protected-area load to a decreasing image-mass target."""
    denominator = max(float(load_max) - float(load_min), 1e-8)
    load = float(np.clip((float(protected_ratio) - float(load_min)) / denominator, 0.0, 1.0))
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
        cfg = augmentation_config()
        if cfg["oacp_protection_policy"] == "size_adaptive" and len(tiny):
            expands, _ = _size_adaptive_expands(
                tiny,
                expand_min=cfg["oacp_size_expand"][0],
                expand_max=cfg["oacp_size_expand"][1],
                size_smax=cfg["oacp_size_expand_smax"],
            )
            protected = _mask_from_boxes(
                boxes, h, w, AUG_CONFIG["safety_expand"]
            ).astype(bool)
            protected |= _mask_from_boxes_per_expand(tiny, expands, h, w).astype(bool)
            expand = float(np.mean(expands))
        else:
            # Preserve the historical OACP control exactly: tiny objects get
            # the configured context expansion, while every GT receives safety
            # cover.
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
    if variant not in {"current", "budget", "density", "mass_adaptive", "load_adaptive", "spatial_load_adaptive", "spacing_adaptive"}:
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
    probability_policy = cfg["oacp_probability_policy"]
    if probability_policy == "load_adaptive":
        probability_load, probability_effective = _load_adaptive_probability(
            num_eligible,
            cfg["oacp_probability_min"],
            cfg["oacp_probability_max"],
            cfg["oacp_load_saturation_count"],
        )
    elif probability_policy == "fixed":
        probability_load, probability_effective = 0.0, float(cfg["oacp_probability"])
    else:
        raise ValueError(f"unknown OACP_PROB_POLICY: {probability_policy}")
    object_load = 0.0
    spatial_load = 0.0
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
    elif variant == "spatial_load_adaptive":
        spatial_load, target_mass = _spatial_load_target_mass(
            protected_ratio,
            cfg["oacp_spatial_load_min"],
            cfg["oacp_spatial_load_max"],
            cfg["oacp_mass_target"][0],
            cfg["oacp_mass_target"][1],
        )
    if variant in {"mass_adaptive", "load_adaptive", "spatial_load_adaptive"}:
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
    spacing_records: list[dict[str, Any]] = []
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
    size_records: list[dict[str, Any]] = []
    if cfg["oacp_protection_policy"] == "size_adaptive" and len(tiny):
        _, size_records = _size_adaptive_expands(
            tiny,
            expand_min=cfg["oacp_size_expand"][0],
            expand_max=cfg["oacp_size_expand"][1],
            size_smax=cfg["oacp_size_expand_smax"],
        )
    normalized_spacings = [
        r["normalized_spacing"] for r in spacing_records
        if r["normalized_spacing"] is not None
    ]
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
        "adaptive_budget": budget if variant in {"mass_adaptive", "load_adaptive", "spatial_load_adaptive"} else 0.0,
        "budget_clipped": budget_clipped,
        "num_eligible": num_eligible,
        "oacp_probability_policy": probability_policy,
        "oacp_probability_effective": probability_effective,
        "oacp_probability_min": float(cfg["oacp_probability_min"]),
        "oacp_probability_max": float(cfg["oacp_probability_max"]),
        "oacp_probability_load": probability_load,
        "oacp_load_saturation_count": int(cfg["oacp_load_saturation_count"]),
        "oacp_applied": False,
        "oacp_effect_policy": cfg["oacp_effect_policy"],
        "oacp_strength_policy": cfg["oacp_strength_policy"],
        "oacp_target_effect": float(cfg["oacp_target_effect"]),
        "oacp_protection_policy": cfg["oacp_protection_policy"],
        "raw_effect": 0.0,
        "effective_strength": 0.0,
        "actual_effect": 0.0,
        "object_load": object_load,
        "spatial_load": spatial_load,
        "spatial_load_min": float(cfg["oacp_spatial_load_min"]),
        "spatial_load_max": float(cfg["oacp_spatial_load_max"]),
        "mean_nearest_distance_norm": float(np.mean(normalized_spacings)) if normalized_spacings else None,
        "median_nearest_distance_norm": float(np.median(normalized_spacings)) if normalized_spacings else None,
        "mean_adaptive_expand": float(np.mean(adaptive_expands)) if adaptive_expands else 0.0,
        "min_adaptive_expand": float(np.min(adaptive_expands)) if adaptive_expands else 0.0,
        "max_adaptive_expand": float(np.max(adaptive_expands)) if adaptive_expands else 0.0,
        "spacing_objects": spacing_records,
        "size_objects": size_records,
        "gt_area_ratio": float(gt_mask.mean()),
        "perturb_gt_overlap_ratio": float((perturb.astype(bool) & gt_mask).mean()),
        "eligible": eligible,
        "would_apply": would_apply,
        "skip_reason": skip_reason,
        "density_occupancy": occupancy,
        "density_target_protected_ratio": density_target,
        "mean_object_size": float(np.mean(np.sqrt(np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])))) if len(boxes) else 0.0,
        "protected_expand": expand,
        "mean_size_adaptive_expand": float(np.mean([r["expand"] for r in size_records])) if size_records else 0.0,
        "min_size_adaptive_expand": float(np.min([r["expand"] for r in size_records])) if size_records else 0.0,
        "max_size_adaptive_expand": float(np.max([r["expand"] for r in size_records])) if size_records else 0.0,
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


def calibrate_effect_target(path: str, quantile: float = 0.50) -> dict[str, float | int]:
    """Summarize applied fixed-R2 effect diagnostics for adaptive calibration.

    The input should be JSONL generated with ``OACP_EFFECT_POLICY=fixed`` and
    the R2 mask/scale/protection settings.  Only samples that actually passed
    the probability/geometry gates contribute to the target distribution.
    """
    values: list[float] = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("oacp_applied"):
                continue
            value = float(record.get("actual_effect", 0.0))
            if np.isfinite(value) and value >= 0.0:
                values.append(value)
    if not values:
        raise ValueError(f"no applied actual_effect records found in {path}")
    quantile = float(np.clip(quantile, 0.0, 1.0))
    result = {
        "count": int(len(values)),
        "actual_effect_q40": float(np.quantile(values, 0.40)),
        "actual_effect_median": float(np.quantile(values, 0.50)),
        "actual_effect_q60": float(np.quantile(values, 0.60)),
        "target_effect": float(np.quantile(values, quantile)),
        "quantile": quantile,
    }
    return result


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

    def __init__(self, p: float = 0.20, shared_state: OACPSharedState | None = None) -> None:
        self.p = float(p)
        self.shared_state = shared_state

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        cfg = _oacp_config()
        img = labels.get("img")
        if img is None or img.ndim != 3:
            return labels
        h, w = img.shape[:2]
        boxes = _boxes(labels, h, w)
        full_cfg = augmentation_config()
        variant = full_cfg["oacp_variant"]
        valid_variants = {
            "current", "budget", "density", "mass_adaptive", "load_adaptive", "spatial_load_adaptive", "spacing_adaptive"
        }
        if variant not in valid_variants:
            raise ValueError(f"unknown OACP_VARIANT: {variant}")
        target_mass = None
        if variant == "mass_adaptive":
            target_mass = random.uniform(*augmentation_config()["oacp_mass_target"])
            budget = None
        elif variant == "load_adaptive":
            budget = None
        elif variant == "spatial_load_adaptive":
            budget = None
        else:
            budget = random.uniform(*cfg["budget"]) if variant != "current" else None
        protected, tiny, _, _, _ = _protection_for_variant(boxes, h, w, variant)
        diagnostics = oacp_diagnostics(
            (h, w), boxes, variant=variant, budget=budget, target_mass=target_mass
        )
        # A non-default constructor probability is an explicit fixed override,
        # retained for tests and programmatic callers. Training uses the
        # independent policy configured through the environment.
        probability = (
            self.p
            if self.p != 0.20
            else diagnostics["oacp_probability_effective"]
        )
        diagnostics["oacp_probability_effective"] = float(probability)
        if not diagnostics["num_eligible"]:
            diagnostics["skip_reason"] = "no_eligible_tiny"
            _record_oacp_diagnostics(labels, diagnostics)
            return labels
        if random.random() >= probability:
            diagnostics["skip_reason"] = "probability_gate"
            _record_oacp_diagnostics(labels, diagnostics)
            return labels
        if variant in {"mass_adaptive", "load_adaptive", "spatial_load_adaptive"}:
            budget = float(diagnostics["adaptive_budget"])
        if not diagnostics["would_apply"]:
            _record_oacp_diagnostics(labels, diagnostics)
            return labels
        mask = _far_mask(protected, h, w)
        if variant in {
            "budget", "density", "mass_adaptive", "load_adaptive", "spatial_load_adaptive", "spacing_adaptive"
        }:
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
        diagnostics["oacp_applied"] = True
        degraded = _resize_degrade(img, random.uniform(*cfg["resolution_scale"]))
        mask_pixels = mask > 0
        delta = np.mean(np.abs(
            img.astype(np.float32) - degraded.astype(np.float32)
        ), axis=2)
        raw_effect = float(np.mean(mask[mask_pixels] * delta[mask_pixels])) if mask_pixels.any() else 0.0
        attenuation = 1.0 - float(protected.mean()) if variant == "current" else 1.0
        policy = full_cfg["oacp_strength_policy"]
        strength_min, strength_max = cfg["strength"]
        strength_load = float(diagnostics.get("object_load", 0.0))
        curriculum_phase = "none"
        hardness = 0.5
        if policy == "fixed":
            base_strength = random.uniform(strength_min, strength_max)
        elif policy == "effect_adaptive":
            # Solve for the pre-attenuation strength.  The actual blend uses
            # base_strength * attenuation, so the attenuation is included in
            # the denominator rather than applied a second time afterwards.
            base_strength = _effect_adaptive_strength(
                raw_effect * attenuation,
                full_cfg["oacp_target_effect"],
                strength_min,
                strength_max,
            )
        elif policy == "load_adaptive":
            strength_load, strength_min, strength_max = _load_adaptive_strength_range(
                diagnostics["num_eligible"],
                cfg["strength"],
                cfg["oacp_load_strength"],
                cfg["oacp_load_saturation_count"],
            )
            base_strength = random.uniform(strength_min, strength_max)
        elif policy == "curriculum":
            total_epochs = int(self.shared_state.total_epochs.value) if self.shared_state else 0
            total_epochs = total_epochs or int(os.environ.get("OACP_TOTAL_EPOCHS", "100"))
            epoch = int(self.shared_state.epoch.value) if self.shared_state else 0
            curriculum_phase, strength_min, strength_max = _curriculum_strength_range(
                epoch,
                total_epochs,
                cfg["strength"],
                cfg["oacp_curriculum_strength"],
                cfg["oacp_curriculum_warmup_fraction"],
                cfg["oacp_curriculum_cooldown_fraction"],
            )
            base_strength = random.uniform(strength_min, strength_max)
        elif policy == "hardness_adaptive":
            epoch = int(self.shared_state.epoch.value) if self.shared_state else 0
            if epoch < full_cfg["oacp_hardness_warmup_epochs"]:
                strength_min, strength_max = cfg["strength"]
            else:
                index = labels.get("dataset_idx", -1)
                hardness = self.shared_state.read_hardness(index) if self.shared_state else 0.5
                strength_min, strength_max = _hardness_strength_range(
                    hardness, cfg["strength"], cfg["oacp_curriculum_strength"]
                )
            base_strength = random.uniform(strength_min, strength_max)
        else:  # guarded by augmentation_config, retained for defensive callers
            raise ValueError(f"unknown OACP_STRENGTH_POLICY: {policy}")
        strength = float(base_strength * attenuation)
        out = img.astype(np.float32) * (1 - strength * mask[..., None]) + degraded.astype(np.float32) * (strength * mask[..., None])
        output_img = np.clip(out, 0, 255).astype(img.dtype)
        labels["img"] = output_img
        diagnostics["raw_effect"] = raw_effect
        diagnostics["effective_strength"] = float(strength)
        diagnostics["base_strength"] = float(base_strength)
        diagnostics["strength_min"] = float(strength_min)
        diagnostics["strength_max"] = float(strength_max)
        diagnostics["strength_load"] = float(strength_load)
        diagnostics["curriculum_phase"] = curriculum_phase
        diagnostics["epoch"] = int(self.shared_state.epoch.value) if self.shared_state else 0
        diagnostics["hardness"] = float(hardness)
        diagnostics["actual_effect"] = float(np.mean(
            np.abs(output_img[mask_pixels].astype(np.float32) - img[mask_pixels].astype(np.float32))
        )) if mask_pixels.any() else 0.0
        _record_oacp_diagnostics(labels, diagnostics)
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
    if mode == "oacp":
        state = getattr(dataset, "oacp_shared_state", None)
        if state is None:
            state = OACPSharedState(
                len(dataset),
                beta=augmentation_config()["oacp_hardness_beta"],
            )
            dataset.oacp_shared_state = state
        return [OACP(shared_state=state)]
    if mode == "cea": return [CEA(dataset)]
    if mode == "lea": return [LEA()]
    return []
