"""Matched Copy-Paste screening protocol shared by LEVIR and TinyPerson runners."""
from __future__ import annotations

COMMON_CP = {
    "mosaic": 0.0,
    "close_mosaic": 0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste_p": 0.5,
    "copy_paste_scale": 1.0,
    "copy_paste_padding": 0.0,
    "copy_paste_blend": "hard",
    "copy_paste_placement": "random",
    "copy_paste_max_overlap": 0.0,
    "copy_paste_max_trials": 30,
    "copy_paste_allow_empty_target": True,
    "copy_paste_allow_same_source": True,
    "copy_paste_mode": "single",
    "negcp": 0.30,
    "negcp_num": 1,
    "negcp_scale": 1.0,
    "negcp_max_gt_ioa": 0.05,
    "negcp_same_source": False,
    "crowd_num": 1,
    "crowd_overlap_min": 0.10,
    "crowd_overlap_max": 0.30,
    "crowd_min_visibility": 0.60,
    "crowd_size_ratio_min": 0.75,
    "crowd_size_ratio_max": 1.33,
    "crowd_trials": 30,
    "scale_cp_num": 1,
    "scale_cp_target_max_size": 20.0,
    "scale_cp_source_min_ratio": 1.25,
    "scale_cp_factor_min": 0.50,
    "scale_cp_factor_max": 0.90,
    "scale_cp_max_overlap": 0.0,
    "scale_cp_trials": 30,
}

VARIANTS = {
    "cp0": {"copy_paste_enabled": False, "copy_paste_unit": "single", "copy_paste_copies": 1},
    "cp1_single1": {"copy_paste_enabled": True, "copy_paste_unit": "single", "copy_paste_copies": 1},
    "cp2_single2": {"copy_paste_enabled": True, "copy_paste_unit": "single", "copy_paste_copies": 2},
    "cp3_cluster1": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "cluster",
        "copy_paste_copies": 1,
        "copy_paste_cluster_expand": 3.0,
        "copy_paste_cluster_min_objects": 2,
    },
    "negcp_offline": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "copy_paste_mode": "negative",
        "copy_paste_p": 0.30,
        "negcp": 0.30,
        "negcp_num": 1,
        "negcp_scale": 1.0,
        "negcp_max_gt_ioa": 0.05,
    },
    "negative_canvas_r1": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "copy_paste_mode": "negative_canvas",
        "negative_cp_p": 0.30,
        "negative_cp_target_policy": "empirical",
        "negative_cp_donor_policy": "matched",
        "negative_cp_target_max_size": 20.0,
        "negative_cp_degradation": "none",
    },
    "negative_canvas_r2": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "copy_paste_mode": "negative_canvas",
        "negative_cp_p": 0.30,
        "negative_cp_target_policy": "deficit",
        "negative_cp_donor_policy": "matched",
        "negative_cp_target_max_size": 20.0,
        "negative_cp_degradation": "none",
    },
    "negative_canvas_r3": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "copy_paste_mode": "negative_canvas",
        "negative_cp_p": 0.30,
        "negative_cp_target_policy": "deficit",
        "negative_cp_donor_policy": "larger",
        "negative_cp_target_max_size": 20.0,
        "negative_cp_degradation": "none",
    },
    "negative_canvas_r4": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "copy_paste_mode": "negative_canvas",
        "negative_cp_p": 0.30,
        "negative_cp_target_policy": "deficit",
        "negative_cp_donor_policy": "larger",
        "negative_cp_target_max_size": 20.0,
        "negative_cp_degradation": "weak_blur",
        "negative_cp_blur_sigma": 0.5,
    },
    "crowd_mild": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "copy_paste_mode": "crowded",
        "copy_paste_p": 0.30,
        "crowd_num": 1,
        "crowd_overlap_min": 0.10,
        "crowd_overlap_max": 0.30,
        "crowd_min_visibility": 0.60,
    },
    "crowd_moderate": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "copy_paste_mode": "crowded",
        "copy_paste_p": 0.30,
        "crowd_num": 1,
        "crowd_overlap_min": 0.20,
        "crowd_overlap_max": 0.40,
        "crowd_min_visibility": 0.60,
    },
    "scale_matched": {
        "copy_paste_enabled": True,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "copy_paste_mode": "scale_matched",
        "copy_paste_p": 0.50,
        "scale_cp_num": 1,
        "scale_cp_target_max_size": 20.0,
        "scale_cp_source_min_ratio": 1.25,
        "scale_cp_factor_min": 0.50,
        "scale_cp_factor_max": 0.90,
        "scale_cp_max_overlap": 0.0,
    },
}


def variant_overrides(name: str) -> dict:
    if name not in VARIANTS:
        raise ValueError(f"unknown Copy-Paste variant: {name}")
    settings = {**COMMON_CP, **VARIANTS[name]}
    validate_settings(settings)
    return settings


def validate_settings(settings: dict) -> None:
    required = {
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste_p": 0.5,
        "copy_paste_scale": 1.0,
        "copy_paste_padding": 0.0,
        "copy_paste_blend": "hard",
        "copy_paste_placement": "random",
    }
    for key, expected in required.items():
        if key == "copy_paste_p" and settings.get("copy_paste_mode") in {
            "negative", "crowded", "scale_matched", "negative_canvas",
        }:
            continue
        if settings.get(key) != expected:
            raise ValueError(f"{key} must be explicitly {expected!r} for Copy-Paste screening")
    if settings["copy_paste_unit"] == "cluster" and settings["copy_paste_copies"] != 1:
        raise ValueError("cluster screening is fixed to copies=1")


def effective_settings(dataset: str, variant: str, seed: int, split_seed: int, **run) -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "split_seed": split_seed,
        "nms_iou": 0.5,
        "augmentation": variant_overrides(variant),
        **run,
    }


__all__ = ["COMMON_CP", "VARIANTS", "effective_settings", "validate_settings", "variant_overrides"]
