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
