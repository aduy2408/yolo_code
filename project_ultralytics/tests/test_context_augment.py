"""Focused behavioral tests for OACP, CEA and LEA."""
from __future__ import annotations

import numpy as np
from ultralytics.utils.instance import Instances

from project_ultralytics.context_augment import CEA, LEA, OACP, oacp_diagnostics


def _labels(img, boxes):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return {"img": img.copy(), "instances": Instances(b, None, None, bbox_format="xywh", normalized=True)}


def test_oacp_skips_empty_and_protects_tiny_box():
    rng = np.random.default_rng(11)
    img = rng.integers(20, 100, (128, 128, 3), dtype=np.uint8)
    img[60:70, 60:70] = 220
    empty = OACP(p=1.0)(_labels(img, []))
    assert np.array_equal(empty["img"], img)
    out = OACP(p=1.0)(_labels(img, [[65 / 128, 65 / 128, 10 / 128, 10 / 128]]))
    assert np.array_equal(out["img"][60:70, 60:70], img[60:70, 60:70])
    assert np.abs(out["img"].astype(np.int16) - img).sum() > 0


def test_budget_limits_perturbation_to_valid_background():
    boxes = np.asarray([[64 - 5, 64 - 5, 64 + 5, 64 + 5]], dtype=np.float32)
    current = oacp_diagnostics((128, 128), boxes, "current")
    budget = oacp_diagnostics((128, 128), boxes, "budget", budget=0.4)
    assert budget["actual_perturbed_area_ratio"] <= budget["perturbable_area_ratio"]
    assert budget["actual_perturbed_area_ratio"] < current["actual_perturbed_area_ratio"]
    assert budget["protected_area_ratio"] == current["protected_area_ratio"]


def test_density_variant_reduces_expansion_for_crowded_scene():
    sparse = np.asarray([[64, 64, 68, 68]], dtype=np.float32)
    crowded = np.asarray([
        [x, y, x + 8, y + 8]
        for y in range(8, 120, 12)
        for x in range(8, 120, 12)
    ], dtype=np.float32)
    sparse_stats = oacp_diagnostics((128, 128), sparse, "density")
    crowded_stats = oacp_diagnostics((128, 128), crowded, "density")
    assert crowded_stats["protected_expand"] < sparse_stats["protected_expand"]
    assert crowded_stats["perturbable_area_ratio"] >= 0.0


def test_lea_changes_low_frequency_but_keeps_shape():
    rng = np.random.default_rng(7)
    img = rng.integers(20, 80, (128, 128, 3), dtype=np.uint8)
    img[60:70, 60:70] = 220
    out = LEA(p=1.0)(_labels(img, [[65 / 128, 65 / 128, 10 / 128, 10 / 128]]))
    assert out["img"].shape == img.shape
    assert np.abs(out["img"].astype(np.int16) - img).sum() > 0


def test_cea_keeps_source_protected_region():
    class Dataset:
        labels = [{"bboxes": np.empty((0, 4), dtype=np.float32)}]

        def __len__(self):
            return 1

    img = np.full((128, 128, 3), 80, dtype=np.uint8)
    img[60:70, 60:70] = 220
    # Dataset length one must be a no-op, proving the guard does not invent a donor.
    out = CEA(Dataset(), p=1.0)(_labels(img, [[65 / 128, 65 / 128, 10 / 128, 10 / 128]]))
    assert np.array_equal(out["img"], img)
