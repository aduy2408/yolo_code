"""Focused behavioral tests for OACP, CEA and LEA."""
from __future__ import annotations

import numpy as np
import pytest
from ultralytics.utils.instance import Instances

from project_ultralytics.context_augment import (
    CEA, LEA, OACP, _protection, _protection_for_variant, oacp_diagnostics,
)


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
    assert budget["budget_fraction_of_valid_bg"] == 0.4
    assert budget["target_perturbed_area_ratio_image"] == pytest.approx(0.4 * budget["perturbable_area_ratio"])
    assert budget["perturb_gt_overlap_ratio"] == 0.0


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


def test_density_variant_controls_union_protection_and_logs_budget_fields():
    boxes = np.asarray([
        [x, y, x + 8, y + 8]
        for y in range(8, 120, 12)
        for x in range(8, 120, 12)
    ], dtype=np.float32)
    stats = oacp_diagnostics((128, 128), boxes, "density", budget=0.4)
    assert stats["density_occupancy"] > 0.0
    assert stats["density_target_protected_ratio"] > 0.0
    assert stats["budget_fraction_of_valid_bg"] == 0.4
    if stats["would_apply"]:
        assert stats["target_perturbed_area_ratio_image"] == pytest.approx(0.4 * stats["perturbable_area_ratio"])
    else:
        assert stats["target_perturbed_area_ratio_image"] == 0.0
    assert stats["perturb_gt_overlap_ratio"] == 0.0


def test_budget_transform_runs_and_writes_sample_record(monkeypatch, tmp_path):
    monkeypatch.setenv("OACP_VARIANT", "budget")
    monkeypatch.setenv("OACP_P", "1.0")
    log = tmp_path / "oacp.jsonl"
    monkeypatch.setenv("OACP_DIAGNOSTICS_PATH", str(log))
    img = np.random.default_rng(9).integers(20, 100, (64, 64, 3), dtype=np.uint8)
    labels = {
        "img": img,
        "bboxes": np.asarray([[0.5, 0.5, 0.05, 0.05]], dtype=np.float32),
        "im_file": "sample.jpg",
    }
    out = OACP(p=1.0)(labels)
    assert out["img"].shape == img.shape
    record = __import__("json").loads(log.read_text().strip())
    assert record["budget_fraction_of_valid_bg"] > 0.0
    assert record["target_perturbed_area_ratio_image"] > 0.0
    assert record["perturb_gt_overlap_ratio"] == 0.0


def test_current_matches_historical_tiny_plus_safety_protection():
    boxes = np.asarray([[8, 8, 12, 12], [64, 64, 56, 56]], dtype=np.float32)
    old_mask, old_tiny = _protection(boxes, 128, 128)
    new_mask, new_tiny, _, _, _ = _protection_for_variant(boxes, 128, 128, "current")
    assert np.array_equal(new_mask, old_mask)
    assert np.array_equal(new_tiny, old_tiny)


def test_diagnostics_reports_training_skip_as_zero_perturbation():
    boxes = np.asarray([
        [x, y, x + 24, y + 24]
        for y in range(0, 128, 24)
        for x in range(0, 128, 24)
    ], dtype=np.float32)
    stats = oacp_diagnostics((128, 128), boxes, "budget", budget=0.4)
    assert stats["would_apply"] is False
    assert stats["skip_reason"] == "protected_coverage_gt_0.55"
    assert stats["actual_perturbed_area_ratio_image"] == 0.0


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
