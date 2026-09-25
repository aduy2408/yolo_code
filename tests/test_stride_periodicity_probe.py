from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "misc" / "stride_periodicity_probe.py"
SPEC = importlib.util.spec_from_file_location("stride_periodicity_probe", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_periodicity_distances_match_definitions():
    d1, d4 = MODULE.periodicity_distances([0.74, 0.52, 0.47, 0.58, 0.72, 0.50, 0.46, 0.57])
    np.testing.assert_allclose(d1, np.mean([0.22, 0.05, 0.11, 0.14, 0.22, 0.04, 0.11]))
    np.testing.assert_allclose(d4, np.mean([0.02, 0.02, 0.01, 0.01]))


def test_periodicity_distances_requires_eight_phases():
    with pytest.raises(ValueError, match="exactly 8"):
        MODULE.periodicity_distances([0.1, 0.2])


def test_validity_rejects_x_side_padding_contacts():
    assert MODULE.valid_for_x_sweep(np.array([7, 0, 93, 100]), 100, 100)
    assert not MODULE.valid_for_x_sweep(np.array([6, 0, 93, 100]), 100, 100)
    assert not MODULE.valid_for_x_sweep(np.array([7, 0, 94, 100]), 100, 100)


def test_raw_rows_restore_shifted_boxes_and_compute_periodicity():
    gt = np.array([[20, 20, 30, 30]], dtype=float)
    phase_boxes = [np.array([[20 + dx, 20, 30 + dx, 30]], dtype=float) for dx in range(8)]
    rows = MODULE._object_rows(gt, phase_boxes, list(range(8)), 100, 100, 0.75)
    assert len(rows) == 1
    assert rows[0]["d1"] == 0.0
    assert rows[0]["d4"] == 0.0
    assert rows[0]["oracle_iou_std"] == 0.0
