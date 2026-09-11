from __future__ import annotations

import random

import numpy as np

from project_ultralytics.mosaic_policy import effective_count, mosaic_crops, simulate_visible_boxes


def metadata(boxes, shape=(100, 100)):
    return [{"shape": shape, "bboxes": np.asarray(boxes, dtype=np.float32)}]


def test_mosaic_crops_are_inside_source_images():
    crops = mosaic_crops([(100, 100)] * 4, 100, 100, 100)
    assert crops == [(0, 0, 100, 100)] * 4


def test_full_box_visibility_is_one():
    values = simulate_visible_boxes(metadata([[0.5, 0.5, 0.2, 0.2]] * 4), [0, 0, 0, 0], 100, 100, 100)
    assert np.allclose(values, 1.0)


def test_removed_box_visibility_is_zero():
    values = simulate_visible_boxes(metadata([[0.1, 0.1, 0.2, 0.2]] * 4), [0, 0, 0, 0], 100, 0, 0)
    assert np.min(values) == 0.0


def test_effective_count_softly_counts_partial_boxes():
    assert effective_count(np.asarray([1.0, 0.7, 0.35]), threshold=0.7) == 2.5


def test_policy_geometry_is_seed_reproducible():
    random.seed(7)
    first = [random.uniform(0, 10) for _ in range(4)]
    random.seed(7)
    second = [random.uniform(0, 10) for _ in range(4)]
    assert first == second
