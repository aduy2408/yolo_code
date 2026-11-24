from __future__ import annotations

import numpy as np

from project_ultralytics.mosaic_policy import effective_count, load_context_cache, mosaic_crops, simulate_visible_boxes


def metadata(boxes, shape=(100, 100)):
    return [{"shape": shape, "bboxes": np.asarray(boxes, dtype=np.float32)}]


def test_mosaic_crops_are_inside_source_images():
    crops = mosaic_crops([(100, 100)] * 4, 100, 100, 100)
    assert crops == [(0, 0, 100, 100)] * 4


def test_mosaic_crops_use_correct_source_corner_for_each_quadrant():
    crops = mosaic_crops([(100, 120)] * 4, 100, 80, 70)
    assert crops == [(40, 30, 120, 100), (0, 30, 120, 100), (40, 0, 120, 100), (0, 0, 120, 100)]


def test_full_box_visibility_is_one():
    values = simulate_visible_boxes(metadata([[0.5, 0.5, 0.2, 0.2]] * 4), [0, 0, 0, 0], 100, 100, 100)
    assert np.allclose(values, 1.0)


def test_removed_box_visibility_is_zero():
    values = simulate_visible_boxes(metadata([[0.1, 0.1, 0.2, 0.2]] * 4), [0, 0, 0, 0], 100, 0, 0)
    assert np.min(values) == 0.0


def test_effective_count_softly_counts_partial_boxes():
    assert effective_count(np.asarray([1.0, 0.7, 0.35]), threshold=0.7) == 2.5


def test_context_cache_paths_are_canonicalized(tmp_path):
    image = tmp_path / "image.jpg"
    cache = tmp_path / "context.npz"
    np.savez(cache, im_file=np.asarray([str(image)]), descriptor=np.asarray([[1.0, 2.0]]))
    loaded = load_context_cache(cache)
    assert loaded["im_file"].tolist() == [str(image.resolve())]


def test_missing_context_cache_fails_loudly(tmp_path):
    missing = tmp_path / "missing.npz"
    try:
        load_context_cache(missing)
    except FileNotFoundError as error:
        assert "does not exist" in str(error)
    else:
        raise AssertionError("missing context cache must fail loudly")
