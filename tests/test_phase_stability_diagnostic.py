from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "misc" / "phase_stability_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("phase_stability_diagnostic", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_box_iou_handles_empty_and_exact_boxes():
    assert MODULE.box_iou(np.empty((0, 4)), np.ones((2, 4))).shape == (0, 2)
    result = MODULE.box_iou(np.array([[0, 0, 4, 2]]), np.array([[0, 0, 4, 2], [2, 0, 6, 2]]))
    np.testing.assert_allclose(result, [[1.0, 1.0 / 3.0]])


def test_read_yolo_labels_converts_normalized_coordinates(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("0 0.5 0.25 0.5 0.5\n", encoding="utf-8")
    np.testing.assert_allclose(MODULE.read_yolo_labels(path, 100, 80), [[25, 0, 75, 40]])


def test_translate_image_expands_canvas_and_preserves_pixels():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    shifted = MODULE.translate_image(image, 2, 1)
    assert shifted.shape == (3, 4)
    np.testing.assert_array_equal(shifted, [[0, 0, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]])


def test_object_rows_compensate_prediction_translation():
    gt = np.array([[10, 10, 18, 18]], dtype=float)
    predictions = [
        {"boxes": np.array([[10, 10, 18, 18]], dtype=float), "scores": np.array([0.8])},
        {"boxes": np.array([[12, 11, 20, 19]], dtype=float), "scores": np.array([0.6])},
    ]
    rows = MODULE._object_rows(gt, predictions, [(0, 0), (2, 1)])
    assert rows[0]["oracle_iou_std"] == 0.0
    assert rows[0]["top_score_iou_std"] == 0.0
    assert rows[0]["top_score_std"] > 0.0
    assert rows[0]["size_bucket"] == "tiny_8_12"


def test_summarize_ignores_non_finite_values():
    summary = MODULE.summarize([1.0, float("nan"), 3.0])
    assert summary["n"] == 2
    assert summary["mean"] == 2.0


def test_label_path_for_image_replaces_images_component():
    image = Path("dataset/images/test/example.png")
    assert MODULE.label_path_for_image(image) == Path("dataset/labels/test/example.txt")
