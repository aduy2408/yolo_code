"""Focused tests for the independent LEVIR augmentation transforms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models_related" / "ultralytics"))

from ultralytics.data.augment import (  # noqa: E402
    BBoxPartialOcclusion,
    RandomViewport,
    ResolutionDegrade,
    crop_pad,
    local_mean_fill,
    viewport_boxes,
)
from ultralytics.utils.instance import Instances  # noqa: E402


def sample(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    return {"img": np.full((100, 100, 3), 80, dtype=np.uint8), "cls": np.zeros((len(boxes), 1), dtype=np.float32),
            "instances": Instances(boxes, bbox_format="xyxy", normalized=False)}


def test_crop_pad_supports_negative_origin():
    image = np.ones((3, 3, 3), dtype=np.uint8) * 7
    result = crop_pad(image, -1, -1, 3, 3, fill=114)
    assert result.shape == (3, 3, 3)
    assert result[0, 0, 0] == 114
    assert result[1:, 1:, 0].tolist() == [[7, 7], [7, 7]]


def test_viewport_partial_clipping_keeps_half_visible_box():
    boxes, keep, visibility = viewport_boxes(np.array([[10, 10, 30, 30]], dtype=np.float32), 20, 0, 100, 100)
    assert keep.tolist() == [True]
    assert np.allclose(boxes[0], [0, 10, 10, 30])
    assert np.isclose(visibility[0], 0.5)


def test_viewport_drops_disappeared_box():
    _, keep, visibility = viewport_boxes(np.array([[10, 10, 20, 20]], dtype=np.float32), 30, 0, 100, 100)
    assert keep.tolist() == [False]
    assert visibility[0] == 0


def test_viewport_identity_preserves_image_and_labels(monkeypatch):
    labels = sample([[10, 20, 30, 40]])
    original = labels["img"].copy()
    monkeypatch.setattr("ultralytics.data.augment.random.random", lambda: 0.0)
    transformed = RandomViewport(p=1.0, scale=(1.0, 1.0), translate=0.0)(labels)
    assert np.array_equal(transformed["img"], original)
    assert np.allclose(transformed["instances"].bboxes, [[10, 20, 30, 40]])


def test_viewport_transform_updates_instances_and_cls(monkeypatch):
    labels = sample([[10, 10, 30, 30], [90, 10, 100, 20]])
    monkeypatch.setattr("ultralytics.data.augment.random.random", lambda: 0.0)
    transformed = RandomViewport(p=1.0, scale=(1.0, 1.0), translate=0.0)(labels)
    assert len(transformed["instances"]) == 2
    assert transformed["cls"].shape == (2, 1)
    assert transformed["instances"].normalized is False


def test_occlusion_keeps_bbox_labels(monkeypatch):
    labels = sample([[20, 20, 60, 60]])
    labels["img"][:] = 0
    monkeypatch.setattr("ultralytics.data.augment.random.random", lambda: 0.0)
    monkeypatch.setattr("ultralytics.data.augment.random.uniform", lambda a, b: a)
    monkeypatch.setattr("ultralytics.data.augment.random.choice", lambda values: "left")
    original_box = labels["instances"].bboxes.copy()
    transformed = BBoxPartialOcclusion(p=1.0, object_prob=1.0, occ_ratio=(0.25, 0.25), fill=123)(labels)
    assert np.allclose(transformed["instances"].bboxes, original_box)
    assert np.all(transformed["img"][20:60, 20:30] == 123)


def test_resolution_degrade_keeps_shape_and_boxes(monkeypatch):
    labels = sample([[20, 20, 60, 60]])
    monkeypatch.setattr("ultralytics.data.augment.random.random", lambda: 0.0)
    monkeypatch.setattr("ultralytics.data.augment.random.uniform", lambda a, b: a)
    transformed = ResolutionDegrade(p=1.0, scale=(0.65, 0.65))(labels)
    assert transformed["img"].shape == (100, 100, 3)
    assert np.allclose(transformed["instances"].bboxes, [[20, 20, 60, 60]])


def test_local_mean_fill_uses_bbox_ring_not_whole_image():
    image = np.full((20, 20, 3), 20, dtype=np.uint8)
    image[8:12, 8:12] = 200
    assert np.allclose(local_mean_fill(image, 8, 8, 12, 12), [20, 20, 20])
