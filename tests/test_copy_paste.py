from pathlib import Path

import cv2
import numpy as np
import pytest

from project_ultralytics.copy_paste import SmallObjectCopyPaste
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances


def _labels(image, boxes=None, classes=None):
    boxes = np.asarray(boxes or [], dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(classes or [], dtype=np.float32).reshape(-1, 1)
    segments = np.zeros((len(boxes), 1000, 2), dtype=np.float32)
    return {
        "img": image,
        "instances": Instances(boxes, segments, bbox_format="xyxy", normalized=False),
        "cls": classes,
    }


def _dataset(tmp_path, source_boxes, source_classes=None):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[2:6, 3:7] = (10, 20, 30)
    image[10:14, 12:16] = (40, 50, 60)
    path = Path(tmp_path) / "source.png"
    assert cv2.imwrite(str(path), image)
    return type("Dataset", (), {
        "labels": [{
            "bboxes": np.asarray(source_boxes, dtype=np.float32),
            "cls": np.asarray(source_classes or [0] * len(source_boxes), dtype=np.float32).reshape(-1, 1),
            "bbox_format": "xyxy",
            "normalized": False,
            "shape": image.shape[:2],
        }],
        "im_files": [str(path)],
    })()


def test_single_paste_appends_box_and_pixels(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    labels = _labels(np.zeros((20, 20, 3), dtype=np.uint8))
    transform = SmallObjectCopyPaste(dataset, p=1.0, rng=__import__("random").Random(4))
    out = transform(labels)
    assert len(out["instances"]) == 1
    assert out["cls"].shape == (1, 1)
    assert np.any(out["img"] == (10, 20, 30))
    box = out["instances"].bboxes[0]
    assert 0 <= box[0] < box[2] <= 20 and 0 <= box[1] < box[3] <= 20


def test_two_copies_append_exactly_two_instances(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    labels = _labels(np.zeros((20, 20, 3), dtype=np.uint8))
    transform = SmallObjectCopyPaste(dataset, p=1.0, copies=2, rng=__import__("random").Random(1))
    out = transform(labels)
    assert len(out["instances"]) == 2
    assert out["cls"].shape == (2, 1)
    assert out["cls"].dtype == np.float32


def test_empty_target_can_receive_copy(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    labels = _labels(np.zeros((20, 20, 3), dtype=np.uint8))
    transform = SmallObjectCopyPaste(dataset, p=1.0, allow_empty_target=True, rng=__import__("random").Random(2))
    out = transform(labels)
    assert len(out["instances"]) == 1
    assert transform.diagnostics()["empty_target_count"] == 1


def test_collision_aware_rejects_full_canvas(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    labels = _labels(np.zeros((20, 20, 3), dtype=np.uint8), [[0, 0, 20, 20]])
    transform = SmallObjectCopyPaste(
        dataset, p=1.0, placement="collision_aware", max_trials=3, rng=__import__("random").Random(2)
    )
    before = labels["img"].copy()
    out = transform(labels)
    assert len(out["instances"]) == 1
    assert np.array_equal(out["img"], before)
    assert transform.diagnostics()["rejected_collision"] == 3


def test_cluster_preserves_relative_geometry(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6], [12, 10, 16, 14]])
    labels = _labels(np.zeros((20, 20, 3), dtype=np.uint8))
    transform = SmallObjectCopyPaste(
        dataset, p=1.0, unit="cluster", cluster_expand=10.0, rng=__import__("random").Random(3)
    )
    out = transform(labels)
    assert len(out["instances"]) == 2
    boxes = out["instances"].bboxes
    assert np.allclose(boxes[1] - boxes[0], [9, 8, 9, 8])
    assert transform.diagnostics()["pasted_clusters"] == 1


def test_normalized_xywh_source_metadata_crops_expected_pixels(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    dataset.labels[0]["bboxes"] = np.array([[0.25, 0.2, 0.2, 0.2]], dtype=np.float32)
    dataset.labels[0]["bbox_format"] = "xywh"
    dataset.labels[0]["normalized"] = True
    labels = _labels(np.zeros((20, 20, 3), dtype=np.uint8))
    transform = SmallObjectCopyPaste(dataset, p=1.0, rng=__import__("random").Random(4))
    out = transform(labels)
    assert len(out["instances"]) == 1
    assert np.any(out["img"] == (10, 20, 30))


def test_baseline_rejects_scale_and_padding_ablation_values(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    with pytest.raises(ValueError, match="scale=1.0"):
        SmallObjectCopyPaste(dataset, scale=0.5)
    with pytest.raises(ValueError, match="padding=0.0"):
        SmallObjectCopyPaste(dataset, padding=0.1)


def test_paste_near_border_stays_valid(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    labels = _labels(np.zeros((4, 4, 3), dtype=np.uint8))
    transform = SmallObjectCopyPaste(dataset, p=1.0, rng=__import__("random").Random(0))
    out = transform(labels)
    box = out["instances"].bboxes[0]
    assert np.allclose(box, [0, 0, 4, 4])


def test_same_image_source_does_not_mutate_source_annotation(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    labels = _labels(np.zeros((20, 20, 3), dtype=np.uint8), [[1, 1, 2, 2]])
    before = labels["instances"].bboxes.copy()
    transform = SmallObjectCopyPaste(dataset, p=1.0, allow_same_source=True, rng=__import__("random").Random(6))
    out = transform(labels)
    assert np.array_equal(out["instances"].bboxes[0], before[0])
    assert len(out["instances"]) == 2


def test_failed_boundary_placement_leaves_labels_unchanged(tmp_path):
    dataset = _dataset(tmp_path, [[3, 2, 7, 6]])
    labels = _labels(np.zeros((3, 3, 3), dtype=np.uint8))
    before_img = labels["img"].copy()
    before_boxes = labels["instances"].bboxes.copy()
    transform = SmallObjectCopyPaste(dataset, p=1.0, rng=__import__("random").Random(5))
    out = transform(labels)
    assert np.array_equal(out["img"], before_img)
    assert np.array_equal(out["instances"].bboxes, before_boxes)
    assert transform.diagnostics()["rejected_boundary"] == 1


def test_real_yolo_dataset_pipeline_appends_after_spatial_transforms(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[4:8, 5:9] = (10, 20, 30)
    assert cv2.imwrite(str(image_dir / "sample.png"), image)
    (label_dir / "sample.txt").write_text("0 0.21875 0.1875 0.125 0.125\n")
    hyp = get_cfg(overrides={
        "imgsz": 32,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "copy_paste": 0.0,
        "copy_paste_enabled": True,
        "copy_paste_p": 1.0,
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
    })
    dataset = YOLODataset(
        img_path=str(image_dir), imgsz=32, data={"names": {0: "ship"}},
        task="detect", augment=True, hyp=hyp, batch_size=1, rect=False, cache=False,
    )
    sample = dataset[0]
    assert sample["img"].shape == (3, 32, 32)
    assert len(sample["bboxes"]) == len(sample["cls"]) == 2
