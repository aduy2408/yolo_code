from pathlib import Path
import random
from types import SimpleNamespace

import cv2
import numpy as np

from project_ultralytics.copy_paste import build_small_object_copy_paste, copy_paste_config
from project_ultralytics.stcp_copy_paste import ScaleTransferCopyPaste
from ultralytics.utils.instance import Instances


def _labels(image, boxes=()):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return {
        "img": image,
        "instances": Instances(
            boxes.copy(), np.zeros((len(boxes), 0, 2), dtype=np.float32),
            bbox_format="xyxy", normalized=False,
        ),
        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
    }


def _dataset(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[2:6, 2:6] = (20, 30, 40)
    image[10:18, 10:18] = (50, 60, 70)
    image[24:40, 24:40] = (100, 110, 120)
    path = Path(tmp_path) / "source.png"
    assert cv2.imwrite(str(path), image)
    return SimpleNamespace(
        imgsz=64,
        labels=[{
            "bboxes": np.array([[2, 2, 6, 6], [10, 10, 18, 18], [24, 24, 40, 40]], np.float32),
            "cls": np.array([[0], [0], [0]], np.float32),
            "bbox_format": "xyxy", "normalized": False, "shape": image.shape[:2],
        }],
        im_files=[str(path)],
    )


def _mismatched_resolution_dataset(tmp_path):
    image = np.zeros((128, 192, 3), dtype=np.uint8)
    image[10:40, 10:40] = (20, 30, 40)
    image[50:110, 60:120] = (100, 110, 120)
    path = Path(tmp_path) / "mismatched.png"
    assert cv2.imwrite(str(path), image)
    return SimpleNamespace(
        imgsz=64,
        labels=[{
            "bboxes": np.array([[10, 10, 40, 40], [60, 50, 120, 110]], np.float32),
            "cls": np.array([[0], [0]], np.float32),
            "bbox_format": "xyxy", "normalized": False, "shape": image.shape[:2],
        }],
        im_files=[str(path)],
    )


def test_stcp_retrieves_same_class_large_donor_and_uses_target_scale(tmp_path):
    dataset = _dataset(tmp_path)
    transform = ScaleTransferCopyPaste(
        dataset, p=1.0, min_pastes=1, max_pastes=1,
        target_max_scale=20.0, ratio_min=1.5, ratio_max=2.5,
        max_trials=100, rng=random.Random(4),
    )
    transform._build_pool()
    transform.target_scale_pool[0] = [8.0]
    out = transform(_labels(np.zeros((64, 64, 3), dtype=np.uint8)))
    assert len(out["instances"]) == 1
    assert transform.stats["successful_pastes"] == 1
    assert transform.stats["target_scale_sum"] == 8.0
    assert 1.5 <= transform.stats["source_target_ratio_sum"] <= 2.5
    box = out["instances"].bboxes[0]
    assert np.sqrt(np.prod(box[2:] - box[:2])) == 8.0


def test_stcp_resizes_raw_crop_to_target_final_scale(tmp_path):
    dataset = _mismatched_resolution_dataset(tmp_path)
    transform = ScaleTransferCopyPaste(
        dataset, p=1.0, min_pastes=1, max_pastes=1,
        target_max_scale=20.0, ratio_min=1.5, ratio_max=2.5,
        max_trials=100, rng=random.Random(2),
    )
    transform._build_pool()
    transform.target_scale_pool[0] = [10.0]
    out = transform(_labels(np.zeros((64, 64, 3), dtype=np.uint8)))
    box = out["instances"].bboxes[0]
    assert np.sqrt(np.prod(box[2:] - box[:2])) == 10.0
    assert transform.stats["source_target_ratio_sum"] == 2.0


def test_stcp_rejects_collision_and_does_not_fallback_to_random_donor(tmp_path):
    dataset = _dataset(tmp_path)
    transform = ScaleTransferCopyPaste(
        dataset, p=1.0, min_pastes=1, max_pastes=1,
        target_max_scale=20.0, ratio_min=1.5, ratio_max=2.5,
        max_trials=20, max_ioa=0.10, rng=random.Random(3),
    )
    transform._build_pool()
    transform.target_scale_pool[0] = [8.0]
    before = _labels(np.zeros((64, 64, 3), dtype=np.uint8), [[0, 0, 64, 64]])
    out = transform(before)
    assert len(out["instances"]) == 1
    assert transform.stats["successful_pastes"] == 0
    assert transform.stats["placement_fail"] == 1
    assert transform.stats["no_donor"] == 0


def test_stcp_builder_and_manifest_are_explicit(tmp_path):
    hyp = SimpleNamespace(
        copy_paste_enabled=True, copy_paste_mode="stcp", stcp_p=0.30,
        stcp_min_pastes=1, stcp_max_pastes=2, stcp_target_max_scale=20.0,
        stcp_ratio_min=1.5, stcp_ratio_max=2.5, stcp_blur_sigma=0.5,
        stcp_blur_min_side=5, stcp_max_ioa=0.10, stcp_max_trials=30,
    )
    transform = build_small_object_copy_paste(_dataset(tmp_path), hyp)
    assert isinstance(transform, ScaleTransferCopyPaste)
    config = copy_paste_config(hyp)
    assert config["mode"] == "stcp"
    assert config["stcp_position"] == "post_geometry"
    assert config["stcp_ratio_min"] == 1.5
    assert config["stcp_ratio_max"] == 2.5
