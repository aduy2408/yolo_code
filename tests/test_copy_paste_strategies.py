from pathlib import Path
import random

import cv2
import numpy as np
import pytest
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import v8_transforms
from ultralytics.data.dataset import YOLODataset

from project_ultralytics.copy_paste import CrowdedCopyPaste, ScaleMatchedCopyPaste
from project_ultralytics.negative_copy_paste import (
    HardNegativeBank,
    HardNegativeMiner,
    HardNegativeRecord,
    NegativeCopyPaste,
)
from ultralytics.utils.instance import Instances


def _labels(image, boxes=(), classes=()):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return {
        "img": image,
        "instances": Instances(boxes.copy(), np.zeros((len(boxes), 0, 2), dtype=np.float32), bbox_format="xyxy", normalized=False),
        "cls": np.asarray(classes or [0] * len(boxes), dtype=np.float32).reshape(-1, 1),
    }


def _dataset(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[4:24, 4:24] = (30, 40, 50)  # sqrt(area)=20
    image[36:44, 36:44] = (100, 110, 120)  # sqrt(area)=8
    path = Path(tmp_path) / "source.png"
    assert cv2.imwrite(str(path), image)
    return type("Dataset", (), {
        "labels": [{
            "bboxes": np.array([[4, 4, 24, 24], [36, 36, 44, 44]], dtype=np.float32),
            "cls": np.array([[0], [0]], dtype=np.float32),
            "bbox_format": "xyxy", "normalized": False, "shape": image.shape[:2],
        }],
        "im_files": [str(path)],
    })()


def test_crowded_copy_paste_adds_one_and_preserves_anchor(tmp_path):
    dataset = _dataset(tmp_path)
    original = np.array([[24, 24, 44, 44]], dtype=np.float32)
    labels = _labels(np.zeros((64, 64, 3), dtype=np.uint8), original)
    transform = CrowdedCopyPaste(
        dataset, p=1.0, overlap_min=0.10, overlap_max=0.30,
        min_visibility=0.60, trials=300, rng=random.Random(4),
    )
    out = transform(labels)
    assert len(out["instances"]) == 2
    assert np.array_equal(out["instances"].bboxes[0], original[0])
    pasted = out["instances"].bboxes[1]
    inter = max(0, min(original[0, 2], pasted[2]) - max(original[0, 0], pasted[0])) * max(
        0, min(original[0, 3], pasted[3]) - max(original[0, 1], pasted[1])
    )
    measured = inter / min(400, (pasted[2] - pasted[0]) * (pasted[3] - pasted[1]))
    assert 0.10 <= measured <= 0.30


def test_scale_matched_adds_box_with_pixel_scale(tmp_path):
    dataset = _dataset(tmp_path)
    labels = _labels(np.zeros((64, 64, 3), dtype=np.uint8), [[48, 48, 56, 56]])
    transform = ScaleMatchedCopyPaste(
        dataset, p=1.0, target_max_size=8.0, source_min_ratio=1.25,
        factor_min=0.40, factor_max=0.90, trials=100, rng=random.Random(2),
    )
    out = transform(labels)
    assert len(out["instances"]) == 2
    box = out["instances"].bboxes[1]
    result_size = np.sqrt((box[2] - box[0]) * (box[3] - box[1]))
    assert result_size == 8.0
    assert transform.diagnostics()["scale/mean_factor"] == 0.4


def test_negative_copy_paste_does_not_change_labels(tmp_path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[4:12, 4:12] = (200, 10, 20)
    path = Path(tmp_path) / "negative.png"
    assert cv2.imwrite(str(path), image)
    bank = HardNegativeBank([HardNegativeRecord(
        image_path=str(path), crop_xyxy=(4, 4, 12, 12), pred_xyxy=(5, 5, 11, 11),
        conf=0.9, pred_size=6.0, source_image_id="other",
    )])
    labels = _labels(np.zeros((32, 32, 3), dtype=np.uint8), [[14, 14, 18, 18]])
    before_boxes = labels["instances"].bboxes.copy()
    before_cls = labels["cls"].copy()
    out = NegativeCopyPaste(bank, p=1.0, rng=random.Random(5))(labels)
    assert np.array_equal(out["instances"].bboxes, before_boxes)
    assert np.array_equal(out["cls"], before_cls)
    assert len(out["instances"]) == 1
    assert np.any(out["img"] == (200, 10, 20))


def test_negative_copy_paste_preserves_normalized_instance_representation(tmp_path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[4:12, 4:12] = (200, 10, 20)
    path = Path(tmp_path) / "negative_normalized.png"
    assert cv2.imwrite(str(path), image)
    bank = HardNegativeBank([HardNegativeRecord(
        image_path=str(path), crop_xyxy=(4, 4, 12, 12), pred_xyxy=(5, 5, 11, 11),
        conf=0.9, pred_size=6.0, source_image_id="other",
    )])
    instances = Instances(
        np.array([[0.5, 0.5, 0.125, 0.125]], dtype=np.float32),
        np.zeros((1, 0, 2), dtype=np.float32), bbox_format="xywh", normalized=True,
    )
    labels = {"img": np.zeros((32, 32, 3), dtype=np.uint8), "instances": instances, "cls": np.array([[0]], np.float32)}
    before_boxes = instances.bboxes.copy()
    before_format = instances._bboxes.format
    before_normalized = instances.normalized
    NegativeCopyPaste(bank, p=1.0, rng=random.Random(5))(labels)
    assert np.array_equal(instances.bboxes, before_boxes)
    assert instances._bboxes.format == before_format
    assert instances.normalized == before_normalized


def test_hard_negative_miner_filters_gt_and_ignore_and_keeps_top_k(tmp_path):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    path = Path(tmp_path) / "mine.png"
    assert cv2.imwrite(str(path), image)
    predictions = [np.array([
        [2, 2, 6, 6, 0.8],      # accepted
        [20, 20, 25, 25, 0.7],  # overlaps valid GT
        [30, 30, 35, 35, 0.9],  # overlaps ignore
        [8, 8, 12, 12, 0.6],    # accepted but below per-image top-k
    ], dtype=np.float32)]
    miner = HardNegativeMiner(bank_per_image=1, crop_expand=1.5)
    bank = miner.mine([path], predictions, [np.array([[20, 20, 25, 25]])], [np.array([[30, 30, 35, 35]])])
    assert len(bank) == 1
    assert bank.records[0].conf == pytest.approx(0.8)
    assert miner.stats["rejected_gt"] == 1
    assert miner.stats["rejected_ignore"] == 1


def test_hard_negative_bank_round_trip(tmp_path):
    bank = HardNegativeBank([HardNegativeRecord("a.png", (1, 2, 3, 4), (1, 2, 3, 4), 0.5, 2.0, "a")])
    path = Path(tmp_path) / "bank.json"
    bank.save(path)
    restored = HardNegativeBank.load(path)
    assert restored.records == bank.records


def test_custom_mode_bypasses_upstream_copy_paste_assertion(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[4:8, 5:9] = (10, 20, 30)
    assert cv2.imwrite(str(image_dir / "sample.png"), image)
    (label_dir / "sample.txt").write_text("0 0.21875 0.1875 0.125 0.125\n")
    hyp = get_cfg(overrides={
        "imgsz": 32, "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
        "perspective": 0.0, "fliplr": 0.0, "flipud": 0.0,
        "copy_paste": 0.0, "copy_paste_mode": "crowded",
    })
    hyp.copy_paste_enabled = True
    hyp.copy_paste_p = 1.0
    hyp.copy_paste_unit = "single"
    hyp.copy_paste_copies = 1
    hyp.copy_paste_max_trials = 30
    hyp.copy_paste_allow_empty_target = True
    hyp.copy_paste_allow_same_source = True
    hyp.crowd_overlap_min = 0.10
    hyp.crowd_overlap_max = 0.30
    hyp.crowd_min_visibility = 0.60
    hyp.crowd_size_ratio_min = 0.75
    hyp.crowd_size_ratio_max = 1.33
    hyp.crowd_trials = 30
    dataset = YOLODataset(
        img_path=str(image_dir), imgsz=32, data={"names": {0: "ship"}},
        task="detect", augment=True, hyp=hyp, batch_size=1, rect=False, cache=False,
    )
    pipeline = v8_transforms(dataset, 32, hyp)
    assert any(type(transform).__name__ == "CrowdedCopyPaste" for transform in pipeline.transforms)
    sample = dataset[0]
    assert sample["img"].shape == (3, 32, 32)
