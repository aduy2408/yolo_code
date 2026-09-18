from pathlib import Path
import random

import cv2
import numpy as np

from project_ultralytics.adaptive_copy_paste import AdaptiveCopyPaste, AdaptivePasteBudget
from project_ultralytics.online_negative_bank import (
    OnlineHardNegativeBank,
    OnlineHardNegativeCollector,
    OnlineHardNegativeRecord,
    OnlineNegativeCopyPaste,
)
from ultralytics.utils.instance import Instances


def _labels(image, boxes=()):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return {
        "img": image,
        "instances": Instances(boxes.copy(), np.zeros((len(boxes), 0, 2), dtype=np.float32), bbox_format="xyxy", normalized=False),
        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
    }


def _dataset(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[4:20, 4:20] = (20, 40, 60)
    image[32:40, 32:40] = (80, 100, 120)
    path = Path(tmp_path) / "source.png"
    assert cv2.imwrite(str(path), image)
    return type("Dataset", (), {
        "labels": [{"bboxes": np.array([[4, 4, 20, 20], [32, 32, 40, 40]], np.float32), "cls": np.array([[0], [0]]), "bbox_format": "xyxy", "normalized": False}],
        "im_files": [str(path)],
    })()


def test_budget_is_bounded_and_empty_when_scene_meets_target():
    budget = AdaptivePasteBudget([0, 1, 3], max_objects=4)
    assert budget(3, random.Random(2)) == 0
    assert 0 <= budget(0, random.Random(2)) <= 3


def test_scale_conditioned_adaptive_paste_adds_budgeted_objects(tmp_path):
    transform = AdaptiveCopyPaste(
        _dataset(tmp_path), budget=AdaptivePasteBudget([2], max_objects=2),
        policy="scale_conditioned", p=1.0, allow_empty_target=True,
        allow_same_source=True, max_trials=100, rng=random.Random(4),
    )
    out = transform(_labels(np.zeros((64, 64, 3), dtype=np.uint8)))
    assert len(out["instances"]) == 2
    assert transform.stats["pasted_instances"] == 2


def test_online_bank_commits_and_scale_ranks(tmp_path):
    bank = OnlineHardNegativeBank(tmp_path / "bank", max_size=2)
    patch = np.full((4, 4, 3), 100, dtype=np.uint8)
    bank.add_patch(patch, OnlineHardNegativeRecord("p.jpg", 8.0, 0.8, 0.8, "img"))
    bank.add_patch(patch, OnlineHardNegativeRecord("q.jpg", 16.0, 0.9, 0.9, "img2"))
    manifest = bank.commit()
    restored = OnlineHardNegativeBank.load(manifest.parent)
    assert len(restored) == 2
    assert restored.ranked(target_size=8.0)[0].pred_size == 8.0


def test_online_collector_and_negative_paste_leave_labels_unchanged(tmp_path):
    bank = OnlineHardNegativeBank(tmp_path / "online")
    image = np.full((16, 16, 3), 120, dtype=np.uint8)
    collector = OnlineHardNegativeCollector(bank)
    assert collector.collect(image, np.array([[3, 3, 7, 7, 0.9]], np.float32), "source", epoch=2) == 1
    labels = _labels(np.zeros((32, 32, 3), dtype=np.uint8), [[14, 14, 18, 18]])
    before = labels["instances"].bboxes.copy()
    OnlineNegativeCopyPaste(bank, p=1.0, scale_matched=True, rng=random.Random(3))(labels)
    assert np.array_equal(before, labels["instances"].bboxes)
    assert np.any(labels["img"] == 120)
