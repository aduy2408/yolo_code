from pathlib import Path
import random
import copy
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import pytest

from project_ultralytics.adaptive_copy_paste import AdaptiveCopyPaste, AdaptivePasteBudget
from project_ultralytics.copy_paste import build_small_object_copy_paste, copy_paste_config
from project_ultralytics.online_negative_bank import (
    OnlineHardNegativeBank,
    OnlineHardNegativeCollector,
    OnlineHardNegativeRecord,
    OnlineNegativeCopyPaste,
    collect_online_hard_negatives,
    configure_online_hard_negative_training,
    deduplicate_candidate_indices,
    finish_online_hard_negative_epoch,
    iter_online_negative_transforms,
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
    assert AdaptivePasteBudget([4, 5], max_objects=4)(4, random.Random(2)) == 1


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


def test_online_collector_uses_unique_source_identity_and_updates_ema(tmp_path):
    bank = OnlineHardNegativeBank(tmp_path / "bank")
    collector = OnlineHardNegativeCollector(bank, beta=0.5)
    image = np.full((16, 16, 3), 100, dtype=np.uint8)
    row = np.array([[3, 3, 7, 7, 0.8]], dtype=np.float32)
    assert collector.collect(image, row, "dataset-1", epoch=2) == 1
    first_path = bank.records[0].patch_path
    assert collector.collect(image, row, "dataset-2", epoch=2) == 1
    assert len(bank.records) == 2
    assert bank.records[0].patch_path != bank.records[1].patch_path
    assert collector.collect(image, np.array([[3, 3, 7, 7, 0.9]], dtype=np.float32), "dataset-1", epoch=3) == 1
    record = next(item for item in bank.records if item.source_image_id == "dataset-1")
    assert record.times_seen == 2
    assert record.ema_hardness != record.hardness
    assert Path(first_path).exists()


def test_online_worker_refresh_sees_committed_epoch(tmp_path):
    bank = OnlineHardNegativeBank(tmp_path / "bank")
    bank.add_patch(np.full((4, 4, 3), 80, np.uint8), OnlineHardNegativeRecord("a.jpg", 4.0, 0.8, 0.8, "a"))
    bank.commit()
    transform = OnlineNegativeCopyPaste(OnlineHardNegativeBank.load(tmp_path / "bank"), p=1.0, rng=random.Random(1))
    worker = copy.copy(transform)
    bank.add_patch(np.full((4, 4, 3), 90, np.uint8), OnlineHardNegativeRecord("b.jpg", 4.0, 0.9, 0.9, "b"))
    bank.commit()
    transform.refresh()
    worker._ensure_fresh()
    assert len(worker.bank) == 2


def test_online_training_helpers_collect_and_commit_epoch(tmp_path):
    bank = OnlineHardNegativeBank(tmp_path / "bank")
    bank.commit()
    transform = OnlineNegativeCopyPaste(bank, p=1.0, rng=random.Random(1))
    dataset = type("Dataset", (), {"transforms": type("Compose", (), {"transforms": [transform]})()})()
    trainer = type("Trainer", (), {
        "args": SimpleNamespace(copy_paste_mode="online_negative"),
        "train_loader": type("Loader", (), {"dataset": dataset})(),
    })()
    assert configure_online_hard_negative_training(trainer) is True
    criterion = type("Criterion", (), {"last_hard_negative_candidates": [[[2, 2, 6, 6, 0.8, 1.2]]]})()
    batch = {"img": torch.full((1, 3, 16, 16), 0.5), "dataset_idx": torch.tensor([7])}
    assert collect_online_hard_negatives(trainer, batch, criterion, epoch=1) == 1
    assert finish_online_hard_negative_epoch(trainer) == 1
    assert len(OnlineHardNegativeBank.load(tmp_path / "bank")) == 1


def test_online_transform_discovery_handles_wrapped_yolodataset_pipeline(tmp_path):
    bank = OnlineHardNegativeBank(tmp_path / "bank")
    transform = OnlineNegativeCopyPaste(bank, p=1.0, rng=random.Random(1))
    compose = type("Compose", (), {"transforms": [transform]})()
    wrapper = type("AlternatePartialClipPipeline", (), {"normal_pipeline": compose})()
    dataset = type("Dataset", (), {"transforms": type("Compose", (), {"transforms": [wrapper]})()})()
    assert list(iter_online_negative_transforms(dataset)) == [transform]


def test_cluster_does_not_overspend_and_matches_member_scale(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[4:20, 4:20] = 50
    image[24:40, 24:40] = 100
    path = Path(tmp_path) / "cluster.png"
    assert cv2.imwrite(str(path), image)
    dataset = type("Dataset", (), {
        "labels": [{"bboxes": np.array([[4, 4, 20, 20], [24, 24, 40, 40]], np.float32), "cls": np.array([[0], [0]]), "bbox_format": "xyxy", "normalized": False}],
        "im_files": [str(path)],
    })()
    labels = _labels(np.zeros((64, 64, 3), dtype=np.uint8), [[48, 48, 56, 56]])
    transform = AdaptiveCopyPaste(dataset, budget=AdaptivePasteBudget([3], max_objects=2), policy="cluster", p=1.0, max_trials=100, rng=random.Random(3))
    out = transform(labels)
    assert len(out["instances"]) == 3
    added = out["instances"].bboxes[1:]
    sizes = np.sqrt((added[:, 2] - added[:, 0]) * (added[:, 3] - added[:, 1]))
    assert np.allclose(sizes, 8.0, atol=1.5)


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


def test_public_builder_exposes_adaptive_and_online_modes(tmp_path):
    dataset = _dataset(tmp_path)
    adaptive_hyp = SimpleNamespace(
        copy_paste_enabled=True, copy_paste_mode="adaptive_scale_deficit",
        copy_paste_p=1.0, adaptive_cp_target_counts=[2], adaptive_cp_max_objects=2,
        copy_paste_allow_empty_target=True, copy_paste_allow_same_source=True,
        copy_paste_max_trials=20,
    )
    transform = build_small_object_copy_paste(dataset, adaptive_hyp)
    assert isinstance(transform, AdaptiveCopyPaste)
    assert transform.policy == "scale_deficit"
    assert copy_paste_config(adaptive_hyp)["adaptive_max_objects"] == 2

    bank_root = tmp_path / "bank"
    bank = OnlineHardNegativeBank(bank_root)
    bank.add_patch(np.full((4, 4, 3), 70, np.uint8), OnlineHardNegativeRecord("p.jpg", 4.0, 0.8, 0.8, "src"))
    bank.commit()
    online_hyp = SimpleNamespace(
        copy_paste_enabled=True, copy_paste_mode="online_negative_scale_matched",
        online_negcp_bank_path=str(bank_root), copy_paste_p=1.0,
    )
    online = build_small_object_copy_paste(dataset, online_hyp)
    assert isinstance(online, OnlineNegativeCopyPaste)
    assert online.scale_matched is True


def test_adaptive_builder_rejects_unmatched_mosaic_statistics(tmp_path):
    hyp = SimpleNamespace(copy_paste_enabled=True, copy_paste_mode="adaptive", mosaic=1.0)
    with pytest.raises(ValueError, match="post-Mosaic"):
        build_small_object_copy_paste(_dataset(tmp_path), hyp)


def test_scale_conditioned_defaults_to_shrink_only(tmp_path):
    transform = AdaptiveCopyPaste(_dataset(tmp_path), budget=AdaptivePasteBudget([2], 1), p=1.0)
    assert transform.factor_max <= 1.0


def test_online_builder_rejects_raw_budget_distribution_after_mosaic(tmp_path):
    root = tmp_path / "bank"
    bank = OnlineHardNegativeBank(root)
    bank.commit()
    hyp = SimpleNamespace(
        copy_paste_enabled=True, copy_paste_mode="online_negative", mosaic=1.0,
        online_negcp_bank_path=str(root), copy_paste_p=1.0,
    )
    with pytest.raises(ValueError, match="post-Mosaic"):
        build_small_object_copy_paste(_dataset(tmp_path), hyp)


def test_online_negative_placement_rejects_overlap_between_patches(tmp_path):
    class FixedRng:
        def random(self):
            return 0.0
        def randint(self, low, high):
            return 0

    bank = OnlineHardNegativeBank(tmp_path / "bank")
    bank.add_patch(np.full((4, 4, 3), 40, np.uint8), OnlineHardNegativeRecord("a.jpg", 4, 0.8, 0.8, "a"))
    bank.add_patch(np.full((4, 4, 3), 200, np.uint8), OnlineHardNegativeRecord("b.jpg", 4, 0.7, 0.7, "b"))
    labels = _labels(np.zeros((16, 16, 3), np.uint8))
    OnlineNegativeCopyPaste(bank, p=1.0, max_objects=2, max_trials=1, rng=FixedRng())(labels)
    assert int(labels["img"][0, 0, 0]) < 100


def test_adaptive_scale_conditioning_stays_on_original_destination_scales(tmp_path):
    transform = AdaptiveCopyPaste(
        _dataset(tmp_path), budget=AdaptivePasteBudget([3], 2), p=1.0,
        policy="scale_conditioned", max_trials=100, rng=random.Random(5),
    )
    seen = []
    original = transform._target_scale
    transform._target_scale = lambda scales: (seen.append(tuple(scales)) or original(scales))
    transform(_labels(np.zeros((64, 64, 3), np.uint8), [[48, 48, 56, 56]]))
    assert len(seen) == 2
    assert all(scales == (8.0,) for scales in seen)


def test_hard_negative_candidate_dedup_keeps_spatially_distinct_boxes():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept = deduplicate_candidate_indices(boxes, scores, torch.ones(3, dtype=torch.bool), limit=3)
    assert kept == [0, 2]
