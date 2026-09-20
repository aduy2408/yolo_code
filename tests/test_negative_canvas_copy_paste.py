from pathlib import Path
import random
from types import SimpleNamespace

import cv2
import numpy as np

from project_ultralytics.copy_paste import build_small_object_copy_paste, copy_paste_config
from project_ultralytics.negative_canvas_copy_paste import NegativeCanvasCopyPaste
from ultralytics.utils.instance import Instances


def _labels(image, boxes=(), im_file=None, image_index=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    labels = {
        "img": image,
        "instances": Instances(
            boxes.copy(),
            np.zeros((len(boxes), 0, 2), dtype=np.float32),
            bbox_format="xyxy",
            normalized=False,
        ),
        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
    }
    if im_file is not None:
        labels["im_file"] = str(im_file)
    if image_index is not None:
        labels["image_index"] = image_index
    return labels


def _dataset(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.indices((16, 16))
    image[4:20, 4:20] = np.stack((xx * 10, yy * 10, (xx + yy) * 5), axis=-1)
    image[28:36, 28:36] = (40, 80, 120)
    image[40:56, 40:56] = (150, 100, 50)
    path = Path(tmp_path) / "source.png"
    negative_path = Path(tmp_path) / "negative.png"
    assert cv2.imwrite(str(path), image)
    assert cv2.imwrite(str(negative_path), np.zeros_like(image))
    return type("Dataset", (), {
        "labels": [{
            "bboxes": np.array([[4, 4, 20, 20], [28, 28, 36, 36], [40, 40, 56, 56]], np.float32),
            "cls": np.array([[0], [0], [0]], np.float32),
            "bbox_format": "xyxy",
            "normalized": False,
            "shape": image.shape[:2],
        }, {"bboxes": [], "cls": [], "bbox_format": "xyxy", "normalized": False, "shape": image.shape[:2]}],
        "im_files": [str(path), str(negative_path)],
    })()


def test_positive_image_is_untouched_and_probability_is_conditional(tmp_path):
    transform = NegativeCanvasCopyPaste(_dataset(tmp_path), p=1.0, rng=random.Random(1))
    labels = _labels(np.full((64, 64, 3), 7, np.uint8), [[1, 1, 5, 5]], im_file=_dataset(tmp_path).im_files[0])
    before_image = labels["img"].copy()
    before_boxes = labels["instances"].bboxes.copy()
    transform(labels)
    assert np.array_equal(labels["img"], before_image)
    assert np.array_equal(labels["instances"].bboxes, before_boxes)
    assert transform.stats["negative_seen"] == 0

    empty = _labels(np.zeros((64, 64, 3), np.uint8), im_file=_dataset(tmp_path).im_files[1])
    gated = NegativeCanvasCopyPaste(_dataset(tmp_path), p=0.0, rng=random.Random(1))
    gated(empty)
    assert gated.stats["negative_seen"] == 1
    assert gated.stats["negative_selected"] == 0


def test_r1_matches_donor_and_adds_one_target_sized_instance(tmp_path):
    transform = NegativeCanvasCopyPaste(
        _dataset(tmp_path), p=1.0, target_policy="empirical", donor_policy="matched",
        target_max_size=8.0, rng=random.Random(4), max_trials=100,
    )
    out = transform(_labels(np.zeros((64, 64, 3), np.uint8), im_file=transform.dataset.im_files[1]))
    assert len(out["instances"]) == 1
    assert transform.stats["applied_images"] == 1
    assert 1.0 <= transform.stats["source_size_sum"] / transform.stats["target_size_sum"] < 1.5
    result_size = np.sqrt(np.prod(out["instances"].bboxes[0, 2:] - out["instances"].bboxes[0, :2]))
    assert result_size == 8.0


def test_donor_policy_ranges_are_disjoint(tmp_path):
    transform = NegativeCanvasCopyPaste(_dataset(tmp_path), target_max_size=8.0)
    assert transform._donor_valid(8.0, 8.0)
    assert transform._donor_valid(11.99, 8.0)
    assert not transform._donor_valid(12.0, 8.0)
    assert not transform._donor_valid(16.0, 8.0)
    transform.donor_policy = "larger"
    assert not transform._donor_valid(11.99, 8.0)
    assert transform._donor_valid(12.0, 8.0)
    assert transform._donor_valid(20.0, 8.0)
    assert not transform._donor_valid(20.01, 8.0)


def test_r3_and_r4_share_selection_and_location_but_differ_in_pixels(tmp_path):
    dataset = _dataset(tmp_path)
    base = _labels(np.zeros((64, 64, 3), np.uint8))
    r3 = NegativeCanvasCopyPaste(
        dataset, p=1.0, target_policy="deficit", donor_policy="larger",
        target_max_size=8.0, degradation="none", rng=random.Random(42), max_trials=100,
    )
    r4 = NegativeCanvasCopyPaste(
        dataset, p=1.0, target_policy="deficit", donor_policy="larger",
        target_max_size=8.0, degradation="weak_blur", blur_sigma=0.5,
        rng=random.Random(42), max_trials=100,
    )
    out3 = r3(_labels(base["img"].copy(), im_file=dataset.im_files[1]))
    out4 = r4(_labels(base["img"].copy(), im_file=dataset.im_files[1]))
    assert np.array_equal(out3["instances"].bboxes, out4["instances"].bboxes)
    assert np.array_equal(out3["cls"], out4["cls"])
    assert not np.array_equal(out3["img"], out4["img"])
    assert r4.stats["degradation_applied"] == 1
    assert 1.5 <= r3.stats["source_size_sum"] / r3.stats["target_size_sum"] <= 2.5


def test_builder_and_manifest_record_all_negative_canvas_fields(tmp_path):
    hyp = SimpleNamespace(
        copy_paste_enabled=True,
        copy_paste_mode="negative_canvas",
        copy_paste_p=0.30,
        negative_cp_target_policy="deficit",
        negative_cp_donor_policy="larger",
        negative_cp_target_max_size=20.0,
        negative_cp_deficit_gamma=0.5,
        negative_cp_max_weight_ratio=3.0,
        negative_cp_matched_ratio_max=1.5,
        negative_cp_large_ratio_min=1.5,
        negative_cp_large_ratio_max=2.5,
        negative_cp_degradation="weak_blur",
        negative_cp_blur_sigma=0.5,
        copy_paste_max_trials=30,
    )
    transform = build_small_object_copy_paste(_dataset(tmp_path), hyp)
    assert isinstance(transform, NegativeCanvasCopyPaste)
    config = copy_paste_config(hyp)
    assert config["mode"] == "negative_canvas"
    assert config["negative_cp_target_policy"] == "deficit"
    assert config["negative_cp_donor_policy"] == "larger"
    assert config["negative_cp_degradation"] == "weak_blur"
    assert config["negative_cp_large_ratio_max"] == 2.5


def test_original_positive_with_dropped_current_boxes_is_not_negative(tmp_path):
    dataset = _dataset(tmp_path)
    transform = NegativeCanvasCopyPaste(dataset, p=1.0, rng=random.Random(7))
    labels = _labels(np.zeros((64, 64, 3), np.uint8), im_file=dataset.im_files[0])
    transform(labels)
    assert len(labels["instances"]) == 0
    assert transform.stats["negative_seen"] == 0


def test_negative_canvas_uses_mode_specific_probability_default(tmp_path):
    hyp = SimpleNamespace(copy_paste_enabled=True, copy_paste_mode="negative_canvas")
    transform = build_small_object_copy_paste(_dataset(tmp_path), hyp)
    assert transform.p == 0.30
    assert copy_paste_config(hyp)["negative_cp_p"] == 0.30


def test_donor_filter_uses_effective_rounded_crop_geometry(tmp_path):
    transform = NegativeCanvasCopyPaste(_dataset(tmp_path), donor_policy="matched")
    record = type("Record", (), {"bbox_xyxy": (0.49, 0.49, 15.60, 15.60)})()
    effective = transform._effective_source_size(record)
    assert effective == 16.0
    assert not transform._donor_valid(effective, 10.0)


def test_stats_keep_base_placement_keys_and_reset_custom_keys(tmp_path):
    transform = NegativeCanvasCopyPaste(_dataset(tmp_path), p=1.0, rng=random.Random(2))
    transform(_labels(np.zeros((64, 64, 3), np.uint8), im_file=transform.dataset.im_files[1]))
    assert {"rejected_boundary", "rejected_collision", "failed_trials"}.issubset(transform.stats)
    transform.reset_stats()
    assert transform.stats["negative_seen"] == 0
    assert transform.stats["source_target_ratio_sum"] == 0.0
    assert "failed_trials" in transform.stats


def test_diagnostic_ratio_is_mean_of_ratios(tmp_path):
    transform = NegativeCanvasCopyPaste(_dataset(tmp_path))
    transform.stats.update({
        "applied_images": 2,
        "negative_seen": 2,
        "negative_selected": 2,
        "target_size_sum": 10.0,
        "source_size_sum": 15.0,
        "source_target_ratio_sum": 3.0,
    })
    assert transform.diagnostics()["negcanvas/mean_source_target_ratio"] == 1.5
