import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from copy_paste_protocol import variant_overrides
from project_ultralytics.copy_paste import build_small_object_copy_paste
from project_ultralytics.r5 import ScaleBinSampler, ScaleBinSpec, ScaleDifficultyController
from ultralytics.utils.instance import Instances


def _labels(image, im_file=None):
    labels = {
        "img": image,
        "instances": Instances(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 0, 2), dtype=np.float32),
            bbox_format="xyxy",
            normalized=False,
        ),
        "cls": np.zeros((0, 1), dtype=np.float32),
    }
    if im_file is not None:
        labels["im_file"] = str(im_file)
    return labels


def _dataset(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[4:20, 4:20] = (20, 40, 60)
    source_path = Path(tmp_path) / "source.png"
    negative_path = Path(tmp_path) / "negative.png"
    assert cv2.imwrite(str(source_path), image)
    assert cv2.imwrite(str(negative_path), np.zeros_like(image))
    return type("Dataset", (), {
        "labels": [
            {"bboxes": np.array([[4, 4, 20, 20]], np.float32), "cls": np.array([[0]], np.float32), "bbox_format": "xyxy", "normalized": False, "shape": image.shape[:2]},
            {"bboxes": [], "cls": [], "bbox_format": "xyxy", "normalized": False, "shape": image.shape[:2]},
        ],
        "im_files": [str(source_path), str(negative_path)],
    })()


def test_scale_bins_use_shared_edges_and_include_final_edge():
    spec = ScaleBinSpec((0, 8, 12, 16, 20))
    assert spec.index(0) == 0
    assert spec.index(7.999) == 0
    assert spec.index(8) == 1
    assert spec.index(12) == 2
    assert spec.index(20) == 3
    assert spec.index(20.001) is None


def test_controller_aggregates_sum_and_count_before_ema():
    controller = ScaleDifficultyController(num_bins=2, beta=0.5, exploration=0.0)
    controller.accumulate([0, 0, 1], [1.0, 0.0, 1.0])
    difficulty = controller.end_epoch()
    assert np.allclose(difficulty, [0.5, 0.75])
    assert controller.count.tolist() == [0, 0]


def test_controller_rejects_invalid_mapping_bin_ids():
    controller = ScaleDifficultyController(num_bins=2)
    try:
        controller.accumulate_mapping({-1: (1.0, 1)})
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("invalid bin id was accepted")


def test_controller_masks_infeasible_bins_and_keeps_exploration():
    controller = ScaleDifficultyController(num_bins=3, exploration=0.2)
    probabilities = controller.probabilities(difficulty=[0.1, 0.8, 0.2], feasible=[True, False, True])
    assert probabilities[1] == 0.0
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities[[0, 2]] > 0.0)


def test_sampler_respects_probabilities_and_feasible_values():
    spec = ScaleBinSpec((0, 8, 12))
    sampler = ScaleBinSampler(spec, rng=random.Random(4), probabilities=[0.0, 1.0])
    values = {0: [4.0], 1: []}
    assert sampler.sample(values) == 4.0
    sampler.set_probabilities([1.0, 0.0])
    assert sampler.sample({0: [], 1: [10.0]}) == 10.0


def test_r5_uniform_protocol_has_explicit_shared_binning():
    settings = variant_overrides("negative_canvas_r5_uniform")
    assert settings["copy_paste_mode"] == "adaptive_negative_canvas"
    assert settings["r5_scale_bin_edges"] == [0.0, 8.0, 12.0, 16.0, 20.0]
    assert settings["r5_scale_probabilities"] == [0.25, 0.25, 0.25, 0.25]


def test_r5_builder_preserves_negative_canvas_mechanics(tmp_path):
    dataset = _dataset(tmp_path)
    settings = variant_overrides("negative_canvas_r5_uniform")
    settings["negative_cp_p"] = 1.0
    transform = build_small_object_copy_paste(dataset, SimpleNamespace(**settings))
    out = transform(_labels(np.zeros((64, 64, 3), np.uint8), im_file=dataset.im_files[1]))
    assert len(out["instances"]) == 1
    assert transform.target_policy == "empirical"
    assert transform.donor_policy == "matched"
    assert transform.max_trials == 30
    assert transform.scale_sampler.probabilities.tolist() == [0.25, 0.25, 0.25, 0.25]
