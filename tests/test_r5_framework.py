import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from copy_paste_protocol import variant_overrides
from project_ultralytics.copy_paste import build_small_object_copy_paste
from project_ultralytics.adaptive_negative_canvas import AdaptiveNegativeCanvasCopyPaste
from project_ultralytics.r5 import ScaleBinSampler, ScaleBinSpec, ScaleDifficultyController
from project_ultralytics.r5 import SharedProbabilityState
from project_ultralytics.r5.diagnostics import r5_diagnostics
from ultralytics.utils.instance import Instances
from ultralytics.data.augment import Format, MixUp


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


def test_controller_zero_weight_fallback_stays_within_feasible_mask():
    controller = ScaleDifficultyController(num_bins=3, exploration=0.0)
    probabilities = controller.probabilities(difficulty=[0.0, 0.0, 0.0], feasible=[True, False, True])
    assert np.allclose(probabilities, [0.5, 0.0, 0.5])


def test_controller_rejects_non_bounded_difficulty():
    controller = ScaleDifficultyController(num_bins=2)
    for values in ([np.nan, 0.2], [1.2, 0.2]):
        try:
            controller.probabilities(difficulty=values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid difficulty was accepted")


def test_sampler_respects_probabilities_and_feasible_values():
    spec = ScaleBinSpec((0, 8, 12))
    sampler = ScaleBinSampler(spec, rng=random.Random(4), probabilities=[0.0, 1.0])
    values = {0: [4.0], 1: []}
    assert sampler.sample(values) == 4.0
    sampler.set_probabilities([1.0, 0.0])
    assert sampler.sample({0: [], 1: [10.0]}) == 10.0


def test_seeded_sampler_is_reproducible_and_resumeable():
    spec = ScaleBinSpec((0, 8, 12))
    first = ScaleBinSampler(spec, seed=42)
    second = ScaleBinSampler(spec, seed=42)
    values = {0: [4.0], 1: [10.0]}
    assert [first.sample(values) for _ in range(12)] == [second.sample(values) for _ in range(12)]
    state = first.state_dict()
    restored = ScaleBinSampler.from_state_dict(state)
    assert [first.sample(values) for _ in range(8)] == [restored.sample(values) for _ in range(8)]


def test_shared_probability_state_restore_preserves_snapshot_not_stale_version():
    state = SharedProbabilityState(4, [0.1, 0.2, 0.3, 0.4])
    state.update([0.4, 0.3, 0.2, 0.1])
    restored = SharedProbabilityState.from_state_dict(state.state_dict())
    assert np.allclose(restored.snapshot(), [0.4, 0.3, 0.2, 0.1])


def test_r5_builder_reuses_explicit_shared_probability_state(tmp_path):
    from project_ultralytics.copy_paste import build_small_object_copy_paste

    dataset = _dataset(tmp_path)
    settings = variant_overrides("negative_canvas_r5_uniform")
    shared_state = SharedProbabilityState(4, [0.25, 0.25, 0.25, 0.25])
    settings["r5_shared_probability_state"] = shared_state
    transform = build_small_object_copy_paste(dataset, SimpleNamespace(**settings))
    assert transform.scale_sampler.shared_state is shared_state


def test_format_rejects_r5_provenance_alignment_mismatch():
    formatter = Format(normalize=False, batch_idx=False)
    instances = Instances(
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 0, 2), dtype=np.float32),
        bbox_format="xyxy",
        normalized=False,
    )
    with pytest.raises(RuntimeError, match="provenance/GT alignment"):
        formatter.apply_instances(
            {
                "instances": instances,
                "r5_original_gt_mask": np.array([False], dtype=bool),
            },
            {"cls": np.zeros((2, 1), dtype=np.float32), "instances": instances, "nl": 2, "h": 32, "w": 32},
        )


def test_mixup_preserves_synthetic_provenance_when_merging_instances():
    instances = Instances(
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 0, 2), dtype=np.float32),
        bbox_format="xyxy",
        normalized=False,
    )
    labels = {
        "instances": instances,
        "cls": np.zeros((1, 1), dtype=np.float32),
        "r5_original_gt_mask": np.array([False], dtype=bool),
        "mix_labels": [
            {
                "instances": Instances(
                    np.zeros((1, 4), dtype=np.float32),
                    np.zeros((1, 0, 2), dtype=np.float32),
                    bbox_format="xyxy",
                    normalized=False,
                ),
                "cls": np.ones((1, 1), dtype=np.float32),
            }
        ],
    }
    MixUp(None, p=1.0).apply_instances(labels, {})
    assert labels["r5_original_gt_mask"].tolist() == [False, True]


def test_r5_uniform_protocol_has_explicit_shared_binning():
    settings = variant_overrides("negative_canvas_r5_uniform")
    assert settings["copy_paste_mode"] == "adaptive_negative_canvas"
    assert settings["r5_scale_bin_edges"] == [0.0, 8.0, 12.0, 16.0, 20.0]
    assert settings["r5_scale_probabilities"] == [0.25, 0.25, 0.25, 0.25]


def test_controller_updates_adaptive_sampler_using_only_feasible_bins(tmp_path):
    dataset = _dataset(tmp_path)
    transform = AdaptiveNegativeCanvasCopyPaste(dataset, p=1.0, donor_policy="matched", seed=42)
    controller = ScaleDifficultyController(num_bins=4, exploration=0.0)
    controller.accumulate([0, 1, 2, 3], [0.1, 0.2, 0.8, 0.9])
    probabilities = transform.end_epoch_from_controller(controller)
    assert np.isclose(sum(probabilities), 1.0)
    assert np.allclose(probabilities, transform.scale_sampler.probabilities)
    assert np.isclose(sum(probabilities[1:]), 1.0)


def test_diagnostics_keep_failure_causes_separate():
    result = r5_diagnostics(
        [0.5, 0.5], [4, 3], [3, 1],
        donor_failed=[1, 0], placement_failed=[0, 2], crop_failed=[0, 0],
    )
    assert result["donor_failed"] == [1.0, 0.0]
    assert result["placement_failed"] == [0.0, 2.0]
    assert result["total_failures"] == 3.0
    try:
        r5_diagnostics([0.5, 0.5], [1, -1], [1, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("negative diagnostic count was accepted")


def test_adaptive_pool_excludes_target_bins_without_matching_donors(tmp_path):
    dataset = _dataset(tmp_path)
    transform = AdaptiveNegativeCanvasCopyPaste(
        dataset, p=1.0, donor_policy="larger", target_max_size=16.0, seed=42,
    )
    transform._build_pool()
    assert transform.scale_bin_values[3]
    assert not transform.donor_feasible_bin_values[3]


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
    assert out["r5_original_gt_mask"].tolist() == [False]
