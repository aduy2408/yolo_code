import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from project_ultralytics.r5 import (
    R5ProbeCallback,
    R5StateCheckpointCallback,
    RecallEvaluator,
    RecallProfile,
    ScaleBinSpec,
    ScaleDifficultyController,
    SharedProbabilityState,
)
from project_ultralytics.r5.offline_profile import build_offline_profile
from train_scripts.train_r5_matrix import _build_components, _fixed_r5_overrides


def test_recall_evaluator_is_class_aware_and_one_to_one():
    evaluator = RecallEvaluator(ScaleBinSpec((0, 8, 12, 16, 20)))
    profile = evaluator.evaluate_records(
        [
            {
                "pred_boxes": [[0, 0, 4, 4], [10, 10, 20, 20]],
                "pred_scores": [0.9, 0.8],
                "pred_classes": [0, 1],
                "gt_boxes": [[0, 0, 4, 4], [10, 10, 20, 20]],
                "gt_classes": [0, 0],
            }
        ],
        method="offline_recall",
        source_split="train",
    )
    assert profile.gt_count == (1, 1, 0, 0)
    assert profile.tp_count == (1, 0, 0, 0)
    assert profile.global_recall == 0.5


def test_recall_profile_round_trip_and_smoothed_difficulty(tmp_path):
    profile = RecallProfile(
        method="offline_recall",
        edges=(0.0, 8.0, 12.0),
        gt_count=(10, 2),
        tp_count=(5, 0),
        iou_threshold=0.5,
        confidence_threshold=0.01,
        source_split="train",
        image_count=4,
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    restored = RecallProfile.load(path)
    assert restored.to_dict() == profile.to_dict()
    difficulty = restored.difficulty(kappa=10.0, gamma=1.0)
    assert np.all((difficulty >= 0.0) & (difficulty <= 1.0))
    assert difficulty[1] > difficulty[0]


def test_offline_profile_keeps_raw_difficulty_for_single_controller_gamma():
    payload = build_offline_profile(
        [{
            "pred_boxes": [],
            "pred_scores": [],
            "pred_classes": [],
            "gt_boxes": [[0, 0, 4, 4]],
            "gt_classes": [0],
        }],
        kappa=0.0,
        gamma=2.0,
    )
    assert payload["difficulty_gamma"] == 1.0
    assert payload["difficulty"] == [1.0, 0.0, 0.0, 0.0]
    controller = ScaleDifficultyController(num_bins=4, gamma=2.0, exploration=0.0)
    probabilities = controller.probabilities(difficulty=payload["difficulty"])
    assert probabilities[0] == 1.0


def test_offline_overrides_preserve_published_profile_probabilities():
    shared = SharedProbabilityState(4, [0.6, 0.2, 0.1, 0.1])
    args = Namespace(seed=42, workers=2)
    overrides = _fixed_r5_overrides(args, shared)
    assert np.allclose(overrides["r5_scale_probabilities"], shared.snapshot())


def test_probe_runner_predictor_uses_current_training_model(tmp_path):
    class Wrapper:
        def __init__(self):
            self.model = object()
            self.seen = []

        def predict(self, source, **kwargs):
            self.seen.append(self.model)
            return [{
                "pred_boxes": [[0, 0, 4, 4]],
                "pred_scores": [0.9],
                "pred_classes": [0],
            }]

    args = Namespace(
        r5_mode="online_probe_recall",
        profile_path=None,
        probe_records=None,
        ema_beta=0.0,
        gamma=1.0,
        exploration=0.0,
        warmup=0,
        log_path=None,
        probe_iou=0.5,
        probe_confidence=0.01,
        nms_iou=0.5,
        imgsz=64,
        device="cpu",
        probe_every=1,
        kappa=0.0,
    )
    wrapper = Wrapper()
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps([{"path": "image.png", "coord_system": "original_xyxy", "gt_boxes": [[0, 0, 4, 4]], "gt_classes": [0]}]))
    args.probe_records = probe_path
    shared = SharedProbabilityState(4, [0.25] * 4)
    _, probe, _ = _build_components(args, wrapper, shared)
    current_model = object()
    probe.records = [{"path": "image.png", "coord_system": "original_xyxy", "gt_boxes": [[0, 0, 4, 4]], "gt_classes": [0]}]
    probe.evaluate(current_model, epoch=1)
    assert wrapper.seen == [current_model]
    assert wrapper.model is not current_model


def test_runner_rejects_probe_records_without_coordinate_contract(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps([{"path": "image.png", "gt_boxes": [], "gt_classes": []}]))
    args = Namespace(
        r5_mode="online_probe_recall",
        profile_path=None,
        probe_records=path,
        ema_beta=0.9,
        gamma=1.0,
        exploration=0.2,
        warmup=0,
        log_path=None,
        probe_iou=0.5,
        probe_confidence=0.01,
        nms_iou=0.5,
        imgsz=64,
        device="cpu",
        probe_every=1,
        kappa=10.0,
        profile_iou=0.5,
        profile_confidence=0.01,
        profile_source_split="train",
    )
    with pytest.raises(ValueError, match="coord_system"):
        _build_components(args, object(), SharedProbabilityState(4, [0.25] * 4))


def test_real_yolo_cached_predictor_is_rebound_to_current_model(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[1]
    yaml = root / "models_related" / "ultralytics" / "ultralytics" / "cfg" / "models" / "v8" / "yolov8.yaml"
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    wrapper = YOLO(str(yaml))
    wrapper.predict(image, imgsz=64, conf=0.01, device="cpu", verbose=False)
    assert wrapper.predictor is not None
    cached_model = wrapper.predictor.model
    current = YOLO(str(yaml))
    records = tmp_path / "probe.json"
    records.write_text(json.dumps([{"coord_system": "original_xyxy", "gt_boxes": [], "gt_classes": []}]))
    args = Namespace(
        r5_mode="online_probe_recall",
        profile_path=None,
        probe_records=records,
        ema_beta=0.9,
        gamma=1.0,
        exploration=0.2,
        warmup=0,
        log_path=None,
        probe_iou=0.5,
        probe_confidence=0.01,
        nms_iou=0.5,
        imgsz=64,
        device="cpu",
        probe_every=1,
        kappa=10.0,
        profile_iou=0.5,
        profile_confidence=0.01,
        profile_source_split="train",
    )
    _, probe, _ = _build_components(args, wrapper, SharedProbabilityState(4, [0.25] * 4))
    probe.predict_fn(current.model, {"image": image, "coord_system": "original_xyxy", "gt_boxes": [], "gt_classes": []})
    assert wrapper.predictor is None
    assert cached_model is not current.model


def test_probe_callback_updates_shared_state_and_restores_model_mode(tmp_path):
    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    records = [
        {
            "image": "image.png",
            "gt_boxes": [[0, 0, 4, 4]],
            "gt_classes": [0],
        }
    ]
    callback = R5ProbeCallback(
        evaluator=RecallEvaluator(ScaleBinSpec((0, 8, 12, 16, 20))),
        controller=ScaleDifficultyController(num_bins=4, beta=0.0, exploration=0.0),
        shared_state=SharedProbabilityState(4, [0.25] * 4),
        records=records,
        predict_fn=lambda model, record: {
            "pred_boxes": [[0, 0, 4, 4]],
            "pred_scores": [0.9],
            "pred_classes": [0],
        },
        probe_every=1,
        log_path=tmp_path / "probe.jsonl",
    )
    model = Model()
    result = callback.evaluate(model, epoch=10)
    assert result.profile.tp_count[0] == 1
    assert model.training is True
    assert callback.probe_count == 1
    assert json.loads((tmp_path / "probe.jsonl").read_text().splitlines()[0])["epoch"] == 10


def test_probe_callback_state_round_trip_restores_count_and_last_result(tmp_path):
    records = [{"gt_boxes": [[0, 0, 4, 4]], "gt_classes": [0]}]
    kwargs = dict(
        evaluator=RecallEvaluator(ScaleBinSpec((0, 8, 12, 16, 20))),
        controller=ScaleDifficultyController(num_bins=4, beta=0.0, exploration=0.0),
        shared_state=SharedProbabilityState(4, [0.25] * 4),
        records=records,
        predict_fn=lambda model, record: {
            "pred_boxes": [[0, 0, 4, 4]],
            "pred_scores": [0.9],
            "pred_classes": [0],
        },
        probe_every=1,
        log_path=tmp_path / "probe.jsonl",
    )
    original = R5ProbeCallback(**kwargs)
    original.evaluate(object(), epoch=3)
    restored = R5ProbeCallback(**kwargs)
    restored.load_state_dict(original.state_dict())
    assert restored.probe_count == 1
    assert restored.last_result is not None
    assert restored.last_result.epoch == 3


def test_r5_checkpoint_callback_round_trips_state(tmp_path):
    state = SharedProbabilityState(4, [0.1, 0.2, 0.3, 0.4])
    callback = R5StateCheckpointCallback(state)

    class Trainer:
        save_dir = tmp_path

    callback.on_train_end(Trainer())
    assert (tmp_path / "r5_state.json").is_file()
    restored = SharedProbabilityState(4, [0.25] * 4)
    loader = R5StateCheckpointCallback(restored)
    loader.on_pretrain_routine_start(Trainer())
    assert np.allclose(restored.snapshot(), state.snapshot())
