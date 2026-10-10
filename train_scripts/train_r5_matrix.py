#!/usr/bin/env python3
"""Unified R5-A/B/C/E training runner.

This runner owns one SharedProbabilityState and passes that same object through
actual dataset construction and the R5 feedback callbacks. It does not run a
training job during import. Live training still requires the repository's
normal environment and, for upload workflows, the Marimo gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _prefer_local_ultralytics() -> None:
    root = Path(__file__).resolve().parents[1]
    local = root / "models_related" / "ultralytics"
    if str(local) not in sys.path:
        sys.path.insert(0, str(local))


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("probe/profile records must be a list or an object with records")
    return records


def _validate_probe_records(records: list[dict[str, Any]]) -> None:
    """Require the coordinate contract used by YOLO Results.boxes.xyxy."""
    for index, record in enumerate(records):
        if record.get("coord_system") != "original_xyxy":
            raise ValueError(
                f"probe record {index} must declare coord_system='original_xyxy'"
            )


def _fixed_r5_overrides(args: argparse.Namespace, shared_state) -> dict[str, Any]:
    return {
        "copy_paste_enabled": True,
        "copy_paste_mode": "adaptive_negative_canvas",
        "copy_paste_unit": "single",
        "copy_paste_copies": 1,
        "negative_cp_p": 0.30,
        "negative_cp_donor_policy": "matched",
        "negative_cp_target_max_size": 20.0,
        "negative_cp_matched_ratio_max": 1.5,
        "negative_cp_degradation": "none",
        "copy_paste_max_trials": 30,
        "r5_scale_bin_edges": [0.0, 8.0, 12.0, 16.0, 20.0],
        # Pass the already-published distribution. The dataset builder may
        # construct a sampler with this value and publish it again, so a
        # uniform placeholder would erase R5-A's offline initialization.
        "r5_scale_probabilities": shared_state.snapshot().tolist(),
        "r5_seed": int(args.seed),
        "r5_shared_state": True,
        "r5_shared_probability_state": shared_state,
        "mosaic": 0.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "cutmix": 0.0,
        "workers": int(args.workers),
    }


def _build_components(args: argparse.Namespace, model, shared_state):
    from project_ultralytics.r5 import ScaleBinSpec, ScaleDifficultyController, TALDifficultyCollector
    from project_ultralytics.r5.checkpoint import R5StateCheckpointCallback
    from project_ultralytics.r5.online_probe import R5ProbeCallback
    from project_ultralytics.r5.online_tal import R5EpochFeedbackAdapter
    from project_ultralytics.r5.recall_evaluator import RecallEvaluator

    bin_spec = ScaleBinSpec((0.0, 8.0, 12.0, 16.0, 20.0))
    controller = None
    feedback = None
    probe = None
    stateful: list[Any] = [shared_state]
    log_path = args.log_path
    if args.r5_mode == "offline_recall":
        if args.profile_path is None:
            raise ValueError("offline_recall requires --profile-path")
        profile = json.loads(args.profile_path.read_text(encoding="utf-8"))
        from project_ultralytics.r5.recall_evaluator import RecallProfile

        loaded_profile = RecallProfile.from_dict(profile)
        expected_edges = tuple(float(value) for value in (0.0, 8.0, 12.0, 16.0, 20.0))
        if loaded_profile.edges != expected_edges:
            raise ValueError("offline profile bins do not match the R5 scale-bin protocol")
        if not np.isclose(loaded_profile.iou_threshold, args.profile_iou):
            raise ValueError("offline profile IoU threshold does not match the requested protocol")
        if not np.isclose(loaded_profile.confidence_threshold, args.profile_confidence):
            raise ValueError("offline profile confidence threshold does not match the requested protocol")
        if loaded_profile.source_split != args.profile_source_split:
            raise ValueError("offline profile source split does not match the requested protocol")
        # Profiles persist raw smoothed miss-rate. Gamma belongs to the
        # controller, where it is applied exactly once.
        difficulty = loaded_profile.difficulty(kappa=args.kappa, gamma=1.0).tolist()
        controller = ScaleDifficultyController(
            num_bins=bin_spec.num_bins,
            beta=args.ema_beta,
            gamma=args.gamma,
            exploration=args.exploration,
        )
        probabilities = controller.probabilities(difficulty=difficulty, feasible=[True] * bin_spec.num_bins)
        shared_state.update(probabilities.tolist())
        stateful.append(controller)
    elif args.r5_mode in {"online_tal_iou", "online_tal_alignment"}:
        controller = ScaleDifficultyController(
            num_bins=bin_spec.num_bins,
            beta=args.ema_beta,
            gamma=args.gamma,
            exploration=args.exploration,
        )
        collector = TALDifficultyCollector(
            bin_spec=bin_spec,
            mode="iou" if args.r5_mode == "online_tal_iou" else "alignment",
            scores_are_logits=False,
        )
        feedback = R5EpochFeedbackAdapter(
            transform=None,
            controller=controller,
            collector=collector,
            shared_state=shared_state,
            warmup_epochs=args.warmup,
            log_path=log_path,
        )
        stateful.append(feedback)
    else:
        if args.probe_records is None:
            raise ValueError("online_probe_recall requires --probe-records")
        controller = ScaleDifficultyController(
            num_bins=bin_spec.num_bins,
            beta=args.ema_beta,
            gamma=args.gamma,
            exploration=args.exploration,
        )
        records = _load_records(args.probe_records)
        _validate_probe_records(records)
        evaluator = RecallEvaluator(
            bin_spec=bin_spec,
            iou_threshold=args.probe_iou,
            confidence_threshold=args.probe_confidence,
        )

        def predict_fn(current_model, record):
            source = record.get("image")
            if source is None:
                source = record.get("path")
            if source is None:
                raise ValueError("probe records require image or path")
            wrapper_model = getattr(model, "model", None)
            try:
                # Ultralytics' public predict API is attached to the wrapper,
                # while the callback receives trainer.model. Temporarily bind
                # the wrapper to that exact current model and restore it after
                # the prediction so R5-E cannot probe stale weights.
                if wrapper_model is not current_model:
                    model.model = current_model
                # YOLO.predict reuses a cached predictor whose ``model`` is an
                # AutoBackend created during the first call. Drop that cache
                # so setup_model wraps the current training model now.
                model.predictor = None
                return model.predict(
                    source,
                    imgsz=args.imgsz,
                    conf=args.probe_confidence,
                    iou=args.nms_iou,
                    device=args.device,
                    verbose=False,
                )[0]
            finally:
                if wrapper_model is not current_model:
                    model.model = wrapper_model
                # Do not leave a predictor bound to a temporary training model.
                model.predictor = None

        probe = R5ProbeCallback(
            evaluator=evaluator,
            controller=controller,
            shared_state=shared_state,
            records=records,
            predict_fn=predict_fn,
            probe_every=args.probe_every,
            kappa=args.kappa,
            gamma=args.gamma,
            feasible=[True] * bin_spec.num_bins,
            log_path=log_path,
        )
        stateful.extend([controller, probe])
    checkpoint = R5StateCheckpointCallback(*stateful)
    return feedback, probe, checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset YAML path or dataset identifier accepted by Ultralytics")
    parser.add_argument("--model", required=True, help="YOLO model checkpoint or YAML")
    parser.add_argument("--r5-mode", required=True, choices=("offline_recall", "online_tal_iou", "online_tal_alignment", "online_probe_recall"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/r5"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-path", type=Path)
    parser.add_argument("--probe-records", type=Path)
    parser.add_argument("--probe-every", type=int, default=10)
    parser.add_argument("--probe-iou", type=float, default=0.5)
    parser.add_argument("--probe-confidence", type=float, default=0.01)
    parser.add_argument("--profile-iou", type=float, default=0.5)
    parser.add_argument("--profile-confidence", type=float, default=0.01)
    parser.add_argument("--profile-source-split", default="train")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--exploration", type=float, default=0.2)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _prefer_local_ultralytics()
    from ultralytics import YOLO
    from project_ultralytics.r5 import SharedProbabilityState
    from project_ultralytics.training import train_with_loss_adapter

    args.project = args.project.resolve()
    args.name = args.name or args.r5_mode
    args.log_path = args.log_path or args.project / args.name / "r5_feedback.jsonl"
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    shared_state = SharedProbabilityState(4, [0.25, 0.25, 0.25, 0.25])
    feedback, probe, checkpoint = _build_components(args, model, shared_state)
    overrides = _fixed_r5_overrides(args, shared_state)
    overrides.update({
        "data": str(args.dataset),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "project": str(args.project),
        "name": args.name,
        "resume": args.resume,
        "deterministic": True,
        "plots": False,
        "amp": True,
        "r5_feedback_adapter": feedback,
        "r5_probe_callback": probe,
        "r5_state_callback": checkpoint,
    })
    loss_adapter = "upstream" if args.r5_mode == "offline_recall" else "ftal"
    train_with_loss_adapter(model, loss_adapter=loss_adapter, **overrides)


if __name__ == "__main__":
    main()
