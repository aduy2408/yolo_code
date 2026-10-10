"""Periodic current-model probe recall updates for R5-E."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .recall_evaluator import RecallEvaluator, RecallProfile


@dataclass(frozen=True)
class ProbeResult:
    epoch: int
    profile: RecallProfile
    difficulty: tuple[float, ...]
    probabilities: tuple[float, ...]


def _prediction_record(model: Any, record: Mapping[str, Any], predict_fn: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
    prediction = predict_fn(model, record)
    if isinstance(prediction, Mapping):
        return {
            **record,
            "pred_boxes": prediction.get("pred_boxes", []),
            "pred_scores": prediction.get("pred_scores", []),
            "pred_classes": prediction.get("pred_classes", []),
        }
    boxes = getattr(prediction, "boxes", None)
    if boxes is None:
        return {**record, "pred_boxes": [], "pred_scores": [], "pred_classes": []}
    return {
        **record,
        "pred_boxes": boxes.xyxy.detach().cpu().numpy(),
        "pred_scores": boxes.conf.detach().cpu().numpy(),
        "pred_classes": boxes.cls.detach().cpu().numpy(),
    }


class R5ProbeCallback:
    """Evaluate a fixed train-internal probe set and update shared probabilities."""

    def __init__(
        self,
        evaluator: RecallEvaluator,
        controller,
        shared_state,
        records: Iterable[Mapping[str, Any]],
        predict_fn: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]],
        *,
        probe_every: int = 10,
        kappa: float = 10.0,
        gamma: float = 1.0,
        feasible=None,
        log_path: str | Path | None = None,
    ) -> None:
        if probe_every < 1:
            raise ValueError("probe_every must be positive")
        self.evaluator = evaluator
        self.controller = controller
        self.shared_state = shared_state
        self.records = list(records)
        self.predict_fn = predict_fn
        self.probe_every = int(probe_every)
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.feasible = feasible
        self.log_path = Path(log_path) if log_path else None
        self.last_result: ProbeResult | None = None
        self.probe_count = 0

    def evaluate(self, model: Any, epoch: int) -> ProbeResult:
        was_training = getattr(model, "training", None)
        if hasattr(model, "eval"):
            model.eval()
        try:
            records = [_prediction_record(model, record, self.predict_fn) for record in self.records]
            profile = self.evaluator.evaluate_records(records, method="online_probe_recall", source_split="train_probe")
        finally:
            if was_training is True and hasattr(model, "train"):
                model.train()
        difficulty = profile.difficulty(kappa=self.kappa, gamma=self.gamma)
        mapping = {
            index: (float(value), 1)
            for index, value in enumerate(difficulty)
            if profile.gt_count[index] > 0
        }
        self.controller.accumulate_mapping(mapping)
        self.controller.end_epoch()
        feasible = self.feasible
        probabilities = self.controller.probabilities(feasible=feasible)
        self.shared_state.update(probabilities.tolist())
        result = ProbeResult(
            epoch=int(epoch),
            profile=profile,
            difficulty=tuple(float(value) for value in difficulty),
            probabilities=tuple(float(value) for value in probabilities),
        )
        self.last_result = result
        self.probe_count += 1
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "epoch": result.epoch,
                    "probe_count": self.probe_count,
                    "gt_count": list(profile.gt_count),
                    "tp_count": list(profile.tp_count),
                    "difficulty": list(result.difficulty),
                    "probabilities": list(result.probabilities),
                }, sort_keys=True) + "\n")
        return result

    def on_train_epoch_end(self, trainer=None) -> ProbeResult | None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        if epoch % self.probe_every != 0:
            return None
        model = getattr(trainer, "model", trainer)
        return self.evaluate(model, epoch)

    def state_dict(self) -> dict[str, object]:
        return {
            "probe_every": self.probe_every,
            "probe_count": self.probe_count,
            "last_result": None if self.last_result is None else {
                "epoch": self.last_result.epoch,
                "difficulty": list(self.last_result.difficulty),
                "probabilities": list(self.last_result.probabilities),
                "profile": self.last_result.profile.to_dict(),
            },
        }


__all__ = ["ProbeResult", "R5ProbeCallback"]
