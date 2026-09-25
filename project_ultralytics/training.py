"""Project training hooks for selecting custom loss adapters.

The hook patches only a model instance at the project boundary. It never edits
Ultralytics source files and keeps the default upstream criterion available.
"""

from __future__ import annotations

from types import MethodType
from functools import partial
from typing import Any

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.detect.train import DetectionTrainer

from .detection_loss_adapter import FactorizedTALDetectionLoss, P2SlotsDetectionLoss
from .parser import project_parser, project_runtime

LOSS_ADAPTERS: dict[str, type] = {
    "upstream": v8DetectionLoss,
    "default": v8DetectionLoss,
    "ftal": FactorizedTALDetectionLoss,
    "factorized_tal": FactorizedTALDetectionLoss,
    "p2_slots": P2SlotsDetectionLoss,
    "p2slots": P2SlotsDetectionLoss,
}


class ProjectDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that reinstalls a project loss on reconstructed models."""

    def __init__(self, *args, loss_adapter: str = "ftal", **kwargs):
        self.project_loss_adapter = loss_adapter
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        install_loss_adapter(model, self.project_loss_adapter)
        return model


def get_loss_adapter(name: str) -> type:
    """Resolve a project loss adapter name to its criterion class."""
    key = str(name).strip().lower()
    try:
        return LOSS_ADAPTERS[key]
    except KeyError as exc:
        known = ", ".join(sorted(LOSS_ADAPTERS))
        raise ValueError(f"Unknown loss adapter {name!r}. Choose from: {known}") from exc


def _unwrap_detection_model(model: Any) -> Any:
    """Return an Ultralytics DetectionModel from either a YOLO wrapper or model."""
    candidate = model if hasattr(model, "init_criterion") else getattr(model, "model", None)
    if not hasattr(candidate, "init_criterion"):
        raise TypeError("Expected a YOLO wrapper or Ultralytics DetectionModel with init_criterion()")
    return candidate


def install_loss_adapter(model: Any, name: str = "ftal") -> Any:
    """Install a project loss adapter on one model instance and return that model.

    The original bound method is saved on the instance so ``name='upstream'`` can
    restore the clean default. This is intentionally instance-local, making it
    safe for experiments that share the installed upstream package.
    """
    detection_model = _unwrap_detection_model(model)
    adapter = get_loss_adapter(name)
    if adapter is FactorizedTALDetectionLoss and bool(getattr(detection_model, "end2end", False)):
        raise ValueError("The FTAL adapter currently supports standard v8 detection, not end2end loss")

    original = getattr(detection_model, "_project_original_init_criterion", None)
    if original is None:
        original = detection_model.init_criterion
        detection_model._project_original_init_criterion = original

    if adapter is v8DetectionLoss:
        detection_model.init_criterion = original
    else:
        detection_model.init_criterion = MethodType(lambda self: adapter(self), detection_model)
    detection_model._project_loss_adapter = adapter.__name__
    return model


def train_with_loss_adapter(model: Any, *, loss_adapter: str = "ftal", **train_kwargs: Any):
    """Install a project loss adapter and delegate to the wrapper's ``train``."""
    install_loss_adapter(model, loss_adapter)
    train = getattr(model, "train", None)
    if not callable(train):
        raise TypeError("Expected a YOLO wrapper with train()")
    from ultralytics.nn import tasks

    # Ultralytics' trainer reconstructs DetectionModel from the YAML. Keep the
    # project parser active for that second construction as well.
    with project_parser(tasks), project_runtime():
        if loss_adapter in {"upstream", "default"}:
            return train(**train_kwargs)
        trainer = partial(ProjectDetectionTrainer, loss_adapter=loss_adapter)
        return train(trainer=trainer, **train_kwargs)
