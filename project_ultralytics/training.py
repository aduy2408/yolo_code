"""Project training hooks for selecting custom loss adapters.

The hook patches only a model instance at the project boundary. It never edits
Ultralytics source files and keeps the default upstream criterion available.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

from ultralytics.utils.loss import v8DetectionLoss

from .detection_loss_adapter import FactorizedTALDetectionLoss

LOSS_ADAPTERS: dict[str, type] = {
    "upstream": v8DetectionLoss,
    "default": v8DetectionLoss,
    "ftal": FactorizedTALDetectionLoss,
    "factorized_tal": FactorizedTALDetectionLoss,
}


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
    return train(**train_kwargs)
