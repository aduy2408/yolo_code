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
from ultralytics.nn.tasks import DetectionModel

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
    """DetectionTrainer that reconstructs a model with a project criterion."""

    def __init__(self, *args, loss_adapter: str = "ftal", project_model_args: dict[str, Any] | None = None, **kwargs):
        self.project_loss_adapter = loss_adapter
        self.project_model_args = dict(project_model_args or {})
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ProjectDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            loss_adapter=self.project_loss_adapter,
        )
        model.project_model_args = dict(self.project_model_args)
        if weights:
            model.load(weights)
        return model


def get_loss_adapter(name: str) -> type:
    """Resolve a project loss adapter name to its criterion class."""
    key = str(name).strip().lower()
    try:
        return LOSS_ADAPTERS[key]
    except KeyError as exc:
        known = ", ".join(sorted(LOSS_ADAPTERS))
        raise ValueError(f"Unknown loss adapter {name!r}. Choose from: {known}") from exc


class ProjectDetectionModel(DetectionModel):
    """Pickle-safe DetectionModel carrying its project loss selection."""

    def __init__(self, cfg, ch=3, nc=None, verbose=True, loss_adapter: str = "ftal"):
        self.project_loss_adapter = str(loss_adapter)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return get_loss_adapter(self.project_loss_adapter)(self)


def _project_init_criterion(self):
    """Importable instance hook retained for explicit non-trainer use."""
    adapter = getattr(self, "_project_loss_adapter_type", None)
    if adapter is None:
        return DetectionModel.init_criterion(self)
    return adapter(self)


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

    if adapter is v8DetectionLoss:
        detection_model._project_loss_adapter_type = None
    else:
        detection_model._project_loss_adapter_type = adapter
        detection_model.init_criterion = MethodType(_project_init_criterion, detection_model)
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
        project_names = (
            "gradient_mode_balance",
            "gradient_mode_count",
            "gradient_mode_iterations",
            "gradient_mode_tiny_size",
            "gradient_mode_min_objects",
        )
        project_model_args = {
            name: train_kwargs.pop(name)
            for name in project_names
            if name in train_kwargs
        }
        trainer = partial(
            ProjectDetectionTrainer,
            loss_adapter=loss_adapter,
            project_model_args=project_model_args,
        )
        return train(trainer=trainer, **train_kwargs)
