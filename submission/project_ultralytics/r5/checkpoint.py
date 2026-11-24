"""Checkpoint sidecar support for R5 controller, probe, and sampler state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class R5StateCheckpointCallback:
    """Persist R5 state beside the Ultralytics run artifacts."""

    def __init__(self, *stateful: Any, filename: str = "r5_state.json") -> None:
        self.stateful = tuple(stateful)
        self.filename = filename
        self.loaded = False

    def _path(self, trainer) -> Path:
        return Path(getattr(trainer, "save_dir", ".")) / self.filename

    def state_dict(self) -> dict[str, object]:
        return {
            type(item).__name__: item.state_dict()
            for item in self.stateful
            if hasattr(item, "state_dict")
        }

    def _load(self, trainer) -> None:
        path = self._path(trainer)
        if not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in self.stateful:
            loader = getattr(item, "load_state_dict", None)
            if not callable(loader):
                continue
            key = type(item).__name__
            if key in payload:
                loader(payload[key])
        self.loaded = True

    def _save(self, trainer) -> None:
        path = self._path(trainer)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def on_pretrain_routine_start(self, trainer=None) -> None:
        if trainer is not None and not self.loaded:
            self._load(trainer)

    def on_model_save(self, trainer=None) -> None:
        if trainer is not None:
            self._save(trainer)

    def on_train_end(self, trainer=None) -> None:
        if trainer is not None:
            self._save(trainer)


__all__ = ["R5StateCheckpointCallback"]
