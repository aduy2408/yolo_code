from pathlib import Path

import torch

from project_ultralytics import load_project_model


ROOT = Path(__file__).resolve().parents[1]


def test_load_project_model_resolves_weighted_add_yaml() -> None:
    model = load_project_model(ROOT / "tests/assets/project_weighted_add.yaml", task="detect", verbose=False)
    assert any(layer.__class__.__name__ == "WeightedAdd" for layer in model.model.model)
    outputs = model.model(torch.zeros(1, 3, 64, 64))
    assert outputs is not None
