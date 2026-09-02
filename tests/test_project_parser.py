from pathlib import Path

import torch

from project_ultralytics import load_project_model


ROOT = Path(__file__).resolve().parents[1]


def test_load_project_model_resolves_weighted_add_yaml() -> None:
    model = load_project_model(ROOT / "tests/assets/project_weighted_add.yaml", task="detect", verbose=False)
    assert any(layer.__class__.__name__ == "WeightedAdd" for layer in model.model.model)
    outputs = model.model(torch.zeros(1, 3, 64, 64))
    assert outputs is not None


def test_load_project_model_resolves_kv_and_channel_attention_yaml() -> None:
    path = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_surgical_a_p3_context.yaml"
    model = load_project_model(path, task="detect", verbose=False)
    names = [layer.__class__.__name__ for layer in model.model.model]
    assert "KVCompressedAttention" in names
    assert "ChannelAttention" in names
