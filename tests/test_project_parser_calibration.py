from pathlib import Path

import pytest

from project_ultralytics import load_project_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_full_self_attention.yaml",
        "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_probe_context.yaml",
        "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_repdw5_gap.yaml",
    ],
)
def test_load_project_model_resolves_attention_and_calibration_yaml(relative_path: str) -> None:
    model = load_project_model(ROOT / relative_path, task="detect", verbose=False)
    assert len(model.model.model) > 0
