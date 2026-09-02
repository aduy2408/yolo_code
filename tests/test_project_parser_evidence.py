from pathlib import Path

import pytest

from project_ultralytics import load_project_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "models_related/models_config/yolov8/levir/yolov8n_p2_gradient_isolated_evidence.yaml",
        "models_related/models_config/yolov8/levir/yolov8n_p2_multicue.yaml",
        "models_related/models_config/yolov8/levir/yolov8n_p2_scale_disappearance_evidence.yaml",
    ],
)
def test_load_project_model_resolves_image_aware_evidence_yaml(relative_path: str) -> None:
    model = load_project_model(ROOT / relative_path, task="detect", verbose=False)
    assert len(model.model.model) > 0
    assert hasattr(model.model, "_project_original_predict_once")
