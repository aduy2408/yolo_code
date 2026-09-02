from pathlib import Path

from project_ultralytics import load_project_model


ROOT = Path(__file__).resolve().parents[1]


def test_load_project_model_resolves_input_cue_stem_yaml() -> None:
    path = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_input_cue_template.yaml"
    model = load_project_model(path, task="detect", verbose=False)
    assert model.model.model[0].__class__.__name__ == "InputCueConv"
