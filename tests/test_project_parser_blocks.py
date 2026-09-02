from pathlib import Path

from project_ultralytics import load_project_model


ROOT = Path(__file__).resolve().parents[1]


def test_load_project_model_resolves_cbam_block_yaml() -> None:
    path = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_block.yaml"
    model = load_project_model(path, task="detect", verbose=False)
    names = [layer.__class__.__name__ for layer in model.model.model]
    assert "C2fCBAM" in names or "C3CBAM" in names


def test_load_project_model_resolves_pconv_yaml() -> None:
    path = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_pconv_gap.yaml"
    model = load_project_model(path, task="detect", verbose=False)
    assert any(layer.__class__.__name__ == "C2f_PConv" for layer in model.model.model)
