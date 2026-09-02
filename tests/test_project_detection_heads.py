from pathlib import Path

import pytest

from project_ultralytics import load_project_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path, variant",
    [
        ("models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_kvca_clsonly.yaml", "kvca"),
        ("models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_clsonly.yaml", "cbam"),
    ],
)
def test_load_project_detect_cls_attention_head(relative_path: str, variant: str) -> None:
    model = load_project_model(ROOT / relative_path, task="detect", verbose=False)
    head = model.model.model[-1]
    assert head.__class__.__name__ == "DetectClsAttention"
    assert head.attn_type == variant


def test_load_project_p2_nudfl_head() -> None:
    model = load_project_model(ROOT / "tests/assets/project_p2_nudfl.yaml", task="detect", verbose=False)
    head = model.model.model[-1]
    assert head.__class__.__name__ == "P2NUDFLDetect"
    assert head.p2_dfl_bins.numel() == 16
