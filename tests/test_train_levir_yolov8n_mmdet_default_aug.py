from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_levir_ship_yolov8n_mmdet_default_aug",
    ROOT / "train_levir_yolov8n_mmdet_default_aug.py",
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_mmdet_default_mapping_is_explicit_and_deterministic() -> None:
    assert runner.MMDET_DEFAULT_AUGMENTATION == {
        "mosaic": 1.0,
        "mixup": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "close_mosaic": 0,
        "erasing": 0.0,
        "auto_augment": None,
    }


def test_defaults_match_baseline_contract() -> None:
    args = runner.parse_args([])
    assert args.pretrained == "yolov8n.pt"
    assert args.epochs == 100
    assert args.imgsz == 640
    assert args.batch_size == 8
    assert args.seed == 42
    assert args.patience == 0
    assert args.upload is False


def test_dataset_yaml_is_present_and_has_all_splits() -> None:
    runner.validate_dataset(runner.DATA_YAML)
