import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "utils/evaluate_levir_yolo_baselines_coco.py"
SPEC = importlib.util.spec_from_file_location("evaluate_levir_yolo_baselines_coco", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_checkpoint_filename():
    assert MODULE.checkpoint_filename("yolov8n", 43) == "train/yolov8n_seed43/weights/best.pt"


def test_aggregate_uses_sample_standard_deviation():
    rows = []
    for seed, value in ((42, 0.2), (43, 0.3), (44, 0.4)):
        metrics = {
            "map_50_95": value,
            "ap50": value,
            "ap75": value,
            "ap_small": value,
            "ap_medium": value,
            "ap_large": -1.0,
        }
        rows.append({"model": "yolov8n", "seed": seed, "validation": metrics, "test": metrics})
    summary = MODULE.aggregate(rows)[0]
    assert summary["test"]["map_50_95"] == {"mean": 0.3, "sample_std": 0.1}
    assert summary["test"]["ap_large"] == {"mean": -1.0, "sample_std": 0.0}


def test_result_paths_cover_each_model_seed_and_split():
    paths = MODULE.result_paths(["yolov8n"], [42, 43])
    assert paths == [
        "manifest.json",
        "results.json",
        "summary.json",
        "yolov8n/seed_42/metrics.json",
        "yolov8n/seed_42/validation_predictions.json",
        "yolov8n/seed_42/test_predictions.json",
        "yolov8n/seed_43/metrics.json",
        "yolov8n/seed_43/validation_predictions.json",
        "yolov8n/seed_43/test_predictions.json",
    ]
