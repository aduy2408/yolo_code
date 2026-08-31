import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "utils/measure_yolo_baselines_complexity.py"
SPEC = importlib.util.spec_from_file_location("measure_yolo_baselines_complexity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_checkpoint_filename():
    assert MODULE.checkpoint_filename("yolo11n", 44) == "train/yolo11n_seed44/weights/best.pt"


def test_summarize_checks_parameter_consistency():
    rows = [
        {"model": "yolov8n", "seed": seed, "layers": 10, "parameters": 100, "gflops": 2.0}
        for seed in (42, 43, 44)
    ]
    assert MODULE.summarize(rows) == [
        {
            "model": "yolov8n",
            "seeds": [42, 43, 44],
            "layers": 10,
            "parameters": 100,
            "parameters_m": 0.0001,
            "gflops": 2.0,
            "gflops_sample_std": 0.0,
            "identical_parameter_count_across_seeds": True,
        }
    ]
