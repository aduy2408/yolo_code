from __future__ import annotations

from pathlib import Path

from train_scripts import train_all_yolo_baselines_no_mosaic as runner
from evaluate_test.size_bucket_evaluator import AREA_LABELS, AREA_RANGES, BUCKET_LABELS, IOU_THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]


def test_requested_models_datasets_and_seeds() -> None:
    assert tuple(runner.MODELS) == ("yolov5", "yolov8", "yolov9", "yolov10", "yolov11")
    assert runner.DATASETS == ("varroa", "tinyperson", "levirship")
    assert runner.SEEDS == (42, 43, 44)
    assert len(runner.MODELS) * len(runner.DATASETS) * len(runner.SEEDS) == 45


def test_default_training_settings_disable_mosaic() -> None:
    args = runner.parse_args(["--hf-repo-id", "duyle2408/yolo-baselines-no-mosaic-runs"])
    assert args.seeds == [42, 43, 44]
    assert args.datasets == list(runner.DATASETS)
    assert args.models == list(runner.MODELS)
    assert runner.SPLIT_SEED == 42
    source = (ROOT / "train_scripts/train_all_yolo_baselines_no_mosaic.py").read_text(encoding="utf-8")
    assert "mosaic=0.0" in source
    assert "close_mosaic=0" in source
    assert "model_from_baseline_yaml" in source
    assert runner.OPTIMIZER == "auto"
    assert "optimizer=OPTIMIZER" in source
    assert 'optimizer_expected": "MuSGD' in source
    assert "patch_musgd_noncontiguous" in source


def test_all_standard_metrics_are_split_qualified() -> None:
    source = (ROOT / "train_scripts/train_all_yolo_baselines_no_mosaic.py").read_text(encoding="utf-8")
    for key in ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95"):
        assert key in source


def test_size_bucket_protocol_matches_tinyperson_ranges() -> None:
    assert AREA_LABELS == ("all", "tiny", "tiny1", "tiny2", "tiny3", "small", "medium", "reasonable")
    assert AREA_RANGES == (
        (1**2, 1e5**2),
        (1**2, 20**2),
        (1**2, 8**2),
        (8**2, 12**2),
        (12**2, 20**2),
        (20**2, 32**2),
        (32**2, 96**2),
        (32**2, 1e5**2),
    )
    assert IOU_THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
    assert BUCKET_LABELS == ("Tiny1", "Tiny2", "Tiny3", "Small", "Medium")


def test_non_tinyperson_protocol_is_explicitly_size_bucketed() -> None:
    source = (ROOT / "train_scripts/train_all_yolo_baselines_no_mosaic.py").read_text(encoding="utf-8")
    size_source = (ROOT / "evaluate_test/size_bucket_evaluator.py").read_text(encoding="utf-8")
    assert '"test_size/AP50-{' in size_source
    assert '"test_size/AP-{' in size_source
    assert "Ultralytics native test split plus TinyBenchmark area buckets" in source


def test_two_server_sharding_is_disjoint_and_complete() -> None:
    all_jobs = set(runner.selected_jobs(list(runner.DATASETS), list(runner.MODELS), [42, 43, 44], 0, 1))
    shards = [
        set(runner.selected_jobs(list(runner.DATASETS), list(runner.MODELS), [42, 43, 44], index, 2))
        for index in (0, 1)
    ]
    assert len(all_jobs) == 45
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == all_jobs
