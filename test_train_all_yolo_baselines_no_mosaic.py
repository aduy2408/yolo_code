from __future__ import annotations

import train_all_yolo_baselines_no_mosaic as runner


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
    source = open("train_all_yolo_baselines_no_mosaic.py", encoding="utf-8").read()
    assert "mosaic=0.0" in source
    assert "close_mosaic=0" in source
    assert "model_from_baseline_yaml" in source


def test_all_standard_metrics_are_split_qualified() -> None:
    source = open("train_all_yolo_baselines_no_mosaic.py", encoding="utf-8").read()
    for key in ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95"):
        assert key in source


def test_two_server_sharding_is_disjoint_and_complete() -> None:
    all_jobs = set(runner.selected_jobs(list(runner.DATASETS), list(runner.MODELS), [42, 43, 44], 0, 1))
    shards = [
        set(runner.selected_jobs(list(runner.DATASETS), list(runner.MODELS), [42, 43, 44], index, 2))
        for index in (0, 1)
    ]
    assert len(all_jobs) == 45
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == all_jobs
