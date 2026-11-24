from __future__ import annotations

from train_scripts import train_all_visdrone_yolo_baselines as runner


def test_model_registry_includes_future_baseline_families() -> None:
    assert runner.MODELS["yolov9t"] == (
        "yolov9t.pt",
        "vendor/ultralytics_upstream/ultralytics/cfg/models/v9/yolov9t.yaml",
    )
    assert runner.MODELS["yolov10n"] == (
        "yolov10n.pt",
        "vendor/ultralytics_upstream/ultralytics/cfg/models/v10/yolov10n.yaml",
    )
    assert runner.MODELS["yolo11n"] == (
        "yolo11n.pt",
        "vendor/ultralytics_upstream/ultralytics/cfg/models/11/yolo11.yaml",
    )


def test_explicit_future_model_queue_preserves_declaration_order() -> None:
    jobs = runner.explicit_jobs(
        ["yolov9t:42,43,44", "yolov10n:42,43,44", "yolo11n:42,43,44"],
        ["mosaic"],
    )
    assert jobs == [
        ("yolov9t", 42, "mosaic"),
        ("yolov9t", 43, "mosaic"),
        ("yolov9t", 44, "mosaic"),
        ("yolov10n", 42, "mosaic"),
        ("yolov10n", 43, "mosaic"),
        ("yolov10n", 44, "mosaic"),
        ("yolo11n", 42, "mosaic"),
        ("yolo11n", 43, "mosaic"),
        ("yolo11n", 44, "mosaic"),
    ]


def test_future_baseline_defaults_keep_batch_and_split_contract() -> None:
    args = runner.parse_args(
        [
            "--data-root",
            "/marimo/VisDrone2019",
            "--hf-repo-id",
            "duyle2408/visdrone-yolov9-yolov10-yolo11-runs",
            "--job",
            "yolov9t:42,43,44",
            "--job",
            "yolov10n:42,43,44",
            "--job",
            "yolo11n:42,43,44",
            "--augmentations",
            "mosaic",
        ]
    )
    assert args.batch_size == 8
    assert args.epochs == 100
    assert args.patience == 0
    assert args.imgsz == 640
    assert args.augmentations == ["mosaic"]
