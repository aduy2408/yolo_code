from pathlib import Path
import os
import subprocess
import sys
import textwrap

import marimo as mo

root = Path("/marimo/yolo_code")
config = root / "models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_pooling.yaml"
assert root.exists(), root
assert config.exists(), config

run_name = "yolov8n_varroa_compare_baseline_p3_api_boxgrad_edge_pooling"
log_dir = root / "runs/detect/yolo_related/runs/train" / run_name
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "train_launch.log"
script_path = root / "tmp_train_p3_pooling.py"

script_path.write_text(
    textwrap.dedent(
        f"""
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        ROOT = Path("/marimo/yolo_code").resolve()
        LOCAL = ROOT / "models_related" / "ultralytics"
        sys.path.insert(0, str(LOCAL))
        for key in list(sys.modules.keys()):
            if key == "ultralytics" or key.startswith("ultralytics."):
                del sys.modules[key]

        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import DEFAULT_CFG_DICT

        print("ultralytics path:", ultralytics.__file__, flush=True)
        print("bbox_iou_loss:", "bbox_iou_loss" in DEFAULT_CFG_DICT, flush=True)
        print("wiou_monotonous:", "wiou_monotonous" in DEFAULT_CFG_DICT, flush=True)

        prepare_dataset = ROOT / "prepare_dataset.py"
        if not prepare_dataset.exists():
            prepare_dataset = ROOT / "misc" / "prepare_dataset.py"
        print("prepare_dataset:", prepare_dataset, flush=True)
        subprocess.run(
            [
                sys.executable,
                str(prepare_dataset),
                "--root",
                "/marimo/data",
                "--out-dir",
                "datasets/varroa_yolo",
            ],
            check=True,
        )

        config_dir = ROOT / "models_related" / "models_config" / "yolov8"
        original_yaml = config_dir / "yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_pooling.yaml"
        model_yaml = config_dir / "yolov8n_varroa_compare_baseline_p3_api_boxgrad_edge_pooling.yaml"
        shutil.copy(original_yaml, model_yaml)
        print("model_yaml:", model_yaml, flush=True)

        model = YOLO(str(model_yaml))
        model.load("yolov8n.pt", smart_transfer=True)
        model.train(
            data="/marimo/data/datasets/varroa_yolo/varroa.yaml",
            epochs=200,
            imgsz=640,
            batch=16,
            workers=4,
            device="cuda",
            patience=40,
            bbox_iou_loss="wiou",
            wiou_monotonous=False,
            project=str(ROOT / "runs/detect/yolo_related/runs/train"),
            name="{run_name}",
        )
        """
    )
)

with log_path.open("ab", buffering=0) as log:
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],
        cwd=str(root),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

print("started pid=", proc.pid)
print("script=", script_path)
print("log=", log_path)
mo.status.toast(f"Started P3 pooling train PID {proc.pid}", kind="success")
