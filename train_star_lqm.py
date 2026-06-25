import sys
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/marimo/yolo_code").resolve()
LOCAL = ROOT / "models_related" / "ultralytics"

sys.path.insert(0, str(LOCAL))
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]

subprocess.run(
    [
        "python",
        str(ROOT / "prepare_dataset.py"),
        "--root",
        "/marimo/data",
        "--out-dir",
        "datasets/varroa_yolo",
    ],
    check=True,
)

from ultralytics import YOLO


MODEL_SCALE = "n"
config_dir = ROOT / "models_related" / "models_config"
data_path = "/marimo/data/datasets/varroa_yolo/varroa.yaml"

config_name = "yolov8_varroa_kvheads_repc2f_bifpn_asf_ensimam_star.yaml"
original_yaml = config_dir / config_name
new_config_name = config_name.replace("yolov8_", f"yolov8{MODEL_SCALE}_")
model_yaml = config_dir / new_config_name
shutil.copy(original_yaml, model_yaml)

model = YOLO(str(model_yaml))
model.load(f"yolov8{MODEL_SCALE}.pt", smart_transfer=True)

run_name = new_config_name.replace(".yaml", "") + "_lqm005"

model.train(
    data=data_path,
    epochs=200,
    imgsz=640,
    batch=16,
    workers=4,
    device="cuda",
    patience=40,
    bbox_iou_loss="wiou",
    wiou_monotonous=False,
    # EXPERIMENTAL: train-only Localization Quality Map supervision.
    loc_quality=0.05,
    loc_quality_levels=2,
    loc_quality_sigma=0.45,
    loc_quality_loss="mse",
    # Keep boundary contrast disabled for a clean LQM ablation.
    boundary_contrast=0.0,
    project=str(ROOT / "runs/detect/yolo_related/runs/train"),
    name=run_name,
)
