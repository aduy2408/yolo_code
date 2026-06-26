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

shrinkage_values = [0.05, 0.1]

for shrinkage in shrinkage_values:
    model = YOLO(str(model_yaml))
    model.load(f"yolov8{MODEL_SCALE}.pt", smart_transfer=True)
    
    shrink_str = str(shrinkage).replace(".", "")
    run_name = new_config_name.replace(".yaml", "") + f"_bcon005_shrink{shrink_str}"

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
        # EXPERIMENTAL: boundary-aware contrastive localization ablation.
        boundary_contrast=0.05,
        boundary_levels=2,
        boundary_ring=1.0,
        boundary_samples=16,
        boundary_tau=0.2,
        boundary_shrinkage=shrinkage,
        project=str(ROOT / "runs/detect/yolo_related/runs/train"),
        name=run_name,
    )
