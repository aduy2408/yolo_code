
import shutil
import sys
from pathlib import Path

ROOT = Path("/marimo/yolo_code").resolve()
LOCAL = ROOT / "models_related" / "ultralytics"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCAL))
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]

data_path = ROOT / "datasets" / "varroa_yolo" / "varroa.yaml"
if not data_path.exists():
    from misc.prepare_dataset import prepare_dataset
    prepare_dataset("/marimo/data", str(ROOT / "datasets" / "varroa_yolo"))

from ultralytics import YOLO

config_dir = ROOT / "models_related" / "models_config" / "yolov8"
src = config_dir / "yolov8_varroa_compare_baseline_p2_api_boxgrad.yaml"
dst = config_dir / "yolov8n_varroa_compare_baseline_p2_api_boxgrad.yaml"
shutil.copy(src, dst)

model = YOLO(str(dst))
model.load("yolov8n.pt", smart_transfer=True)

model.train(
    data=str(data_path),
    epochs=100,
    imgsz=640,
    batch=16,
    workers=4,
    device="cuda",
    patience=20,
    bbox_iou_loss="wiou",
    wiou_monotonous=False,
    boundary_contrast=0.02,
    boundary_levels=2,
    boundary_ring=1.0,
    boundary_samples=16,
    boundary_tau=0.2,
    project=str(ROOT / "runs/detect/yolo_related/runs/train"),
    name="yolov8n_varroa_compare_baseline_p2_api_boxgrad_bcon002",
    exist_ok=True,
)
