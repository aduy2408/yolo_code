import sys
import shutil
from pathlib import Path

# Đưa local ultralytics lên sys.path
LOCAL = Path("/mnt/data/varroa/yolo_related/models_related/ultralytics").resolve()
sys.path.insert(0, str(LOCAL))

from ultralytics import YOLO

config_dir = Path("/mnt/data/varroa/yolo_related/models_related/models_config")
base_yaml = config_dir / "yolov8_varroa_p2_neck_nat_heads.yaml"
x_yaml = config_dir / "yolov8x_varroa_p2_neck_nat_heads.yaml"

# 1. Tạo file config size X
shutil.copy(base_yaml, x_yaml)

print("\n=== KHỞI TẠO MODEL SIZE X ===")
try:
    model = YOLO(str(x_yaml))
    print(f"Tổng số tham số model X: {sum(p.numel() for p in model.model.parameters()):,}")
    
    print("\n=== LOAD TRỌNG SỐ TỪ yolov8x.pt ===")
    model.load("yolov8x.pt")
except Exception as e:
    print(f"Lỗi: {e}")
finally:
    # Xoá file yaml tạm sau khi test xong
    if x_yaml.exists():
        x_yaml.unlink()

