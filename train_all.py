import os
import sys
import subprocess
import shutil
from pathlib import Path

ROOT = Path("/marimo/yolo_code").resolve()
LOCAL = ROOT / "models_related" / "ultralytics"

# Đưa local ultralytics lên đầu sys.path
sys.path.insert(0, str(LOCAL))

# Xóa toàn bộ module ultralytics đã cache
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]

# Import lại từ đầu
import ultralytics
from ultralytics.utils import DEFAULT_CFG_DICT

print("ultralytics path:", ultralytics.__file__)
print("bbox_iou_loss:", "bbox_iou_loss" in DEFAULT_CFG_DICT)
print("wiou_monotonous:", "wiou_monotonous" in DEFAULT_CFG_DICT)
subprocess.run(
    [
        "python",
        "/marimo/yolo_code/prepare_dataset.py",
        "--root", "/marimo/data",
        "--out-dir", "datasets/varroa_yolo",
    ],
    check=True,
)
from ultralytics import YOLO

# ==========================================
# CẤU HÌNH SIZE MÔ HÌNH Ở ĐÂY (n, s, m, l, x)
MODEL_SCALE = "n"
# ==========================================

# Đường dẫn tới thư mục chứa các file config
config_dir = Path("/marimo/yolo_code/models_related/models_config")
data_path = "/marimo/data/datasets/varroa_yolo/varroa.yaml"

COMBO_CONFIGS = [
    "yolov8_varroa_repensimam_kvheads_concat.yaml",
    "yolov8_varroa_repensimam_kvheads_concat_asf.yaml",
    "yolov8_varroa_repensimam_kvheads_bifpn.yaml",
    "yolov8_varroa_repensimam_kvheads_bifpn_asf.yaml",
    "yolov8_varroa_repensimam_kvheads_bifpn_asf_p2p4.yaml",
]

KVPLACE_CONFIGS = [
    "yolov8_varroa_kvheads_repc2f.yaml",
    "yolov8_varroa_kvheads_repc2f_ensimam_p5.yaml",
    "yolov8_varroa_kvheads_repc2f_ensimam_p2p3.yaml",
    "yolov8_varroa_kvheads_repc2f_bifpn.yaml",
    "yolov8_varroa_kvheads_repc2f_bifpn_ensimam_p2p3.yaml",
    "yolov8_varroa_kvheads_repc2f_bifpn_asf.yaml",
    "yolov8_varroa_kvheads_repc2f_bifpn_asf_ensimam.yaml",
]

SPECIAL_CONFIGS = set(COMBO_CONFIGS + KVPLACE_CONFIGS)

# Lấy danh sách tất cả các file yaml gốc (chỉ lấy các file bắt đầu bằng yolov8_ để bỏ qua các file yolov8x_ sinh ra)
all_configs = sorted([
    f for f in os.listdir(config_dir) 
    if f.endswith('.yaml') and f.startswith('yolov8_') and f not in SPECIAL_CONFIGS
])

# Chia thành 2 nửa
part1_configs = all_configs[:7]
part2_configs = all_configs[7:]

# Kiểm tra đối số dòng lệnh để xác định đang chạy phần nào (mặc định là phần 1)
part = sys.argv[1] if len(sys.argv) > 1 else "1"

if part == "1":
    configs_to_run = part1_configs
    print(f"[*] ĐANG CHẠY PHẦN 1: {len(configs_to_run)} configs")
elif part == "2":
    configs_to_run = part2_configs
    print(f"[*] ĐANG CHẠY PHẦN 2: {len(configs_to_run)} configs")
elif part == "combo":
    configs_to_run = COMBO_CONFIGS
    print(f"[*] ĐANG CHẠY COMBO YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")
elif part == "combo1":
    configs_to_run = COMBO_CONFIGS[:2]
    print(f"[*] ĐANG CHẠY COMBO 1 YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")
elif part == "combo2":
    configs_to_run = COMBO_CONFIGS[2:]
    print(f"[*] ĐANG CHẠY COMBO 2 YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")
elif part == "kvplace":
    configs_to_run = KVPLACE_CONFIGS
    print(f"[*] ĐANG CHẠY KVPLACE YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")
elif part == "kvplace1":
    configs_to_run = KVPLACE_CONFIGS[:3]
    print(f"[*] ĐANG CHẠY KVPLACE 1 YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")
elif part == "kvplace2":
    configs_to_run = KVPLACE_CONFIGS[3:]
    print(f"[*] ĐANG CHẠY KVPLACE 2 YOLOv8{MODEL_SCALE}: {len(configs_to_run)} configs")
else:
    print(
        "Vui lòng truyền đối số là 1, 2, combo, combo1, combo2, kvplace, kvplace1 hoặc kvplace2. "
        "Ví dụ: python train_all.py kvplace1"
    )
    sys.exit(1)

# Chạy vòng lặp qua từng config
for config_name in configs_to_run:
    print("\n" + "="*50)
    print(f"[*] Bắt đầu training với config: {config_name}")
    print("="*50 + "\n")
    
    original_yaml = config_dir / config_name
    
    # Tạo tên file mới có scale tương ứng (vd: yolov8_ -> yolov8x_)
    new_config_name = config_name.replace("yolov8_", f"yolov8{MODEL_SCALE}_")
    model_yaml = config_dir / new_config_name
    
    # Tự động copy nội dung sang file yaml mới
    shutil.copy(original_yaml, model_yaml)
    
    # Truyền file có chữ 'x' vào YOLO
    model = YOLO(str(model_yaml))
    
    # Load đúng file pretrained weights theo scale
    model.load(f"yolov8{MODEL_SCALE}.pt",smart_transfer=True) 
    
    # Đặt tên run bỏ đuôi .yaml
    run_name = new_config_name.replace('.yaml', '')
    
    model.train(
        data=data_path,
        epochs=200,
        imgsz=640,
        batch=16,
        workers=4,
        device="cuda",

        patience=40,

        # box=10.0,
        # dfl=2.0,
        # cls=0.25,
        bbox_iou_loss="wiou",
        wiou_monotonous=False,

        # mosaic=0.2,
        # close_mosaic=40,
        # mixup=0.0,
        # copy_paste=0.0,
        # degrees=0.0,
        # perspective=0.0,
        # translate=0.1,
        # scale=0.25,
        # shear=0.0,

        # optimizer="AdamW",
        # lr0=0.001,
        # lrf=0.01,
        # cos_lr=True,
        # weight_decay=0.0005,
        # warmup_epochs=3,

        # project="runs/detect/yolo_related/runs/train",
        # name=run_name,
    )
