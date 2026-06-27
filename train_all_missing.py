import os
import sys
import subprocess
import csv
import argparse
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
from ultralytics import YOLO

print("ultralytics path:", ultralytics.__file__)

subprocess.run(
    [
        "python",
        "/marimo/yolo_code/prepare_dataset.py",
        "--root", "/marimo/data",
        "--out-dir", "datasets/varroa_yolo",
    ],
    check=True,
)

def find_best_weights(root: Path) -> list[Path]:
    """Find trained best.pt files across known Ultralytics output roots."""
    train_roots = [
        root / "runs/detect/runs/detect/yolo_related/runs/train",
        root / "runs/detect/yolo_related/runs/train",
        root / "runs/detect/train",
        Path("runs/detect/yolo_related/runs/train"),
        Path("runs/detect/train"),
    ]
    best_paths = []
    seen = set()
    for train_root in train_roots:
        for best_path in train_root.glob("*/weights/best.pt"):
            resolved = best_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                best_paths.append(best_path)
    return sorted(best_paths)


def run_test_inference(root: Path, data_test: str) -> Path | None:
    """Evaluate all discovered best.pt checkpoints on the test split and save a summary CSV."""
    test_root = root / "runs/detect/yolo_related/runs/test"
    best_paths = find_best_weights(root)

    print(f"Found {len(best_paths)} models")
    if not best_paths:
        return None

    results = []
    for best_path in best_paths:
        run_name = best_path.parents[1].name

        print("\n" + "=" * 60)
        print(f"Testing: {run_name}")
        print(f"Weight: {best_path}")
        print("=" * 60)

        model = YOLO(str(best_path))
        metrics = model.val(
            data=data_test,
            split="test",
            imgsz=640,
            batch=16,
            device="cuda",
            conf=0.001,
            iou=0.5,
            project=str(test_root),
            name=run_name,
            exist_ok=True,
        )

        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)

        results.append((run_name, map50, map5095, precision, recall))

        print(f"mAP50:    {map50:.4f}")
        print(f"mAP50-95: {map5095:.4f}")
        print(f"Precision:{precision:.4f}")
        print(f"Recall:   {recall:.4f}")

    print("\n\n===== SUMMARY =====")
    results = sorted(results, key=lambda x: x[1], reverse=True)

    for run_name, map50, map5095, precision, recall in results:
        print(
            f"{run_name:45s} | "
            f"mAP50={map50:.4f} | "
            f"mAP50-95={map5095:.4f} | "
            f"P={precision:.4f} | "
            f"R={recall:.4f}"
        )

    test_root.mkdir(parents=True, exist_ok=True)
    summary_path = test_root / "test_summary_missing.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
        writer.writerows(results)
    print(f"\nSaved test summary: {summary_path}")
    return test_root


def upload_runs_to_hf(root: Path, part_name: str, hf_token: str | None = None, repo_id: str | None = None) -> None:
    """Upload train/test outputs to Hugging Face when HF_TOKEN is available."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not set; skipping Hugging Face upload.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed; skipping Hugging Face upload.")
        return

    repo_id = repo_id or os.environ.get("HF_REPO_ID", f"duyle2408/varroa-yolo-{part_name}-missing")
    folder_path = root / "runs/detect/yolo_related/runs"
    if not folder_path.exists():
        print(f"Upload folder does not exist: {folder_path}")
        return

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    print(f"Uploading {folder_path} to https://huggingface.co/datasets/{repo_id} (excluding .pt files)")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset", ignore_patterns=["*.pt"])
    else:
        api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset", ignore_patterns=["*.pt"])
    print("Hugging Face upload complete.")


data_path = "/marimo/data/datasets/varroa_yolo/varroa.yaml"

parser = argparse.ArgumentParser()
parser.add_argument(
    "part",
    nargs="?",
    default="1",
    choices=("1", "2", "3"),
    help="Phần models để chạy (1, 2 hoặc 3 cho 3 máy).",
)
parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
args = parser.parse_args()
part = args.part

# Danh sách các mô hình nguyên bản còn thiếu từ train_all_full.py.
# train_all_full.py đã có n/s/l cho YOLOv5, YOLOv8, YOLOv10, YOLO11
# và t/s/c cho YOLOv9. Script này bổ sung các scale còn thiếu có sẵn
# trong Ultralytics: m/x cho v5, v8, v10; m/e cho v9.
# Riêng YOLO11 chỉ chạy m/x vì n/s/l đã có trong train_all_full.py.
# YOLOv9 không có weight x chuẩn trong repo Ultralytics này.
all_models = [
    # YOLOv5 (Sử dụng bản 'u' - updated cho tương thích Ultralytics)
    "yolov5mu.pt", "yolov5xu.pt",
    # YOLOv8
    "yolov8m.pt", "yolov8x.pt",
    # YOLOv9
    "yolov9m.pt", "yolov9e.pt",
    # YOLOv10
    "yolov10m.pt", "yolov10x.pt",
    # YOLO11
    "yolo11m.pt", "yolo11x.pt",
]

model_parts = {
    "1": [
        "yolov5mu.pt", "yolov5xu.pt",
        "yolov8m.pt",
    ],
    "2": [
        "yolov9m.pt", "yolov9e.pt",
        "yolov10m.pt", "yolov10x.pt",
    ],
    "3": [
        "yolov8x.pt",
        "yolo11m.pt", "yolo11x.pt",
    ],
}
models_to_run = model_parts[part]
print(f"[*] ĐANG CHẠY MÁY {part} (Phần {part}/3): {len(models_to_run)} models")

print(f"[*] Danh sách: {models_to_run}")

seeds = [42, 43, 44]

for model_name in models_to_run:
    for seed in seeds:
        print("\n" + "="*50)
        print(f"[*] Bắt đầu training với model: {model_name} | Seed: {seed}")
        print("="*50 + "\n")
        
        # Khởi tạo mô hình mới mỗi lần để tránh rò rỉ state
        model = YOLO(model_name)
        
        # Tên run sẽ là tên file bỏ đuôi .pt, ví dụ: yolov8n_seed42
        run_name = f"{model_name.replace('.pt', '')}_seed{seed}"
        
        # Train mô hình với settings default (như yêu cầu), 100 epoch, early stop 20
        model.train(
            data=data_path,
            epochs=100,
            patience=20,     # Early stopping = 20 epochs
            imgsz=640,
            batch=8,
            workers=4,
            device="cuda",
            seed=seed,
            project=str(ROOT / "runs/detect/yolo_related/runs/train"),
            name=run_name,
        )

print("\n[*] Hoàn thành training tất cả các mô hình. Đang chạy test inference...")
test_output_root = run_test_inference(ROOT, data_path)

if test_output_root is not None:
    upload_runs_to_hf(ROOT, f"baselines-missing-part{part}", hf_token=args.hf_token, repo_id=args.hf_repo_id)
