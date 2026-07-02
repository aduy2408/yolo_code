"""Train, test, and upload the YOLOv8 P3-only edge Varroa configs."""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("YOLO_CODE_ROOT", "/marimo/yolo_code")).resolve()
LOCAL = ROOT / "models_related" / "ultralytics"

sys.path.insert(0, str(LOCAL))

for key in list(sys.modules.keys()):
    if key == "ultralytics" or key.startswith("ultralytics."):
        del sys.modules[key]

import ultralytics
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT

print("ultralytics path:", ultralytics.__file__)
print("bbox_iou_loss:", "bbox_iou_loss" in DEFAULT_CFG_DICT)
print("wiou_monotonous:", "wiou_monotonous" in DEFAULT_CFG_DICT)

PREPARE_DATASET = ROOT / "prepare_dataset.py"
if not PREPARE_DATASET.exists():
    PREPARE_DATASET = ROOT / "misc" / "prepare_dataset.py"

subprocess.run(
    [
        "python",
        str(PREPARE_DATASET),
        "--root",
        "/marimo/data",
        "--out-dir",
        "datasets/varroa_yolo",
    ],
    check=True,
)

MODEL_SCALE = "n"
CONFIG_DIR = ROOT / "models_related" / "models_config" / "yolov8"
DATA_PATH = "/marimo/data/datasets/varroa_yolo/varroa.yaml"
TRAIN_PROJECT = ROOT / "runs/detect/yolo_related/runs/train"
TEST_PROJECT = ROOT / "runs/detect/yolo_related/runs/test_p3_edge"

P3_EDGE_CONFIGS = [
    "yolov8_varroa_compare_baseline_p3_api_boxgrad_ensimam.yaml",
    "yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_ensimam.yaml",
    "yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_pooling.yaml",
]


def scaled_config_name(config_name: str, scale: str) -> str:
    """Return the temporary scale-specific YAML name used for a training run."""
    return config_name.replace("yolov8_", f"yolov8{scale}_", 1)


def find_best_weights(run_names: list[str]) -> list[Path]:
    """Find best.pt files for the requested P3-only runs."""
    best_paths = []
    for run_name in run_names:
        best = TRAIN_PROJECT / run_name / "weights" / "best.pt"
        if best.exists():
            best_paths.append(best)
        else:
            print(f"Missing best.pt, skipping test: {best}")
    return best_paths


def run_test_inference(run_names: list[str], data_test: str) -> Path | None:
    """Evaluate trained P3-only checkpoints on the test split and save a summary CSV."""
    best_paths = find_best_weights(run_names)
    print(f"\nFound {len(best_paths)} P3-only edge models")
    if not best_paths:
        return None

    results = []
    for best_path in best_paths:
        run_name = best_path.parents[1].name

        print("\n" + "=" * 60)
        print(f"Testing: {run_name}")
        print(f"Weight: {best_path}")
        print("=" * 60)

        metrics = YOLO(str(best_path)).val(
            data=data_test,
            split="test",
            imgsz=640,
            batch=16,
            device="cuda",
            conf=0.001,
            iou=0.5,
            project=str(TEST_PROJECT),
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

    results = sorted(results, key=lambda row: row[1], reverse=True)
    print("\n\n===== P3 EDGE SUMMARY =====")
    for run_name, map50, map5095, precision, recall in results:
        print(
            f"{run_name:60s} | "
            f"mAP50={map50:.4f} | "
            f"mAP50-95={map5095:.4f} | "
            f"P={precision:.4f} | "
            f"R={recall:.4f}"
        )

    TEST_PROJECT.mkdir(parents=True, exist_ok=True)
    summary_path = TEST_PROJECT / "test_summary_p3_edge.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
        writer.writerows(results)
    print(f"\nSaved test summary: {summary_path}")
    return TEST_PROJECT


def upload_runs_to_hf(hf_token: str | None = None, repo_id: str | None = None) -> None:
    """Upload P3-only edge train/test outputs to Hugging Face when HF_TOKEN is available."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not set; skipping Hugging Face upload.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed; skipping Hugging Face upload.")
        return

    folder_path = ROOT / "runs/detect/yolo_related/runs"
    if not folder_path.exists():
        print(f"Upload folder does not exist: {folder_path}")
        return

    repo_id = repo_id or os.environ.get("HF_REPO_ID", f"duyle2408/varroa-yolo-p3-edge-yolov8{MODEL_SCALE}")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    print(f"Uploading {folder_path} to https://huggingface.co/datasets/{repo_id}")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
    else:
        api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
    print("Hugging Face upload complete.")


parser = argparse.ArgumentParser()
parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"), help="YOLOv8 model scale.")
parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
parser.add_argument("--no-test", action="store_true", help="Skip test split evaluation after training.")
parser.add_argument("--no-upload", action="store_true", help="Skip Hugging Face upload after testing.")
args = parser.parse_args()
MODEL_SCALE = args.scale

run_names = []
print(f"[*] ĐANG CHẠY P3-ONLY EDGE YOLOv8{MODEL_SCALE}: {len(P3_EDGE_CONFIGS)} configs")

for config_name in P3_EDGE_CONFIGS:
    print("\n" + "=" * 50)
    print(f"[*] Bắt đầu training với config: {config_name}")
    print("=" * 50 + "\n")

    original_yaml = CONFIG_DIR / config_name
    new_config_name = scaled_config_name(config_name, MODEL_SCALE)
    model_yaml = CONFIG_DIR / new_config_name
    shutil.copy(original_yaml, model_yaml)

    model = YOLO(str(model_yaml))
    model.load(f"yolov8{MODEL_SCALE}.pt", smart_transfer=True)

    run_name = new_config_name.replace(".yaml", "")
    run_names.append(run_name)

    model.train(
        data=DATA_PATH,
        epochs=200,
        imgsz=640,
        batch=16,
        workers=4,
        device="cuda",
        patience=40,
        bbox_iou_loss="wiou",
        wiou_monotonous=False,
        project=str(TRAIN_PROJECT),
        name=run_name,
    )

if not args.no_test:
    test_output_root = run_test_inference(run_names, DATA_PATH)
    if test_output_root is not None and not args.no_upload:
        upload_runs_to_hf(hf_token=args.hf_token, repo_id=args.hf_repo_id)
