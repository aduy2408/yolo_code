import argparse
import os
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


def upload_runs_to_hf(root: Path, hf_token: str | None = None, repo_id: str | None = None) -> None:
    """Upload LQM train outputs to Hugging Face when HF_TOKEN is available."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not set; skipping Hugging Face upload.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed; skipping Hugging Face upload.")
        return

    repo_id = repo_id or os.environ.get("HF_REPO_ID", f"duyle2408/varroa-yolo-star-lqm-yolov8{MODEL_SCALE}")
    folder_path = root / "runs/detect/yolo_related/runs"
    if not folder_path.exists():
        print(f"Upload folder does not exist: {folder_path}")
        return

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    print(f"Uploading {folder_path} to https://huggingface.co/datasets/{repo_id}")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
    else:
        api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
    print("Hugging Face upload complete.")


parser = argparse.ArgumentParser()
parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
args = parser.parse_args()

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

upload_runs_to_hf(ROOT, hf_token=args.hf_token, repo_id=args.hf_repo_id)
