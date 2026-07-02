#!/usr/bin/env python3
"""Train VFL TAL alpha=0 ablations and periodically upload runs to Hugging Face.

The Hugging Face token is intentionally not stored in this file. Pass it via
HF_TOKEN/HUGGINGFACE_HUB_TOKEN or --hf-token when running the script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path


DEFAULT_ROOT = Path("/marimo/yolo_code")
DEFAULT_DATA = Path("/marimo/data/datasets/varroa_yolo/varroa.yaml")
DEFAULT_REPO_ID = "duyle2408/yolov8n_shuffled_dataset_run"
MODEL_SCALE = "n"

CONFIGS = {
    "b6": "yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b6.yaml",
    "b4": "yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b4.yaml",
    "b2": "yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b2.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Repo root on the training machine.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO dataset YAML.")
    parser.add_argument("--configs", nargs="+", default=["b6", "b4", "b2"], choices=tuple(CONFIGS), help="Ablations to run.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run directories.")
    parser.add_argument("--prepare-data", action="store_true", help="Run prepare_dataset.py before training.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not smart-transfer yolov8n.pt before training.")
    parser.add_argument("--no-upload", action="store_true", help="Disable Hugging Face uploads.")
    parser.add_argument("--upload-every-min", type=float, default=30.0, help="Periodic upload interval in minutes.")
    parser.add_argument("--upload-folder", type=Path, default=None, help="Folder to upload. Defaults to ROOT/runs.")
    parser.add_argument("--hf-token", default=None, help="HF token. Prefer HF_TOKEN/HUGGINGFACE_HUB_TOKEN env vars.")
    parser.add_argument("--hf-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--hf-private", action="store_true", help="Create target dataset repo as private.")
    return parser.parse_args()


def setup_local_ultralytics(root: Path) -> None:
    local = root / "models_related" / "ultralytics"
    sys.path.insert(0, str(local))
    sys.path.insert(0, str(root))
    for key in list(sys.modules):
        if key == "ultralytics" or key.startswith("ultralytics."):
            del sys.modules[key]


def maybe_prepare_data(root: Path) -> None:
    prepare = root / "prepare_dataset.py"
    subprocess.run(
        ["python", str(prepare), "--root", "/marimo/data", "--out-dir", "datasets/varroa_yolo"],
        check=True,
    )


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    return args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def upload_runs_to_hf(
    folder_path: Path,
    repo_id: str,
    hf_token: str | None,
    private: bool = False,
    label: str = "manual",
) -> None:
    if not hf_token:
        print(f"[hf:{label}] HF token is not set; skipping upload.", flush=True)
        return
    if not folder_path.exists():
        print(f"[hf:{label}] Upload folder does not exist: {folder_path}", flush=True)
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(f"[hf:{label}] huggingface_hub is not installed; skipping upload.", flush=True)
        return

    api = HfApi(token=hf_token)
    print(f"[hf:{label}] Uploading {folder_path} to https://huggingface.co/datasets/{repo_id}", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
    else:
        api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
    print(f"[hf:{label}] Upload complete.", flush=True)


class PeriodicUploader:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.folder_path = args.upload_folder or (args.root / "runs")
        self.hf_token = resolve_hf_token(args)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.args.no_upload or self.args.upload_every_min <= 0:
            return
        self.thread = threading.Thread(target=self._loop, name="hf-periodic-upload", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        interval = max(60.0, float(self.args.upload_every_min) * 60.0)
        while not self.stop_event.wait(interval):
            self.upload(label="periodic")

    def upload(self, label: str) -> None:
        if self.args.no_upload:
            return
        if not self.lock.acquire(blocking=False):
            print(f"[hf:{label}] Previous upload still running; skipping this tick.", flush=True)
            return
        try:
            upload_runs_to_hf(
                folder_path=self.folder_path,
                repo_id=self.args.hf_repo_id,
                hf_token=self.hf_token,
                private=self.args.hf_private,
                label=label,
            )
        except Exception as exc:  # keep training alive if HF upload fails
            print(f"[hf:{label}] Upload failed: {exc!r}", flush=True)
        finally:
            self.lock.release()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def make_scaled_config(root: Path, config_name: str) -> Path:
    config_dir = root / "models_related" / "models_config" / "yolov8"
    src = config_dir / config_name
    if not src.exists():
        raise FileNotFoundError(src)
    dst = config_dir / config_name.replace("yolov8_", f"yolov8{MODEL_SCALE}_")
    if src != dst:
        shutil.copy(src, dst)
    return dst


def train_one(args: argparse.Namespace, key: str) -> None:
    from ultralytics import YOLO

    config_name = CONFIGS[key]
    model_yaml = make_scaled_config(args.root, config_name)
    run_name = model_yaml.stem

    print("\n" + "=" * 80, flush=True)
    print(f"Training {key}: {model_yaml}", flush=True)
    print(f"Run name: {run_name}", flush=True)
    print("=" * 80, flush=True)

    model = YOLO(str(model_yaml))
    if not args.no_pretrained:
        model.load(f"yolov8{MODEL_SCALE}.pt", smart_transfer=True)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        project=str(args.root / "runs/detect/yolo_related/runs/train"),
        name=run_name,
        exist_ok=args.exist_ok,
    )


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.upload_folder = (args.upload_folder or (args.root / "runs")).resolve()

    setup_local_ultralytics(args.root)
    if args.prepare_data:
        maybe_prepare_data(args.root)

    import ultralytics
    from ultralytics.utils import DEFAULT_CFG_DICT

    print(f"ultralytics path: {ultralytics.__file__}", flush=True)
    print(f"tal_alpha in DEFAULT_CFG_DICT: {'tal_alpha' in DEFAULT_CFG_DICT}", flush=True)
    print(f"upload folder: {args.upload_folder}", flush=True)
    print(f"hf repo: {args.hf_repo_id}", flush=True)

    uploader = PeriodicUploader(args)
    uploader.start()
    try:
        for key in args.configs:
            train_one(args, key)
            uploader.upload(label=f"after-{key}")
    finally:
        uploader.stop()
        uploader.upload(label="final")


if __name__ == "__main__":
    main()
