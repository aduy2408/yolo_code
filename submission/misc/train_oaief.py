#!/usr/bin/env python3
"""Train, evaluate, and upload the matched OAIEF W1/W2 experiments."""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys
from pathlib import Path
from misc.prepare_levir_ship import prepare
from utils.marimo_ops import require_training_context
ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
EXPERIMENTS = {
    "W1": CONFIG_ROOT / "yolov8n_p2_levir_oaief_w1.yaml",
    "W2": CONFIG_ROOT / "yolov8n_p2_levir_oaief_w2.yaml",
}
TRAIN_REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv")
COMPLETE_REQUIRED = (*TRAIN_REQUIRED, "evaluation_metrics.json", "manifest.json")
def local_ultralytics(): sys.path.insert(0, str(ULTRALYTICS))
def has_files(path, names): return all((path / item).is_file() for item in names)
def seed_everything(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def evaluate(run_dir, data, args):
    local_ultralytics(); from ultralytics import YOLO
    model = YOLO(run_dir / "weights/best.pt"); metrics = {}
    for split in ("val", "test"):
        result = model.val(data=str(data), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True)
        metrics.update({f"{split}/{key}": float(value) for key, value in (result.results_dict or {}).items()})
        metrics[f"{split}/AP75"] = float(result.box.map75)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics
def upload_and_verify(run_dir, name, repo_id):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"]); api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    prefix = f"runs/{name}"; api.upload_folder(folder_path=str(run_dir), path_in_repo=prefix, repo_id=repo_id, repo_type="dataset")
    remote = {item.rfilename for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=prefix, recursive=True) if hasattr(item, "rfilename")}
    missing = {f"{prefix}/{item}" for item in COMPLETE_REQUIRED} - remote
    if missing: raise RuntimeError(f"Remote upload verification failed for {name}: {sorted(missing)}")
    (run_dir / "upload_complete.json").write_text(json.dumps({"repo_id": repo_id, "remote_prefix": prefix}, indent=2) + "\n")
def run(args):
    require_training_context(hf_repo_id=args.hf_repo_id)
    data = prepare(args.data_root, args.dataset_root / f"levir_ship_yolo_seed{args.seed}", args.seed)
    local_ultralytics(); from ultralytics import YOLO
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name, config in EXPERIMENTS.items():
        run_dir = args.project / name; run_dir.mkdir(parents=True, exist_ok=True)
        if has_files(run_dir, COMPLETE_REQUIRED) and (run_dir / "upload_complete.json").is_file(): continue
        (run_dir / "manifest.json").write_text(json.dumps({"experiment": name, "config": str(config), "commit_sha": sha, "seed": args.seed, "split": ["val", "test"], "nms_iou": 0.5, "epochs": args.epochs, "patience": args.patience, "hf_repo_id": args.hf_repo_id}, indent=2) + "\n")
        seed_everything(args.seed)
        if not has_files(run_dir, TRAIN_REQUIRED):
            model = YOLO(str(config)); model.load("yolov8n.pt", smart_transfer=True)
            model.train(data=str(data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch_size,
                device=args.device, workers=args.workers, amp=args.amp, seed=args.seed, deterministic=True,
                project=str(args.project), name=name, exist_ok=True)
        if not has_files(run_dir, TRAIN_REQUIRED): raise FileNotFoundError(f"Incomplete training artifacts: {run_dir}")
        if not (run_dir / "evaluation_metrics.json").is_file(): evaluate(run_dir, data, args)
        if not has_files(run_dir, COMPLETE_REQUIRED): raise FileNotFoundError(f"Incomplete evaluation artifacts: {run_dir}")
        upload_and_verify(run_dir, name, args.hf_repo_id); print(f"COMPLETE {name}", flush=True)
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--dataset-root", type=Path, required=True); p.add_argument("--project", type=Path, required=True); p.add_argument("--hf-repo-id", required=True); p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=0); p.add_argument("--imgsz", type=int, default=512); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--device", default="0"); p.add_argument("--workers", type=int, default=4); p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True); return p.parse_args()
if __name__ == "__main__": run(parse_args())
