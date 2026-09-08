#!/usr/bin/env python3
"""Run R1/R2/R3 DMM P2 ablations with OACP and mosaic disabled."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRA = ROOT / "models_related/ultralytics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from misc.prepare_levir_ship import prepare
from project_ultralytics.parser import load_project_model, project_parser
from utils.marimo_ops import require_training_context

CONFIGS = {
    "r1_dmm_lite": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_dmm_lite.yaml",
    "r2_kvca_dmm_lite": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_kvca_dmm_lite.yaml",
    "r3_kvca_dmm_gated": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_kvca_dmm_gated.yaml",
}
REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv")
COMPLETE = (*REQUIRED, "evaluation_metrics.json", "manifest.json")


def _has(root: Path, names: tuple[str, ...]) -> bool:
    return all((root / name).is_file() for name in names)


def _seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(run: Path, data: Path, args: argparse.Namespace) -> None:
    metrics = {}
    for split in ("val", "test"):
        model = load_project_model(run / "weights/best.pt", verbose=False)
        result = model.val(data=str(data), split=split, imgsz=args.imgsz, batch=args.batch_size,
                           device=args.device, workers=args.workers, plots=False, iou=0.5,
                           project=str(run / "evaluation"), name=split, exist_ok=True)
        metrics.update({f"{split}/{k}": float(v) for k, v in (result.results_dict or {}).items()})
        metrics[f"{split}/AP75"] = float(result.box.map75)
    (run / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _upload(run: Path, name: str, repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    prefix = f"runs/{name}"
    api.upload_folder(folder_path=str(run), path_in_repo=prefix, repo_id=repo_id, repo_type="dataset")
    remote = {x.rfilename for x in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=prefix, recursive=True)
              if hasattr(x, "rfilename")}
    missing = {f"{prefix}/{item}" for item in COMPLETE} - remote
    if missing:
        raise RuntimeError(f"Upload verification failed for {name}: {sorted(missing)}")
    marker = run / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": repo_id, "remote_prefix": prefix}, indent=2) + "\n")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{prefix}/upload_complete.json",
                    repo_id=repo_id, repo_type="dataset")


def run(args: argparse.Namespace) -> None:
    require_training_context(hf_repo_id=args.hf_repo_id)
    data = prepare(args.data_root, args.dataset_root / "levir_ship_yolo", args.split_seed)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, config in CONFIGS.items():
        run_dir = args.project / name
        run_dir.mkdir(parents=True, exist_ok=True)
        if _has(run_dir, COMPLETE) and (run_dir / "upload_complete.json").is_file():
            continue
        _seed(args.seed)
        os.environ["YOLO_CONTEXT_AUG"] = "oacp"
        manifest = {
                "experiment": name, "config": str(config), "augmentation": "oacp",
                "mosaic": 0.0, "close_mosaic": 0, "deterministic": args.deterministic,
                "commit_sha": sha, "seed": args.seed, "split_seed": args.split_seed,
                "split": ["val", "test"], "nms_iou": 0.5, "epochs": args.epochs,
                "patience": args.patience, "hf_repo_id": args.hf_repo_id,
            }
        manifest_path = run_dir / "manifest.json"
        old_manifest = {}
        if manifest_path.is_file():
            try:
                old_manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                old_manifest = {}
        if old_manifest.get("commit_sha") != sha or old_manifest.get("mosaic") != 0.0:
            manifest["resumed_from_manifest"] = old_manifest.get("commit_sha")
            manifest["resumed_at_commit"] = sha
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        elif not manifest_path.is_file():
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if not _has(run_dir, REQUIRED):
            model = load_project_model(config, verbose=False)
            model.load("yolov8n.pt", smart_transfer=True)
            from ultralytics.nn import tasks
            with project_parser(tasks):
                model.train(data=str(data), epochs=args.epochs, patience=args.patience,
                            imgsz=args.imgsz, batch=args.batch_size, device=args.device,
                            workers=args.workers, amp=args.amp, seed=args.seed,
                            deterministic=args.deterministic, project=str(args.project),
                            name=name, exist_ok=True, mosaic=0.0, close_mosaic=0)
        if not _has(run_dir, REQUIRED):
            raise FileNotFoundError(run_dir)
        if not (run_dir / "evaluation_metrics.json").is_file():
            _evaluate(run_dir, data, args)
        if not _has(run_dir, COMPLETE):
            raise FileNotFoundError(run_dir)
        _upload(run_dir, name, args.hf_repo_id)
        print(f"COMPLETE {name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
