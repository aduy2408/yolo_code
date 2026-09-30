#!/usr/bin/env python3
"""Upload-required TPH-YOLOv5 queue for Varroa, TinyPerson, LevirShip, and VisDrone.

This runner intentionally uses the public upstream ``cv516Buaa/tph-yolov5``
implementation rather than the project's canonical Ultralytics baseline runner.
Dataset preparation is reused from the project so split provenance and the
TinyPerson corner-window protocol remain explicit.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPH_URL = "https://github.com/cv516Buaa/tph-yolov5.git"
TPH_COMMIT = "052dfeb375e51756f17e9ca4f96b7e3e3a7cf3c4"
SPLIT_SEED = 42
DATASETS = ("varroa", "tinyperson", "levirship", "visdrone")
IMAGE_SIZES = {"varroa": 640, "tinyperson": 640, "levirship": 512, "visdrone": 640}
DEFAULT_ROOTS = {
    "varroa": "/marimo/Varroa",
    "tinyperson": "/marimo/TinyPerson",
    "levirship": "/marimo/LevirShip/LevirShipData",
    "visdrone": "/marimo/VisDrone2019",
}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "experiment_manifest.json",
)


def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    env = {**os.environ, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": "0"}
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    result = subprocess.run(command, cwd=cwd, env=env, text=True, check=True, capture_output=capture)
    return (result.stdout + result.stderr) if capture else ""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def ensure_tph_repo(path: Path) -> None:
    if not path.exists():
        run(["git", "clone", TPH_URL, str(path)])
    run(["git", "fetch", "--depth", "1", "origin", TPH_COMMIT], cwd=path)
    run(["git", "checkout", "--detach", TPH_COMMIT], cwd=path)
    dirty = run(["git", "status", "--porcelain"], cwd=path, capture=True).strip()
    train = path / "train.py"
    text = train.read_text(encoding="utf-8")
    allowed = {"M train.py", "M models/experimental.py", "M utils/general.py", "M utils/datasets.py", "M utils/loss.py", "M utils/metrics.py", "M utils/plots.py"}
    dirty_lines = [line for line in dirty.splitlines() if "__pycache__/" not in line and not line.rstrip().endswith(".pyc") and line.strip() != "?? yolov5l.pt"]
    if dirty_lines and not (all(line.strip() in allowed for line in dirty_lines) and "getattr(opt, 'seed', 42)" in text):
        raise RuntimeError(f"TPH checkout is dirty: {path}")
    if "getattr(opt, 'seed', 42)" not in text:
        text = text.replace("init_seeds(1 + RANK)", "init_seeds(getattr(opt, 'seed', 42) + 1 + RANK)")
        marker = "parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')"
        if marker not in text:
            raise RuntimeError("Unable to patch deterministic seed option in upstream train.py")
        text = text.replace(marker, marker + "\n    parser.add_argument('--seed', type=int, default=42, help='training seed')")
        train.write_text(text, encoding="utf-8")
    compatibility = {
        path / "models/experimental.py": [("torch.load(attempt_download(w), map_location=map_location)", "torch.load(attempt_download(w), map_location=map_location, weights_only=False)")],
        path / "utils/datasets.py": [("np.int", "int"), ("if segment:", "if segment is not None and len(segment):")],
        path / "utils/loss.py": [("gj.clamp_(0, gain[3] - 1)", "gj.clamp_(0, gain[3].long() - 1)"), ("gi.clamp_(0, gain[2] - 1)", "gi.clamp_(0, gain[2].long() - 1)")],
        path / "utils/general.py": [("torch.load(f, map_location=torch.device('cpu'))", "torch.load(f, map_location=torch.device('cpu'), weights_only=False)"), ("np.int", "int")],
        path / "utils/metrics.py": [("np.trapz", "np.trapezoid")],
        path / "utils/plots.py": [("self.font.getsize(text)", "self.font.getbbox(text)[2:4]"), ("self.font.getsize(label)", "self.font.getbbox(label)[2:4]")],
        train: [("torch.load(weights, map_location=device)", "torch.load(weights, map_location=device, weights_only=False)")],
    }
    for target, replacements in compatibility.items():
        source = target.read_text(encoding="utf-8")
        for old, new in replacements:
            source = source.replace(old, new)
        target.write_text(source, encoding="utf-8")


def _ensure_tph_dataset_schema(path: Path, dataset: str) -> Path:
    text = path.read_text(encoding="utf-8")
    if "\nnc:" not in f"\n{text}":
        nc = 10 if dataset == "visdrone" else 1
        text = text.replace("names:\n", f"nc: {nc}\nnames:\n", 1)
        path.write_text(text, encoding="utf-8")
    return path


def prepare_dataset(name: str, data_root: Path, dataset_root: Path) -> Path:
    sys.path.insert(0, str(ROOT))
    if name == "varroa":
        from misc.prepare_dataset import prepare_dataset
        output = prepare_dataset(data_root, dataset_root / "varroa_split_42", gt_source="gt_one", only_positives=True, class_policy="map-3-to-1", seed=SPLIT_SEED).resolve()
    elif name == "levirship":
        from misc.prepare_levir_ship import prepare
        output = prepare(data_root, dataset_root / "levirship_split_42", SPLIT_SEED).resolve()
    elif name == "visdrone":
        from train_scripts.train_visdrone_scripts.train_all_visdrone_verifier import prepare_dataset
        output = prepare_dataset(data_root, dataset_root / "visdrone_split_42").resolve()
    else:
        import train_scripts.train_all_tinyperson as tiny
        test_dir = tiny.prepare_test_set(data_root, dataset_root)
        output = tiny.prepare_seed_dataset(data_root, dataset_root, test_dir, SPLIT_SEED).joinpath("tinyperson.yaml").resolve()
    return _ensure_tph_dataset_schema(output, name)


def patch_model_yaml(tph_root: Path, dataset: str, output: Path) -> None:
    import yaml
    config = yaml.safe_load((tph_root / "models/yolov5l-xs-tph.yaml").read_text(encoding="utf-8"))
    config["nc"] = 10 if dataset == "visdrone" else 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def write_results_csv(run_dir: Path) -> None:
    source = run_dir / "results.txt"
    target = run_dir / "results.csv"
    if target.is_file():
        return
    rows = source.read_text(encoding="utf-8", errors="replace").splitlines() if source.is_file() else []
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "line"])
        writer.writerows([["results.txt", row] for row in rows])


def parse_metrics(output: str) -> tuple[float, float]:
    matches = re.findall(r"\ball\s+\d+\s+\d+\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", output)
    if not matches:
        raise RuntimeError("TPH val.py output did not contain a split summary line")
    return float(matches[-1][0]), float(matches[-1][1])


def evaluate(tph_root: Path, run_dir: Path, data_yaml: Path, dataset: str, imgsz: int, batch: int, workers: int, device: str) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {"nms_iou": 0.5}
    for split in ("val", "test"):
        command = [sys.executable, "val.py", "--data", str(data_yaml), "--weights", str(run_dir / "weights/best.pt"), "--img", str(imgsz), "--batch-size", str(batch), "--device", device, "--task", split, "--iou-thres", "0.5", "--project", str(run_dir / "evaluation"), "--name", split, "--exist-ok", "--verbose"]
        output = run(command, cwd=tph_root, capture=True)
        ap50, map5095 = parse_metrics(output)
        metrics[f"{split}/AP50"] = ap50
        metrics[f"{split}/mAP50-95"] = map5095
    metrics["test_protocol"] = "TinyPerson official corner-window test" if dataset == "tinyperson" else "native YOLO test split"
    return metrics


def upload(repo_id: str, remote: str, run_dir: Path) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing local artifacts before upload: {missing}")
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected = {f"{remote}/{path}" for path in REQUIRED}
    missing_remote = sorted(expected - files)
    if missing_remote:
        raise RuntimeError(f"Missing remote artifacts after upload: {missing_remote}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": repo_id, "remote_prefix": remote, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo_id, repo_type="dataset")


def train_one(args: argparse.Namespace, tph_root: Path, dataset: str, seed: int, data_yaml: Path) -> None:
    run_dir = args.project / dataset / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    remote = f"runs/{dataset}/seed_{seed}"
    if (run_dir / "upload_complete.json").is_file():
        print(f"SKIP_VERIFIED {remote}", flush=True)
        return
    config = args.model_yaml or (args.dataset_root / "tph_configs" / f"yolov5l-xs-tph_{dataset}.yaml")
    patch_model_yaml(tph_root, dataset, config)
    if not (run_dir / "weights/best.pt").is_file():
        seed_everything(seed)
        command = [sys.executable, "train.py", "--img", str(IMAGE_SIZES[dataset]), "--adam", "--batch", str(args.batch_size), "--epochs", str(args.epochs), "--patience", str(args.patience), "--data", str(data_yaml), "--weights", "yolov5l.pt", "--hyp", "data/hyps/hyp.VisDrone.yaml", "--cfg", str(config), "--name", f"{dataset}_seed_{seed}", "--project", str(args.project), "--workers", str(args.workers), "--device", args.device, "--seed", str(seed), "--single-cls"]
        if dataset == "visdrone":
            command.remove("--single-cls")
        run(command, cwd=tph_root)
        produced = args.project / f"{dataset}_seed_{seed}"
        if produced != run_dir and produced.is_dir():
            if run_dir.is_dir() and not any(run_dir.iterdir()):
                run_dir.rmdir()
                shutil.move(str(produced), str(run_dir))
            else:
                for item in produced.iterdir():
                    target = run_dir / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(target))
                produced.rmdir()
    write_results_csv(run_dir)
    metrics = evaluate(tph_root, run_dir, data_yaml, dataset, IMAGE_SIZES[dataset], args.batch_size, args.workers, args.device)
    manifest = {"experiment_id": "tph_yolov5_multidataset", "dataset": dataset, "baseline": "none", "variant": "upstream TPH-YOLOv5 yolov5l-xs-tph", "source_commit": args.source_commit, "tph_upstream_commit": TPH_COMMIT, "runner": "train_scripts/train_tph_yolov5_multidataset.py", "model_yaml": str(config), "pretrained_source": "yolov5l.pt", "data_yaml": str(data_yaml), "data_root": args.data_roots[dataset], "split_seed": SPLIT_SEED, "seed": seed, "image_size": IMAGE_SIZES[dataset], "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience, "amp": True, "optimizer": "Adam", "nms_iou": 0.5, "hf_repo_id": args.hf_repo_id, "remote_prefix": remote, "test_protocol": metrics["test_protocol"], **metrics}
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upload(args.hf_repo_id, remote, run_dir)
    print(f"COMPLETE {remote}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-yaml", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--data-root", action="append", metavar="DATASET=PATH")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/tph_yolov5")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/tph_yolov5_multidataset")
    parser.add_argument("--tph-root", type=Path, default=ROOT / "vendor/tph-yolov5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.seed is not None and args.seeds is not None:
        raise ValueError("Use either --seed or --seeds, not both")
    if args.seed is not None:
        args.seeds = [args.seed]
    elif args.seeds is None:
        args.seeds = [42, 43, 44]
    if args.split_seed != SPLIT_SEED:
        raise ValueError(f"This TPH matrix requires split-seed={SPLIT_SEED}")
    roots = dict(DEFAULT_ROOTS)
    for item in args.data_root or []:
        dataset, sep, path = item.partition("=")
        if dataset not in DATASETS or not sep or not path:
            raise ValueError(f"Invalid --data-root {item!r}")
        roots[dataset] = path
    args.data_roots = roots
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.tph_root = args.tph_root.resolve()
    args.model_yaml = args.model_yaml.resolve() if args.model_yaml is not None else None
    return args


def main() -> None:
    args = parse_args()
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch")
    from utils.marimo_ops import ensure_hf_repo, require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    ensure_hf_repo(args.hf_repo_id)
    ensure_tph_repo(args.tph_root)
    for dataset in args.datasets:
        data_yaml = prepare_dataset(dataset, Path(args.data_roots[dataset]), args.dataset_root)
        for seed in args.seeds:
            train_one(args, args.tph_root, dataset, seed, data_yaml)


if __name__ == "__main__":
    main()
