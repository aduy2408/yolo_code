#!/usr/bin/env python3
"""Train YOLOv8 configs with fewer than 2M params using WIoU."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import torch


MARIMO_ROOT = Path("/marimo/yolo_code")
ROOT = MARIMO_ROOT.resolve() if MARIMO_ROOT.exists() else Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = ROOT / "models_related" / "ultralytics"

MODEL_SCALE = "n"
PARAM_LIMIT = 2_000_000
EPOCHS = 200
IMGSZ = 640
BATCH = 16
WORKERS = 4
DEVICE = "cuda"
PATIENCE = 30
SEEDS = (42, 69)

DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))
DATASET_OUT_DIR = ROOT / "datasets" / "varroa_yolo"
DATA_PATH = DATASET_OUT_DIR / "varroa.yaml"
CONFIG_ROOT = ROOT / "models_related" / "models_config" / "yolov8"

PROJECT = ROOT / "runs" / "detect" / "yolo_related" / "runs" / "train_under_2m_wiou"
TEST_PROJECT = ROOT / "runs" / "detect" / "yolo_related" / "runs" / "test_under_2m_wiou"
GENERATED_CONFIG_DIR = Path("/tmp") / "varroa_under_2m_wiou_scaled_configs"


def prefer_local_ultralytics() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            del sys.modules[name]


prefer_local_ultralytics()

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import DEFAULT_CFG_DICT  # noqa: E402


def prepare_data() -> None:
    from misc.prepare_dataset import prepare_dataset

    print(f"Preparing dataset from {DATA_ROOT} -> {DATASET_OUT_DIR}")
    prepare_dataset(str(DATA_ROOT), str(DATASET_OUT_DIR))


def candidate_configs(exclude_tried: bool) -> list[Path]:
    configs = []
    for config in sorted(CONFIG_ROOT.rglob("yolov8*.yaml")):
        if config.stem.endswith("_train"):
            continue
        rel = config.relative_to(CONFIG_ROOT)
        if exclude_tried and "tried" in rel.parts:
            continue
        configs.append(config)
    return configs


def config_for_scale(config: Path, model_scale: str) -> Path:
    text = config.read_text()
    out_lines = []
    inserted = False

    for line in text.splitlines():
        if line.startswith("scale:"):
            out_lines.append(f"scale: {model_scale}")
            inserted = True
            continue

        out_lines.append(line)
        if not inserted and line.startswith("nc:"):
            out_lines.append(f"scale: {model_scale}")
            inserted = True

    if not inserted:
        out_lines.insert(0, f"scale: {model_scale}")

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rel_stem = "_".join(config.relative_to(CONFIG_ROOT).with_suffix("").parts)
    scaled_config = GENERATED_CONFIG_DIR / f"{rel_stem}_{model_scale}{config.suffix}"
    scaled_config.write_text("\n".join(out_lines) + "\n")
    return scaled_config


def run_name_for(config: Path, model_scale: str, seed: int | None = None) -> str:
    rel = config.relative_to(CONFIG_ROOT).with_suffix("")
    name = f"{'_'.join(rel.parts)}_{model_scale}"
    return f"{name}_seed{seed}" if seed is not None else name


def write_selected_csv(rows: list[dict[str, str]]) -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    path = PROJECT / "selected_configs.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "generated_yaml", "params", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved config selection CSV: {path}")


def discover_under_limit(args: argparse.Namespace) -> list[tuple[Path, Path, int]]:
    selected = []
    rows = []
    forward_device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    for config in candidate_configs(args.exclude_tried):
        model_yaml = config_for_scale(config, args.scale)
        rel_config = str(config.relative_to(ROOT))
        try:
            model = YOLO(str(model_yaml))
            params = sum(p.numel() for p in model.model.parameters())
            status = "selected" if params < args.param_limit else "skipped"
            reason = "" if status == "selected" else f">= {args.param_limit}"

            if status == "selected":
                model.model.to(forward_device).eval()
                with torch.inference_mode():
                    model.model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=forward_device))
                selected.append((config, model_yaml, params))
                print(f"SELECT {rel_config}: params={params}")
            else:
                print(f"SKIP   {rel_config}: params={params}")
        except Exception as exc:
            params = ""
            status = "error"
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            print(f"ERROR  {rel_config}: {reason}")

        rows.append(
            {
                "config": rel_config,
                "generated_yaml": str(model_yaml),
                "params": str(params),
                "status": status,
                "reason": reason,
            }
        )

    write_selected_csv(rows)
    return selected


def selected_shard(configs: list[tuple[Path, Path, int]], args: argparse.Namespace) -> list[tuple[Path, Path, int]]:
    if args.num_machines < 1:
        raise ValueError("--num-machines must be >= 1")
    if not 0 <= args.machine_index < args.num_machines:
        raise ValueError("--machine-index must be in [0, num-machines)")

    shard = [item for i, item in enumerate(configs) if i % args.num_machines == args.machine_index]
    print(f"\nSelected {len(shard)}/{len(configs)} configs for shard {args.machine_index}/{args.num_machines}:")
    for config, _, params in shard:
        print(f"  - {config.relative_to(CONFIG_ROOT)} ({params} params)")
    return shard


def train_config(config: Path, model_yaml: Path, seed: int, args: argparse.Namespace) -> None:
    run_name = run_name_for(config, args.scale, seed)

    print("\n" + "=" * 60)
    print(f"Training: {run_name}")
    print(f"Seed:     {seed}")
    print(f"Config:   {model_yaml}")
    print("=" * 60)

    model = YOLO(str(model_yaml))
    model.load(f"yolov8{args.scale}.pt", smart_transfer=True)
    model.train(
        data=str(DATA_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        seed=seed,
        bbox_iou_loss="wiou",
        wiou_monotonous=False,
        project=str(PROJECT),
        name=run_name,
    )


def find_best_weights(configs: list[tuple[Path, Path, int]], args: argparse.Namespace) -> list[Path]:
    run_names = {run_name_for(config, args.scale, seed) for config, _, _ in configs for seed in args.seeds}
    return sorted(path for path in PROJECT.glob("*/weights/best.pt") if path.parents[1].name in run_names)


def run_test_inference(configs: list[tuple[Path, Path, int]], args: argparse.Namespace) -> Path | None:
    best_paths = find_best_weights(configs, args)
    print(f"\nFound {len(best_paths)} trained checkpoints for testing")
    if not best_paths:
        return None

    rows = []
    for best_path in best_paths:
        run_name = best_path.parents[1].name

        print("\n" + "=" * 60)
        print(f"Testing: {run_name}")
        print(f"Weight:  {best_path}")
        print("=" * 60)

        model = YOLO(str(best_path))
        metrics = model.val(
            data=str(DATA_PATH),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            iou=0.5,
            project=str(TEST_PROJECT),
            name=run_name,
            exist_ok=True,
        )

        row = (
            run_name,
            float(metrics.box.map50),
            float(metrics.box.map),
            float(metrics.box.mp),
            float(metrics.box.mr),
        )
        rows.append(row)
        print(f"mAP50={row[1]:.4f} | mAP50-95={row[2]:.4f} | P={row[3]:.4f} | R={row[4]:.4f}")

    rows.sort(key=lambda x: x[1], reverse=True)
    TEST_PROJECT.mkdir(parents=True, exist_ok=True)
    summary_path = TEST_PROJECT / f"test_summary_shard{args.machine_index}_of_{args.num_machines}.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
        writer.writerows(rows)

    print(f"\nSaved test summary: {summary_path}")
    return TEST_PROJECT


def upload_runs_to_hf(args: argparse.Namespace) -> None:
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.no_upload:
        print("Skipping Hugging Face upload (--no-upload).")
        return
    if not hf_token:
        print("HF_TOKEN is not set; skipping Hugging Face upload.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed; skipping Hugging Face upload.")
        return

    repo_id = args.hf_repo_id or os.environ.get("HF_REPO_ID", "duyle2408/varroa-yolo-under-2m-wiou")
    folder_path = ROOT / "runs" / "detect" / "yolo_related" / "runs"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--param-limit", type=int, default=PARAM_LIMIT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--exclude-tried", action="store_true", help="Skip archived configs under tried/.")
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only discover configs and run dummy forwards.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
    parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading train/test outputs to Hugging Face.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("ultralytics path:", ultralytics.__file__)
    print("bbox_iou_loss:", "bbox_iou_loss" in DEFAULT_CFG_DICT)
    print("wiou_monotonous:", "wiou_monotonous" in DEFAULT_CFG_DICT)

    if GENERATED_CONFIG_DIR.exists():
        shutil.rmtree(GENERATED_CONFIG_DIR)

    configs = selected_shard(discover_under_limit(args), args)
    if args.check_only:
        return

    if not args.skip_prepare_data:
        prepare_data()
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_PATH}")

    for config, model_yaml, _ in configs:
        for seed in args.seeds:
            train_config(config, model_yaml, seed, args)

    if run_test_inference(configs, args) is not None:
        upload_runs_to_hf(args)


if __name__ == "__main__":
    main()
