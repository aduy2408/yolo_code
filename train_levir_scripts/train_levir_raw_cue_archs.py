#!/usr/bin/env python3
"""Train raw-cue architecture experiments #3, #4, #6a, #6b on LEVIR-Ship seed 42.

#3 DedicatedCueSlots        — direct injection, no learned mixing
#4 DetachedResidualFusion   — stop-grad residual + L1 recon aux supervision
#6a SplitChannelDetect       — channel split control (no aux supervision)
#6b GTChannelSpecialization  — channel split + D8 GT cue aux supervision
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misc.prepare_levir_ship import prepare


def local_ultralytics() -> None:
    local = ROOT / "models_related/ultralytics"
    if (local / "ultralytics/__init__.py").is_file() and str(local) not in sys.path:
        sys.path.insert(0, str(local))


CFG_DIR = ROOT / "models_related/models_config/yolov8/levir"

VARIANTS = {
    "yolov8n_p2_dedicated_cue_slots":    CFG_DIR / "yolov8n_p2_dedicated_cue_slots.yaml",    # #3
    "yolov8n_p2_detached_residual":      CFG_DIR / "yolov8n_p2_detached_residual.yaml",      # #4
    "yolov8n_p2_split_channel":          CFG_DIR / "yolov8n_p2_split_channel.yaml",          # #6a
    "yolov8n_p2_gt_channel_spec":        CFG_DIR / "yolov8n_p2_gt_channel_spec.yaml",        # #6b
}

SEEDS = [42]


def completed(run_dir: Path) -> bool:
    return all(
        (run_dir / f).is_file()
        for f in ("weights/best.pt", "weights/last.pt", "results.csv", "evaluation_metrics.json")
    )


def evaluate_run(model, data_yaml: str, run_dir: Path) -> dict:
    best_pt = run_dir / "weights/best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Missing best.pt: {run_dir}")

    val_res = model.val(data=data_yaml, split="val", iou=0.5, save=False, verbose=False)
    test_res = model.val(data=data_yaml, split="test", iou=0.5, save=False, verbose=False)

    metrics = {
        "nms_iou": 0.5,
        "val": {
            "mp": float(val_res.results_dict.get("metrics/precision(B)", 0.0)),
            "mr": float(val_res.results_dict.get("metrics/recall(B)", 0.0)),
            "map50": float(val_res.results_dict.get("metrics/mAP50(B)", 0.0)),
            "map75": float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        },
        "test": {
            "mp": float(test_res.results_dict.get("metrics/precision(B)", 0.0)),
            "mr": float(test_res.results_dict.get("metrics/recall(B)", 0.0)),
            "map50": float(test_res.results_dict.get("metrics/mAP50(B)", 0.0)),
            "map75": float(test_res.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        },
    }
    with open(run_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def generate_summary(project_dir: Path) -> dict:
    summary_data = {}
    md_lines = [
        "# Raw-Cue Architecture Experiments #3/#4/#6a/#6b Runs Summary",
        "",
        "| Variant | NMS IoU | VAL mAP50 | VAL mAP50-95 | TEST mAP50 | TEST mAP50-95 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for d in sorted(project_dir.iterdir()):
        if d.is_dir():
            mf = d / "evaluation_metrics.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                summary_data[d.name] = m
                val = m.get("val", {})
                test = m.get("test", {})
                md_lines.append(
                    f"| {d.name} | 0.5 | {val.get('map50', 0):.4f} | {val.get('map75', 0):.4f}"
                    f" | **{test.get('map50', 0):.4f}** | {test.get('map75', 0):.4f} |"
                )

    (project_dir / "summary.json").write_text(json.dumps(summary_data, indent=2))
    (project_dir / "summary.md").write_text("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))
    return summary_data


def run(args: argparse.Namespace) -> None:
    local_ultralytics()

    for seed in SEEDS:
        ds_dir = args.dataset_root / f"levir_ship_yolo_seed{seed}"
        data_root = args.data_root if (args.data_root / "All Images").is_dir() else Path("/marimo/LevirShipData")
        data_yaml = prepare(data_root, ds_dir, seed)

        for variant_name, yaml_cfg in VARIANTS.items():
            # Optionally run only a subset via --variants flag
            if args.variants and variant_name not in args.variants:
                continue

            run_name = f"{variant_name}_seed{seed}"
            run_dir = args.project / run_name
            if completed(run_dir):
                print(f"[SKIP] Already completed: {run_name}")
                continue

            last = run_dir / "weights/last.pt"
            from ultralytics import YOLO

            if last.is_file():
                print(f"[RESUME] {run_name}")
                model = YOLO(str(last))
                model.train(resume=True)
            else:
                print(f"[START] {run_name} with {yaml_cfg.name}")
                model = YOLO(str(yaml_cfg))
                model.train(
                    data=str(data_yaml),
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch_size,
                    device=args.device,
                    workers=args.workers,
                    patience=args.patience,
                    seed=seed,
                    deterministic=True,
                    project=str(args.project),
                    name=run_name,
                    exist_ok=True,
                )

            best_model = YOLO(str(run_dir / "weights/best.pt"))
            print(f"[EVAL] {run_name} nms_iou=0.5")
            metrics = evaluate_run(best_model, str(data_yaml), run_dir)
            print(json.dumps(metrics, indent=2))

    generate_summary(args.project)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",    type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project",      type=Path, default=ROOT / "runs/levir_raw_cue_archs")
    parser.add_argument("--epochs",       type=int,  default=100)
    parser.add_argument("--imgsz",        type=int,  default=512)
    parser.add_argument("--batch-size",   type=int,  default=8)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--workers",      type=int,  default=4)
    parser.add_argument("--patience",     type=int,  default=0)
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help="Which variants to run (default: all). E.g. --variants yolov8n_p2_split_channel"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
