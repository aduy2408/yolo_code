#!/usr/bin/env python3
"""Measure parameters and GFLOPs for standard LEVIR-Ship YOLO baselines."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "duyle2408/levir-ship-yolo-baselines"
MODELS = ("yolov5nu", "yolov8n", "yolov9t", "yolov10n", "yolo11n")
SEEDS = (42, 43, 44)


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def checkpoint_filename(model: str, seed: int) -> str:
    return f"train/{model}_seed{seed}/weights/best.pt"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        params = [row["parameters"] for row in selected]
        flops = [row["gflops"] for row in selected]
        output.append(
            {
                "model": model,
                "seeds": [row["seed"] for row in selected],
                "layers": selected[0]["layers"],
                "parameters": params[0],
                "parameters_m": params[0] / 1e6,
                "gflops": statistics.mean(flops),
                "gflops_sample_std": statistics.stdev(flops) if len(flops) > 1 else 0.0,
                "identical_parameter_count_across_seeds": len(set(params)) == 1,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from utils.marimo_ops import require_training_context

    require_training_context(hf_repo_id=args.hf_repo_id)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    sys.path.insert(0, str(ROOT / "models_related" / "ultralytics"))
    from huggingface_hub import HfApi, hf_hub_download
    from ultralytics import YOLO

    models = comma_list(args.models)
    seeds = [int(seed) for seed in comma_list(args.seeds)]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    info = HfApi(token=token).dataset_info(args.hf_repo_id)
    remote_files = {item.rfilename for item in info.siblings}
    expected = [checkpoint_filename(model, seed) for model in models for seed in seeds]
    missing = sorted(set(expected) - remote_files)
    if missing:
        raise RuntimeError(f"Missing checkpoints: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_name in models:
        for seed in seeds:
            filename = checkpoint_filename(model_name, seed)
            checkpoint = hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=filename,
                repo_type="dataset",
                revision=info.sha,
                token=token,
            )
            model = YOLO(checkpoint)
            measured = model.info(detailed=False, verbose=True, imgsz=args.imgsz)
            if measured is None or len(measured) != 4:
                raise RuntimeError(f"Ultralytics did not return complexity for {model_name} seed {seed}")
            layers, parameters, gradients, gflops = measured
            row = {
                "model": model_name,
                "seed": seed,
                "checkpoint": filename,
                "source_revision": info.sha,
                "input_size": [args.imgsz, args.imgsz],
                "layers": int(layers),
                "parameters": int(parameters),
                "gradients": int(gradients),
                "gflops": float(gflops),
            }
            rows.append(row)
            target = args.output_dir / model_name / f"seed_{seed}"
            target.mkdir(parents=True, exist_ok=True)
            (target / "complexity.json").write_text(json.dumps(row, indent=2) + "\n")
            print(f"{model_name} seed {seed}: {parameters / 1e6:.3f} M, {gflops:.3f} GFLOPs", flush=True)
            del model

    summary = summarize(rows)
    inconsistent = [row["model"] for row in summary if not row["identical_parameter_count_across_seeds"]]
    if inconsistent:
        raise RuntimeError(f"Parameter counts differ across seeds: {inconsistent}")
    manifest = {
        "source_repo": args.hf_repo_id,
        "source_revision": info.sha,
        "models": models,
        "seeds": seeds,
        "input_size": [args.imgsz, args.imgsz],
        "method": "Ultralytics model.info/get_flops",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
