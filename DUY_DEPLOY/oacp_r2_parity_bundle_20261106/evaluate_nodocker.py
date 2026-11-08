#!/usr/bin/env python3
"""Evaluate exported ONNX models with ONNX Runtime TensorRT EP, without Docker."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
DEPLOY = BUNDLE.parent
ONNX = BUNDLE / "output" / "onnx"
OUTPUT = BUNDLE / "output"


def run(command: list[str], env: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tinyperson", "levirship", "all"), default="all")
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    datasets = ("tinyperson", "levirship") if args.dataset == "all" else (args.dataset,)
    precisions = ("fp32", "fp16") if args.precision == "both" else (args.precision,)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(DEPLOY / "docker/jetson_onnx"), env.get("PYTHONPATH", "")]
    )

    for dataset in datasets:
        for precision in precisions:
            scope_root = OUTPUT / "_nodocker_models" / precision
            scope_dataset = scope_root / dataset
            for model_path in sorted((ONNX / dataset).glob(f"*/seed*/{precision}.onnx")):
                relative = model_path.relative_to(ONNX / dataset)
                link = scope_dataset / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(model_path)
            models_root = scope_dataset if dataset == "tinyperson" else scope_root
            results_root = OUTPUT / "metrics" / dataset / precision
            predictions_root = OUTPUT / "predictions" / dataset / precision
            cache_root = OUTPUT / "trt_cache" / dataset / precision
            for path in (results_root, predictions_root, cache_root):
                path.mkdir(parents=True, exist_ok=True)

            if dataset == "tinyperson":
                run(
                    [
                        "python3",
                        str(DEPLOY / "docker/jetson_onnx/evaluate_tinyperson_merged.py"),
                        "--models-root",
                        str(models_root),
                        "--data-root",
                        str(args.data_root),
                        "--results-root",
                        str(results_root),
                        "--predictions-root",
                        str(predictions_root),
                        "--trt-cache-root",
                        str(cache_root),
                        "--precision",
                        precision,
                        "--batch-size",
                        str(args.batch_size),
                        "--conf",
                        str(args.conf),
                        "--nms-iou",
                        str(args.nms_iou),
                        "--max-det",
                        str(args.max_det),
                    ],
                    env,
                )
            else:
                run(
                    [
                        "python3",
                        str(DEPLOY / "docker/jetson_onnx/evaluate.py"),
                        "--models-root",
                        str(models_root),
                        "--data-root",
                        str(args.data_root),
                        "--dataset",
                        dataset,
                        "--split",
                        "val",
                        "test",
                        "--results-root",
                        str(results_root),
                        "--provider",
                        "tensorrt",
                        "--precision",
                        precision,
                        "--trt-cache-root",
                        str(cache_root),
                        "--batch-size",
                        str(args.batch_size),
                        "--conf",
                        str(args.conf),
                        "--nms-iou",
                        str(args.nms_iou),
                        "--max-det",
                        str(args.max_det),
                    ],
                    env,
                )


if __name__ == "__main__":
    main()
