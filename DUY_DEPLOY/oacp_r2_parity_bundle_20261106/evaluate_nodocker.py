#!/usr/bin/env python3
"""Evaluate TensorRT engines with the Ultralytics validators, without Docker."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
DEPLOY = BUNDLE.parent
REPO = DEPLOY.parent
ENGINES = BUNDLE / "output" / "engines"
OUTPUT = BUNDLE / "output"


def run(command: list[str], env: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tinyperson", "levirship", "varroa", "all"), default="all")
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    datasets = ("tinyperson", "levirship", "varroa") if args.dataset == "all" else (args.dataset,)
    precisions = ("fp32", "fp16") if args.precision == "both" else (args.precision,)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO / "vendor/ultralytics_upstream"),
            str(DEPLOY / "docker/pytorch_jetson"),
            env.get("PYTHONPATH", ""),
        ]
    )

    for dataset in datasets:
        evaluator = (
            DEPLOY / "docker/pytorch_jetson/evaluate_ultralytics_tinyperson.py"
            if dataset == "tinyperson"
            else DEPLOY / "docker/pytorch_jetson/evaluate_ultralytics.py"
        )
        for precision in precisions:
            models = sorted((ENGINES / dataset).glob(f"*/seed*/{precision}.engine"))
            if not models:
                raise SystemExit(f"No {precision} engines under {ENGINES / dataset}")
            for model in models:
                relative = model.relative_to(ENGINES / dataset)
                result = OUTPUT / "metrics" / dataset / precision / relative.parent
                result.mkdir(parents=True, exist_ok=True)
                if dataset == "tinyperson":
                    command = [
                        sys.executable,
                        str(evaluator),
                        "--model",
                        str(model),
                        "--data-root",
                        str(args.data_root),
                        "--results-root",
                        str(result),
                        "--evaluator",
                        str(REPO / "vendor/tinyperson_cocoeval.py"),
                        "--device",
                        args.device,
                        "--precision",
                        precision,
                        "--conf",
                        str(args.conf),
                        "--iou",
                        str(args.nms_iou),
                        "--max-det",
                        str(args.max_det),
                    ]
                else:
                    command = [
                        sys.executable,
                        str(evaluator),
                        "--model",
                        str(model),
                        "--data-root",
                        str(args.data_root),
                        "--dataset",
                        dataset,
                        "--split",
                        "val",
                        "test",
                        "--results-root",
                        str(result),
                        "--device",
                        args.device,
                        "--precision",
                        precision,
                        "--conf",
                        str(args.conf),
                        "--iou",
                        str(args.nms_iou),
                        "--max-det",
                        str(args.max_det),
                    ]
                run(command, env)


if __name__ == "__main__":
    main()
