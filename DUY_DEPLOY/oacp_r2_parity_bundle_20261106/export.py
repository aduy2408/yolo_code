#!/usr/bin/env python3
"""Export dataset/model/seed checkpoints to static ONNX on Jetson."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
DEPLOY = BUNDLE.parent
CHECKPOINTS = BUNDLE / "checkpoints"
ONNX = BUNDLE / "output" / "onnx"
SIZES = {"tinyperson": 640, "levirship": 512, "varroa": 640}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*SIZES, "all"), default="all")
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    args = parser.parse_args()
    datasets = tuple(SIZES) if args.dataset == "all" else (args.dataset,)
    precisions = ("fp32", "fp16") if args.precision == "both" else (args.precision,)
    checkpoints = [p for dataset in datasets for p in sorted((CHECKPOINTS / dataset).glob("*/seed*/best.pt"))]
    if not checkpoints:
        raise SystemExit(f"No checkpoints under {CHECKPOINTS}")
    for checkpoint in checkpoints:
        dataset = checkpoint.relative_to(CHECKPOINTS).parts[0]
        model = checkpoint.relative_to(CHECKPOINTS).parts[1]
        seed = checkpoint.relative_to(CHECKPOINTS).parts[2]
        for precision in precisions:
            output = ONNX / dataset / model / seed / f"{precision}.onnx"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "docker", "run", "--rm", "--runtime=nvidia", "--ipc=host",
                "-e", "NVIDIA_VISIBLE_DEVICES=all", "-v", f"{checkpoint}:/models/best.pt:ro",
                "-v", f"{output.parent}:/out",
                "-v", f"{DEPLOY / 'docker/pytorch_jetson/export_onnx.py'}:/app/export_onnx.py:ro",
                "-v", f"{DEPLOY / 'runtime_deps'}:/opt/runtime_deps:ro",
                "-e", "PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream",
                "--entrypoint", "python3", "duy-yolo-pytorch:jetpack6", "/app/export_onnx.py",
                "--checkpoint", "/models/best.pt", "--output", f"/out/{precision}.onnx",
                "--imgsz", str(SIZES[dataset]), "--precision", precision,
            ]
            print(f"EXPORT {dataset}/{model}/{seed} {precision}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
