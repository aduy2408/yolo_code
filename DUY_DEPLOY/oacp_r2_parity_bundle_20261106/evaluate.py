#!/usr/bin/env python3
"""Evaluate PT, TensorRT FP32, and TensorRT FP16 by dataset/model/seed."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
DEPLOY = BUNDLE.parent
CHECKPOINTS = BUNDLE / "checkpoints"
ONNX = BUNDLE / "output" / "onnx"
OUTPUT = BUNDLE / "output"
DATA_ROOT = Path("/home/ait01/jetson_seed42_benchmark/data")


def mounts(model: Path, result: Path, predictions: Path, cache: Path, data_root: Path) -> list[str]:
    return [
        "--rm", "--runtime=nvidia", "--ipc=host", "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-v", f"{model}:/models/{model.name}:ro", "-v", f"{data_root}:/datasets:ro",
        "-v", f"{result}:/results", "-v", f"{predictions}:/predictions", "-v", f"{cache}:/trt_cache",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tinyperson", "levirship", "varroa", "all"), default="tinyperson")
    parser.add_argument("--backend", choices=("pt", "fp32", "fp16", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    datasets = ("tinyperson", "levirship", "varroa") if args.dataset == "all" else (args.dataset,)
    backends = ("pt", "fp32", "fp16") if args.backend == "all" else (args.backend,)
    for dataset in datasets:
        for backend in backends:
            root = CHECKPOINTS if backend == "pt" else ONNX
            models = sorted(root.joinpath(dataset).glob("*/seed*/best.pt" if backend == "pt" else "*/ */"))
            if backend != "pt":
                models = sorted(root.joinpath(dataset).glob("*/ */".replace(" ", "")))
            if backend != "pt":
                models = sorted(root.joinpath(dataset).glob("*/seed*/" + backend + ".onnx"))
            if not models:
                raise SystemExit(f"No {backend} models for dataset={dataset}")
            for model in models:
                relative = model.relative_to(root / dataset)
                model_name, seed = relative.parts[0], relative.parts[1]
                stem = f"{dataset}_{model_name}_{seed}_{backend}"
                result = OUTPUT / "metrics" / dataset / backend / model_name / seed
                predictions = OUTPUT / "predictions" / dataset / backend / model_name / seed
                cache = OUTPUT / "trt_cache" / dataset / backend / model_name / seed
                for path in (result, predictions, cache):
                    path.mkdir(parents=True, exist_ok=True)
                if backend == "pt":
                    if dataset == "tinyperson":
                        command = [
                            "docker", "run", *mounts(model, result, predictions, cache, args.data_root),
                            "-v", f"{DEPLOY / 'docker/pytorch_jetson/evaluate_ultralytics_tinyperson.py'}:/app/evaluate_tinyperson.py:ro",
                            "-v", f"{DEPLOY / 'docker/pytorch_jetson/checkpoint_compat.py'}:/app/checkpoint_compat.py:ro",
                            "-v", f"{DEPLOY.parent / 'vendor/tinyperson_cocoeval.py'}:/app/tinyperson_cocoeval.py:ro",
                            "-v", f"{DEPLOY / 'runtime_deps'}:/opt/runtime_deps:ro",
                            "-e", "PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream",
                            "--entrypoint", "python3", "duy-yolo-pytorch:jetpack6", "/app/evaluate_tinyperson.py",
                            "--model", f"/models/{model.name}", "--data-root", "/datasets", "--results-root", "/results",
                            "--evaluator", "/app/tinyperson_cocoeval.py", "--device", args.device,
                            "--precision", "fp32", "--conf", "0.001", "--iou", "0.5", "--max-det", "300", "--save-only",
                        ]
                    else:
                        command = [
                            "docker", "run", *mounts(model, result, predictions, cache, args.data_root),
                            "-v", f"{DEPLOY / 'docker/pytorch_jetson/evaluate_ultralytics.py'}:/app/evaluate.py:ro",
                            "-v", f"{DEPLOY / 'docker/pytorch_jetson/checkpoint_compat.py'}:/app/checkpoint_compat.py:ro",
                            "-v", f"{DEPLOY / 'runtime_deps'}:/opt/runtime_deps:ro",
                            "-e", "PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream",
                            "--entrypoint", "python3", "duy-yolo-pytorch:jetpack6", "/app/evaluate.py",
                            "--model", f"/models/{model.name}", "--data-root", "/datasets", "--dataset", dataset,
                            "--split", "val", "test", "--results-root", "/results", "--device", args.device,
                            "--precision", "fp32", "--conf", "0.001", "--iou", "0.5", "--max-det", "300",
                        ]
                elif dataset == "tinyperson":
                    command = [
                        "docker", "run", *mounts(model, result, predictions, cache, args.data_root),
                        "-v", f"{DEPLOY / 'docker/jetson_onnx/evaluate_tinyperson_merged.py'}:/app/evaluate_tinyperson.py:ro",
                        "-v", f"{DEPLOY / 'runtime_deps'}:/opt/runtime_deps:ro",
                        "-e", "PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream",
                        "--entrypoint", "python3", "yolo-onnx-jetson:jp6", "/app/evaluate_tinyperson.py",
                        "--model-path", f"/models/{model.name}", "--data-root", "/datasets", "--results-root", "/results",
                        "--predictions-root", "/predictions", "--trt-cache-root", "/trt_cache",
                        "--precision", backend, "--batch-size", "1", "--conf", "0.001", "--nms-iou", "0.5", "--max-det", "300", "--save-only",
                    ]
                else:
                    command = [
                        "docker", "run", *mounts(model, result, predictions, cache, args.data_root),
                        "-v", f"{DEPLOY / 'docker/jetson_onnx/evaluate.py'}:/app/evaluate.py:ro",
                        "-v", f"{DEPLOY / 'runtime_deps'}:/opt/runtime_deps:ro",
                        "-e", "PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream",
                        "--entrypoint", "python3", "yolo-onnx-jetson:jp6", "/app/evaluate.py",
                        "--models-root", "/models", "--data-root", "/datasets", "--dataset", dataset,
                        "--split", "val", "test", "--results-root", "/results", "--provider", "tensorrt",
                        "--precision", backend, "--trt-cache-root", "/trt_cache", "--batch-size", "1",
                        "--conf", "0.001", "--nms-iou", "0.5", "--max-det", "300",
                    ]
                print(f"EVALUATE {dataset}/{model_name}/{seed} backend={backend}", flush=True)
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
