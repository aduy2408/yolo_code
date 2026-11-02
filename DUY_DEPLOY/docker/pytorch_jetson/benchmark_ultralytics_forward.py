from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO

DATASETS = {
    "levirship": 512,
    "tinyperson": 640,
    "varroa": 640,
}
MODELS = ("yolov5nu", "yolov8n", "yolov9t", "yolov10n", "yolo11n")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-root", type=Path, default=Path("/checkpoints"))
    parser.add_argument("--results-root", type=Path, default=Path("/results"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.batch_size < 1:
        raise ValueError("warmup non-negative, iterations and batch-size positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU benchmark")
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    args.results_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, size in DATASETS.items():
        for model_name in MODELS:
            checkpoint = args.checkpoints_root / dataset / model_name / "seed_42" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            loaded = YOLO(str(checkpoint))
            module = loaded.model.to(device).eval()
            if args.precision == "fp16":
                module.half()
            else:
                module.float()
            sample = torch.zeros((args.batch_size, 3, size, size), device=device, dtype=torch.float16 if args.precision == "fp16" else torch.float32)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    module(sample)
                torch.cuda.synchronize(device)
                times = []
                for _ in range(args.iterations):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    module(sample)
                    torch.cuda.synchronize(device)
                    times.append((time.perf_counter() - started) * 1000.0)
            mean_ms = statistics.fmean(times)
            row = {
                "dataset": dataset,
                "model": model_name,
                "seed": 42,
                "checkpoint": str(checkpoint),
                "runtime": "pytorch_ultralytics_forward",
                "scope": "model_forward_only",
                "device": args.device,
                "precision": args.precision,
                "image_size": [size, size],
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "fps": args.batch_size * 1000.0 / mean_ms,
                "latency_median_ms": statistics.median(times),
                "latency_mean_ms": mean_ms,
                "latency_p95_ms": percentile(times, 0.95),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del module, loaded, sample
            torch.cuda.empty_cache()
    output = {
        "protocol": {
            "runtime": "pytorch_ultralytics_forward",
            "scope": "model_forward_only",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "precision": args.precision,
        },
        "results": rows,
    }
    (args.results_root / "benchmark_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with (args.results_root / "benchmark_results.csv").open("w", encoding="utf-8") as stream:
        stream.write("dataset,model,seed,batch_size,precision,fps,latency_median_ms,latency_mean_ms,latency_p95_ms\n")
        for row in rows:
            stream.write(f"{row['dataset']},{row['model']},{row['seed']},{row['batch_size']},{row['precision']},{row['fps']:.6f},{row['latency_median_ms']:.6f},{row['latency_mean_ms']:.6f},{row['latency_p95_ms']:.6f}\n")


if __name__ == "__main__":
    main()
