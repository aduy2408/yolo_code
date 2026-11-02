from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


CUDA_PROVIDER = "CUDAExecutionProvider"
TENSORRT_PROVIDER = "TensorrtExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load ONNX sessions once and run a CPU, CUDA, or TensorRT benchmark.")
    parser.add_argument("--models-root", type=Path, default=Path("/models"))
    parser.add_argument("--results-root", type=Path, default=Path("/results"))
    parser.add_argument("--provider", choices=("cpu", "cuda", "tensorrt"), default="cpu")
    parser.add_argument("--trt-cache-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def providers(provider: str, cache_root: Path | None) -> list[object]:
    names = {
        "cpu": CPU_PROVIDER,
        "cuda": CUDA_PROVIDER,
        "tensorrt": TENSORRT_PROVIDER,
    }
    name = names[provider]
    if name not in ort.get_available_providers():
        raise RuntimeError(f"{name} is unavailable; available={ort.get_available_providers()}")
    if provider == "cpu":
        return [name]
    if provider == "cuda":
        return [(name, {"device_id": 0})]
    options = {"device_id": 0, "trt_fp16_enable": True, "trt_engine_cache_enable": True}
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        options["trt_engine_cache_path"] = str(cache_root)
    return [(name, options)]


def expected_provider(provider: str) -> str:
    return {"cpu": CPU_PROVIDER, "cuda": CUDA_PROVIDER, "tensorrt": TENSORRT_PROVIDER}[provider]


def input_size(path: Path) -> int:
    return 512 if "levirship" in path.parts else 640


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("batch-size must be positive; warmup non-negative; iterations positive")
    model_paths = sorted(args.models_root.rglob("*.onnx"))
    if not model_paths:
        raise FileNotFoundError(f"No ONNX models found under {args.models_root}")
    provider_config = providers(args.provider, args.trt_cache_root)
    sessions = {}
    for path in model_paths:
        session_options = ort.SessionOptions()
        if args.provider == "tensorrt":
            session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(str(path), sess_options=session_options, providers=provider_config)
        active_providers = session.get_providers()
        expected = expected_provider(args.provider)
        if not active_providers or active_providers[0] != expected:
            raise RuntimeError(f"{path} did not initialize {expected} first: {active_providers}")
        sessions[path] = session
    args.results_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, session in sessions.items():
        size = input_size(path)
        input_spec = session.get_inputs()[0]
        input_name = input_spec.name
        dtype = np.float16 if input_spec.type == "tensor(float16)" else np.float32
        sample = np.zeros((args.batch_size, 3, size, size), dtype=dtype)
        for _ in range(args.warmup):
            session.run(None, {input_name: sample})
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            outputs = session.run(None, {input_name: sample})
            times.append((time.perf_counter() - start) * 1000.0)
        times.sort()
        mean_ms = sum(times) / len(times)
        row = {
            "model": str(path),
            "provider": args.provider,
            "providers": session.get_providers(),
            "batch_size": args.batch_size,
            "image_size": [size, size],
            "input_dtype": str(dtype),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "latency_median_ms": times[len(times) // 2],
            "latency_mean_ms": mean_ms,
            "latency_p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))],
            "images_per_second": args.batch_size * 1000.0 / mean_ms,
            "output_shapes": [list(output.shape) for output in outputs],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = args.results_root / f"onnx_{args.provider}_batch{args.batch_size}_benchmark.json"
    output.write_text(json.dumps({"provider_config": provider_config, "results": rows}, indent=2) + "\n")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()




