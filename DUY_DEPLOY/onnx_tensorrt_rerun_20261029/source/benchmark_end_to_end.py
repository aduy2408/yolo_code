from __future__ import annotations

import argparse
import json
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

_EVALUATE_SPEC = importlib.util.spec_from_file_location("_rerun_evaluate", Path(__file__).with_name("evaluate.py"))
if _EVALUATE_SPEC is None or _EVALUATE_SPEC.loader is None:
    raise ImportError("Cannot load the bundled evaluate.py")
evaluate = importlib.util.module_from_spec(_EVALUATE_SPEC)
sys.modules["_rerun_evaluate"] = evaluate
_EVALUATE_SPEC.loader.exec_module(evaluate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time TensorRT with Ultralytics-style preprocessing and postprocessing.")
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("levirship", "tinyperson", "varroa"), required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--trt-cache-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def provider(cache_root: Path) -> list[object]:
    name = "TensorrtExecutionProvider"
    if name not in ort.get_available_providers():
        raise RuntimeError(f"{name} unavailable: {ort.get_available_providers()}")
    cache_root.mkdir(parents=True, exist_ok=True)
    return [(name, {"device_id": 0, "trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache_root)})]


def timed_batch(session: ort.InferenceSession, rows: list[dict], size: int, conf: float, nms_iou: float, max_det: int) -> dict[str, float]:
    input_spec = session.get_inputs()[0]
    dtype = np.float16 if input_spec.type == "tensor(float16)" else np.float32
    tensors = []
    transforms = []
    started = time.perf_counter()
    for row in rows:
        with Image.open(row["path"]) as source:
            tensor, scale, pad_x, pad_y = evaluate.letterbox(source.convert("RGB"), size, dtype)
        tensors.append(tensor)
        transforms.append((scale, pad_x, pad_y))
    batch_tensor = np.stack(tensors)
    preprocess_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    raw = session.run(None, {input_spec.name: batch_tensor})
    inference_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    for offset, (row, transform) in enumerate(zip(rows, transforms)):
        boxes, scores = evaluate.decode_outputs(raw, offset, size)
        selected = scores >= conf
        boxes, scores = boxes[selected], scores[selected]
        scale, pad_x, pad_y = transform
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, row["width"])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, row["height"])
        if len(boxes):
            evaluate.nms(boxes, scores, nms_iou, max_det)
    postprocess_ms = (time.perf_counter() - started) * 1000.0
    return {"preprocess_ms": preprocess_ms, "inference_ms": inference_ms, "postprocess_ms": postprocess_ms, "total_ms": preprocess_ms + inference_ms + postprocess_ms}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("batch-size positive; warmup non-negative; iterations positive")
    size = evaluate.image_size(args.dataset)
    images, _ = evaluate.load_coco(args.data_root, args.dataset, "test")
    model_paths = sorted(args.models_root.rglob("*.onnx"))
    if not model_paths:
        raise FileNotFoundError(f"No ONNX models under {args.models_root}")
    args.results_root.mkdir(parents=True, exist_ok=True)
    for model_path in model_paths:
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=provider(args.trt_cache_root))
        if hasattr(session, "disable_fallback"):
            session.disable_fallback()
        if session.get_providers()[0] != "TensorrtExecutionProvider":
            raise RuntimeError(f"TensorRT was not first for {model_path}: {session.get_providers()}")
        batches = [[images[(index * args.batch_size + offset) % len(images)] for offset in range(args.batch_size)] for index in range(args.warmup + args.iterations)]
        for batch in batches[:args.warmup]:
            timed_batch(session, batch, size, args.conf, args.nms_iou, args.max_det)
        measured = [timed_batch(session, batch, size, args.conf, args.nms_iou, args.max_det) for batch in batches[args.warmup:]]
        def stats(field: str) -> dict[str, float]:
            values = np.asarray([x[field] for x in measured], dtype=np.float64)
            return {
                f"mean_{field}": float(np.mean(values)),
                f"median_{field}": float(np.median(values)),
                f"p95_{field}": float(np.percentile(values, 95)),
            }
        timing = {}
        for field in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
            timing.update(stats(field))
        row = {
            "model": str(model_path), "dataset": args.dataset, "split": "test", "provider": "TensorrtExecutionProvider",
            "session_providers": session.get_providers(), "batch_size": args.batch_size, "image_size": [size, size],
            "warmup": args.warmup, "iterations": args.iterations, "conf": args.conf, "nms_iou": args.nms_iou,
            "max_det": args.max_det, **timing,
            "images_per_second_end_to_end": args.batch_size * 1000.0 / timing["mean_total_ms"],
            "images_per_second_inference_only": args.batch_size * 1000.0 / timing["mean_inference_ms"],
        }
        output = args.results_root / model_path.name.replace(".onnx", ".json")
        output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
