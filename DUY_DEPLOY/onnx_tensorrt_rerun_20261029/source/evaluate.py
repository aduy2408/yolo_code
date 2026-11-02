from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

CUDA_PROVIDER = "CUDAExecutionProvider"
TENSORRT_PROVIDER = "TensorrtExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
DATASET_DIRS = {"levirship": "levir_ship", "tinyperson": "tinyperson", "varroa": "varroa"}
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def provider_config(provider: str, cache_root: Path | None) -> tuple[str, list[object]]:
    names = {"cpu": CPU_PROVIDER, "cuda": CUDA_PROVIDER, "tensorrt": TENSORRT_PROVIDER}
    name = names[provider]
    if name not in ort.get_available_providers():
        raise RuntimeError(f"{name} unavailable: {ort.get_available_providers()}")
    if provider == "cpu":
        return name, [name]
    if provider == "cuda":
        return name, [(name, {"device_id": 0})]
    options = {"device_id": 0, "trt_fp16_enable": True, "trt_engine_cache_enable": True}
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        options["trt_engine_cache_path"] = str(cache_root)
    return name, [(name, options)]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO ONNX models with CPU, CUDA, or TensorRT.")
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("levirship", "tinyperson", "varroa"), required=True)
    parser.add_argument("--split", choices=("val", "test"), nargs="+", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument("--provider", choices=("cpu", "cuda", "tensorrt"), default="cpu")
    parser.add_argument("--trt-cache-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def image_size(dataset: str) -> int:
    return 512 if dataset == "levirship" else 640


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def letterbox(image: Image.Image, size: int, dtype: np.dtype) -> tuple[np.ndarray, float, float, float]:
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    pad_x = (size - resized.width) / 2.0
    pad_y = (size - resized.height) / 2.0
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (round(pad_x), round(pad_y)))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)).astype(dtype), scale, pad_x, pad_y


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + other - intersection, 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, max_det: int) -> np.ndarray:
    order = np.argsort(-scores)
    kept: list[int] = []
    while order.size and len(kept) < max_det:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        candidates = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[candidates, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[candidates, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[candidates, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[candidates, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(0.0, boxes[current, 3] - boxes[current, 1])
        area_candidates = np.maximum(0.0, boxes[candidates, 2] - boxes[candidates, 0]) * np.maximum(0.0, boxes[candidates, 3] - boxes[candidates, 1])
        overlap = intersection / np.maximum(area_current + area_candidates - intersection, 1e-12)
        order = candidates[overlap <= threshold]
    return np.asarray(kept, dtype=np.int64)


def load_coco(data_root: Path, dataset: str, split: str) -> tuple[list[dict[str, Any]], dict[int, list[np.ndarray]]]:
    dataset_dir = DATASET_DIRS[dataset]
    annotation_path = data_root / dataset_dir / "annotations" / "seed_42" / f"{split}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations: dict[int, list[np.ndarray]] = {}
    for row in payload["annotations"]:
        x, y, width, height = row["bbox"]
        annotations.setdefault(int(row["image_id"]), []).append(
            np.asarray([x, y, x + width, y + height], dtype=np.float32)
        )
    images = [{**row, "path": data_root / dataset_dir / row["file_name"]} for row in payload["images"]]
    return images, annotations


def decode_outputs(outputs: list[np.ndarray], offset: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(outputs) == 1:
        values = outputs[0][offset]
        if values.shape[-1] == 6:
            return values[:, :4].astype(np.float32), values[:, 4].astype(np.float32)
        values = values.T.astype(np.float32)
        return (
            np.column_stack((values[:, 0] - values[:, 2] / 2, values[:, 1] - values[:, 3] / 2, values[:, 0] + values[:, 2] / 2, values[:, 1] + values[:, 3] / 2)),
            values[:, 4],
        )
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for scale_index, stride in enumerate((8, 16, 32)):
        distribution = outputs[scale_index * 2][offset].astype(np.float32)
        logits = outputs[scale_index * 2 + 1][offset].astype(np.float32)
        _, height, width = distribution.shape
        bins = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
        distribution = distribution.reshape(4, 16, height, width)
        distribution -= distribution.max(axis=1, keepdims=True)
        probabilities = np.exp(distribution)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        distances = (probabilities * bins).sum(axis=1) * stride
        grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        centers_x = (grid_x + 0.5) * stride
        centers_y = (grid_y + 0.5) * stride
        boxes.append(np.stack((centers_x - distances[0], centers_y - distances[1], centers_x + distances[2], centers_y + distances[3]), axis=-1).reshape(-1, 4))
        scores.append((1.0 / (1.0 + np.exp(-logits))).reshape(-1))
    return np.concatenate(boxes), np.concatenate(scores)


def predictions(session: ort.InferenceSession, images: list[dict[str, Any]], size: int, batch_size: int, conf: float, nms_iou: float, max_det: int) -> dict[int, list[tuple[np.ndarray, float]]]:
    input_spec = session.get_inputs()[0]
    input_name = input_spec.name
    dtype = np.float16 if input_spec.type == "tensor(float16)" else np.float32
    output: dict[int, list[tuple[np.ndarray, float]]] = {}
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors: list[np.ndarray] = []
        transforms: list[tuple[float, float, float]] = []
        for image in chunk:
            with Image.open(image["path"]) as source:
                tensor, scale, pad_x, pad_y = letterbox(source.convert("RGB"), size, dtype)
            tensors.append(tensor)
            transforms.append((scale, pad_x, pad_y))
        raw = session.run(None, {input_name: np.stack(tensors)})
        for offset, (image, transform) in enumerate(zip(chunk, transforms)):
            boxes, scores = decode_outputs(raw, offset, size)
            selected = scores >= conf
            boxes, scores = boxes[selected], scores[selected]
            scale, pad_x, pad_y = transform
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image["width"])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image["height"])
            kept = nms(boxes, scores, nms_iou, max_det) if len(boxes) else np.empty(0, dtype=np.int64)
            output[int(image["id"])] = [(boxes[index], float(scores[index])) for index in kept]
    return output


def ap_for_area(images: list[dict[str, Any]], ground_truth: dict[int, list[np.ndarray]], predicted: dict[int, list[tuple[np.ndarray, float]]], threshold: float, area_limit: tuple[float, float]) -> float:
    min_area, max_area = area_limit
    total_gt = 0
    ranked: list[tuple[float, int, np.ndarray]] = []
    for image in images:
        gt = [box for box in ground_truth.get(int(image["id"]), []) if min_area <= float((box[2] - box[0]) * (box[3] - box[1])) < max_area]
        total_gt += len(gt)
        for box, score in predicted.get(int(image["id"]), []):
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            if min_area <= area < max_area:
                ranked.append((score, int(image["id"]), box))
    if total_gt == 0:
        return -1.0
    ranked.sort(key=lambda row: -row[0])
    matched: dict[int, set[int]] = {}
    true_positive = np.zeros(len(ranked), dtype=np.float64)
    for index, (_, image_id, box) in enumerate(ranked):
        gt = [candidate for candidate in ground_truth.get(image_id, []) if min_area <= float((candidate[2] - candidate[0]) * (candidate[3] - candidate[1])) < max_area]
        available_indices = [
            candidate_index
            for candidate_index, candidate in enumerate(gt)
            if candidate_index not in matched.setdefault(image_id, set())
        ]
        if available_indices:
            overlaps = iou(box, np.asarray([gt[candidate_index] for candidate_index in available_indices]))
            best = int(np.argmax(overlaps))
            if overlaps[best] >= threshold:
                true_positive[index] = 1.0
                matched[image_id].add(available_indices[best])
    false_positive = 1.0 - true_positive
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive + false_positive), 1e-12)
    recall = np.cumsum(true_positive) / total_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean([envelope[recall >= point][0] if np.any(recall >= point) else 0.0 for point in np.linspace(0, 1, 101)]))


def evaluate(images: list[dict[str, Any]], ground_truth: dict[int, list[np.ndarray]], predicted: dict[int, list[tuple[np.ndarray, float]]]) -> dict[str, float]:
    areas = {"all": (0.0, 1e12), "small": (0.0, 32.0 * 32.0)}
    metrics: dict[str, float] = {}
    for label, area in areas.items():
        values = [ap_for_area(images, ground_truth, predicted, threshold, area) for threshold in IOU_THRESHOLDS]
        suffix = "Small" if label == "small" else "all"
        metrics[f"AP50-{suffix}"] = values[0]
        metrics[f"mAP50-95-{suffix}"] = float(np.mean([value for value in values if value >= 0]))
    return metrics


def main() -> None:
    parsed = args()
    provider_name, provider_settings = provider_config(parsed.provider, parsed.trt_cache_root)
    sessions: dict[Path, ort.InferenceSession] = {}
    for model_path in sorted(parsed.models_root.rglob("*.onnx")):
        options = ort.SessionOptions()
        options.intra_op_num_threads = parsed.intra_op_threads
        options.inter_op_num_threads = parsed.inter_op_threads
        if parsed.provider == "tensorrt":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=provider_settings)
        if parsed.provider == "tensorrt" and hasattr(session, "disable_fallback"):
            session.disable_fallback()
        if session.get_providers()[0] != provider_name:
            raise RuntimeError(f"{model_path} did not initialize {provider_name} first: {session.get_providers()}")
        sessions[model_path] = session
    if not sessions:
        raise FileNotFoundError(f"No ONNX models under {parsed.models_root}")
    parsed.results_root.mkdir(parents=True, exist_ok=True)
    for split in parsed.split:
        images, ground_truth = load_coco(parsed.data_root, parsed.dataset, split)
        for model_path, session in sessions.items():
            started = time.perf_counter()
            predicted = predictions(session, images, image_size(parsed.dataset), parsed.batch_size, parsed.conf, parsed.nms_iou, parsed.max_det)
            metrics = evaluate(images, ground_truth, predicted)
            row = {"model": str(model_path), "dataset": parsed.dataset, "split": split, "images": len(images), "ground_truth_boxes": sum(map(len, ground_truth.values())), "provider": provider_name, "nms_iou": parsed.nms_iou, "conf": parsed.conf, "batch_size": parsed.batch_size, "image_size": image_size(parsed.dataset), "elapsed_seconds": time.perf_counter() - started, **metrics}
            model_parts = model_path.parts
            root_name = "onnx_cpu_fp32" if "onnx_cpu_fp32" in model_parts else "onnx_cuda_fp16"
            model_key = Path(*model_parts[model_parts.index(root_name) + 1 :]) if root_name in model_parts else model_path.relative_to(parsed.models_root)
            output = parsed.results_root / parsed.dataset / split / model_key.with_suffix(".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            row["model_sha256"] = sha256(model_path)
            output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if parsed.predictions_root is not None:
                prediction_output = parsed.predictions_root / parsed.dataset / split / model_key.with_suffix(".json")
                prediction_output.parent.mkdir(parents=True, exist_ok=True)
                prediction_payload = {
                    "model": str(model_path),
                    "dataset": parsed.dataset,
                    "split": split,
                    "provider": provider_name,
                    "conf": parsed.conf,
                    "nms_iou": parsed.nms_iou,
                    "max_det": parsed.max_det,
                    "image_size": image_size(parsed.dataset),
                    "predictions": {
                        str(image_id): [
                            {"bbox": [float(value) for value in box], "score": float(score)}
                            for box, score in values
                        ]
                        for image_id, values in predicted.items()
                    },
                }
                prediction_output.write_text(json.dumps(prediction_payload, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
