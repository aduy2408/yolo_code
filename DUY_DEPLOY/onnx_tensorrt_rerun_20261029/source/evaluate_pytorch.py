from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

_EVALUATE_SPEC = importlib.util.spec_from_file_location("_shared_evaluate", Path(__file__).with_name("evaluate.py"))
if _EVALUATE_SPEC is None or _EVALUATE_SPEC.loader is None:
    raise ImportError("Cannot load shared evaluate.py")
_shared = importlib.util.module_from_spec(_EVALUATE_SPEC)
sys.modules["_shared_evaluate"] = _shared
_EVALUATE_SPEC.loader.exec_module(_shared)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO .pt model with the shared ONNX/TensorRT pipeline.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(_shared.DATASET_DIRS), required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="Legacy prepared root; prefer --dataset-yaml.")
    parser.add_argument("--dataset-yaml", type=Path, default=None, help="Prepared single-dataset YOLO YAML.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()

def adapt_raw_outputs(raw: object, size: int) -> list[np.ndarray]:
    """Adapt Ultralytics model outputs into the shared ONNX decoder contract."""
    tensor = raw[0] if isinstance(raw, tuple) else raw
    if tensor.ndim == 3 and tensor.shape[-1] == 6:
        return [tensor.detach().cpu().numpy()]
    if isinstance(raw, tuple) and len(raw) > 1 and isinstance(raw[1], dict):
        one2one = raw[1].get("one2one")
        if isinstance(one2one, dict) and {"boxes", "scores"} <= one2one.keys():
            boxes = one2one["boxes"]
            scores = one2one["scores"]
            outputs: list[np.ndarray] = []
            start = 0
            for stride in (8, 16, 32):
                height = width = size // stride
                count = height * width
                outputs.extend(
                    (
                        boxes[:, :, start : start + count].reshape(-1, 64, height, width).detach().cpu().numpy(),
                        scores[:, :, start : start + count].reshape(-1, 1, height, width).detach().cpu().numpy(),
                    )
                )
                start += count
            return outputs
    return [tensor.detach().cpu().numpy()]


def main() -> None:
    parsed = args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU evaluation")
    if parsed.dataset_yaml is not None:
        images, ground_truth = _shared.load_yolo_yaml(parsed.dataset_yaml, parsed.split)
    elif parsed.data_root is not None:
        images, ground_truth = _shared.load_coco(parsed.data_root, parsed.dataset, parsed.split)
    else:
        raise ValueError("Pass --dataset-yaml for one prepared dataset (or legacy --data-root).")
    model = YOLO(str(parsed.model))
    model.model.to("cuda").eval()
    predicted: dict[int, list[tuple[np.ndarray, float]]] = {}
    with torch.inference_mode():
        for start in range(0, len(images), parsed.batch_size):
            chunk = images[start : start + parsed.batch_size]
            tensors: list[np.ndarray] = []
            transforms: list[tuple[float, float, float]] = []
            for image in chunk:
                from PIL import Image
                with Image.open(image["path"]) as source:
                    tensor, scale, pad_x, pad_y = _shared.letterbox(source.convert("RGB"), _shared.image_size(parsed.dataset), np.dtype(np.float32))
                tensors.append(tensor)
                transforms.append((scale, pad_x, pad_y))
            raw = model.model(torch.from_numpy(np.stack(tensors)).to("cuda"))
            raw_outputs = adapt_raw_outputs(raw, _shared.image_size(parsed.dataset))
            for offset, (image, transform) in enumerate(zip(chunk, transforms)):
                boxes, scores = _shared.decode_outputs(raw_outputs, offset, _shared.image_size(parsed.dataset))
                selected = scores >= parsed.conf
                boxes, scores = boxes[selected], scores[selected]
                scale, pad_x, pad_y = transform
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image["width"])
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image["height"])
                kept = _shared.nms(boxes, scores, parsed.nms_iou, parsed.max_det) if len(boxes) else np.empty(0, dtype=np.int64)
                predicted[int(image["id"])] = [(boxes[index], float(scores[index])) for index in kept]
    metrics = _shared.evaluate(images, ground_truth, predicted)
    row = {
        "model": str(parsed.model), "dataset": parsed.dataset, "split": parsed.split,
        "provider": "PyTorch/raw-model+shared-pipeline", "precision": "FP32",
        "batch_size": parsed.batch_size, "image_size": _shared.image_size(parsed.dataset),
        "conf": parsed.conf, "nms_iou": parsed.nms_iou, "max_det": parsed.max_det,
        "images": len(images), "ground_truth_boxes": sum(map(len, ground_truth.values())), **metrics,
    }
    parsed.results_root.mkdir(parents=True, exist_ok=True)
    (parsed.results_root / "metrics.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
