"""Causal/context intervention probe using paired object/context image edits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import DatasetSpec, ProbeConfig, image_paths, load_yolo, metadata, read_boxes, size_bucket, write_json, write_rows


def _masked(image: np.ndarray, boxes: list[dict[str, float]], mode: str) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        x1, y1 = max(0, int(box["x1"])), max(0, int(box["y1"]))
        x2, y2 = min(image.shape[1], int(box["x2"])), min(image.shape[0], int(box["y2"]))
        mask[y1:y2, x1:x2] = 1
    if mode == "object_only":
        return image * mask[..., None]
    if mode == "context_only":
        return image * (1 - mask[..., None])
    if mode == "object_removed":
        fill = np.median(image.reshape(-1, 3), axis=0).astype(image.dtype)
        output = image.copy()
        output[mask.astype(bool)] = fill
        return output
    return image


def _score(model: Any, image: np.ndarray, config: ProbeConfig) -> float:
    result = model.predict(image, imgsz=config.imgsz, conf=config.conf, device=config.device, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0
    return float(boxes.conf.detach().float().max().cpu())


def run(spec: DatasetSpec, checkpoint: Path, output: Path, config: ProbeConfig) -> dict[str, Any]:
    model = load_yolo(checkpoint, config.device)
    rows: list[dict[str, Any]] = []
    for image_path in image_paths(spec, config.max_images):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes = read_boxes(image_path, spec)
        if not boxes:
            continue
        original = _score(model, image, config)
        variants = {mode: _score(model, _masked(image, boxes, mode), config) for mode in ("object_removed", "object_only", "context_only")}
        for index, box in enumerate(boxes):
            rows.append({
                "dataset": spec.name,
                "image": image_path.name,
                "object_id": index,
                "size_bucket": size_bucket(box),
                "original_score": original,
                "object_removed_score": variants["object_removed"],
                "object_only_score": variants["object_only"],
                "context_only_score": variants["context_only"],
                "object_sensitivity": original - variants["object_removed"],
                "background_dependence": variants["context_only"] / max(original, 1e-8),
            })
    write_rows(output / "causal_intervention.csv", rows)
    summary: dict[str, Any] = {"metadata": metadata(spec, config, checkpoint, "causal_intervention"), "rows": len(rows), "by_size": {}}
    for bucket in sorted({row["size_bucket"] for row in rows}):
        subset = [row for row in rows if row["size_bucket"] == bucket]
        summary["by_size"][bucket] = {key: float(np.mean([row[key] for row in subset])) for key in ("object_sensitivity", "background_dependence", "original_score")}
    write_json(output / "causal_intervention.json", summary)
    return summary
