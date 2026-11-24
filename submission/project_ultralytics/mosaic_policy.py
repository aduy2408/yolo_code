"""Policy selection helpers for project-owned Mosaic variants.

The helpers in this module are deliberately independent from Ultralytics' label
objects. They operate on cached YOLO metadata and are therefore cheap enough to
use before donor images are loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable
import json

import numpy as np


@dataclass(frozen=True)
class MosaicProposal:
    """A candidate donor/center choice for a four-image Mosaic."""

    donor_indices: tuple[int, ...]
    xc: int
    yc: int
    visibility: np.ndarray | None = None
    effective_count: float | None = None
    score: float | None = None


def resized_shape(shape: Iterable[int], imgsz: int) -> tuple[int, int]:
    """Return the long-side-preserving shape used by ``BaseDataset.load_image``."""
    h, w = (int(shape[0]), int(shape[1]))
    r = float(imgsz) / max(h, w)
    return min(int(np.ceil(h * r)), imgsz), min(int(np.ceil(w * r)), imgsz)


def _xywh_to_xyxy(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    h, w = shape
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    return out


def _visible_fraction(boxes: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Compute box area retained by a source crop, without filtering boxes."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    x1, y1, x2, y2 = crop
    clipped = boxes.copy()
    clipped[:, 0] = np.maximum(clipped[:, 0], x1)
    clipped[:, 1] = np.maximum(clipped[:, 1], y1)
    clipped[:, 2] = np.minimum(clipped[:, 2], x2)
    clipped[:, 3] = np.minimum(clipped[:, 3], y2)
    before = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    after = np.maximum(clipped[:, 2] - clipped[:, 0], 0) * np.maximum(clipped[:, 3] - clipped[:, 1], 0)
    return (after / np.maximum(before, 1e-9)).astype(np.float32)


def mosaic_crops(shapes: list[tuple[int, int]], imgsz: int, xc: int, yc: int) -> list[tuple[int, int, int, int]]:
    """Return source-space visible crops for a four-image Mosaic."""
    crops = []
    for i, (h, w) in enumerate(shapes):
        if i == 0:
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, imgsz * 2), yc
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(imgsz * 2, yc + h)
        else:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, imgsz * 2), min(imgsz * 2, yc + h)
        pasted_w, pasted_h = max(x2a - x1a, 0), max(y2a - y1a, 0)
        if i == 0:
            x1b, y1b = w - pasted_w, h - pasted_h
        elif i == 1:
            x1b, y1b = 0, h - pasted_h
        elif i == 2:
            x1b, y1b = w - pasted_w, 0
        else:
            x1b, y1b = 0, 0
        crops.append((int(x1b), int(y1b), int(x1b + pasted_w), int(y1b + pasted_h)))
    return crops


def simulate_visible_boxes(metadata: list[dict], indices: list[int], imgsz: int, xc: int, yc: int) -> np.ndarray:
    """Simulate all GT visibility fractions for anchor followed by three donors."""
    shapes = [resized_shape(metadata[i].get("shape", (imgsz, imgsz)), imgsz) for i in indices]
    crops = mosaic_crops(shapes, imgsz, xc, yc)
    values = []
    for index, crop, shape in zip(indices, crops, shapes):
        boxes = _xywh_to_xyxy(np.asarray(metadata[index].get("bboxes", [])), shape)
        values.append(_visible_fraction(boxes, crop))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def effective_count(visibility: np.ndarray, threshold: float = 0.7) -> float:
    """Count visible evidence softly, saturating at one per source object."""
    if len(visibility) == 0:
        return 0.0
    return float(np.minimum(visibility / max(threshold, 1e-6), 1.0).sum())


def candidate_centers(imgsz: int, border: tuple[int, int], count: int) -> list[tuple[int, int]]:
    """Sample centers with the exact standard Mosaic support."""
    return [
        (int(random.uniform(-border[0], 2 * imgsz + border[0])), int(random.uniform(-border[1], 2 * imgsz + border[1])))
        for _ in range(max(int(count), 1))
    ]


def load_context_cache(path: str | Path | None) -> dict[str, np.ndarray] | None:
    """Load a descriptor cache created by ``build_mosaic_context_cache.py``."""
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mosaic context cache does not exist: {path}")
    data = np.load(path, allow_pickle=False)
    files = np.asarray([str(Path(item).expanduser().resolve()) for item in data["im_file"]])
    return {"im_file": files, "descriptor": data["descriptor"]}


def load_scale_statistics(path: str | Path | None) -> dict | None:
    """Load training-only object scale statistics used by post-scale Mosaic."""
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mosaic scale statistics do not exist: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or "sqrt_area_quantiles" not in value:
        raise ValueError(f"Invalid Mosaic scale statistics: {path}")
    return value


def source_priority(metadata: dict) -> float:
    """Return inverse median object scale, with empty images assigned zero."""
    boxes = np.asarray(metadata.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return 0.0
    scales = np.sqrt(np.maximum(boxes[:, 2] * boxes[:, 3], 1e-12))
    return float(1.0 / max(float(np.median(scales)), 1e-6))


def cluster_crop(boxes: np.ndarray, shape: tuple[int, int], anchor: int, imgsz: int,
                 min_fraction: float = 0.25, max_fraction: float = 0.65,
                 context_expand: float = 1.5) -> tuple[int, int, int, int] | None:
    """Choose a square source crop around an object and its nearby cluster."""
    boxes = _xywh_to_xyxy(np.asarray(boxes), shape)
    if not len(boxes):
        return None
    anchor = int(np.clip(anchor, 0, len(boxes) - 1))
    ax1, ay1, ax2, ay2 = boxes[anchor]
    radius = max(float(ax2 - ax1), float(ay2 - ay1)) * max(context_expand, 0.1)
    cx, cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    nearby = ((boxes[:, 0] + boxes[:, 2]) / 2 - cx) ** 2 + ((boxes[:, 1] + boxes[:, 3]) / 2 - cy) ** 2 <= radius**2
    selected = boxes[nearby]
    x1, y1 = selected[:, 0].min(), selected[:, 1].min()
    x2, y2 = selected[:, 2].max(), selected[:, 3].max()
    side = np.clip(max(x2 - x1, y2 - y1) * max(context_expand, 1.0), imgsz * min_fraction, imgsz * max_fraction)
    h, w = shape
    side = min(float(side), float(min(h, w)))
    x1 = (cx + (x1 + x2) / 2) / 2 - side / 2
    y1 = (cy + (y1 + y2) / 2) / 2 - side / 2
    x1 = int(np.clip(x1, 0, max(w - side, 0)))
    y1 = int(np.clip(y1, 0, max(h - side, 0)))
    return x1, y1, int(min(w, x1 + side)), int(min(h, y1 + side))
