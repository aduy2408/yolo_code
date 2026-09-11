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
        x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
        crops.append((int(x1b), int(y1b), int(x1b + max(x2a - x1a, 0)), int(y1b + max(y2a - y1a, 0))))
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
        return None
    data = np.load(path, allow_pickle=False)
    return {"im_file": data["im_file"], "descriptor": data["descriptor"]}
