"""Derive scene-load thresholds for adaptive Copy-Paste.

The input must contain final post-Mosaic/post-affine canvas boxes, not raw image
boxes.  A producer can write either a JSON list of records or JSONL records,
where each record is ``{"boxes": [[x1, y1, x2, y2], ...]}``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


def normalized_nearest_spacing(boxes: np.ndarray) -> float | None:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) < 2:
        return None
    sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 1e-6)
    gaps = []
    for i, box in enumerate(boxes):
        dx = np.maximum(np.maximum(box[0] - boxes[:, 2], boxes[:, 0] - box[2]), 0.0)
        dy = np.maximum(np.maximum(box[1] - boxes[:, 3], boxes[:, 1] - box[3]), 0.0)
        distance = np.sqrt(dx * dx + dy * dy)
        distance[i] = np.inf
        nearest = int(np.argmin(distance))
        scale = float(np.sqrt(sizes[i, 0] * sizes[i, 1]))
        gaps.append(float(distance[nearest]) / max(scale, 1e-6))
    return float(np.median(gaps))


def derive_scene_stats(box_sets: Iterable[np.ndarray]) -> dict[str, float | int]:
    counts = []
    spacings = []
    for boxes in box_sets:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        counts.append(len(boxes))
        spacing = normalized_nearest_spacing(boxes)
        if spacing is not None:
            spacings.append(spacing)
    if not counts:
        raise ValueError("no scene records supplied")
    if not spacings:
        raise ValueError("at least one scene must contain two boxes for spacing_q50")
    return {
        "count_q33": float(np.quantile(counts, 1 / 3)),
        "count_q67": float(np.quantile(counts, 2 / 3)),
        "spacing_q50": float(np.quantile(spacings, 0.5)),
        "scene_count": len(counts),
        "spacing_scene_count": len(spacings),
    }


def _load_records(path: Path) -> list[np.ndarray]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [np.asarray(record["boxes"], dtype=np.float32) for record in records]


def _resolve_split(root: Path, value):
    if isinstance(value, list):
        return [str((root / item).resolve()) for item in value]
    return str((root / value).resolve())


def collect_post_mosaic_boxes(data_yaml: Path, split: str, imgsz: int, samples: int, seed: int) -> list[np.ndarray]:
    """Run the project's real YOLODataset transform with CP/OACP disabled."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    random.seed(seed)
    np.random.seed(seed)
    hyp = get_cfg(overrides={
        "imgsz": imgsz,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "copy_paste_enabled": False,
        "close_mosaic": 0,
    })
    dataset = YOLODataset(
        img_path=_resolve_split(root, config[split]),
        imgsz=imgsz,
        data=config,
        task="detect",
        augment=True,
        hyp=hyp,
        batch_size=1,
        rect=False,
        cache=False,
    )
    limit = len(dataset) if samples <= 0 else min(samples, len(dataset))
    records = []
    for index in range(limit):
        sample = dataset[index]
        if "instances" in sample:
            boxes = np.asarray(sample["instances"].bboxes, dtype=np.float32)
        else:
            boxes = np.asarray(sample.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
        records.append(boxes)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON or JSONL post-Mosaic scene records")
    parser.add_argument("--data-yaml", type=Path, help="dataset YAML; collects real post-Mosaic canvases")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--samples", type=int, default=0, help="0 means all dataset samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True, help="output JSON threshold file")
    args = parser.parse_args()
    if bool(args.input) == bool(args.data_yaml):
        parser.error("provide exactly one of --input or --data-yaml")
    records = _load_records(args.input) if args.input else collect_post_mosaic_boxes(
        args.data_yaml, args.split, args.imgsz, args.samples, args.seed
    )
    stats = derive_scene_stats(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
