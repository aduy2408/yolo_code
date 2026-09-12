#!/usr/bin/env python3
"""Measure scene-statistics shift caused by Copy-Paste variants.

The probe works on YOLO detection datasets and compares raw train, sampled
Copy-Paste train, and val/test distributions. It intentionally measures label
geometry only, so it is train-free and independent of model checkpoints.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from copy_paste_protocol import variant_overrides


FEATURES = (
    "objects_per_image",
    "empty_image",
    "mean_bbox_area_fraction",
    "nearest_neighbor_distance",
    "mean_pairwise_distance",
    "mean_overlap",
    "mean_border_distance",
    "mean_bbox_width",
    "mean_bbox_height",
    "max_cluster_size",
)


def _resolve_split(root: Path, value: str | list[str]) -> list[Path]:
    values = [value] if isinstance(value, str) else value
    paths: list[Path] = []
    for item in values:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        # YOLO YAMLs conventionally point at images while this probe consumes
        # the paired normalized-label files.
        if path.name == "images" or "images" in path.parts:
            parts = ["labels" if part == "images" else part for part in path.parts]
            path = Path(*parts)
        if path.suffix:
            paths.append(path.with_suffix(".txt"))
        elif path.is_dir():
            paths.extend(sorted(path.glob("*.txt")))
        else:
            paths.extend(sorted(path.glob("*.txt")))
    return paths


def _image_path(label_path: Path) -> Path:
    parts = list(label_path.parts)
    for i, part in enumerate(parts):
        if part == "labels":
            parts[i] = "images"
            break
    path = Path(*parts).with_suffix(".png")
    if path.exists():
        return path
    for suffix in (".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find image for {label_path}")


def _read_boxes(path: Path, width: int, height: int) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) >= 5:
            _, xc, yc, w, h = map(float, values[:5])
            rows.append([(xc - w / 2) * width, (yc - h / 2) * height,
                         (xc + w / 2) * width, (yc + h / 2) * height])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def _pairwise_distances(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) < 2:
        return np.empty(0, dtype=np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt((delta * delta).sum(axis=2))
    distances[np.diag_indices_from(distances)] = np.inf
    return distances[np.triu_indices(len(boxes), 1)]


def _overlap(boxes: np.ndarray) -> float:
    if len(boxes) < 2:
        return 0.0
    values = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            x1, y1 = np.maximum(boxes[i, :2], boxes[j, :2])
            x2, y2 = np.minimum(boxes[i, 2:], boxes[j, 2:])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area = max(1e-8, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
            values.append(inter / area)
    return float(np.mean(values)) if values else 0.0


def _max_cluster_size(boxes: np.ndarray, width: int, height: int, radius_fraction: float) -> float:
    if len(boxes) < 2:
        return float(len(boxes))
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    radius = radius_fraction * max(width, height)
    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if np.linalg.norm(centers[i] - centers[j]) <= radius:
                union(i, j)
    sizes: dict[int, int] = {}
    for index in range(len(boxes)):
        root = find(index)
        sizes[root] = sizes.get(root, 0) + 1
    return float(max(sizes.values(), default=0))


def _record(boxes: np.ndarray, width: int, height: int, cluster_radius_fraction: float) -> dict[str, float]:
    count = len(boxes)
    if count == 0:
        return {name: 0.0 for name in FEATURES} | {"empty_image": 1.0}
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    border = np.minimum.reduce((centers[:, 0], width - centers[:, 0], centers[:, 1], height - centers[:, 1]))
    distances = _pairwise_distances(boxes)
    return {
        "objects_per_image": float(count),
        "empty_image": 0.0,
        "mean_bbox_area_fraction": float(np.mean(areas) / (width * height)),
        "nearest_neighbor_distance": float(np.min(distances)) if len(distances) else float(max(width, height)),
        "mean_pairwise_distance": float(np.mean(distances)) if len(distances) else 0.0,
        "mean_overlap": _overlap(boxes),
        "mean_border_distance": float(np.mean(border) / max(width, height)),
        "mean_bbox_width": float(np.mean(boxes[:, 2] - boxes[:, 0]) / width),
        "mean_bbox_height": float(np.mean(boxes[:, 3] - boxes[:, 1]) / height),
        "max_cluster_size": _max_cluster_size(boxes, width, height, cluster_radius_fraction),
    }


def _summarize(records: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([record[name] for record in records])) for name in FEATURES}


def _wasserstein(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    a, b = np.sort(np.asarray(list(values_a), dtype=np.float64)), np.sort(np.asarray(list(values_b), dtype=np.float64))
    if not len(a) or not len(b):
        return float("nan")
    grid = np.unique(np.concatenate([a, b]))
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    if len(grid) == 1:
        return 0.0
    return float(np.sum(np.abs(ca[:-1] - cb[:-1]) * np.diff(grid)))


def _js(values_a: Iterable[float], values_b: Iterable[float], bins: int = 32) -> float:
    a, b = np.asarray(list(values_a), dtype=np.float64), np.asarray(list(values_b), dtype=np.float64)
    if not len(a) or not len(b):
        return float("nan")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, edges); pb, _ = np.histogram(b, edges)
    pa = pa / max(pa.sum(), 1); pb = pb / max(pb.sum(), 1)
    m = (pa + pb) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_a = np.sum(np.where(pa > 0, pa * np.log2(pa / np.maximum(m, 1e-12)), 0.0))
        kl_b = np.sum(np.where(pb > 0, pb * np.log2(pb / np.maximum(m, 1e-12)), 0.0))
    return float((kl_a + kl_b) / 2)


def _load_records(label_paths: list[Path], cluster_radius_fraction: float) -> list[dict[str, float]]:
    records = []
    for label_path in label_paths:
        image = cv2.imread(str(_image_path(label_path)), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {_image_path(label_path)}")
        h, w = image.shape[:2]
        records.append(_record(_read_boxes(label_path, w, h), w, h, cluster_radius_fraction))
    return records


def _load_augmented_records(label_paths: list[Path], variant: str, seed: int, cluster_radius_fraction: float) -> list[dict[str, float]]:
    # Import the project transform only when this mode is requested. This keeps
    # raw-statistics use independent of Ultralytics installation details.
    from project_ultralytics.copy_paste import SmallObjectCopyPaste
    from ultralytics.utils.instance import Instances

    source_paths = [_image_path(path) for path in label_paths]
    labels = []
    for label_path, image_path in zip(label_paths, source_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        h, w = image.shape[:2]
        boxes = _read_boxes(label_path, w, h)
        labels.append({
            "bboxes": boxes, "cls": np.zeros((len(boxes), 1), dtype=np.float32),
            "bbox_format": "xyxy", "normalized": False, "shape": (h, w),
        })
    dataset = type("ProbeDataset", (), {"labels": labels, "im_files": [str(p) for p in source_paths]})()
    settings = variant_overrides(variant)
    transform = None if variant == "cp0" else SmallObjectCopyPaste(
        dataset=dataset, p=settings["copy_paste_p"], unit=settings["copy_paste_unit"],
        copies=settings["copy_paste_copies"], placement=settings["copy_paste_placement"],
        max_overlap=settings["copy_paste_max_overlap"], padding=settings["copy_paste_padding"],
        scale=settings["copy_paste_scale"], blend=settings["copy_paste_blend"],
        max_trials=settings["copy_paste_max_trials"], allow_empty_target=settings["copy_paste_allow_empty_target"],
        allow_same_source=settings["copy_paste_allow_same_source"],
        cluster_expand=settings.get("copy_paste_cluster_expand", 3.0),
        cluster_min_objects=settings.get("copy_paste_cluster_min_objects", 2),
        rng=random.Random(seed),
    )
    records = []
    for index, (label_path, image_path) in enumerate(zip(label_paths, source_paths)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        boxes = _read_boxes(label_path, w, h)
        instances = Instances(boxes.copy(), np.zeros((len(boxes), 0, 2), dtype=np.float32), bbox_format="xyxy", normalized=False)
        sample = {"img": image, "instances": instances, "cls": np.zeros((len(boxes), 1), dtype=np.float32), "image_index": index}
        if transform is not None:
            sample = transform(sample)
        records.append(_record(np.asarray(sample["instances"].bboxes), w, h, cluster_radius_fraction))
    return records


def _distance_table(groups: dict[str, list[dict[str, float]]], reference: str) -> dict[str, dict[str, float]]:
    ref = groups[reference]
    output = {}
    for group, records in groups.items():
        output[group] = {}
        for feature in FEATURES:
            output[group][feature] = _wasserstein((r[feature] for r in records), (r[feature] for r in ref))
        output[group]["mean_js"] = float(np.mean([_js((r[feature] for r in records), (r[feature] for r in ref)) for feature in FEATURES]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--variant", choices=("cp0", "cp1_single1", "cp2_single2", "cp3_cluster1"), default="cp0")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--samples", type=int, default=0, help="sample this many train images; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-radius-fraction", type=float, default=0.1,
                        help="center-distance threshold as a fraction of max image dimension")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cluster_radius_fraction < 0:
        parser.error("--cluster-radius-fraction must be non-negative")
    if args.samples < 0:
        parser.error("--samples must be non-negative")
    config = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", args.data_yaml.parent))
    if not root.is_absolute():
        root = (args.data_yaml.parent / root).resolve()
    split_paths = {name: _resolve_split(root, config[name]) for name in ("train", "val", "test") if name in config}
    train_paths = split_paths["train"]
    if args.samples and args.samples < len(train_paths):
        train_paths = random.Random(args.seed).sample(train_paths, args.samples)
    groups = {
        "train_raw": _load_records(train_paths, args.cluster_radius_fraction),
        "train_augmented": _load_augmented_records(train_paths, args.variant, args.seed, args.cluster_radius_fraction),
        args.eval_split: _load_records(split_paths[args.eval_split], args.cluster_radius_fraction),
    }
    result = {
        "data_yaml": str(args.data_yaml.resolve()), "variant": args.variant, "seed": args.seed,
        "cluster_radius_fraction": args.cluster_radius_fraction,
        "counts": {name: len(records) for name, records in groups.items()},
        "means": {name: _summarize(records) for name, records in groups.items()},
        "distance_to_eval": _distance_table(groups, args.eval_split),
        "augmentation_delta_vs_raw": {
            feature: groups["train_augmented"] and float(np.mean([r[feature] for r in groups["train_augmented"]]) - np.mean([r[feature] for r in groups["train_raw"]]))
            for feature in FEATURES
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
