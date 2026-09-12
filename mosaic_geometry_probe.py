#!/usr/bin/env python3
"""Train-free probe for Mosaic-policy geometry survival through RandomPerspective.

The probe samples four-image LEVIR Mosaics, records geometry immediately after
Mosaic, then applies the project's canonical RandomPerspective implementation
with current and milder settings. It never loads a model or starts training.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "models_related" / "ultralytics"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from project_ultralytics.mosaic_policy import (  # noqa: E402
    effective_count,
    mosaic_crops,
    resized_shape,
    simulate_visible_boxes,
)
from ultralytics.data.augment import RandomPerspective  # noqa: E402

POLICIES = ("M0_standard", "M1_visibility", "M2_occupancy")
POLICY_SEEDS = {"M0_standard": 11, "M1_visibility": 23, "M2_occupancy": 37}
GEOMETRIES = {
    "current": {"scale": 0.5, "translate": 0.10},
    "mild": {"scale": 0.15, "translate": 0.05},
}
THRESHOLD = 0.70
SIZE_THRESHOLDS = (20.0, 12.0, 8.0)


def _label_paths(data_yaml: Path) -> list[Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train = Path(config["train"])
    if not train.is_absolute():
        train = root / train
    if "images" in train.parts:
        train = Path(*["labels" if part == "images" else part for part in train.parts])
    return sorted(train.glob("*.txt"))


def _metadata(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for label_path in paths:
        image_path = Path(*["images" if part == "labels" else part for part in label_path.parts]).with_suffix(".png")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read paired image for {label_path}: {image_path}")
        h, w = image.shape[:2]
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) >= 5:
                boxes.append([float(x) for x in values[1:5]])
        records.append({"shape": (h, w), "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4)})
    return records


def _center(imgsz: int) -> tuple[int, int]:
    # Mirrors Mosaic._sample_center: return (yc, xc).
    border = -imgsz // 2
    return int(random.uniform(-border, 2 * imgsz + border)), int(random.uniform(-border, 2 * imgsz + border))


def _candidate_centers(imgsz: int, count: int) -> list[tuple[int, int]]:
    border = -imgsz // 2
    return [
        (int(random.uniform(-border, 2 * imgsz + border)), int(random.uniform(-border, 2 * imgsz + border)))
        for _ in range(max(count, 1))
    ]


def _xywh_to_xyxy(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    return out


def _mosaic_boxes(metadata: list[dict[str, Any]], indices: list[int], imgsz: int, xc: int, yc: int) -> list[dict[str, float]]:
    shapes = [resized_shape(metadata[i]["shape"], imgsz) for i in indices]
    crops = mosaic_crops(shapes, imgsz, xc, yc)
    records: list[dict[str, float]] = []
    for quadrant, (index, shape, crop) in enumerate(zip(indices, shapes, crops)):
        h, w = shape
        if quadrant == 0:
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
        elif quadrant == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, imgsz * 2), yc
        elif quadrant == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(imgsz * 2, yc + h)
        else:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, imgsz * 2), min(imgsz * 2, yc + h)
        x1b, y1b, x2b, y2b = crop
        padw, padh = x1a - x1b, y1a - y1b
        boxes = _xywh_to_xyxy(metadata[index]["bboxes"], shape)
        for box in boxes:
            ox1, oy1, ox2, oy2 = map(float, box)
            original_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            cx1, cy1 = max(ox1, x1b), max(oy1, y1b)
            cx2, cy2 = min(ox2, x2b), min(oy2, y2b)
            visible_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            if original_area <= 0:
                continue
            if visible_area <= 0:
                cx = min(max((ox1 + ox2) / 2, x1b), x2b)
                cy = min(max((oy1 + oy2) / 2, y1b), y2b)
                cx1 = cx2 = cx
                cy1 = cy2 = cy
            records.append({
                "original_area": original_area,
                "pre_area": visible_area,
                "pre_box": [cx1 + padw, cy1 + padh, cx2 + padw, cy2 + padh],
            })
    return records


def _select_layout(metadata: list[dict[str, Any]], policy: str, imgsz: int) -> tuple[list[int], int, int, float | None]:
    n = len(metadata)
    anchor = random.randrange(n)
    indices = [anchor] + [random.randrange(n) for _ in range(3)]
    if policy == "M0_standard":
        yc, xc = _center(imgsz)
        return indices, xc, yc, None
    if policy == "M1_visibility":
        yc, xc = _center(imgsz)
        reference = simulate_visible_boxes(metadata, indices, imgsz, xc, yc)
        reference_count = int(np.count_nonzero(reference > 0))
        proposals = []
        for xc0, yc0 in _candidate_centers(imgsz, 16):
            visibility = simulate_visible_boxes(metadata, indices, imgsz, xc0, yc0)
            visible_count = int(np.count_nonzero(visibility > 0))
            if visible_count == reference_count and not np.any((reference > 0) & (visibility <= 0)):
                partial = np.count_nonzero((visibility > 0) & (visibility < THRESHOLD))
                score = float(visibility[reference > 0].sum() - 0.1 * partial)
                proposals.append((score, xc0, yc0))
        if not proposals:
            return indices, xc, yc, None
        proposals.sort(reverse=True)
        _, xc, yc = random.choice(proposals[:4])
        return indices, xc, yc, None
    if policy == "M2_occupancy":
        target = float(random.choice([len(item["bboxes"]) for item in metadata]))
        proposals = []
        for _ in range(16):
            candidates = [anchor] + [random.randrange(n) for _ in range(3)]
            yc, xc = _center(imgsz)
            visibility = simulate_visible_boxes(metadata, candidates, imgsz, xc, yc)
            count = effective_count(visibility, THRESHOLD)
            proposals.append((abs(count - target), candidates, xc, yc))
        valid = [item for item in proposals if item[0] <= 1.0]
        chosen = sorted(valid or proposals, key=lambda item: item[0])[:4]
        _, indices, xc, yc = random.choice(chosen)
        return indices, xc, yc, target
    raise ValueError(f"unknown policy: {policy}")


def _transform_polygon(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ matrix.T
    transformed[:, :2] /= np.maximum(transformed[:, 2:3], 1e-12)
    return transformed[:, :2].astype(np.float32)


def _clipped_area(polygon: np.ndarray, size: int) -> float:
    if len(polygon) < 3:
        return 0.0
    canvas = np.asarray([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    try:
        area, _ = cv2.intersectConvexConvex(polygon, canvas)
        return max(float(area), 0.0)
    except cv2.error:
        return 0.0


def _metrics(records: list[dict[str, float]], matrix: np.ndarray | None, output_size: int, target: float | None) -> dict[str, Any]:
    original = np.asarray([item["original_area"] for item in records], dtype=np.float64)
    pre_area = np.asarray([item["pre_area"] for item in records], dtype=np.float64)
    if matrix is None:
        final_area = pre_area.copy()
    else:
        final_area = []
        for item in records:
            x1, y1, x2, y2 = item["pre_box"]
            polygon = _transform_polygon(np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32), matrix)
            final_area.append(_clipped_area(polygon, output_size))
        final_area = np.asarray(final_area, dtype=np.float64)
    pre_visibility = np.minimum(pre_area / np.maximum(original, 1e-12), 1.0)
    post_visibility = np.minimum(final_area / np.maximum(original, 1e-12), 1.0)
    pre_sqrt = np.sqrt(pre_area)
    post_sqrt = np.sqrt(final_area)
    result: dict[str, Any] = {
        "gt_count": int(len(records)),
        "mean_visibility": float(np.mean(post_visibility)) if len(post_visibility) else 0.0,
        "partial_fraction": float(np.mean((post_visibility > 0) & (post_visibility < THRESHOLD))) if len(post_visibility) else 0.0,
        "removed_fraction": float(np.mean(post_visibility <= 0)) if len(post_visibility) else 0.0,
        "fullish_fraction": float(np.mean(post_visibility >= THRESHOLD)) if len(post_visibility) else 0.0,
        "visible_gt_count": float(np.count_nonzero(post_visibility > 0)),
        "effective_count": float(effective_count(post_visibility, THRESHOLD)),
        "mean_sqrt_area": float(np.mean(post_sqrt)) if len(post_sqrt) else 0.0,
        "pre_mean_visibility": float(np.mean(pre_visibility)) if len(pre_visibility) else 0.0,
        "pre_partial_fraction": float(np.mean((pre_visibility > 0) & (pre_visibility < THRESHOLD))) if len(pre_visibility) else 0.0,
        "pre_removed_fraction": float(np.mean(pre_visibility <= 0)) if len(pre_visibility) else 0.0,
        "pre_fullish_fraction": float(np.mean(pre_visibility >= THRESHOLD)) if len(pre_visibility) else 0.0,
        "pre_visible_gt_count": float(np.count_nonzero(pre_visibility > 0)),
        "pre_effective_count": float(effective_count(pre_visibility, THRESHOLD)),
        "pre_mean_sqrt_area": float(np.mean(pre_sqrt)) if len(pre_sqrt) else 0.0,
        "size_crossings": {
            f">={int(threshold)}_to_<{int(threshold)}": int(np.count_nonzero((pre_sqrt >= threshold) & (post_sqrt < threshold)))
            for threshold in SIZE_THRESHOLDS
        },
    }
    if target is not None:
        result["occupancy_target"] = target
        result["occupancy_error"] = abs(result["effective_count"] - target)
        result["pre_occupancy_error"] = abs(result["pre_effective_count"] - target)
    return result


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        "mean_visibility", "partial_fraction", "removed_fraction", "fullish_fraction",
        "visible_gt_count", "effective_count", "mean_sqrt_area", "pre_mean_visibility",
        "pre_partial_fraction", "pre_removed_fraction", "pre_fullish_fraction",
        "pre_visible_gt_count", "pre_effective_count", "pre_mean_sqrt_area",
    ]
    output = {key: float(np.mean([item[key] for item in samples])) for key in scalar_keys}
    for key in ("occupancy_error", "pre_occupancy_error", "occupancy_target"):
        values = [item[key] for item in samples if key in item]
        if values:
            output[key] = float(np.mean(values))
    output["sample_count"] = len(samples)
    output["size_crossings"] = {
        key: int(sum(item["size_crossings"][key] for item in samples))
        for key in samples[0]["size_crossings"]
    } if samples else {}
    return output


def _decision_metrics(summaries: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    m0, m1, m2 = summaries["M0_standard"], summaries["M1_visibility"], summaries["M2_occupancy"]
    retention: dict[str, dict[str, float | None]] = {}
    for metric in ("mean_visibility", "partial_fraction", "removed_fraction", "fullish_fraction", "visible_gt_count", "effective_count", "mean_sqrt_area"):
        pre_delta = m1["pre"][metric] - m0["pre"][metric]
        retention[metric] = {}
        for geometry in ("current", "mild"):
            post_delta = m1[geometry][metric] - m0[geometry][metric]
            retention[metric][geometry] = None if abs(pre_delta) < 1e-12 else post_delta / pre_delta
    return {
        "m1_advantage_delta": {
            metric: {
                "pre": m1["pre"][metric] - m0["pre"][metric],
                "current": m1["current"][metric] - m0["current"][metric],
                "mild": m1["mild"][metric] - m0["mild"][metric],
            }
            for metric in retention
        },
        "m1_advantage_retention": retention,
        "m2_occupancy_error": {
            "pre": m2["pre"].get("occupancy_error", m2["pre"].get("pre_occupancy_error")),
            "current": m2["current"]["occupancy_error"],
            "mild": m2["mild"]["occupancy_error"],
            "current_delta": m2["current"]["occupancy_error"] - m2["pre"].get("occupancy_error", m2["pre"].get("pre_occupancy_error")),
            "mild_delta": m2["mild"]["occupancy_error"] - m2["pre"].get("occupancy_error", m2["pre"].get("pre_occupancy_error")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.imgsz <= 0:
        parser.error("--samples and --imgsz must be positive")
    paths = _label_paths(args.data_yaml)
    metadata = _metadata(paths)
    if len(metadata) < 1:
        raise RuntimeError("no training labels found")
    results: dict[str, dict[str, list[dict[str, Any]]]] = {
        policy: {"pre": [], "current": [], "mild": []} for policy in POLICIES
    }
    for sample_index in range(args.samples):
        for policy in POLICIES:
            random.seed(args.seed + sample_index * 1009 + POLICY_SEEDS[policy])
            indices, xc, yc, target = _select_layout(metadata, policy, args.imgsz)
            records = _mosaic_boxes(metadata, indices, args.imgsz, xc, yc)
            pre = _metrics(records, None, args.imgsz, target)
            results[policy]["pre"].append(pre)
            for geometry, config in GEOMETRIES.items():
                random.seed(args.seed + sample_index * 1009 + POLICY_SEEDS[policy])
                transform = RandomPerspective(
                    degrees=0.0, translate=config["translate"], scale=config["scale"],
                    shear=0.0, perspective=0.0, size=(args.imgsz, args.imgsz),
                )
                params = transform.get_params({"img": np.full((args.imgsz * 2, args.imgsz * 2, 3), 114, dtype=np.uint8)})
                results[policy][geometry].append(_metrics(records, params["M"], args.imgsz, target))
        if (sample_index + 1) % 500 == 0:
            print(f"PROGRESS {sample_index + 1}/{args.samples}", flush=True)
    summaries = {policy: {stage: _summary(values) for stage, values in stages.items()} for policy, stages in results.items()}
    report = {
        "data_yaml": str(args.data_yaml.resolve()),
        "seed": args.seed,
        "samples": args.samples,
        "imgsz": args.imgsz,
        "geometry": GEOMETRIES,
        "visibility_threshold": THRESHOLD,
        "policies": summaries,
        "decision_metrics": _decision_metrics(summaries),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["policies"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
