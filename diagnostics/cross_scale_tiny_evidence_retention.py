#!/usr/bin/env python3
"""Full fixed-split frozen P2->P3 evidence-retention probe.

The script is checkpointable by dataset/split. It never trains the detector. It
extracts frozen P2-pre, AvgPool2(P2), and actual P3-post representations, fits
small linear object/background probes on the train split, and evaluates on test.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def letterbox(image: Image.Image, size: int):
    width, height = image.size
    scale = min(size / width, size / height)
    new_size = (round(width * scale), round(height * scale))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_size[0]) // 2, (size - new_size[1]) // 2
    canvas.paste(resized, (pad_x, pad_y))
    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)) / 255
    return tensor, (scale, pad_x, pad_y, width, height)


def levir_records(root: Path, split: str):
    records = []
    image_root, label_root = root / "images" / split, root / "labels" / split
    for image in sorted(image_root.glob("*")):
        label = label_root / f"{image.stem}.txt"
        if not label.is_file():
            continue
        boxes = []
        for line in label.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            _, cx, cy, width, height = map(float, values)
            boxes.append((cx, cy, width, height, width * 512 * height * 512))
        if boxes:
            records.append((image, boxes, None))
    return records


def tinyperson_records(root: Path, split: str):
    annotation = root / "annotations" / "corner" / f"tiny_set_{split}_sw640_sh512.json"
    data = json.loads(annotation.read_text())
    images = {item["id"]: item for item in data["images"]}
    by_image = {key: [] for key in images}
    for item in data["annotations"]:
        x, y, width, height = item.get("bbox", [0, 0, 0, 0])
        if item.get("ignore") or item.get("uncertain") or item.get("logo"):
            continue
        if width <= 0 or height <= 0:
            continue
        size = float(item.get("size", math.sqrt(width * height)))
        by_image.setdefault(item["image_id"], []).append(((x, y, width, height), size))
    records = []
    for image_id, info in images.items():
        if not by_image.get(image_id):
            continue
        source = root / split / info["file_name"]
        if not source.is_file():
            continue
        x0, y0, x1, y1 = info["corner"]
        crop_width, crop_height = x1 - x0, y1 - y0
        boxes = []
        for (x, y, width, height), size in by_image[image_id]:
            boxes.append(((x + width / 2) / crop_width, (y + height / 2) / crop_height,
                          width / crop_width, height / crop_height, size))
        records.append((source, boxes, (x0, y0, x1, y1)))
    return records


def prepare(item, size: int):
    source, boxes, crop = item
    image = Image.open(source).convert("RGB")
    if crop is not None:
        image = image.crop(crop)
    tensor, metadata = letterbox(image, size)
    return tensor, metadata, boxes


def extract(model, records, size: int, batch_size: int):
    captures = {}
    handles = [
        model.model.model[index].register_forward_hook(
            lambda _, __, output, index=index: captures.__setitem__(index, output.detach().cpu())
        )
        for index in (2, 3)
    ]
    samples = {name: [] for name in ("p2", "avg", "p3")}
    locations = {name: [] for name in ("p2", "avg", "p3")}
    device = next(model.model.parameters()).device
    model.model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                tensors, metadata, all_boxes = [], [], []
                for item in chunk:
                    tensor, meta, boxes = prepare(item, size)
                    tensors.append(tensor)
                    metadata.append(meta)
                    scale, pad_x, pad_y, width, height = meta
                    all_boxes.append([
                        ((cx - bw / 2) * width * scale + pad_x,
                         (cy - bh / 2) * height * scale + pad_y,
                         (cx + bw / 2) * width * scale + pad_x,
                         (cy + bh / 2) * height * scale + pad_y,
                         object_size)
                        for cx, cy, bw, bh, object_size in boxes
                    ])
                captures.clear()
                model.model(torch.stack(tensors).to(device))
                feature_maps = {
                    "p2": captures[2],
                    "avg": torch.nn.functional.avg_pool2d(captures[2], 2, 2),
                    "p3": captures[3],
                }
                for batch_index, boxes in enumerate(all_boxes):
                    for name, batch_features in feature_maps.items():
                        feature = batch_features[batch_index].numpy()
                        _, height, width = feature.shape
                        grid_x, grid_y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
                        for object_index, (x1, y1, x2, y2, object_size) in enumerate(boxes):
                            fx1, fx2 = x1 * width / size, x2 * width / size
                            fy1, fy2 = y1 * height / size, y2 * height / size
                            positive = (grid_x >= fx1) & (grid_x <= fx2) & (grid_y >= fy1) & (grid_y <= fy2)
                            ex1, ex2 = fx1 - (fx2 - fx1), fx2 + (fx2 - fx1)
                            ey1, ey2 = fy1 - (fy2 - fy1), fy2 + (fy2 - fy1)
                            other = np.zeros((height, width), dtype=bool)
                            for other_index, (ox1, oy1, ox2, oy2, _) in enumerate(boxes):
                                if other_index != object_index:
                                    other |= ((grid_x * size / width >= ox1) & (grid_x * size / width <= ox2) &
                                              (grid_y * size / height >= oy1) & (grid_y * size / height <= oy2))
                            negative = ((grid_x >= ex1) & (grid_x <= ex2) & (grid_y >= ey1) &
                                        (grid_y <= ey2) & ~positive & ~other)
                            positives, negatives = np.argwhere(positive), np.argwhere(negative)
                            if not len(positives):
                                positives = np.array([[min(height - 1, max(0, int((y1 + y2) * height / (2 * size)))),
                                                       min(width - 1, max(0, int((x1 + x2) * width / (2 * size))))]])
                            np.random.shuffle(positives)
                            np.random.shuffle(negatives)
                            for yy, xx in positives[:5]:
                                samples[name].append((feature[:, yy, xx], 1, object_size))
                            for yy, xx in negatives[:5]:
                                samples[name].append((feature[:, yy, xx], 0, object_size))
                            center_x, center_y = (x1 + x2) * width / (2 * size), (y1 + y2) * height / (2 * size)
                            radius = max(2, int(max(x2 - x1, y2 - y1) * max(width, height) / size) + 2)
                            locations[name].append((feature, center_x, center_y, radius, object_size))
                if start % (batch_size * 10) == 0:
                    print(f"extracted {start + len(chunk)}/{len(records)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return samples, locations


def bootstrap(values, repeats=300):
    values = np.asarray(values, dtype=float)
    estimates = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(repeats)]
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def fit_and_score(train, test, locations, input_size):
    result = {}
    for name in ("p2", "avg", "p3"):
        train_x = np.stack([row[0] for row in train[name]])
        train_y = np.asarray([row[1] for row in train[name]])
        classifier = LogisticRegression(max_iter=150, class_weight="balanced", solver="liblinear", random_state=42)
        classifier.fit(train_x, train_y)
        test_x = np.stack([row[0] for row in test[name]])
        test_y = np.asarray([row[1] for row in test[name]])
        logits = classifier.decision_function(test_x)
        scores = 1 / (1 + np.exp(-logits))
        errors = []
        for feature, center_x, center_y, radius, _ in locations[name]:
            heatmap = classifier.decision_function(feature.reshape(feature.shape[0], -1).T).reshape(feature.shape[1:])
            y0, y1 = max(0, int(center_y - radius)), min(heatmap.shape[0], int(center_y + radius + 1))
            x0, x1 = max(0, int(center_x - radius)), min(heatmap.shape[1], int(center_x + radius + 1))
            yy, xx = np.unravel_index(np.argmax(heatmap[y0:y1, x0:x1]), (y1 - y0, x1 - x0))
            errors.append(float(math.hypot(xx + x0 + 0.5 - center_x, yy + y0 + 0.5 - center_y) * input_size / feature.shape[2]))
        errors = np.asarray(errors)
        result[name] = {
            "pr_auc": float(average_precision_score(test_y, scores)),
            "positive_mean_logit": float(logits[test_y == 1].mean()),
            "hard_negative_margin": float(logits[test_y == 1].mean() - logits[test_y == 0].mean()),
            "mean_center_error_px": float(errors.mean()),
            "mean_center_error_ci": bootstrap(errors),
            "median_center_error_px": float(np.median(errors)),
            "hit_at_4px": float(np.mean(errors <= 4)),
            "hit_at_8px": float(np.mean(errors <= 8)),
            "hit_at_8_ci": bootstrap((errors <= 8).astype(float)),
            "n_objects": int(len(errors)),
        }
    result["deltas"] = {
        f"p2_minus_{name}": {
            "pr_auc": result["p2"]["pr_auc"] - result[name]["pr_auc"],
            "mean_center_error_px": result["p2"]["mean_center_error_px"] - result[name]["mean_center_error_px"],
            "hit_at_8px": result["p2"]["hit_at_8px"] - result[name]["hit_at_8px"],
        }
        for name in ("avg", "p3")
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=Path("/marimo"))
    parser.add_argument("--output", type=Path, default=Path("runs/cross_scale_probe_full.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", choices=("levir", "tinyperson"), default=("levir", "tinyperson"))
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    sys_path = str(args.project_root / "models_related/ultralytics")
    import sys
    sys.path[:0] = [str(args.project_root), sys_path]
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    specs = {
        "levir": ("LEVIR-Ship", args.dataset_root / "yolo_code/datasets/levir_ship_yolo_seed42", "duyle2408/levir-ship-yolo-baselines", "train/yolov8n_p2_baseline_seed42/weights/best.pt", 512),
        "tinyperson": ("TinyPerson", args.dataset_root / "TinyPerson", "duyle2408/tinyperson-yolov8n-baselines", "runs/yolov8n_p2p3p4_plain/seed_42/weights/best.pt", 640),
    }
    output = {"protocol": "full fixed split; P2, AvgPool2(P2), actual P3; max 5 positive and 5 hard-negative cells/object; seed=42", "results": []}
    for key in args.datasets:
        name, root, repo, checkpoint, image_size = specs[key]
        records = (levir_records(root, "train"), levir_records(root, "test")) if key == "levir" else (tinyperson_records(root, "train"), tinyperson_records(root, "test"))
        print(f"{name}: train={len(records[0])}, test={len(records[1])}", flush=True)
        model = YOLO(hf_hub_download(repo, checkpoint, repo_type="dataset", token=os.environ["HF_TOKEN"]))
        train_samples, _ = extract(model, records[0], image_size, args.batch_size)
        test_samples, test_locations = extract(model, records[1], image_size, args.batch_size)
        metrics = fit_and_score(train_samples, test_samples, test_locations, image_size)
        output["results"].append({"dataset": name, "checkpoint": f"{repo}:{checkpoint}", "train_images": len(records[0]), "test_images": len(records[1]), "metrics": metrics})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    import os
    main()
