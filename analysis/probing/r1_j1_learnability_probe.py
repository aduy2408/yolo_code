"""Frozen R1/J1 candidate learnability probe.

Fits a small held-out ridge ranker to candidate evidence without changing the
 detector. It compares raw GT-class score ordering with the learned ordering on
 TAL-positive candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from analysis.probing.candidate_evidence_probe import _candidate_tensors, box_iou
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics import YOLO


def read_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, cx, cy, bw, bh = map(float, values[:5])
        cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
        rows.append((int(cls), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return rows


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(0)
    scale = x.std(0) + 1e-6
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y), mean, scale


def predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    weights, mean, scale = model
    z = (x - mean) / scale
    return np.column_stack((np.ones(len(z)), z)) @ weights


def rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--patience", default="0")
    parser.add_argument("--workers", default="8")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--split-seed", default="42")
    parser.add_argument("--hf-repo-id", default="")
    parser.add_argument("--max-images", type=int, default=808)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = args.dataset_yaml.parent
    label_root = split / "labels/val"
    images = sorted((split / "images/val").glob("*.jpg"))[: args.max_images]
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    summary: dict[str, Any] = {"images": len(images), "models": {}}
    device = torch.device(args.device)

    for name, checkpoint in checkpoints.items():
        model = YOLO(checkpoint)
        model.model.to(device).eval()
        head = model.model.model[-1]
        assigner = TaskAlignedAssigner(topk=10, num_classes=head.nc, alpha=0.5, beta=6.0, stride=head.stride.tolist())
        rows: list[dict[str, Any]] = []
        for image_index, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            labels = read_labels(label_root / f"{image_path.stem}.txt", width, height)
            if not labels:
                continue
            resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
            with torch.no_grad():
                boxes, class_scores, evidence, anchors, strides = _candidate_tensors(model, tensor)
            gt = torch.tensor([[x[1] * 640 / width, x[2] * 640 / height, x[3] * 640 / width, x[4] * 640 / height] for x in labels], device=device, dtype=boxes.dtype)
            gt_labels = torch.tensor([[[x[0]] for x in labels]], device=device, dtype=torch.long)
            mask = torch.ones((1, len(labels), 1), device=device, dtype=torch.bool)
            with torch.no_grad():
                _, _, _, tal_fg, tal_gt_idx = assigner(class_scores.T.unsqueeze(0), boxes.unsqueeze(0), anchors * strides, gt_labels, gt.unsqueeze(0), mask)
            ious = box_iou(gt, boxes)
            widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1)
            heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
            base_features = torch.stack((class_scores, evidence[None, :].expand_as(class_scores)), dim=-1) if False else None
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            for object_id, label in enumerate(labels):
                if max(gt[object_id, 2] - gt[object_id, 0], gt[object_id, 3] - gt[object_id, 1]) > 16:
                    continue
                pool = (tal_fg[0] & (tal_gt_idx[0] == object_id)).nonzero(as_tuple=False).flatten()
                if pool.numel() < 2:
                    continue
                cls = label[0]
                score = class_scores[cls, pool]
                box_width = widths[pool]
                box_height = heights[pool]
                features = torch.stack((score, evidence[pool], score * evidence[pool], torch.log((box_width * box_height).sqrt()), torch.log(box_width / box_height)), dim=1)
                utility = ious[object_id, pool]
                for candidate, feature, utility_value in zip(pool.tolist(), features.detach().cpu().numpy(), utility.detach().cpu().numpy()):
                    rows.append({"image_index": image_index, "object_id": object_id, "candidate": candidate, "features": feature.tolist(), "utility": float(utility_value), "score": float(class_scores[cls, candidate])})
        train = [row for row in rows if row["image_index"] % 2 == 0]
        test = [row for row in rows if row["image_index"] % 2 == 1]
        if not train or not test:
            summary["models"][name] = {"rows": len(rows), "status": "insufficient_split"}
            continue
        model_fit = ridge_fit(np.asarray([row["features"] for row in train]), np.asarray([row["utility"] for row in train]))
        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in test:
            row["prediction"] = float(predict(model_fit, np.asarray([row["features"]]))[0])
            groups.setdefault((row["image_index"], row["object_id"]), []).append(row)
        raw_top, learned_top, oracle = [], [], []
        raw_hit, learned_hit = [], []
        score_values, utility_values, pred_values = [], [], []
        for group in groups.values():
            raw = max(group, key=lambda row: row["score"])
            learned = max(group, key=lambda row: row["prediction"])
            best = max(group, key=lambda row: row["utility"])
            raw_top.append(raw["utility"]); learned_top.append(learned["utility"]); oracle.append(best["utility"])
            raw_hit.append(float(raw["candidate"] == best["candidate"])); learned_hit.append(float(learned["candidate"] == best["candidate"]))
            score_values.extend(row["score"] for row in group); utility_values.extend(row["utility"] for row in group); pred_values.extend(row["prediction"] for row in group)
        summary["models"][name] = {"status": "ok", "train_candidates": len(train), "test_candidates": len(test), "test_groups": len(groups), "raw_top_iou": float(np.mean(raw_top)), "learned_top_iou": float(np.mean(learned_top)), "oracle_top_iou": float(np.mean(oracle)), "raw_oracle_hit": float(np.mean(raw_hit)), "learned_oracle_hit": float(np.mean(learned_hit)), "learned_delta_iou": float(np.mean(learned_top) - np.mean(raw_top)), "learned_delta_hit": float(np.mean(learned_hit) - np.mean(raw_hit)), "raw_rank_corr": rank_corr(np.asarray(score_values), np.asarray(utility_values)), "learned_rank_corr": rank_corr(np.asarray(pred_values), np.asarray(utility_values))}
        del model
    (args.output_dir / "probe_c_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
