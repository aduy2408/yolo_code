"""Probe C2: frozen learnability before TAL top-k selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner

from analysis.probing.candidate_evidence_probe import _candidate_tensors, box_iou


def labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, cx, cy, bw, bh = map(float, values[:5])
        cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
        out.append((int(cls), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y), mean, scale


def predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    weights, mean, scale = model
    return np.column_stack((np.ones(len(x)), (x - mean) / scale)) @ weights


def rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--patience", default="0")
    parser.add_argument("--workers", default="8")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--split-seed", default="42")
    parser.add_argument("--hf-repo-id", default="")
    parser.add_argument("--max-images", type=int, default=808)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = args.dataset_yaml.parent
    images = sorted((split / "images/val").glob("*.jpg"))[: args.max_images]
    checkpoint_map = dict(item.split("=", 1) for item in args.checkpoint)
    device = torch.device(args.device)
    result: dict[str, Any] = {"images": len(images), "topk": args.topk, "models": {}}

    for name, checkpoint in checkpoint_map.items():
        model = YOLO(checkpoint)
        model.model.to(device).eval()
        head = model.model.model[-1]
        assigner = TaskAlignedAssigner(topk=args.topk, num_classes=head.nc, alpha=0.5, beta=6.0, stride=head.stride.tolist())
        rows: list[dict[str, Any]] = []
        for image_index, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            gt_rows = labels(split / "labels/val" / f"{image_path.stem}.txt", width, height)
            if not gt_rows:
                continue
            resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
            with torch.no_grad():
                boxes, class_scores, evidence, anchors, strides = _candidate_tensors(model, tensor)
            gt = torch.tensor([[x[1] * 640 / width, x[2] * 640 / height, x[3] * 640 / width, x[4] * 640 / height] for x in gt_rows], device=device, dtype=boxes.dtype)
            gt_labels = torch.tensor([[[x[0]] for x in gt_rows]], device=device, dtype=torch.long)
            mask = torch.ones((1, len(gt_rows), 1), device=device, dtype=torch.bool)
            assigner.bs = 1
            assigner.n_max_boxes = len(gt_rows)
            with torch.no_grad():
                eligible = assigner.select_candidates_in_gts(anchors * strides, gt.unsqueeze(0), mask)
                align, overlaps = assigner.get_box_metrics(class_scores.T.unsqueeze(0), boxes.unsqueeze(0), gt_labels, gt.unsqueeze(0), eligible * mask)
                tal_topk = assigner.select_topk_candidates(align, topk_mask=mask.expand(-1, -1, args.topk).bool())
            ious = box_iou(gt, boxes)
            widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1)
            heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            for object_id, gt_row in enumerate(gt_rows):
                gt_width = gt[object_id, 2] - gt[object_id, 0]
                gt_height = gt[object_id, 3] - gt[object_id, 1]
                if max(gt_width, gt_height) > 16:
                    continue
                pool = eligible[0, object_id].nonzero(as_tuple=False).flatten()
                if pool.numel() < 2:
                    continue
                cls = gt_row[0]
                score = class_scores[cls, pool]
                dx = (centers[pool, 0] - (gt[object_id, 0] + gt[object_id, 2]) / 2) / gt_width.clamp(min=1)
                dy = (centers[pool, 1] - (gt[object_id, 1] + gt[object_id, 3]) / 2) / gt_height.clamp(min=1)
                features = torch.stack((score, evidence[pool], score * evidence[pool], torch.log((widths[pool] * heights[pool]).sqrt()), torch.log(widths[pool] / heights[pool]), dx, dy, strides[pool, 0]), dim=1)
                utility = ious[object_id, pool]
                tal_values = align[0, object_id, pool]
                for candidate, feature, utility_value, score_value, tal_value in zip(pool.tolist(), features.detach().cpu().numpy(), utility.detach().cpu().numpy(), score.detach().cpu().numpy(), tal_values.detach().cpu().numpy()):
                    rows.append({"image_index": image_index, "object_id": object_id, "candidate": candidate, "features": feature.tolist(), "utility": float(utility_value), "score": float(score_value), "tal": float(tal_value)})
        train = [row for row in rows if row["image_index"] % 2 == 0]
        test = [row for row in rows if row["image_index"] % 2 == 1]
        if not train or not test:
            result["models"][name] = {"status": "insufficient_split", "rows": len(rows)}
            continue
        fit = fit_ridge(np.asarray([row["features"] for row in train]), np.asarray([row["utility"] for row in train]))
        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in test:
            row["prediction"] = float(predict(fit, np.asarray([row["features"]]))[0])
            groups.setdefault((row["image_index"], row["object_id"]), []).append(row)
        metrics = {"tal_best_iou": [], "learned_best_iou": [], "tal_oracle_recall": [], "learned_oracle_recall": [], "tal_cls_mean": [], "learned_cls_mean": [], "topk_overlap": [], "tal_values": [], "learned_values": [], "utilities": []}
        for group in groups.values():
            k = min(args.topk, len(group))
            tal_group = sorted(group, key=lambda row: row["tal"], reverse=True)[:k]
            learned_group = sorted(group, key=lambda row: row["prediction"], reverse=True)[:k]
            oracle = max(row["utility"] for row in group)
            metrics["tal_best_iou"].append(max(row["utility"] for row in tal_group))
            metrics["learned_best_iou"].append(max(row["utility"] for row in learned_group))
            metrics["tal_oracle_recall"].append(float(max(row["utility"] for row in tal_group) >= oracle - 1e-7))
            metrics["learned_oracle_recall"].append(float(max(row["utility"] for row in learned_group) >= oracle - 1e-7))
            metrics["tal_cls_mean"].append(float(np.mean([row["score"] for row in tal_group])))
            metrics["learned_cls_mean"].append(float(np.mean([row["score"] for row in learned_group])))
            metrics["topk_overlap"].append(len({row["candidate"] for row in tal_group} & {row["candidate"] for row in learned_group}) / k)
            metrics["tal_values"].extend(row["tal"] for row in group)
            metrics["learned_values"].extend(row["prediction"] for row in group)
            metrics["utilities"].extend(row["utility"] for row in group)
        result["models"][name] = {"status": "ok", "train_candidates": len(train), "test_candidates": len(test), "test_groups": len(groups), "tal_best_iou": float(np.mean(metrics["tal_best_iou"])), "learned_best_iou": float(np.mean(metrics["learned_best_iou"])), "delta_best_iou": float(np.mean(metrics["learned_best_iou"]) - np.mean(metrics["tal_best_iou"])), "tal_oracle_recall": float(np.mean(metrics["tal_oracle_recall"])), "learned_oracle_recall": float(np.mean(metrics["learned_oracle_recall"])), "delta_oracle_recall": float(np.mean(metrics["learned_oracle_recall"]) - np.mean(metrics["tal_oracle_recall"])), "tal_cls_mean": float(np.mean(metrics["tal_cls_mean"])), "learned_cls_mean": float(np.mean(metrics["learned_cls_mean"])), "topk_overlap": float(np.mean(metrics["topk_overlap"])), "tal_rank_corr": rank_corr(np.asarray(metrics["tal_values"]), np.asarray(metrics["utilities"])), "learned_rank_corr": rank_corr(np.asarray(metrics["learned_values"]), np.asarray(metrics["utilities"]))}
        del model
    (args.output_dir / "probe_c2_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
