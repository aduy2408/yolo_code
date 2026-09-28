"""Run fixed-subset candidate responsibility diagnostics over saved checkpoints.

The script deliberately measures mechanism-level signals without changing detector
behavior. It reuses the project's candidate extraction path and evaluates the
same deterministically selected images for every checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .candidate_evidence_probe import _candidate_tensors, _rank_corr, box_iou
from .common import DATASETS, ProbeConfig, image_paths, load_yolo, read_boxes, size_bucket, write_json, write_rows


def _entropy(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    probs = torch.softmax(values.float(), dim=0)
    return float((-(probs * probs.clamp_min(1e-12).log()).sum()).cpu())


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def run_checkpoint(dataset: str, checkpoint: Path, output: Path, config: ProbeConfig) -> dict[str, Any]:
    spec = DATASETS[dataset]
    model = load_yolo(checkpoint, config.device)
    rows: list[dict[str, Any]] = []
    for image_path in image_paths(spec, config.max_images):
        image = cv2.imread(str(image_path))
        boxes = read_boxes(image_path, spec)
        if image is None or not boxes:
            continue
        height, width = image.shape[:2]
        resized = cv2.resize(image, (config.imgsz, config.imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized[..., ::-1].copy()).to(config.device).permute(2, 0, 1).float()[None] / 255.0
        candidate_boxes, scores, _ = _candidate_tensors(model, tensor)
        scale = torch.tensor([config.imgsz / width, config.imgsz / height] * 2, device=config.device)
        gt = torch.tensor([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes], device=config.device) * scale
        ious = box_iou(gt, candidate_boxes)
        centers = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) / 2
        for object_id, box in enumerate(boxes):
            gt_box = gt[object_id]
            pad = 8.0
            local = ((centers[:, 0] >= gt_box[0] - pad) & (centers[:, 0] <= gt_box[2] + pad) &
                     (centers[:, 1] >= gt_box[1] - pad) & (centers[:, 1] <= gt_box[3] + pad)).nonzero(as_tuple=False).flatten()
            if local.numel() == 0:
                continue
            local_iou = ious[object_id, local]
            local_scores = scores[local]
            oracle_pos = int(local_iou.argmax().cpu())
            score_pos = int(local_scores.argmax().cpu())
            oracle_iou = float(local_iou[oracle_pos].cpu())
            score_iou = float(local_iou[score_pos].cpu())
            rows.append({
                "dataset": spec.name,
                "checkpoint": str(checkpoint),
                "image": image_path.name,
                "object_id": object_id,
                "size_bucket": size_bucket(box),
                "oracle_iou": oracle_iou,
                "score_iou": score_iou,
                "oracle_gap": oracle_iou - score_iou,
                "score_iou_spearman": _rank_corr(local_scores.cpu().numpy(), local_iou.cpu().numpy()),
                "responsibility_entropy": _entropy(local_scores),
                "effective_positive_mass": float(torch.softmax(local_scores.float(), dim=0).sum().cpu()),
                "num_local_positives": int(local.numel()),
                "alignment_accuracy": float(score_pos == oracle_pos),
                "oracle_score": float(local_scores[oracle_pos].cpu()),
            })
    write_rows(output / "checkpoint_responsibility.csv", rows)
    summary: dict[str, Any] = {
        "dataset": spec.name,
        "checkpoint": str(checkpoint),
        "images": len({row["image"] for row in rows}),
        "objects": len(rows),
        "by_size": {},
    }
    for bucket in sorted({row["size_bucket"] for row in rows}):
        subset = [row for row in rows if row["size_bucket"] == bucket]
        summary["by_size"][bucket] = {
            key: _mean_or_nan([float(row[key]) for row in subset])
            for key in ("oracle_gap", "score_iou", "oracle_iou", "score_iou_spearman", "responsibility_entropy", "effective_positive_mass", "num_local_positives", "alignment_accuracy", "oracle_score")
        }
    summary["overall"] = {
        key: _mean_or_nan([float(row[key]) for row in rows])
        for key in ("oracle_gap", "score_iou", "oracle_iou", "score_iou_spearman", "responsibility_entropy", "effective_positive_mass", "num_local_positives", "alignment_accuracy", "oracle_score")
    }
    write_json(output / "checkpoint_responsibility.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = ProbeConfig(max_images=args.max_images, imgsz=args.imgsz, device=args.device)
    print(json.dumps(run_checkpoint(args.dataset, args.checkpoint, args.output, config), indent=2))


if __name__ == "__main__":
    main()
