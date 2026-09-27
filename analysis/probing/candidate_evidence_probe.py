"""Measure whether intermediate candidate evidence is aligned with detector scores.

This probe targets the refined hypothesis directly. For each GT object it
compares the candidate with the best localization IoU against the candidate
selected by the classification score, and measures whether a simple feature
energy proxy ranks candidates more faithfully than the score. It is inference
only and does not modify the detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .common import DATASETS, DatasetSpec, ProbeConfig, image_paths, load_yolo, metadata, read_boxes, size_bucket, write_json, write_rows


def box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(left[:, None, :2], right[None, :, :2])
    rb = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    area_left = (left[:, 2:] - left[:, :2]).clamp(min=0).prod(1)[:, None]
    area_right = (right[:, 2:] - right[:, :2]).clamp(min=0).prod(1)[None]
    return inter / (area_left + area_right - inter).clamp(min=1e-8)


def _candidate_tensors(model: Any, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model.model(tensor)
    if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], dict):
        raise RuntimeError("Checkpoint did not expose the project raw candidate dictionary")
    raw = output[1]
    head = model.model.model[-1]
    decoded = head._get_decode_boxes(raw)[0].T
    boxes = torch.cat((decoded[:, :2] - decoded[:, 2:] / 2, decoded[:, :2] + decoded[:, 2:] / 2), dim=1)
    scores = raw["scores"][0].sigmoid().max(dim=0).values
    evidence_parts = []
    for feature in raw.get("feats", []):
        energy = feature.detach().float().abs().mean(dim=1).flatten()
        evidence_parts.append(energy)
    evidence = torch.cat(evidence_parts) if evidence_parts else torch.zeros_like(scores)
    if evidence.numel() != scores.numel():
        raise RuntimeError(f"Feature/candidate count mismatch: evidence={evidence.numel()} scores={scores.numel()}")
    return boxes.detach(), scores.detach(), evidence


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def run(spec: DatasetSpec, checkpoint: Path, output: Path, config: ProbeConfig) -> dict[str, Any]:
    model = load_yolo(checkpoint, config.device)
    rows: list[dict[str, Any]] = []
    for image_path in image_paths(spec, config.max_images):
        image = cv2.imread(str(image_path))
        original_boxes = read_boxes(image_path, spec)
        if image is None or not original_boxes:
            continue
        height, width = image.shape[:2]
        resized = cv2.resize(image, (config.imgsz, config.imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized[..., ::-1].copy()).to(config.device).permute(2, 0, 1).float()[None] / 255.0
        candidate_boxes, scores, evidence = _candidate_tensors(model, tensor)
        scale = torch.tensor([config.imgsz / width, config.imgsz / height, config.imgsz / width, config.imgsz / height], device=config.device)
        gt = torch.tensor([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in original_boxes], device=config.device) * scale
        ious = box_iou(gt, candidate_boxes)
        for object_id, ground_truth in enumerate(original_boxes):
            object_iou = ious[object_id]
            # Compare candidates responsible for this object, not the single
            # highest-scoring candidate in the whole image. This follows the
            # existing project diagnostic protocol: use candidate centers in a
            # padded GT neighborhood before measuring score-vs-IoU alignment.
            gt_scaled = gt[object_id]
            centers = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) / 2
            pad = 8.0
            local = ((centers[:, 0] >= gt_scaled[0] - pad) & (centers[:, 0] <= gt_scaled[2] + pad) &
                     (centers[:, 1] >= gt_scaled[1] - pad) & (centers[:, 1] <= gt_scaled[3] + pad)).nonzero(as_tuple=False).flatten()
            if local.numel() == 0:
                continue
            local_iou, local_scores, local_evidence = object_iou[local], scores[local], evidence[local]
            best_iou, oracle_local = local_iou.max(dim=0)
            score_local = local_scores.argmax()
            evidence_local = local_evidence.argmax()
            oracle_index = local[oracle_local]
            rows.append({
                "dataset": spec.name,
                "image": image_path.name,
                "object_id": object_id,
                "size_bucket": size_bucket(ground_truth),
                "oracle_iou": float(best_iou.cpu()),
                "score_iou": float(local_iou[score_local].cpu()),
                "evidence_iou": float(local_iou[evidence_local].cpu()),
                "oracle_gap": float((best_iou - local_iou[score_local]).cpu()),
                "score_at_oracle": float(scores[oracle_index].cpu()),
                "evidence_at_oracle": float(evidence[oracle_index].cpu()),
                "score_rank_corr_iou": _rank_corr(local_scores.cpu().numpy(), local_iou.cpu().numpy()),
                "evidence_rank_corr_iou": _rank_corr(local_evidence.cpu().numpy(), local_iou.cpu().numpy()),
            })
    write_rows(output / "candidate_evidence.csv", rows)
    summary: dict[str, Any] = {"metadata": metadata(spec, config, checkpoint, "candidate_evidence"), "rows": len(rows), "by_size": {}}
    for bucket in sorted({row["size_bucket"] for row in rows}):
        subset = [row for row in rows if row["size_bucket"] == bucket]
        summary["by_size"][bucket] = {key: float(np.nanmean([row[key] for row in subset])) for key in ("oracle_iou", "score_iou", "evidence_iou", "oracle_gap", "score_rank_corr_iou", "evidence_rank_corr_iou")}
    write_json(output / "candidate_evidence.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = ProbeConfig(max_images=args.max_images, device=args.device)
    run(DATASETS[args.dataset], args.checkpoint, args.output, config)


if __name__ == "__main__":
    main()
