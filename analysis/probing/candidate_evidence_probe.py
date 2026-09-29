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
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors


def box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(left[:, None, :2], right[None, :, :2])
    rb = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    area_left = (left[:, 2:] - left[:, :2]).clamp(min=0).prod(1)[:, None]
    area_right = (right[:, 2:] - right[:, :2]).clamp(min=0).prod(1)[None]
    return inter / (area_left + area_right - inter).clamp(min=1e-8)


def _candidate_tensors(
    model: Any, tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model.model(tensor)
    if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], dict):
        raise RuntimeError("Checkpoint did not expose the project raw candidate dictionary")
    raw = output[1]
    head = model.model.model[-1]
    decoded = head._get_decode_boxes(raw)[0].T
    boxes = torch.cat((decoded[:, :2] - decoded[:, 2:] / 2, decoded[:, :2] + decoded[:, 2:] / 2), dim=1)
    class_scores = raw["scores"][0].sigmoid()
    scores = class_scores.max(dim=0).values
    anchor_points, stride_tensor = make_anchors(raw["feats"], head.stride, 0.5)
    evidence_parts = []
    for feature in raw.get("feats", []):
        energy = feature.detach().float().abs().mean(dim=1).flatten()
        evidence_parts.append(energy)
    evidence = torch.cat(evidence_parts) if evidence_parts else torch.zeros_like(scores)
    if evidence.numel() != scores.numel():
        raise RuntimeError(f"Feature/candidate count mismatch: evidence={evidence.numel()} scores={scores.numel()}")
    return boxes.detach(), class_scores.detach(), evidence, anchor_points.detach(), stride_tensor.detach()


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def run(
    spec: DatasetSpec,
    checkpoint: Path,
    output: Path,
    config: ProbeConfig,
    joint_lambda: float = 0.25,
    tiny_max_dim: float = 16.0,
) -> dict[str, Any]:
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
        candidate_boxes, class_scores, evidence, anchor_points, stride_tensor = _candidate_tensors(model, tensor)
        scale = torch.tensor([config.imgsz / width, config.imgsz / height, config.imgsz / width, config.imgsz / height], device=config.device)
        gt = torch.tensor([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in original_boxes], device=config.device) * scale
        ious = box_iou(gt, candidate_boxes)
        gt_labels = torch.tensor(
            [[int(b["class"]) for b in original_boxes]], device=config.device, dtype=torch.float32
        ).unsqueeze(-1)
        gt_bboxes = gt.unsqueeze(0)
        mask_gt = torch.ones((1, len(original_boxes), 1), device=config.device, dtype=torch.bool)
        head = model.model.model[-1]
        assigner = TaskAlignedAssigner(
            topk=10, num_classes=head.nc, alpha=0.5, beta=6.0, stride=head.stride.tolist()
        )
        _, _, _, tal_fg, tal_gt_idx = assigner(
            class_scores.T.unsqueeze(0),
            candidate_boxes.unsqueeze(0),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
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
            gt_width = gt_scaled[2] - gt_scaled[0]
            gt_height = gt_scaled[3] - gt_scaled[1]
            gt_class = int(ground_truth["class"])
            score_values = class_scores[gt_class]
            tal_local = (tal_fg[0] & (tal_gt_idx[0] == object_id)).nonzero(as_tuple=False).flatten()
            pools = {"local_neighborhood": local}
            if tal_local.numel() > 0:
                pools["tal_positive"] = tal_local
            for protocol, pool in pools.items():
                if pool.numel() == 0:
                    continue
                local_iou = object_iou[pool]
                local_scores = score_values[pool]
                local_evidence = evidence[pool]
                best_iou, oracle_local = local_iou.max(dim=0)
                score_local = local_scores.argmax()
                evidence_local = local_evidence.argmax()
                oracle_index = pool[oracle_local]
                if max(float(gt_width), float(gt_height)) <= tiny_max_dim and pool.numel() >= 2:
                    score_logits = local_scores.clamp(1e-6, 1 - 1e-6).logit()
                    normalized_iou = (local_iou - local_iou.mean()) / (local_iou.std(unbiased=False) + 1e-6)
                    joint_scores = score_logits + joint_lambda * normalized_iou.clamp(-0.5, 0.5)
                    joint_local = joint_scores.argmax()
                    joint_iou = local_iou[joint_local]
                else:
                    joint_iou = local_iou[score_local]
                    joint_local = score_local
                score_iou = local_iou[score_local]
                score_hit = float(score_local == oracle_local)
                joint_hit = float(joint_local == oracle_local)
                rows.append({
                    "dataset": spec.name,
                    "image": image_path.name,
                    "object_id": object_id,
                    "candidate_protocol": protocol,
                    "size_bucket": size_bucket(ground_truth),
                    "oracle_iou": float(best_iou.cpu()),
                    "score_iou": float(score_iou.cpu()),
                    "evidence_iou": float(local_iou[evidence_local].cpu()),
                    "joint_iou": float(joint_iou.cpu()),
                    "joint_delta_iou": float((joint_iou - score_iou).cpu()),
                    "score_oracle_hit": score_hit,
                    "joint_oracle_hit": joint_hit,
                    "delta_oracle_hit": joint_hit - score_hit,
                    "oracle_gap": float((best_iou - score_iou).cpu()),
                    "joint_oracle_gap": float((best_iou - joint_iou).cpu()),
                    "score_at_oracle": float(score_values[oracle_index].cpu()),
                    "evidence_at_oracle": float(evidence[oracle_index].cpu()),
                    "score_rank_corr_iou": _rank_corr(local_scores.cpu().numpy(), local_iou.cpu().numpy()),
                    "evidence_rank_corr_iou": _rank_corr(local_evidence.cpu().numpy(), local_iou.cpu().numpy()),
                })
    write_rows(output / "candidate_evidence.csv", rows)
    summary: dict[str, Any] = {
        "metadata": metadata(spec, config, checkpoint, "candidate_evidence"),
        "rows": len(rows),
        "joint_lambda": joint_lambda,
        "joint_tiny_max_dim": tiny_max_dim,
        "protocols": sorted({row["candidate_protocol"] for row in rows}),
        "by_size": {},
    }
    for protocol in sorted({row["candidate_protocol"] for row in rows}):
        summary["by_size"][protocol] = {}
        for bucket in sorted({row["size_bucket"] for row in rows if row["candidate_protocol"] == protocol}):
            subset = [row for row in rows if row["candidate_protocol"] == protocol and row["size_bucket"] == bucket]
            summary["by_size"][protocol][bucket] = {
                key: float(np.nanmean([row[key] for row in subset]))
                for key in (
                    "oracle_iou", "score_iou", "evidence_iou", "joint_iou", "joint_delta_iou",
                    "score_oracle_hit", "joint_oracle_hit", "delta_oracle_hit", "oracle_gap",
                    "joint_oracle_gap", "score_rank_corr_iou", "evidence_rank_corr_iou",
                )
            }
    write_json(output / "candidate_evidence.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--joint-lambda", type=float, default=0.25)
    parser.add_argument("--tiny-max-dim", type=float, default=16.0)
    args = parser.parse_args()
    config = ProbeConfig(max_images=args.max_images, device=args.device)
    run(DATASETS[args.dataset], args.checkpoint, args.output, config, args.joint_lambda, args.tiny_max_dim)


if __name__ == "__main__":
    main()
