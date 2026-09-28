"""Test whether tiny-object candidate utility is stable and learnable.

This is a training-feasibility gate, not a detector method. It applies benign
image perturbations that preserve geometry, measures whether the locally best
candidate remains the same, and fits a frozen, train/test split calibration
from candidate score and feature energy to local IoU. A positive result means
score alignment is worth a matched training experiment. A negative result
means candidate identity or the target itself is too unstable for naive score
alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .candidate_evidence_probe import _candidate_tensors, box_iou
from .common import DATASETS, ProbeConfig, image_paths, load_yolo, metadata, read_boxes, size_bucket, write_json, write_rows


def _augment(image: np.ndarray, name: str) -> np.ndarray:
    if name == "brightness_down":
        return np.clip(image.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    if name == "brightness_up":
        return np.clip(image.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
    if name == "blur":
        return cv2.GaussianBlur(image, (3, 3), 0)
    return image


def _features(boxes: torch.Tensor, scores: torch.Tensor, evidence: torch.Tensor) -> np.ndarray:
    width = (boxes[:, 2] - boxes[:, 0]).clamp(min=1)
    height = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
    area = (width * height).sqrt()
    aspect = torch.log(width / height)
    return torch.stack((scores, evidence, scores * evidence, torch.log(area), aspect), dim=1).cpu().numpy()


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    mean, scale = x.mean(0), x.std(0) + 1e-6
    normalized = (x - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return np.concatenate((weights, mean, scale))


def _ridge_predict(model: np.ndarray, x: np.ndarray) -> np.ndarray:
    width = x.shape[1]
    weights, mean, scale = model[: width + 1], model[width + 1 : 2 * width + 1], model[2 * width + 1 :]
    return np.column_stack((np.ones(len(x)), (x - mean) / scale)) @ weights


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])


def run(dataset_key: str, checkpoint: Path, output: Path, config: ProbeConfig) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    model = load_yolo(checkpoint, config.device)
    augmentations = ("identity", "brightness_down", "brightness_up", "blur")
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for image_index, image_path in enumerate(image_paths(spec, config.max_images)):
        original = cv2.imread(str(image_path))
        gt_rows = read_boxes(image_path, spec)
        if original is None or not gt_rows:
            continue
        height, width = original.shape[:2]
        gt = torch.tensor([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in gt_rows], device=config.device)
        gt *= torch.tensor([config.imgsz / width, config.imgsz / height, config.imgsz / width, config.imgsz / height], device=config.device)
        per_aug: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for augmentation in augmentations:
            image = cv2.resize(_augment(original, augmentation), (config.imgsz, config.imgsz), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(image[..., ::-1].copy()).to(config.device).permute(2, 0, 1).float()[None] / 255.0
            per_aug[augmentation] = _candidate_tensors(model, tensor)
        base_boxes, base_scores, base_evidence = per_aug["identity"]
        base_centers = (base_boxes[:, :2] + base_boxes[:, 2:]) / 2
        base_iou = box_iou(gt, base_boxes)
        for object_id, gt_row in enumerate(gt_rows):
            local = ((base_centers[:, 0] >= gt[object_id, 0] - 8) & (base_centers[:, 0] <= gt[object_id, 2] + 8) & (base_centers[:, 1] >= gt[object_id, 1] - 8) & (base_centers[:, 1] <= gt[object_id, 3] + 8)).nonzero(as_tuple=False).flatten()
            if local.numel() < 2:
                continue
            base_local_iou = base_iou[object_id, local]
            oracle = int(local[base_local_iou.argmax()])
            score_choice = int(local[base_scores[local].argmax()])
            for augmentation in augmentations[1:]:
                aug_boxes, aug_scores, aug_evidence = per_aug[augmentation]
                aug_iou = box_iou(gt[object_id : object_id + 1], aug_boxes)[0]
                aug_local_iou = aug_iou[local]
                aug_oracle = int(local[aug_local_iou.argmax()])
                rows.append({
                    "dataset": spec.name,
                    "image": image_path.name,
                    "object_id": object_id,
                    "size_bucket": size_bucket(gt_row),
                    "augmentation": augmentation,
                    "oracle_identity_stable": float(aug_oracle == oracle),
                    "score_identity_stable": float(int(local[aug_scores[local].argmax()]) == score_choice),
                    "base_oracle_iou": float(base_local_iou.max().cpu()),
                    "aug_oracle_iou": float(aug_local_iou.max().cpu()),
                    "oracle_iou_delta": float((aug_local_iou.max() - base_local_iou.max()).cpu()),
                    "score_rank_corr": _rank_corr(base_scores[local].cpu().numpy(), aug_scores[local].cpu().numpy()),
                })
            features = _features(base_boxes[local], base_scores[local], base_evidence[local])
            targets = base_local_iou.cpu().numpy()
            calibration_rows.extend({"image_index": image_index, "object_id": object_id, "size_bucket": size_bucket(gt_row), "features": row.tolist(), "target": float(target)} for row, target in zip(features, targets))
    write_rows(output / "alignment_stability.csv", rows)
    if not calibration_rows:
        write_json(output / "alignment_feasibility.json", {"metadata": metadata(DATASETS[dataset_key], config, checkpoint, "alignment_feasibility"), "rows": 0})
        return {"rows": 0}
    train = [row for row in calibration_rows if row["image_index"] % 2 == 0]
    test = [row for row in calibration_rows if row["image_index"] % 2 == 1]
    train_x, train_y = np.asarray([row["features"] for row in train]), np.asarray([row["target"] for row in train])
    test_x, test_y = np.asarray([row["features"] for row in test]), np.asarray([row["target"] for row in test])
    calibration = _ridge_fit(train_x, train_y)
    calibrated = _ridge_predict(calibration, test_x)
    test_groups = {}
    for row, raw, prediction in zip(test, test_x[:, 0], calibrated):
        key = (row["image_index"], row["object_id"])
        test_groups.setdefault(key, []).append((row["size_bucket"], raw, prediction, row["target"]))
    raw_top, calibrated_top, oracle_top = [], [], []
    for group in test_groups.values():
        raw_top.append(max(group, key=lambda value: value[1])[3])
        calibrated_top.append(max(group, key=lambda value: value[2])[3])
        oracle_top.append(max(value[3] for value in group))
    summary: dict[str, Any] = {"metadata": metadata(spec, config, checkpoint, "alignment_feasibility"), "stability_rows": len(rows), "calibration_train_candidates": len(train), "calibration_test_candidates": len(test), "calibration_test_groups": len(test_groups), "calibration": {"raw_score_corr": _rank_corr(test_x[:, 0], test_y), "calibrated_corr": _rank_corr(calibrated, test_y), "raw_mse": float(np.mean((test_x[:, 0] - test_y) ** 2)), "calibrated_mse": float(np.mean((calibrated - test_y) ** 2)), "raw_top_iou": float(np.mean(raw_top)), "calibrated_top_iou": float(np.mean(calibrated_top)), "oracle_top_iou": float(np.mean(oracle_top))}, "by_size": {}}
    for bucket in sorted({row["size_bucket"] for row in rows}):
        subset = [row for row in rows if row["size_bucket"] == bucket]
        summary["by_size"][bucket] = {key: float(np.nanmean([row[key] for row in subset])) for key in ("oracle_identity_stable", "score_identity_stable", "base_oracle_iou", "aug_oracle_iou", "score_rank_corr")}
    write_json(output / "alignment_feasibility.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.dataset, args.checkpoint, args.output, ProbeConfig(max_images=args.max_images, device=args.device))


if __name__ == "__main__":
    main()
