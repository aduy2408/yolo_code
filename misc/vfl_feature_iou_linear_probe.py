#!/usr/bin/env python3
"""Linear probes for whether Detect cls/reg features encode TAL assigned IoU.

This diagnostic is intentionally read-only with respect to model behavior: it
loads a checkpoint, extracts frozen per-anchor features immediately before the
final Detect branch convolutions, reuses the existing TaskAlignedAssigner logic,
and fits simple linear regressors from features to assigned IoU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = REPO_ROOT / "models_related" / "ultralytics"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LOCAL_ULTRALYTICS))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.modules import Detect  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.metrics import bbox_iou  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402
from ultralytics.utils.tal import make_anchors  # noqa: E402


def seed_everything(seed: int) -> None:
    """Seed random number generators for repeatable sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="YOLO checkpoint to probe.")
    parser.add_argument("--data", required=True, type=Path, help="YOLO data.yaml.")
    parser.add_argument("--out", default=Path("runs/diagnostics/vfl_feature_iou_linear_probe.json"), type=Path)
    parser.add_argument("--csv", default=None, type=Path, help="Optional CSV summary path.")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--device", default="", help="Torch device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--train-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--eval-split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-train-anchors", default=20000, type=int)
    parser.add_argument("--max-eval-anchors", default=20000, type=int)
    parser.add_argument("--ridge", default=1e-3, type=float, help="L2 regularization for linear probe.")
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device:
        if device.isdigit():
            device = f"cuda:{device}"
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_detect(model: torch.nn.Module) -> Detect:
    detect = model.model[-1]
    if not isinstance(detect, Detect):
        raise TypeError(f"Expected final module to be Detect, got {type(detect).__name__}.")
    return detect


def branch_stem(branch: torch.nn.Module) -> torch.nn.Module:
    """Return a branch without its final prediction conv."""
    if not isinstance(branch, torch.nn.Sequential) or len(branch) < 2:
        raise TypeError(f"Expected Detect branch to be nn.Sequential, got {type(branch).__name__}.")
    return torch.nn.Sequential(*list(branch.children())[:-1])


@torch.no_grad()
def extract_branch_features(detect: Detect, preds: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened reg and cls branch features in Detect anchor order."""
    reg_features = []
    cls_features = []
    for level, feat in enumerate(preds["feats"]):
        reg = branch_stem(detect.cv2[level])(feat)
        cls = branch_stem(detect.cv3[level])(feat)
        bs = feat.shape[0]
        reg_features.append(reg.view(bs, reg.shape[1], -1).permute(0, 2, 1).contiguous())
        cls_features.append(cls.view(bs, cls.shape[1], -1).permute(0, 2, 1).contiguous())
    return torch.cat(reg_features, dim=1), torch.cat(cls_features, dim=1)


def preprocess_targets(
    batch: dict[str, torch.Tensor], batch_size: int, imgsz: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build gt labels, boxes, and mask in the same format as v8DetectionLoss."""
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = targets.to(device)
    nl, ne = targets.shape
    if nl == 0:
        out = torch.zeros(batch_size, 0, ne - 1, device=device)
    else:
        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(imgsz[[1, 0, 1, 0]]))
    gt_labels, gt_bboxes = out.split((1, 4), 2)
    return gt_labels, gt_bboxes, gt_bboxes.sum(2, keepdim=True).gt_(0.0)


@torch.no_grad()
def assigned_iou_and_mask(
    criterion: v8DetectionLoss,
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse YOLOv8 TAL assignment and return per-positive assigned IoU labels."""
    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(preds["feats"], criterion.stride, 0.5)
    batch_size = pred_scores.shape[0]
    dtype = pred_scores.dtype
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=dtype) * criterion.stride[0]
    gt_labels, gt_bboxes, mask_gt = preprocess_targets(batch, batch_size, imgsz, device)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)

    _, target_bboxes, _, fg_mask, _ = criterion.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )
    if not fg_mask.any():
        return fg_mask, pred_scores.new_zeros(0)
    assigned_iou = bbox_iou(
        pred_bboxes.detach()[fg_mask],
        (target_bboxes / stride_tensor)[fg_mask],
        xywh=False,
        CIoU=False,
    ).squeeze(-1).clamp(0)
    return fg_mask, assigned_iou


def make_loader(data: dict[str, Any], cfg: SimpleNamespace, split: str, batch: int, stride: int):
    if split not in data or not data[split]:
        raise KeyError(f"Split '{split}' is not present in data.yaml.")
    dataset = build_yolo_dataset(cfg, data[split], batch, data, mode="val", rect=False, stride=stride)
    return build_dataloader(dataset, batch=batch, workers=cfg.workers, shuffle=False, rank=-1)


def maybe_subsample(x: torch.Tensor, y: torch.Tensor, limit: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] <= limit:
        return x, y
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:limit]
    return x[idx], y[idx]


@torch.no_grad()
def collect_features(
    model: torch.nn.Module,
    criterion: v8DetectionLoss,
    loader,
    device: torch.device,
    limit: int,
    seed: int,
) -> dict[str, torch.Tensor | int]:
    """Collect positive-anchor branch features and assigned IoU labels."""
    detect = get_detect(model)
    reg_parts = []
    cls_parts = []
    y_parts = []
    batches_seen = 0
    empty_batches = 0
    positives = 0
    for batch in loader:
        batches_seen += 1
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=device.type == "cuda")
        batch["img"] = batch["img"].float() / 255.0

        output = model(batch["img"])
        preds = output[1] if isinstance(output, tuple) else output
        if isinstance(preds, dict) and "one2many" in preds:
            preds = preds["one2many"]
        fg_mask, assigned_iou = assigned_iou_and_mask(criterion, preds, batch, device)
        if not fg_mask.any():
            empty_batches += 1
            continue

        reg_features, cls_features = extract_branch_features(detect, preds)
        reg_parts.append(reg_features[fg_mask].detach().cpu())
        cls_parts.append(cls_features[fg_mask].detach().cpu())
        y_parts.append(assigned_iou.detach().cpu().float())
        positives += int(assigned_iou.numel())
        if positives >= limit:
            break

    if not y_parts:
        return {
            "reg": torch.empty(0, 0),
            "cls": torch.empty(0, 0),
            "y": torch.empty(0),
            "batches_seen": batches_seen,
            "empty_batches": empty_batches,
            "positives": 0,
        }
    reg = torch.cat(reg_parts, dim=0).float()
    cls = torch.cat(cls_parts, dim=0).float()
    y = torch.cat(y_parts, dim=0).float()
    reg, y_reg = maybe_subsample(reg, y, limit, seed)
    cls, y_cls = maybe_subsample(cls, y, limit, seed)
    if not torch.equal(y_reg, y_cls):
        raise RuntimeError("Internal sampling mismatch between reg and cls features.")
    return {
        "reg": reg,
        "cls": cls,
        "y": y_reg,
        "batches_seen": batches_seen,
        "empty_batches": empty_batches,
        "positives": int(y_reg.numel()),
    }


def fit_linear_probe(x: torch.Tensor, y: torch.Tensor, ridge: float) -> dict[str, torch.Tensor]:
    """Fit a standardized ridge linear regressor."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    xs = (x - mean) / std
    xb = torch.cat((xs, torch.ones(xs.shape[0], 1, dtype=xs.dtype)), dim=1)
    eye = torch.eye(xb.shape[1], dtype=xb.dtype)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(xb.T @ xb + float(ridge) * eye, xb.T @ y)
    return {"mean": mean, "std": std, "weights": weights}


def predict_linear_probe(probe: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    xs = (x - probe["mean"]) / probe["std"]
    xb = torch.cat((xs, torch.ones(xs.shape[0], 1, dtype=xs.dtype)), dim=1)
    return xb @ probe["weights"]


def rankdata(values: torch.Tensor) -> torch.Tensor:
    """Return average ranks for a 1D tensor, including ties."""
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    sorted_values = values[order]
    n = values.numel()
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) * 0.5
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-12))
    return float((a * b).sum() / denom)


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = torch.mean((pred - target).square())
    var = torch.sum((target - target.mean()).square()).clamp_min(1e-12)
    r2 = 1.0 - torch.sum((pred - target).square()) / var
    return {
        "mse": float(mse),
        "r2": float(r2),
        "pearson": pearson(pred, target),
        "spearman": pearson(rankdata(pred), rankdata(target)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    yolo = YOLO(str(args.weights))
    model = yolo.model.to(device)
    model.eval()
    model.args = getattr(model, "args", SimpleNamespace())
    data = check_det_dataset(str(args.data), autodownload=False)
    stride = max(int(get_detect(model).stride.max()), 32)
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "mode": "val",
            "data": str(args.data),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
        }
    )
    # v8DetectionLoss expects model.args to contain loss hyperparameters.
    model.args = cfg
    criterion = v8DetectionLoss(model)

    train = collect_features(
        model,
        criterion,
        make_loader(data, cfg, args.train_split, args.batch, stride),
        device,
        args.max_train_anchors,
        args.seed,
    )
    eval_data = collect_features(
        model,
        criterion,
        make_loader(data, cfg, args.eval_split, args.batch, stride),
        device,
        args.max_eval_anchors,
        args.seed + 1,
    )
    if int(train["positives"]) == 0 or int(eval_data["positives"]) == 0:
        raise RuntimeError(
            f"No positive anchors collected: train={train['positives']} eval={eval_data['positives']}."
        )

    results: dict[str, Any] = {
        "weights": str(args.weights),
        "data": str(args.data),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "imgsz": args.imgsz,
        "ridge": args.ridge,
        "collection": {
            "train": {
                "positives": train["positives"],
                "batches_seen": train["batches_seen"],
                "empty_batches": train["empty_batches"],
            },
            "eval": {
                "positives": eval_data["positives"],
                "batches_seen": eval_data["batches_seen"],
                "empty_batches": eval_data["empty_batches"],
            },
        },
        "branches": {},
    }
    for branch in ("cls", "reg"):
        probe = fit_linear_probe(train[branch], train["y"], args.ridge)
        pred = predict_linear_probe(probe, eval_data[branch])
        results["branches"][branch] = {
            "feature_dim": int(train[branch].shape[1]),
            **metrics(pred, eval_data["y"]),
        }
    return results


def write_outputs(results: dict[str, Any], out_path: Path, csv_path: Path | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_path is None:
        csv_path = out_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for branch, values in results["branches"].items():
        row = {"branch": branch}
        row.update(values)
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results = run_probe(args)
    write_outputs(results, args.out, args.csv)
    print(json.dumps(results["branches"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
