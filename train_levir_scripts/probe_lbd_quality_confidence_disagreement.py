#!/usr/bin/env python3
"""No-train probe of TAL target, IoU quality, confidence, and rank disagreement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.metrics import bbox_iou  # noqa: E402
from ultralytics.utils.tal import make_anchors  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402


def corr(a: list[float], b: list[float], method: str = "pearson") -> float | None:
    if len(a) < 3:
        return None
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if method == "spearman":
        x, y = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_batch(image: Path, imgsz: int, device: torch.device) -> tuple[torch.Tensor, dict]:
    original = cv2.imread(str(image))
    if original is None:
        raise RuntimeError(f"cannot read {image}")
    h, w = original.shape[:2]
    letterboxed = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=original)
    tensor = torch.from_numpy(letterboxed[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
    labels = []
    label_path = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
    for line in label_path.read_text().splitlines():
        cls, x, y, bw, bh = map(float, line.split()[:5])
        labels.append((cls, x, y, bw, bh))
    if labels:
        cls = torch.tensor([[row[0]] for row in labels], device=device)
        boxes = torch.tensor([row[1:] for row in labels], device=device)
    else:
        cls = torch.zeros((0, 1), device=device)
        boxes = torch.zeros((0, 4), device=device)
    return tensor, {"batch_idx": torch.zeros(len(labels), device=device), "cls": cls, "bboxes": boxes}


def summarize(rows: list[dict], q_thr: float, p_thr: float) -> dict:
    def vals(key: str) -> list[float]:
        return [float(row[key]) for row in rows if math.isfinite(float(row[key]))]

    def mean(key: str) -> float | None:
        values = vals(key)
        return float(np.mean(values)) if values else None

    q = vals("q")
    p = vals("p")
    g = vals("bce_gradient")
    d = vals("q_times_one_minus_p")
    disagreement = vals("relative_disagreement")
    special = [row for row in rows if row["q"] >= q_thr and row["p"] <= p_thr]
    return {
        "candidate_rows": len(rows),
        "gt_count": len({(row["image"], row["gt_index"]) for row in rows}),
        "q_mean": mean("q"), "p_mean": mean("p"), "bce_gradient_mean": mean("bce_gradient"),
        "d_mean": mean("q_times_one_minus_p"), "relative_disagreement_mean": mean("relative_disagreement"),
        "corr_pearson_D_abs_p_minus_y": corr(d, g),
        "corr_spearman_D_abs_p_minus_y": corr(d, g, "spearman"),
        "corr_pearson_D_target": corr(d, vals("target")),
        "corr_spearman_D_target": corr(d, vals("target"), "spearman"),
        "corr_pearson_D_q": corr(d, q),
        "corr_spearman_D_q": corr(d, q, "spearman"),
        "corr_pearson_disagreement_abs_p_minus_y": corr(disagreement, g),
        "corr_spearman_disagreement_abs_p_minus_y": corr(disagreement, g, "spearman"),
        "corr_pearson_disagreement_target": corr(disagreement, vals("target")),
        "corr_spearman_disagreement_target": corr(disagreement, vals("target"), "spearman"),
        "corr_pearson_disagreement_q": corr(disagreement, q),
        "corr_spearman_disagreement_q": corr(disagreement, q, "spearman"),
        "high_quality_low_confidence_count": len(special),
        "high_quality_low_confidence_fraction": len(special) / max(len(rows), 1),
        "high_quality_low_confidence_mean_bce_gradient": float(np.mean([r["bce_gradient"] for r in special])) if special else None,
        "d_top_quartile_count": sum(row["q_times_one_minus_p"] >= np.quantile(d, 0.75) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--q-thr", type=float, default=0.75)
    parser.add_argument("--p-thr", type=float, default=0.25)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    wrapper = YOLO(args.checkpoint)
    net = wrapper.model.to(device).eval()
    if isinstance(net.args, dict):
        net.args = get_cfg(overrides=net.args)
    elif isinstance(net.args, SimpleNamespace) and not hasattr(net.args, "box"):
        net.args = get_cfg(overrides=vars(net.args))
    criterion = v8DetectionLoss(net)
    factorized = bool(getattr(net.args, "factorized_tal_target", False))
    rows: list[dict] = []
    images = sorted(args.images.glob("*.png"))
    for image_index, image in enumerate(images, 1):
        tensor, batch = load_batch(image, args.imgsz, device)
        with torch.no_grad():
            _, preds = net(tensor)
            context, _, _ = criterion.get_assigned_targets_and_loss(preds, batch)
            fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = context
            n_p2 = int(preds["feats"][0].shape[-2] * preds["feats"][0].shape[-1])
            p = preds["scores"].permute(0, 2, 1).sigmoid()[0, :n_p2, 0]
            target = criterion.dbss_assignment_context["p2_target_scores"][0, :n_p2, 0]
            distri = preds["boxes"].permute(0, 2, 1).contiguous()
            residual = preds.get("dfl_residual")
            residual = residual.permute(0, 2, 1).contiguous() if residual is not None else None
            decoded = criterion.bbox_decode(anchor_points, distri, residual, stride_tensor)[0] * stride_tensor[:n_p2]
            pos = fg_mask[0, :n_p2]
            ious = bbox_iou(decoded, target_bboxes[0, :n_p2], xywh=False, CIoU=False).squeeze(-1)
            gt_idx = target_gt_idx[0, :n_p2]
        for gt in torch.unique(gt_idx[pos]).tolist():
            selected = pos & (gt_idx == gt)
            indices = selected.nonzero().flatten().tolist()
            if not indices:
                continue
            q = ious[indices].clamp(0, 1).cpu().numpy()
            scores = p[indices].clamp(1e-6, 1 - 1e-6).cpu().numpy()
            targets = target[indices].clamp(0, 1).cpu().numpy()
            q_norm = q / max(float(q.max()), 1e-6)
            p_norm = scores / max(float(scores.max()), 1e-6)
            rel = np.maximum(q_norm - p_norm, 0)
            for local, index in enumerate(indices):
                rows.append({
                    "image": image.name, "gt_index": int(gt), "candidate_index": int(index),
                    "q": float(q[local]), "p": float(scores[local]), "target": float(targets[local]),
                    "bce_gradient": float(abs(scores[local] - targets[local])),
                    "q_times_one_minus_p": float(q[local] * (1 - scores[local])),
                    "relative_disagreement": float(rel[local]),
                })
        if image_index % 100 == 0:
            print(f"{image_index}/{len(images)} images, {len(rows)} positive P2 candidates", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "images": len(images), "imgsz": args.imgsz,
        "nms_iou": 0.5, "factorized_tal_target": factorized,
        "target_semantics": "FTAL target" if factorized else "standard TAL target (checkpoint metadata)",
        "thresholds": {"q": args.q_thr, "p": args.p_thr},
        "summary": summarize(rows, args.q_thr, args.p_thr),
    }, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summarize(rows, args.q_thr, args.p_thr), indent=2))


if __name__ == "__main__":
    main()
