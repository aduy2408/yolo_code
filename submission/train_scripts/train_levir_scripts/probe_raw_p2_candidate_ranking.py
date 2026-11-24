#!/usr/bin/env python3
"""Raw P2 candidate ranking probes for localized low-confidence misses."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402

from probe_center_ring_cohorts import CHECKPOINTS, iou_matrix, load_samples, scan_cohorts  # noqa: E402


def ring_mean(x: torch.Tensor, radius: int = 5) -> torch.Tensor:
    c, _, _ = x.shape
    size = 2 * radius + 1
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist = ((xx - radius) ** 2 + (yy - radius) ** 2).float().sqrt()
    mask = (dist > 1.0) & (dist <= radius)
    kernel = (mask.float() / mask.sum()).view(1, 1, size, size).repeat(c, 1, 1, 1).to(x.device)
    return F.conv2d(F.pad(x.unsqueeze(0), (radius, radius, radius, radius), mode="replicate"), kernel, groups=c).squeeze(0)


def ap_auc(scores: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    order = torch.argsort(scores, descending=True)
    sorted_y = y[order].float()
    tp = sorted_y.cumsum(0)
    fp = (1 - sorted_y).cumsum(0)
    recall = tp / sorted_y.sum().clamp_min(1)
    precision = tp / (tp + fp).clamp_min(1)
    ap = float(((recall[1:] - recall[:-1]) * precision[1:]).sum().item())
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return ap, float("nan")
    auc = float(((pos[:, None] > neg[None]).float().mean() + 0.5 * (pos[:, None] == neg[None]).float().mean()).item())
    return ap, auc


def train_probe(x: torch.Tensor, y: torch.Tensor, epochs: int) -> nn.Linear:
    model = nn.Linear(x.shape[1], 1).to(x.device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(y == 0).sum().float() / (y == 1).sum().float().clamp_min(1))
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x).squeeze(1), y.float()).backward()
        opt.step()
    return model


def metrics(scores: torch.Tensor, ious: torch.Tensor, y: torch.Tensor) -> dict:
    ap, auc = ap_auc(scores, y)
    best_iou_idx = int(torch.argmax(ious))
    order = torch.argsort(scores, descending=True)
    rank = int((order == best_iou_idx).nonzero(as_tuple=False)[0].item()) + 1
    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    margin = float(pos_scores.max().item() - neg_scores.max().item()) if len(pos_scores) and len(neg_scores) else float("nan")
    out = {"ap": ap, "auc": auc, "best_iou_rank": rank, "pairwise_margin": margin}
    for k in (1, 5, 10):
        top = order[: min(k, len(order))]
        out[f"recall_at_{k}"] = float((ious[top] >= 0.5).any().item())
    return out


def flatten_feature(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    c, h, w = feat.shape
    return feat.permute(1, 2, 0).reshape(h * w, c)[indices]


def raw_p2(net: nn.Module, image_np: np.ndarray, letterbox: LetterBox, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = letterbox(image=image_np)
    tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad():
        decoded, preds = net(tensor)
    p2 = preds["feats"][0].squeeze(0)
    n_p2 = p2.shape[1] * p2.shape[2]
    boxes = xywh2xyxy(decoded[0, :4, :n_p2].T)
    logits = preds["scores"][0, 0, :n_p2]
    return p2, boxes, logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--final-conf", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--near-cells", type=int, default=8)
    parser.add_argument("--top-neg", type=int, default=256)
    parser.add_argument("--extra-checkpoint", nargs="*", default=[], metavar="NAME=PATH")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    checkpoints = dict(CHECKPOINTS)
    for item in args.extra_checkpoint:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--extra-checkpoint must be NAME=PATH, got {item!r}")
        checkpoints[name] = Path(path)
    samples = load_samples(args.dataset_root)
    ref = YOLO(CHECKPOINTS["GAP+FTAL"])
    misses = [m for m in scan_cohorts(ref, samples, device, args.imgsz, args.final_conf) if m["cohort"] == "C_low_conf_localized"]
    if args.max_cases:
        misses = misses[: args.max_cases]

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    summary: dict[str, dict] = {}
    for variant, ckpt in checkpoints.items():
        wrapper = YOLO(ckpt)
        net = wrapper.model.to(device).eval()
        rows: dict[str, list[dict]] = {"raw_logit": [], "F": [], "Ring": [], "[F,Ring]": [], "[F,Ring]_shuffle": []}
        for miss in misses:
            original = cv2.imread(str(miss["image"]))
            h, w = original.shape[:2]
            p2, boxes, logits = raw_p2(net, original, letterbox, device)
            ious = iou_matrix(miss["gt"].to(device).view(1, 4), boxes).squeeze(0)
            c, hp, wp = p2.shape
            yy, xx = torch.meshgrid(torch.arange(hp, device=device), torch.arange(wp, device=device), indexing="ij")
            gx = ((miss["gt"][0] + miss["gt"][2]) * 0.5 * wp / w).to(device)
            gy = ((miss["gt"][1] + miss["gt"][3]) * 0.5 * hp / h).to(device)
            dist = ((xx.flatten() - gx) ** 2 + (yy.flatten() - gy) ** 2).sqrt()
            pos = ious >= 0.5
            neg_pool = (ious < 0.3) & (dist <= args.near_cells)
            if neg_pool.sum() < 10:
                neg_pool = ious < 0.3
            neg_idx = torch.nonzero(neg_pool, as_tuple=False).flatten()
            if len(neg_idx) > args.top_neg:
                top = torch.argsort(logits[neg_idx], descending=True)[: args.top_neg]
                neg_idx = neg_idx[top]
            idx = torch.cat([torch.nonzero(pos, as_tuple=False).flatten(), neg_idx]).unique()
            if pos.sum() == 0 or len(idx) < 2:
                continue
            y = (ious[idx] >= 0.5).long()
            local_ious = ious[idx]
            f = flatten_feature(p2, idx)
            rmap = ring_mean(p2)
            r = flatten_feature(rmap, idx)
            shuffle_r = r[torch.randperm(len(r), device=device)]
            reps = {"F": f, "Ring": r, "[F,Ring]": torch.cat([f, r], dim=1)}
            rows["raw_logit"].append(metrics(logits[idx], local_ious, y))
            for name, x in reps.items():
                model = train_probe(x, y, args.epochs)
                rows[name].append(metrics(torch.sigmoid(model(x).squeeze(1)), local_ious, y))
                if name == "[F,Ring]":
                    rows["[F,Ring]_shuffle"].append(
                        metrics(torch.sigmoid(model(torch.cat([f, shuffle_r], dim=1)).squeeze(1)), local_ious, y)
                    )

        summary[variant] = {}
        for name, vals in rows.items():
            if not vals:
                continue
            keys = vals[0].keys()
            summary[variant][name] = {k: float(np.nanmean([v[k] for v in vals])) for k in keys}
            summary[variant][name]["count"] = len(vals)

    output = {
        "protocol": {
            "split": "test",
            "nms_iou_for_cohort_selection": 0.5,
            "raw_candidate_source": "P2 decoded pre-NMS head output",
            "candidate_conf": None,
            "cohort": "C_low_conf_localized",
            "case_count": len(misses),
            "positive": "IoU >= 0.5",
            "hard_negative": f"IoU < 0.3, prefer cells within {args.near_cells} P2 cells and top {args.top_neg} raw logits",
            "checkpoints": {k: str(v) for k, v in checkpoints.items()},
        },
        "summary": summary,
    }
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_raw_p2_candidate_ranking_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
