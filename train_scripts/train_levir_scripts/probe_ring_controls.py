#!/usr/bin/env python3
"""Controls for the ring signal in localized low-confidence LEVIR misses."""

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

from probe_center_ring_cohorts import CHECKPOINTS, gt_mask_for, load_samples, scan_cohorts  # noqa: E402


def annulus_mean(x: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
    c, _, _ = x.shape
    size = 2 * r1 + 1
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist = ((xx - r1) ** 2 + (yy - r1) ** 2).float().sqrt()
    mask = (dist >= r0) & (dist <= r1)
    kernel = (mask.float() / mask.sum()).view(1, 1, size, size).repeat(c, 1, 1, 1).to(x.device)
    return F.conv2d(F.pad(x.unsqueeze(0), (r1, r1, r1, r1), mode="replicate"), kernel, groups=c).squeeze(0)


def ap_from_probs(probs: torch.Tensor, y: torch.Tensor) -> float:
    order = torch.argsort(probs, descending=True)
    sorted_y = y[order]
    tp = sorted_y.cumsum(0)
    fp = (1 - sorted_y).cumsum(0)
    recall = tp / sorted_y.sum().clamp_min(1)
    precision = tp / (tp + fp).clamp_min(1)
    return float(((recall[1:] - recall[:-1]) * precision[1:]).sum().item())


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


def score(model: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return ap_from_probs(torch.sigmoid(model(x).squeeze(1)), y)


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.permute(1, 2, 0).reshape(-1, x.shape[0])


def erase_gt(original: np.ndarray, gt: torch.Tensor) -> np.ndarray:
    out = original.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in gt]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(max(x2, x1 + 1), w), min(max(y2, y1 + 1), h)
    pad = 8
    xa, ya, xb, yb = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, w), min(y2 + pad, h)
    patch = out[ya:yb, xa:xb].copy()
    patch[y1 - ya:y2 - ya, x1 - xa:x2 - xa] = 0
    keep = np.ones((yb - ya, xb - xa), dtype=bool)
    keep[y1 - ya:y2 - ya, x1 - xa:x2 - xa] = False
    fill = np.median(patch[keep], axis=0) if keep.any() else np.median(out.reshape(-1, 3), axis=0)
    out[y1:y2, x1:x2] = fill.astype(out.dtype)
    return out


def random_ring(ring: torch.Tensor, gt_mask: torch.Tensor, other: torch.Tensor | None = None) -> torch.Tensor:
    source = ring if other is None else other
    bg = torch.nonzero(~gt_mask, as_tuple=False)
    if len(bg) == 0:
        return source.mean(dim=(1, 2), keepdim=True).expand_as(source)
    y, x = bg[random.randrange(len(bg))]
    return source[:, y, x].view(-1, 1, 1).expand_as(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--final-conf", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    samples = load_samples(args.dataset_root)
    ref = YOLO(CHECKPOINTS["GAP+FTAL"])
    misses = [m for m in scan_cohorts(ref, samples, device, args.imgsz, args.final_conf) if m["cohort"] == "C_low_conf_localized"]
    if args.max_cases:
        misses = misses[: args.max_cases]

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    bands = {"near_1_2": (1, 2), "mid_3_4": (3, 4), "far_5_7": (5, 7)}
    summary: dict[str, dict] = {}

    for variant, ckpt in CHECKPOINTS.items():
        wrapper = YOLO(ckpt)
        net = wrapper.model.to(device).eval()
        p2_idx = 19 if type(net.model[19]).__name__ == "ChannelAttention" else 18
        acts = {}

        def hook(_module, _inp, out):
            acts["p2"] = out.squeeze(0)

        handle = net.model[p2_idx].register_forward_hook(hook)
        rows = []
        prev_ring = None
        for miss in misses:
            original = cv2.imread(str(miss["image"]))
            h, w = original.shape[:2]

            def forward(image_np: np.ndarray) -> torch.Tensor:
                image = letterbox(image=image_np)
                tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
                with torch.no_grad():
                    net(tensor)
                return acts["p2"]

            p2 = forward(original)
            mask = gt_mask_for(miss["gt"].to(device), p2.shape[1:], (h, w), device)
            y = mask.flatten().long()
            center = p2
            ring = annulus_mean(p2, 1, 5)
            erased_ring = annulus_mean(forward(erase_gt(original, miss["gt"])), 1, 5)

            ring_model = train_probe(flatten(ring), y, args.epochs)
            pair_model = train_probe(flatten(torch.cat([center, ring], dim=0)), y, args.epochs)
            rows.append(("Ring original", score(ring_model, flatten(ring), y)))
            rows.append(("Ring after object erasure", score(ring_model, flatten(erased_ring), y)))
            rows.append(("[F,R] local", score(pair_model, flatten(torch.cat([center, ring], dim=0)), y)))
            rows.append(("[F,R] same-image random", score(pair_model, flatten(torch.cat([center, random_ring(ring, mask)], dim=0)), y)))
            if prev_ring is not None:
                rows.append(("[F,R] cross-image random", score(pair_model, flatten(torch.cat([center, random_ring(ring, mask, prev_ring)], dim=0)), y)))
            prev_ring = ring.detach()

            for name, (r0, r1) in bands.items():
                band = annulus_mean(p2, r0, r1)
                rows.append((f"{name} Ring", score(train_probe(flatten(band), y, args.epochs), flatten(band), y)))
                rows.append((f"{name} [F,Ring]", score(train_probe(flatten(torch.cat([center, band], dim=0)), y, args.epochs), flatten(torch.cat([center, band], dim=0)), y)))
        handle.remove()

        summary[variant] = {
            name: {
                "mean_ap": float(np.mean([v for k, v in rows if k == name])),
                "count": sum(k == name for k, _ in rows),
            }
            for name in sorted({k for k, _ in rows})
        }

    output = {
        "protocol": {
            "split": "test",
            "nms_iou": 0.5,
            "candidate_conf": 0.001,
            "final_conf": args.final_conf,
            "cohort": "C_low_conf_localized",
            "case_count": len(misses),
            "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
        },
        "summary": summary,
    }
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_ring_controls_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
