#!/usr/bin/env python3
"""Localized linear logistic probing for plain YOLOv8n-P2 baseline checkpoints.

Probes the three stem layers of the standard P2 backbone (cv1, cv2, c2f/P2)
using checkpoints downloaded from HuggingFace (duyle2408/levir-ship-yolo-p2).
Trains probes on VAL, evaluates on TEST, averages across seeds.
"""

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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402

from probe_center_ring_cohorts import iou_matrix, read_labels  # noqa: E402


# ---------------------------------------------------------------------------
# Shared probe utilities (mirrors probe_contrast_basis_stems.py)
# ---------------------------------------------------------------------------

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
    auc = float(
        ((pos[:, None] > neg[None]).float().mean()
         + 0.5 * (pos[:, None] == neg[None]).float().mean()).item()
    )
    return ap, auc


def train_probe(x: torch.Tensor, y: torch.Tensor, epochs: int) -> nn.Linear:
    model = nn.Linear(x.shape[1], 1).to(x.device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=(y == 0).sum().float() / (y == 1).sum().float().clamp_min(1)
    )
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x).squeeze(1), y.float()).backward()
        opt.step()
    return model


def evaluate_probe(model, x, local_ious, y):
    with torch.no_grad():
        scores = torch.sigmoid(model(x).squeeze(1))
    ap, auc = ap_auc(scores, y)
    order = torch.argsort(scores, descending=True)
    best_iou_idx = int(torch.argmax(local_ious))
    rank = int((order == best_iou_idx).nonzero(as_tuple=False)[0].item()) + 1

    pos_indices = (y == 1).nonzero(as_tuple=False).flatten()
    pair_acc = float("nan")
    if len(pos_indices) > 1:
        ious_pos = local_ious[pos_indices]
        scores_pos = scores[pos_indices]
        diff_mask = ious_pos[:, None] > ious_pos[None, :] + 0.05
        if diff_mask.any():
            pair_acc = float(
                (scores_pos[:, None] > scores_pos[None, :])[diff_mask].float().mean().item()
            )

    out = {"ap": ap, "auc": auc, "best_iou_rank": rank, "within_gt_pair_acc": pair_acc}
    for k in (1, 5):
        top = order[: min(k, len(order))]
        out[f"recall_at_{k}"] = float((local_ious[top] >= 0.5).any().item())
    return out


def load_split_samples(dataset_root: Path, split: str) -> list[dict]:
    yaml_path = dataset_root / "levir_ship_yolo_seed42/levir_ship.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    split_dir = dataset_root / "levir_ship_yolo_seed42" / config[split]
    samples = []
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            samples.append({"image": p, "boxes": read_labels(p)})
    return samples


def get_candidates(net, original_shape, letterbox, image_path, gt, device,
                   near_cells=8, top_neg=256):
    h, w = original_shape[:2]
    image = letterbox(image=cv2.imread(str(image_path)))
    tensor = (
        torch.from_numpy(image[..., ::-1].copy())
        .to(device).permute(2, 0, 1).float()[None] / 255.0
    )
    with torch.no_grad():
        decoded, preds = net(tensor)

    p2 = preds["feats"][0].squeeze(0)
    c, hp, wp = p2.shape
    n_p2 = hp * wp
    boxes = xywh2xyxy(decoded[0, :4, :n_p2].T)
    logits = preds["scores"][0, 0, :n_p2]

    ious = iou_matrix(gt.view(1, 4), boxes).squeeze(0)
    yy, xx = torch.meshgrid(
        torch.arange(hp, device=device),
        torch.arange(wp, device=device),
        indexing="ij",
    )
    gx = (gt[0] + gt[2]) * 0.5 * wp / w
    gy = (gt[1] + gt[3]) * 0.5 * hp / h
    dist = ((xx.flatten() - gx) ** 2 + (yy.flatten() - gy) ** 2).sqrt()

    pos = ious >= 0.1
    neg_pool = (ious < 0.05) & (dist <= near_cells)
    if neg_pool.sum() < 10:
        neg_pool = ious < 0.05
    neg_idx = torch.nonzero(neg_pool, as_tuple=False).flatten()
    if len(neg_idx) > top_neg:
        top = torch.argsort(logits[neg_idx], descending=True)[:top_neg]
        neg_idx = neg_idx[top]
    idx = torch.cat([torch.nonzero(pos, as_tuple=False).flatten(), neg_idx]).unique()
    return idx, ious[idx], (ious[idx] >= 0.1).long()


# ---------------------------------------------------------------------------
# Plain P2 probe: hook layers 0/1/2 = cv1/cv2/c2f
# ---------------------------------------------------------------------------
# Plain P2 architecture (fpn_only_plain.yaml):
#   model[0] = Conv(3->32, k=3, s=2)    stride-2, 256x256  (cv1)
#   model[1] = Conv(32->64, k=3, s=2)   stride-4, 128x128  (cv2 = P2 res)
#   model[2] = C2f(64->64)              stride-4, 128x128  (c2f = P2)
# Candidate indices are at P2 head resolution (128x128 for 512px input).
# cv2 and c2f share the same spatial resolution → safe to use idx directly.
# cv1 is 256x256 → idx must be remapped (each P2 cell = 4 cv1 cells).

REP_KEYS = ["cv1", "cv2", "c2f", "c2f_fused"]


def _extract_feat(hooked: dict, key: str, idx: torch.Tensor, p2_hw: tuple[int, int]) -> torch.Tensor | None:
    """Extract flattened feature at candidate positions.

    p2_hw: (hp, wp) at P2 / stride-4 resolution (e.g. 128x128 for 512px).
    For cv1 (stride-2, double res), maps each P2 index to top-left cv1 cell.
    """
    if key not in hooked:
        return None
    t = hooked[key]  # (C, H, W)
    c, hf, wf = t.shape
    hp, wp = p2_hw

    if hf == hp and wf == wp:
        # Same resolution as P2: direct index
        n = hf * wf
        safe = idx[idx < n]
        if len(safe) < 2:
            return None
        return t.permute(1, 2, 0).reshape(n, c)[safe]
    elif hf == hp * 2 and wf == wp * 2:
        # cv1: stride-2 = 2x P2 resolution; map P2 idx -> cv1 top-left
        row = (idx // wp) * 2
        col = (idx % wp) * 2
        cv1_idx = row * wf + col
        n = hf * wf
        safe_mask = cv1_idx < n
        if safe_mask.sum() < 2:
            return None
        return t.permute(1, 2, 0).reshape(n, c)[cv1_idx[safe_mask]]
    else:
        return None


def probe_one_seed(ckpt: Path, val_samples, test_samples, device, letterbox, args) -> dict:
    wrapper = YOLO(str(ckpt))
    net = wrapper.model.to(device).eval()

    hooked: dict = {}
    handles = []
    for key, layer_idx in [("cv1", 0), ("cv2", 1), ("c2f", 2), ("c2f_fused", 18)]:
        def _hook(mod, inp, out, k=key):
            hooked[k] = out if isinstance(out, torch.Tensor) else out[0]
            hooked[k] = hooked[k].squeeze(0)
        handles.append(net.model[layer_idx].register_forward_hook(_hook))

    # ---- VAL ----
    val_data = {k: [] for k in REP_KEYS}
    val_y_list = []
    for sample in val_samples:
        original = cv2.imread(str(sample["image"]))
        if original is None:
            continue
        gt_boxes = sample["boxes"].to(device)
        if len(gt_boxes) == 0:
            continue
        gt_xyxy = xywh2xyxy(gt_boxes)
        gt_xyxy[:, [0, 2]] *= original.shape[1]
        gt_xyxy[:, [1, 3]] *= original.shape[0]
        for gt in gt_xyxy:
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0:
                continue
            # determine P2 resolution from c2f hook
            if "c2f" not in hooked:
                continue
            _, hp, wp = hooked["c2f"].shape
            feats = {k: _extract_feat(hooked, k, idx, (hp, wp)) for k in REP_KEYS}
            if any(v is None or len(v) < 2 for v in feats.values()):
                continue
            for k in REP_KEYS:
                val_data[k].append(feats[k].cpu())
            val_y_list.append(y.cpu())

    if not val_y_list:
        for h in handles: h.remove()
        return {}

    val_X = {k: torch.cat(val_data[k], 0).to(device) for k in REP_KEYS}
    val_y_cat = torch.cat(val_y_list, 0).to(device)
    probes = {k: train_probe(val_X[k], val_y_cat, args.epochs) for k in REP_KEYS}
    print(f"  Trained probes on {val_y_cat.shape[0]} VAL candidates "
          f"({int(val_y_cat.sum())} pos, {int((val_y_cat==0).sum())} neg)")

    # ---- TEST ----
    test_cases = []
    for sample in test_samples:
        original = cv2.imread(str(sample["image"]))
        if original is None:
            continue
        gt_boxes = sample["boxes"].to(device)
        if len(gt_boxes) == 0:
            continue
        gt_xyxy = xywh2xyxy(gt_boxes)
        gt_xyxy[:, [0, 2]] *= original.shape[1]
        gt_xyxy[:, [1, 3]] *= original.shape[0]
        for gt in gt_xyxy:
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0:
                continue
            if "c2f" not in hooked:
                continue
            _, hp, wp = hooked["c2f"].shape
            feats = {k: _extract_feat(hooked, k, idx, (hp, wp)) for k in REP_KEYS}
            if any(v is None or len(v) < 2 for v in feats.values()):
                continue
            test_cases.append((feats, local_ious, y))

    for h in handles:
        h.remove()

    variant_results = {k: [] for k in REP_KEYS}
    for case_feats, local_ious, y in test_cases:
        for k in REP_KEYS:
            m = evaluate_probe(probes[k], case_feats[k], local_ious, y)
            variant_results[k].append(m)

    summary = {}
    for k, rows in variant_results.items():
        if not rows:
            continue
        metric_keys = rows[0].keys()
        summary[k] = {
            mk: float(np.mean([r[mk] for r in rows if not np.isnan(r[mk])]
                               ) if [r[mk] for r in rows if not np.isnan(r[mk])] else float("nan"))
            for mk in metric_keys
        }
        summary[k]["count"] = len(rows)
    return summary


def average_seed_results(seed_results: list[dict]) -> dict:
    valid = [r for r in seed_results if r]
    if not valid:
        return {}
    all_keys = set().union(*(r.keys() for r in valid))
    out = {}
    for k in all_keys:
        rows = [r[k] for r in valid if k in r]
        metric_keys = rows[0].keys()
        out[k] = {
            mk: float(np.mean([r[mk] for r in rows if not np.isnan(r[mk])])
                      if [r[mk] for r in rows if not np.isnan(r[mk])] else float("nan"))
            for mk in metric_keys
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--ckpt-root", type=Path,
                        default=ROOT / "runs/levir_yolov8n_p2_plain")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    random.seed(42)
    torch.manual_seed(42)

    val_samples = load_split_samples(args.dataset_root, "val")
    test_samples = load_split_samples(args.dataset_root, "test")
    print(f"VAL: {len(val_samples)} images, TEST: {len(test_samples)} images")

    if args.smoke_only:
        val_samples = [s for s in val_samples if len(s["boxes"]) > 0][:50]
        test_samples = [s for s in test_samples if len(s["boxes"]) > 0][:50]
        print(f"[SMOKE] VAL: {len(val_samples)}, TEST: {len(test_samples)}")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)

    seed_results = []
    used_ckpts = {}
    for seed in args.seeds:
        ckpt = args.ckpt_root / f"seed_{seed}" / "weights" / "best.pt"
        if not ckpt.is_file():
            print(f"[SKIP] seed {seed}: {ckpt}")
            continue
        print(f"\n=== seed {seed} ===")
        used_ckpts[seed] = str(ckpt)
        result = probe_one_seed(ckpt, val_samples, test_samples, device, letterbox, args)
        seed_results.append(result)
        print(f"  {json.dumps(result, indent=2)}")

    if not seed_results:
        print("No valid seed results.")
        sys.exit(1)

    averaged = average_seed_results(seed_results)
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_plain_p2_results.json"

    output = {
        "protocol": {
            "model": "yolov8n_p2_baseline",
            "hf_repo": "duyle2408/levir-ship-yolo-p2",
            "nms_iou": 0.5,
            "split_eval": "test",
            "probe_trained_on": "val",
            "positive_def": "IoU >= 0.1",
            "rep_keys": REP_KEYS,
            "seeds": list(used_ckpts.keys()),
            "checkpoints": {str(s): p for s, p in used_ckpts.items()},
        },
        "per_seed": seed_results,
        "averaged": averaged,
    }

    out_file.write_text(json.dumps(output, indent=2))
    print("\n=== AVERAGED RESULTS ===")
    print(json.dumps(averaged, indent=2))
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
