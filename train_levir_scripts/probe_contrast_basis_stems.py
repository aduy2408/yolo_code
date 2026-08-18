#!/usr/bin/env python3
"""Unified localized linear logistic probing script for LocalContrastBasisStem and Native Cross-Reconstruction."""

from __future__ import annotations

import argparse
import json
import random
import sys
import yaml
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

from probe_center_ring_cohorts import iou_matrix, read_labels  # noqa: E402


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


def evaluate_probe(model: nn.Linear, x: torch.Tensor, local_ious: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = model(x).squeeze(1)
        scores = torch.sigmoid(logits)
    
    ap, auc = ap_auc(scores, y)
    best_iou_idx = int(torch.argmax(local_ious))
    order = torch.argsort(scores, descending=True)
    rank = int((order == best_iou_idx).nonzero(as_tuple=False)[0].item()) + 1
    
    # Within-GT positive pair accuracy (pairwise accuracy among positives with different IoU values)
    pos_indices = (y == 1).nonzero(as_tuple=False).flatten()
    pair_acc = float("nan")
    if len(pos_indices) > 1:
        ious_pos = local_ious[pos_indices]
        scores_pos = scores[pos_indices]
        diff_mask = ious_pos[:, None] > ious_pos[None, :] + 0.05
        if diff_mask.any():
            correct = (scores_pos[:, None] > scores_pos[None, :])[diff_mask].float().mean().item()
            pair_acc = float(correct)
            
    out = {
        "ap": ap,
        "auc": auc,
        "best_iou_rank": rank,
        "within_gt_pair_acc": pair_acc,
    }
    for k in (1, 5):
        top = order[: min(k, len(order))]
        out[f"recall_at_{k}"] = float((local_ious[top] >= 0.5).any().item())
    return out


def load_split_samples(dataset_root: Path, split: str) -> list[dict]:
    # Read YAML configuration
    yaml_path = dataset_root / "levir_ship_yolo_seed42/levir_ship.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    split_dir = dataset_root / "levir_ship_yolo_seed42" / config[split]
    samples = []
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            samples.append({"image": p, "boxes": read_labels(p)})
    return samples


def get_candidates(net, original_shape, letterbox, image_path, gt, device, near_cells=8, top_neg=256):
    h, w = original_shape[:2]
    image = letterbox(image=cv2.imread(str(image_path)))
    tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255.0
    
    with torch.no_grad():
        decoded, preds = net(tensor)
        
    p2 = preds["feats"][0].squeeze(0)
    c, hp, wp = p2.shape
    n_p2 = hp * wp
    
    boxes = xywh2xyxy(decoded[0, :4, :n_p2].T)
    logits = preds["scores"][0, 0, :n_p2]
    
    ious = iou_matrix(gt.view(1, 4), boxes).squeeze(0)
    yy, xx = torch.meshgrid(torch.arange(hp, device=device), torch.arange(wp, device=device), indexing="ij")
    gx = ((gt[0] + gt[2]) * 0.5 * wp / w)
    gy = ((gt[1] + gt[3]) * 0.5 * hp / h)
    dist = ((xx.flatten() - gx) ** 2 + (yy.flatten() - gy) ** 2).sqrt()
    
    # We define positive candidates relaxed as IoU >= 0.1 for maximum candidate pool size on smoke subsets
    # and IoU >= 0.3 for a cleaner representation during full execution.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--final-conf", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    random.seed(42)
    torch.manual_seed(42)

    checkpoints = {
        "raw_control": Path("runs/levir_yolov8n_p2_contrast_basis/_smoke/raw_control/weights/best.pt"),
        "contrast_no_cross": Path("runs/detect/runs/levir_yolov8n_p2_contrast_basis/_smoke/contrast_no_cross/weights/best.pt"),
        "contrast_basis": Path("runs/levir_yolov8n_p2_contrast_basis/_smoke/contrast_basis/weights/best.pt"),
        "contrast_basis_ffm": Path("runs/levir_yolov8n_p2_contrast_basis_ffm/_smoke/contrast_basis_ffm/weights/best.pt"),
        "native_concat": Path("runs/levir_yolov8n_p2_native_cross/_smoke/native_concat/weights/best.pt"),
        "native_ffm": Path("runs/levir_yolov8n_p2_native_cross/_smoke/native_ffm/weights/best.pt"),
    }

    # Filter out missing checkpoints (if any)
    available_ckpts = {name: Path(ROOT / path) for name, path in checkpoints.items() if (ROOT / path).is_file()}
    if not available_ckpts:
        print("No checkpoints found. Please make sure the paths are correct.")
        sys.exit(1)

    print("Found checkpoints:", list(available_ckpts.keys()))
    
    # Load VAL and TEST splits separately
    val_samples = load_split_samples(args.dataset_root, "val")
    test_samples = load_split_samples(args.dataset_root, "test")
    print(f"VAL samples: {len(val_samples)}, TEST samples: {len(test_samples)}")

    if args.smoke_only:
        val_samples = [s for s in val_samples if len(s["boxes"]) > 0][:50]
        test_samples = [s for s in test_samples if len(s["boxes"]) > 0][:50]
        print(f"[SMOKE] Filtered VAL samples: {len(val_samples)}, TEST samples: {len(test_samples)}")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    final_results = {}

    for variant, ckpt in available_ckpts.items():
        print(f"\n--- Probing variant: {variant} ---")
        wrapper = YOLO(ckpt)
        net = wrapper.model.to(device).eval()

        # Hooks setup
        hooked_tensors = {}
        handles = []

        def register_hook(name, module):
            def hook_fn(_mod, _inp, out):
                if isinstance(out, tuple):
                    hooked_tensors[name] = out[0].squeeze(0)
                else:
                    hooked_tensors[name] = out.squeeze(0)
            handles.append(module.register_forward_hook(hook_fn))

        is_native = variant.startswith("native")
        if is_native:
            # Native targets
            register_hook("A", net.model[1])
            register_hook("B", net.model[2])
            register_hook("F", net.model[3])
            rep_keys = ["A", "B", "[A,B]", "F"]
        else:
            # Contrast basis targets
            stem = net.model[0]
            register_hook("M", stem.main_c2f)
            register_hook("R", stem.scale_formation)
            register_hook("F", stem)
            if variant == "contrast_basis_ffm":
                register_hook("F_ffm", stem.conv_out)
                rep_keys = ["M", "R", "[M,R]", "F", "[M,R_shuffle]", "F_ffm"]
            else:
                rep_keys = ["M", "R", "[M,R]", "F", "[M,R_shuffle]"]

        # Collect features and candidates for VAL
        val_data = {k: [] for k in rep_keys}
        val_ious = []
        val_y = []

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
                
                # Retrieve hooked activations
                feats = {}
                for name, t in hooked_tensors.items():
                    c, h_f, w_f = t.shape
                    # P2 feature dimension matches P2 head resolution (stride 4 -> 128x128 for 512x512)
                    # For deeper levels in native, coordinate scaling is automatically handled by the head's mapping.
                    # We flatten the activation map and slice with candidate indices
                    feats[name] = t.permute(1, 2, 0).reshape(h_f * w_f, c)[idx]

                # Construct representation variants
                if is_native:
                    feats["[A,B]"] = torch.cat([feats["A"], feats["B"]], dim=1)
                else:
                    feats["[M,R]"] = torch.cat([feats["M"], feats["R"]], dim=1)
                    r_feat = feats["R"]
                    shuffled_r = r_feat[torch.randperm(len(r_feat), device=device)]
                    feats["[M,R_shuffle]"] = torch.cat([feats["M"], shuffled_r], dim=1)

                for k in rep_keys:
                    val_data[k].append(feats[k].cpu())
                val_ious.append(local_ious.cpu())
                val_y.append(y.cpu())

        if not val_y:
            print(f"Skipping {variant} because no valid candidates were collected in VAL.")
            for h in handles:
                h.remove()
            continue

        # Concatenate all VAL candidates
        val_X = {k: torch.cat(val_data[k], dim=0).to(device) for k in rep_keys}
        val_y_cat = torch.cat(val_y, dim=0).to(device)

        # Train one unified probe per representation on VAL
        probes = {}
        for k in rep_keys:
            probes[k] = train_probe(val_X[k], val_y_cat, args.epochs)

        # Collect features and candidates for TEST
        test_case_groups = []

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
                
                # Retrieve hooked activations
                feats = {}
                for name, t in hooked_tensors.items():
                    c, h_f, w_f = t.shape
                    feats[name] = t.permute(1, 2, 0).reshape(h_f * w_f, c)[idx]

                # Construct representation variants
                if is_native:
                    feats["[A,B]"] = torch.cat([feats["A"], feats["B"]], dim=1)
                else:
                    feats["[M,R]"] = torch.cat([feats["M"], feats["R"]], dim=1)
                    r_feat = feats["R"]
                    shuffled_r = r_feat[torch.randperm(len(r_feat), device=device)]
                    feats["[M,R_shuffle]"] = torch.cat([feats["M"], shuffled_r], dim=1)

                case_rep = {k: feats[k] for k in rep_keys}
                test_case_groups.append((case_rep, local_ious, y))

        for h in handles:
            h.remove()

        # Evaluate on TEST split GT groups
        variant_results = {k: [] for k in rep_keys}
        for case_rep, local_ious, y in test_case_groups:
            for k in rep_keys:
                eval_metrics = evaluate_probe(probes[k], case_rep[k], local_ious, y)
                variant_results[k].append(eval_metrics)

        # Average results over test cases
        summary_metrics = {}
        for k in rep_keys:
            summary_metrics[k] = {}
            keys = variant_results[k][0].keys()
            for key in keys:
                vals = [r[key] for r in variant_results[k] if not np.isnan(r[key])]
                summary_metrics[k][key] = float(np.mean(vals)) if vals else float("nan")
            summary_metrics[k]["count"] = len(variant_results[k])

        final_results[variant] = summary_metrics

    # Save to file
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_contrast_basis_results.json"
    
    output = {
        "protocol": {
            "split": "test",
            "nms_iou_for_cohort_selection": 0.5,
            "raw_candidate_source": "P2 decoded pre-NMS head output",
            "candidate_conf": None,
            "positive": "IoU >= 0.1",
            "checkpoints": {k: str(v) for k, v in available_ckpts.items()},
        },
        "summary": final_results,
    }
    
    out_file.write_text(json.dumps(output, indent=2))
    print("\nPROBING RESULTS:")
    print(json.dumps(output, indent=2))
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
