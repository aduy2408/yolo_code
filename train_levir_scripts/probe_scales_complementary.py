#!/usr/bin/env python3
"""Unified localized probing for R_s and R_l scale evaluation on contrast_basis."""

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
from probe_contrast_basis_stems import ap_auc, train_probe, evaluate_probe, load_split_samples, get_candidates


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

    ckpt_path = ROOT / "runs/checkpoints_full/contrast_basis_best.pt"
    if not ckpt_path.is_file():
        print(f"contrast_basis checkpoint not found at {ckpt_path}. Exiting.")
        sys.exit(1)

    print("Loading contrast_basis checkpoint...")
    wrapper = YOLO(ckpt_path)
    net = wrapper.model.to(device).eval()

    val_samples = load_split_samples(args.dataset_root, "val")
    test_samples = load_split_samples(args.dataset_root, "test")

    if args.smoke_only:
        val_samples = [s for s in val_samples if len(s["boxes"]) > 0][:20]
        test_samples = [s for s in test_samples if len(s["boxes"]) > 0][:20]
        print(f"[SMOKE] Filtered VAL samples: {len(val_samples)}, TEST samples: {len(test_samples)}")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)

    # Hooks setup to grab rel_small and rel_large inputs/outputs
    # We hook rel_encoder output during forward propagation
    # Since rel_encoder is called twice (first with basis_small, then with basis_large),
    # we intercept both activations sequentially using a list accumulator.
    rel_encoder_calls = []

    def hook_fn(_mod, _inp, out):
        if isinstance(out, tuple):
            rel_encoder_calls.append(out[0].squeeze(0).detach().cpu().clone())
        else:
            rel_encoder_calls.append(out.squeeze(0).detach().cpu().clone())

    stem = net.model[0]
    handle = stem.rel_encoder.register_forward_hook(hook_fn)

    stages = ["R_s", "R_l", "[R_s,R_l]"]
    val_data = {k: [] for k in stages}
    val_ious, val_y = [], []

    # VAL collection
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
            rel_encoder_calls.clear()
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0 or len(rel_encoder_calls) < 2:
                continue

            # rel_encoder_calls[0] is rel_small, rel_encoder_calls[1] is rel_large
            rs_tensor = rel_encoder_calls[0]
            rl_tensor = rel_encoder_calls[1]

            c, h_f, w_f = rs_tensor.shape
            sliced_rs = rs_tensor.permute(1, 2, 0).reshape(h_f * w_f, c)[idx.cpu()]
            sliced_rl = rl_tensor.permute(1, 2, 0).reshape(h_f * w_f, c)[idx.cpu()]

            val_data["R_s"].append(sliced_rs)
            val_data["R_l"].append(sliced_rl)
            val_data["[R_s,R_l]"].append(torch.cat([sliced_rs, sliced_rl], dim=1))
            
            val_ious.append(local_ious.cpu().detach().clone())
            val_y.append(y.cpu().detach().clone())

    if not val_y:
        print("No valid candidates collected in VAL split.")
        handle.remove()
        sys.exit(1)

    val_X = {k: torch.cat(val_data[k], dim=0).to(device) for k in stages}
    val_y_cat = torch.cat(val_y, dim=0).to(device)

    # Train Unified Linear Probes on VAL
    probes = {}
    for k in stages:
        probes[k] = train_probe(val_X[k].detach(), val_y_cat, args.epochs)

    # TEST collection
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
            rel_encoder_calls.clear()
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0 or len(rel_encoder_calls) < 2:
                continue

            rs_tensor = rel_encoder_calls[0]
            rl_tensor = rel_encoder_calls[1]

            c, h_f, w_f = rs_tensor.shape
            sliced_rs = rs_tensor.permute(1, 2, 0).reshape(h_f * w_f, c)[idx.cpu()]
            sliced_rl = rl_tensor.permute(1, 2, 0).reshape(h_f * w_f, c)[idx.cpu()]

            case_rep = {
                "R_s": sliced_rs.to(device),
                "R_l": sliced_rl.to(device),
                "[R_s,R_l]": torch.cat([sliced_rs, sliced_rl], dim=1).to(device),
            }
            test_case_groups.append((case_rep, local_ious, y))

    handle.remove()

    # Evaluate on TEST split
    final_results = {}
    for k in stages:
        case_results = []
        for case_rep, local_ious, y in test_case_groups:
            eval_metrics = evaluate_probe(probes[k], case_rep[k].detach(), local_ious, y)
            case_results.append(eval_metrics)

        # Average metrics
        summary_metrics = {}
        keys = case_results[0].keys()
        for key in keys:
            vals = [r[key] for r in case_results if not np.isnan(r[key])]
            summary_metrics[key] = float(np.mean(vals)) if vals else float("nan")
        summary_metrics["count"] = len(case_results)
        final_results[k] = summary_metrics

    # Save and output
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_scales_results.json"
    
    output = {
        "protocol": {
            "split": "test",
            "nms_iou_for_cohort_selection": 0.5,
            "raw_candidate_source": "P2 decoded pre-NMS head output",
            "candidate_conf": None,
            "positive": "IoU >= 0.1",
            "checkpoint": str(ckpt_path),
        },
        "summary": final_results,
    }
    
    out_file.write_text(json.dumps(output, indent=2))
    print("\nSCALE EVALUATION PROBING RESULTS:")
    print(json.dumps(output, indent=2))
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
