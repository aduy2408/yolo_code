#!/usr/bin/env python3
"""Audit classification supervision for assigned YOLO candidates and background."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.loss import make_anchors, v8DetectionLoss  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

LEVEL_NAMES = ("P2", "P3", "P4", "P5", "P6")
GEOMETRY_GROUPS = ("iou_ge_075", "iou_050_075", "iou_lt_050")
SIZE_GROUPS = ("tiny", "small", "medium", "large")


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def size_group(area: float) -> str:
    if area < 100:
        return "tiny"
    if area <= 400:
        return "small"
    if area <= 1024:
        return "medium"
    return "large"


def geometry_group(iou: float) -> str:
    if iou >= 0.75:
        return "iou_ge_075"
    if iou >= 0.5:
        return "iou_050_075"
    return "iou_lt_050"


def read_labels(image: Path) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    rows = [line.split() for line in labels_for(image).read_text().splitlines() if line.strip()]
    cls = torch.tensor([int(row[0]) for row in rows], dtype=torch.float32).reshape(-1, 1)
    boxes = torch.tensor([[float(value) for value in row[1:5]] for row in rows], dtype=torch.float32).reshape(-1, 4)
    original = cv2.imread(str(image))
    if original is None:
        raise RuntimeError(f"could not read image: {image}")
    height, width = original.shape[:2]
    areas = [float(box[2] * width * box[3] * height) for box in boxes]
    return cls, boxes, areas


def level_slices(feats: list[torch.Tensor]) -> list[tuple[str, int, int]]:
    start = 0
    result = []
    for index, feat in enumerate(feats):
        end = start + math.prod(feat.shape[2:])
        result.append((LEVEL_NAMES[index], start, end))
        start = end
    return result


def inspect(checkpoint: Path, images: list[Path], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    wrapper = YOLO(checkpoint)
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    if args.expected_seed is not None and train_args.get("seed") != args.expected_seed:
        raise RuntimeError(f"expected seed {args.expected_seed}, got {train_args.get('seed')!r}")
    net = wrapper.model.to(args.torch_device).eval()
    loss = v8DetectionLoss(net)
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    positive_rows: list[dict[str, object]] = []
    image_level_rows: list[dict[str, object]] = []
    for image_index, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            continue
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).to(args.torch_device).permute(2, 0, 1).float()[None] / 255
        cls, boxes, areas = read_labels(image_path)
        batch = {
            "batch_idx": torch.zeros((len(boxes),), dtype=torch.long, device=args.torch_device),
            "cls": cls.to(args.torch_device), "bboxes": boxes.to(args.torch_device),
        }
        with torch.inference_mode():
            _, preds = net(tensor)
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            pred_residual = preds.get("dfl_residual")
            pred_residual = pred_residual.permute(0, 2, 1).contiguous() if pred_residual is not None else None
            anchor_points, stride_tensor = make_anchors(preds["feats"], loss.stride, 0.5)
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=args.torch_device, dtype=pred_scores.dtype) * loss.stride[0]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = loss.preprocess(targets, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            pred_bboxes = loss.bbox_decode(anchor_points, pred_distri, pred_residual, stride_tensor)
            pred_pixel = pred_bboxes.detach() * stride_tensor
            _, _, target_scores, fg_mask, target_gt_idx = loss.assigner(
                pred_scores.detach().sigmoid(), pred_pixel.type(gt_bboxes.dtype), anchor_points * stride_tensor,
                gt_labels, gt_bboxes, mask_gt,
            )
        probs = pred_scores[0].sigmoid()
        levels = level_slices(preds["feats"])
        has_gt = bool(len(boxes))
        for level, start, end in levels:
            level_fg = fg_mask[0, start:end].bool()
            p_neg = probs[start:end].amax(dim=1)[~level_fg]
            pos_mask = level_fg
            g_pos = 0.0
            if pos_mask.any():
                pos_indices = torch.arange(start, end, device=args.torch_device)[pos_mask]
                for anchor_index in pos_indices.tolist():
                    gt_index = int(target_gt_idx[0, anchor_index])
                    gt_class = int(cls[gt_index])
                    p = float(probs[anchor_index, gt_class])
                    q = float(target_scores[0, anchor_index, gt_class])
                    u = float(box_iou(pred_pixel[0, anchor_index].view(1, 4), gt_bboxes[0, gt_index].view(1, 4))[0, 0])
                    g_pos += abs(p - q)
                    positive_rows.append({
                        "image": image_path.name, "image_has_gt": has_gt, "gt_index": gt_index,
                        "class": gt_class, "feature_level": level, "area_px2": areas[gt_index],
                        "size_group": size_group(areas[gt_index]), "iou": u,
                        "geometry_group": geometry_group(u), "p": p, "u": u, "q": q, "g_cls": p - q,
                    })
            image_level_rows.append({
                "image": image_path.name, "image_has_gt": has_gt, "feature_level": level,
                "positive_candidate_count": int(pos_mask.sum()), "negative_candidate_count": int((~level_fg).sum()),
                "g_pos": g_pos, "g_neg": float(p_neg.sum()),
            })
        if image_index % 50 == 0:
            print(f"{image_index}/{len(images)} images, {len(positive_rows)} positive candidates", flush=True)
    del net, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return positive_rows, image_level_rows


def aggregate_positive(rows: list[dict[str, object]], size: str, geometry: str) -> dict[str, object]:
    selected = [r for r in rows if (size == "all" or r["size_group"] == size) and (geometry == "all" or r["geometry_group"] == geometry)]
    def mean(key: str) -> float | None:
        values = [float(r[key]) for r in selected]
        return float(np.mean(values)) if values else None
    return {"count": len(selected), "mean_p": mean("p"), "mean_q": mean("q"), "mean_q_minus_p": (mean("q") - mean("p")) if selected else None, "q_lt_050_fraction": (sum(float(r["q"]) < 0.5 for r in selected) / len(selected)) if selected else None}


def aggregate_negative(rows: list[dict[str, object]], level: str, cohort: str) -> dict[str, object]:
    selected = [r for r in rows if (level == "all" or r["feature_level"] == level) and (cohort == "all" or (cohort == "with_gt") == bool(r["image_has_gt"]))]
    g_pos, g_neg = sum(float(r["g_pos"]) for r in selected), sum(float(r["g_neg"]) for r in selected)
    return {"images": len(selected), "g_pos": g_pos, "g_neg": g_neg, "g_neg_over_g_pos": (g_neg / g_pos) if g_pos > 0 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.torch_device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    images = sorted(p for p in (args.dataset_root / "levir_ship_yolo_seed42/images/test").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        images = images[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    positive, image_level = inspect(args.checkpoint, images, args)
    with (args.output / "positive_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(positive[0]) if positive else [])
        writer.writeheader(); writer.writerows(positive)
    with (args.output / "image_level_gradient_mass.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_level[0]) if image_level else [])
        writer.writeheader(); writer.writerows(image_level)
    result = {"checkpoint": str(args.checkpoint), "images": len(images), "positive_candidates": len(positive),
              "positive_by_size_geometry": {size: {geometry: aggregate_positive(positive, size, geometry) for geometry in (*GEOMETRY_GROUPS, "all")} for size in (*SIZE_GROUPS, "all")},
              "negative_gradient_mass": {cohort: {level: aggregate_negative(image_level, level, cohort) for level in (*[x[0] for x in level_slices([torch.empty(1,1,1,1)])], "all")} for cohort in ("with_gt", "empty", "all")}}
    # The level names are determined from the model output, so rebuild this section from CSV rows.
    levels = sorted({str(r["feature_level"]) for r in image_level}, key=lambda x: int(x[1:]))
    result["negative_gradient_mass"] = {cohort: {level: aggregate_negative(image_level, level, cohort) for level in (*levels, "all")} for cohort in ("with_gt", "empty", "all")}
    (args.output / "classification_supervision_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
