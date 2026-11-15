#!/usr/bin/env python3
"""Measure per-GT TAL target mass and effective support for Factorized TAL variants."""

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


VARIANTS = {
    "gap_baseline": None,
    "gap_factorized_ceiling": (0.75, 1.0, 0.5),
    "gap_factorized_separation": (1.0, 1.5, 0.5),
    "gap_factorized_k15": (0.75, 1.5, 0.5),
    "gap_factorized_k20": (0.75, 2.0, 0.5),
}


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def gt_group(area: float) -> str:
    return "tiny" if area < 100 else "small" if area <= 400 else "large"


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


def transform_scores(q: torch.Tensor, gt_bboxes: torch.Tensor, gt_idx: int, setting: tuple[float, float, float] | None) -> torch.Tensor:
    if setting is None or not len(q):
        return q
    tau, kappa, lam = setting
    box = gt_bboxes[gt_idx]
    if (box[2:] - box[:2]).clamp_min(1e-6).prod().sqrt() >= 32.0:
        return q
    u_max = q.max().clamp_min(1e-12)
    q_new = u_max.pow(tau) * (q / u_max).clamp(0, 1).pow(kappa)
    return torch.where(q > 0, q + lam * (q_new - q), q)


def support_metrics(q: torch.Tensor) -> dict[str, float | int]:
    q = q[q > 0].float()
    if not len(q):
        return {"support_count": 0, "target_mass": 0.0, "n_eff": 0.0, "top1_over_mass": 0.0, "target_entropy": 0.0}
    mass = q.sum()
    p = q / mass.clamp_min(1e-12)
    return {
        "support_count": int(len(q)),
        "target_mass": float(mass),
        "n_eff": float(mass.square() / q.square().sum().clamp_min(1e-12)),
        "top1_over_mass": float(q.max() / mass.clamp_min(1e-12)),
        "target_entropy": float(-(p * p.clamp_min(1e-12).log()).sum()),
    }


def inspect_checkpoint(name: str, checkpoint: Path, images: list[Path], args: argparse.Namespace) -> list[dict[str, object]]:
    wrapper = YOLO(checkpoint)
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    expected_seed = getattr(args, "expected_seed", None)
    if expected_seed is not None and train_args.get("seed") != expected_seed:
        raise RuntimeError(f"{name}: expected checkpoint seed {expected_seed}, got {train_args.get('seed')!r}")
    net = wrapper.model.to(args.torch_device).eval()
    head = net.model[-1]
    strides = [float(value) for value in head.stride]
    if not strides or strides[0] != 4.0:
        raise RuntimeError(f"{name}: expected P2 stride first, got {strides}")

    loss = v8DetectionLoss(net)
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    rows = []
    for image_index, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"could not read image: {image_path}")
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).to(args.torch_device).permute(2, 0, 1).float()[None] / 255
        cls, boxes, areas = read_labels(image_path)
        batch = {
            "batch_idx": torch.zeros((len(boxes),), dtype=torch.long, device=args.torch_device),
            "cls": cls.to(args.torch_device),
            "bboxes": boxes.to(args.torch_device),
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
            _, _, target_scores, fg_mask, target_gt_idx = loss.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
        n_p2 = math.prod(preds["feats"][0].shape[2:])
        setting = VARIANTS[name]
        for gt_idx, area in enumerate(areas):
            group = fg_mask[0, :n_p2].bool() & (target_gt_idx[0, :n_p2] == gt_idx)
            q = transform_scores(target_scores[0, :n_p2][group].sum(-1), gt_bboxes[0], gt_idx, setting)
            rows.append(
                {
                    "variant": name,
                    "image": image_path.name,
                    "gt_index": gt_idx,
                    "area_px2": area,
                    "size_group": gt_group(area),
                    **support_metrics(q),
                }
            )
        if image_index % 100 == 0:
            print(f"{name}: {image_index}/{len(images)} images, {len(rows)} GT", flush=True)
    del net, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        out[variant] = {}
        variant_rows = [row for row in rows if row["variant"] == variant]
        for group in ("all", "tiny", "small", "large"):
            selected = variant_rows if group == "all" else [row for row in variant_rows if row["size_group"] == group]
            record: dict[str, float | int | None] = {"gt": len(selected)}
            for metric in ("support_count", "target_mass", "n_eff", "top1_over_mass", "target_entropy"):
                values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
                record[f"{metric}_mean"] = float(values.mean()) if len(values) else None
                record[f"{metric}_median"] = float(np.median(values)) if len(values) else None
            out[variant][group] = record
    return out


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name!r}")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/levir_yolov8n_p2_gap_factorized_tal/target_support")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", action="append", required=True, help="variant=/path/to/best.pt")
    args = parser.parse_args()
    args.torch_device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    checkpoints = parse_checkpoints(args.checkpoint)
    missing = sorted(set(VARIANTS) - set(checkpoints))
    if missing:
        raise ValueError(f"missing checkpoints: {missing}")
    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/test"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        images = images[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, checkpoint in checkpoints.items():
        rows.extend(inspect_checkpoint(name, checkpoint, images, args))
    fields = ["variant", "image", "gt_index", "area_px2", "size_group", "support_count", "target_mass", "n_eff", "top1_over_mass", "target_entropy"]
    with (args.output / "target_support_per_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "target_support_summary.json").write_text(json.dumps(summarize(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
