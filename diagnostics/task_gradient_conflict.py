#!/usr/bin/env python3
"""No-training cls-vs-reg gradient conflict probe on shared detector features."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ULTRALYTICS = ROOT / "models_related/ultralytics"
if str(ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS))

BUCKETS = (
    ("tiny", 0.0, 32.0),
    ("small", 32.0, 64.0),
    ("medium", 64.0, 128.0),
    ("large", 128.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_sample(image_path: Path, label_path: Path, device: torch.device) -> tuple[torch.Tensor, dict, list[dict]]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    height, width = image.shape[:2]
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    padded_height, padded_width = height + pad_h, width + pad_w
    rows = []
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, values[:5])
            rows.append({
                "cls": int(cls),
                "cx": cx * width / padded_width,
                "cy": cy * height / padded_height,
                "w": bw * width / padded_width,
                "h": bh * height / padded_height,
                "sqrt_area": float(np.sqrt(bw * width * bh * height)),
            })
    if rows:
        bboxes = torch.tensor([[r["cx"], r["cy"], r["w"], r["h"]] for r in rows], device=device, dtype=torch.float32)
        classes = torch.tensor([[r["cls"]] for r in rows], device=device, dtype=torch.float32)
        batch_idx = torch.zeros((len(rows),), device=device, dtype=torch.long)
    else:
        bboxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
        classes = torch.zeros((0, 1), device=device, dtype=torch.float32)
        batch_idx = torch.zeros((0,), device=device, dtype=torch.long)
    batch = {"img": tensor, "bboxes": bboxes, "cls": classes, "batch_idx": batch_idx}
    return tensor, batch, rows


def bucket_for(area: float) -> str:
    for name, low, high in BUCKETS:
        if low <= area < high:
            return name
    raise AssertionError(area)


def probe_one(net, criterion, tensor, batch, bucket: str) -> list[dict]:
    selected = [i for i, row in enumerate(batch["_rows"]) if bucket_for(row["sqrt_area"]) == bucket]
    if not selected:
        return []
    results = []
    for selected_index in selected:
        keep = torch.zeros((len(batch["_rows"]),), device=tensor.device, dtype=torch.bool)
        keep[selected_index] = True
        work = {key: value for key, value in batch.items() if not key.startswith("_")}
        work["bboxes"] = work["bboxes"].clone()
        work["bboxes"][~keep] = 0.0
        work["cls"] = work["cls"].clone()
        work["cls"][~keep] = 0.0

        net.zero_grad(set_to_none=True)
        preds = net.predict(tensor)
        if isinstance(preds, tuple):
            preds = preds[1]
        features = preds["feats"][0]
        features.retain_grad()
        _, losses, _ = criterion.get_assigned_targets_and_loss(preds, work)
        cls_loss = losses[1]
        reg_loss = losses[0] + losses[2]
        g_cls = torch.autograd.grad(cls_loss, features, retain_graph=True, allow_unused=True)[0]
        g_reg = torch.autograd.grad(reg_loss, features, retain_graph=False, allow_unused=True)[0]
        if g_cls is None or g_reg is None:
            continue
        a = g_cls.detach().float().reshape(-1)
        b = g_reg.detach().float().reshape(-1)
        cosine = torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)
        results.append({
            "bucket": bucket,
            "gt_index": selected_index,
            "num_gt": 1,
            "sqrt_area": batch["_rows"][selected_index]["sqrt_area"],
            "cls_loss": float(cls_loss.detach().item()),
            "reg_loss": float(reg_loss.detach().item()),
            "cls_norm": float(a.norm().item()),
            "reg_norm": float(b.norm().item()),
            "dot": float(torch.dot(a, b).item()),
            "cosine": float(cosine.item()),
            "conflict": bool(cosine.item() < 0.0),
        })
    return results


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    from ultralytics import YOLO

    loaded = YOLO(str(args.weights))
    net = loaded.model.to(device)
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)
    for name, value in (("box", 7.5), ("cls", 0.5), ("dfl", 1.5)):
        if not hasattr(net.args, name):
            setattr(net.args, name, value)
    net.train()
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    for module in net.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    criterion = net.init_criterion()
    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.limit]
    rows = []
    with torch.enable_grad():
        for image_path in image_paths:
            tensor, batch, objects = read_sample(image_path, args.labels / f"{image_path.stem}.txt", device)
            batch["_rows"] = objects
            for bucket, _, _ in BUCKETS:
                results = probe_one(net, criterion, tensor, batch, bucket)
                for result in results:
                    result["image"] = image_path.name
                    rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["image", "bucket"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"weights": str(args.weights), "images": str(args.images), "limit": args.limit, "rows": len(rows), "buckets": {}}
    for bucket, _, _ in BUCKETS:
        values = [row for row in rows if row["bucket"] == bucket]
        cosines = np.asarray([row["cosine"] for row in values], dtype=np.float64)
        summary["buckets"][bucket] = {
            "n": len(values),
            "conflict_rate": float(np.mean(cosines < 0)) if len(values) else None,
            "cosine_mean": float(np.mean(cosines)) if len(values) else None,
            "cosine_median": float(np.median(cosines)) if len(values) else None,
            "cosine_q25": float(np.quantile(cosines, 0.25)) if len(values) else None,
            "cosine_q75": float(np.quantile(cosines, 0.75)) if len(values) else None,
        }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
