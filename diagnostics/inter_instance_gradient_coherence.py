#!/usr/bin/env python3
"""No-training inter-instance P2 gradient coherence probe."""
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

BUCKETS = (("tiny", 0.0, 32.0), ("small", 32.0, 64.0), ("medium", 64.0, 128.0), ("large", 128.0, float("inf")))


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=128)
    return p.parse_args()


def bucket(area: float) -> str:
    for name, low, high in BUCKETS:
        if low <= area < high:
            return name
    raise AssertionError(area)


def load_sample(image_path: Path, label_path: Path, device: torch.device):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    height, width = image.shape[:2]
    pad_h, pad_w = (32 - height % 32) % 32, (32 - width % 32) % 32
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    if pad_h or pad_w:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    objects = []
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, values[:5])
            objects.append({
                "cls": int(cls),
                "cx": cx * width / padded_w,
                "cy": cy * height / padded_h,
                "w": bw * width / padded_w,
                "h": bh * height / padded_h,
                "sqrt_area": float(np.sqrt(bw * width * bh * height)),
            })
    if objects:
        bboxes = torch.tensor([[o["cx"], o["cy"], o["w"], o["h"]] for o in objects], device=device)
        classes = torch.tensor([[o["cls"]] for o in objects], device=device)
        batch_idx = torch.zeros(len(objects), device=device, dtype=torch.long)
    else:
        bboxes = torch.zeros((0, 4), device=device)
        classes = torch.zeros((0, 1), device=device)
        batch_idx = torch.zeros(0, device=device, dtype=torch.long)
    return image_tensor, {"img": image_tensor, "bboxes": bboxes, "cls": classes, "batch_idx": batch_idx}, objects


def main() -> None:
    cfg = args()
    device = torch.device(cfg.device)
    from ultralytics import YOLO

    loaded = YOLO(str(cfg.weights))
    net = loaded.model.to(device)
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)
    for name, value in (("box", 7.5), ("cls", 0.5), ("dfl", 1.5)):
        if not hasattr(net.args, name):
            setattr(net.args, name, value)
    net.train()
    for p in net.parameters():
        p.requires_grad_(True)
    for module in net.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    criterion = net.init_criterion()
    image_paths = sorted(p for p in cfg.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: cfg.limit]
    vectors = []
    with torch.enable_grad():
        for image_path in image_paths:
            tensor, base_batch, objects = load_sample(image_path, cfg.labels / f"{image_path.stem}.txt", device)
            for index, obj in enumerate(objects):
                batch = {key: value.clone() if torch.is_tensor(value) else value for key, value in base_batch.items()}
                keep = torch.zeros(len(objects), device=device, dtype=torch.bool)
                keep[index] = True
                batch["bboxes"][~keep] = 0.0
                batch["cls"][~keep] = 0.0
                net.zero_grad(set_to_none=True)
                preds = net.predict(tensor)
                if isinstance(preds, tuple):
                    preds = preds[1]
                p2 = preds["feats"][0]
                p2.retain_grad()
                _, losses, _ = criterion.get_assigned_targets_and_loss(preds, batch)
                total = losses[0] + losses[1] + losses[2]
                grad = torch.autograd.grad(total, p2, allow_unused=True)[0]
                if grad is None:
                    continue
                vector = grad.detach().float().mean(dim=(2, 3)).flatten()
                norm = vector.norm().clamp_min(1e-12)
                vector = vector / norm
                vectors.append({"image": image_path.name, "gt_index": index, "bucket": bucket(obj["sqrt_area"]), "sqrt_area": obj["sqrt_area"], "vector": vector.cpu().numpy()})
    csv_rows = []
    summary = {"weights": str(cfg.weights), "images": str(cfg.images), "limit": cfg.limit, "rows": len(vectors), "buckets": {}}
    for name, _, _ in BUCKETS:
        items = [item for item in vectors if item["bucket"] == name]
        matrix = np.stack([item["vector"] for item in items]) if items else np.zeros((0, 1))
        if len(items) >= 2:
            similarities = matrix @ matrix.T
            upper = similarities[np.triu_indices(len(items), k=1)]
            aggregation_ratio = float(np.linalg.norm(matrix.sum(axis=0)) / len(items))
            summary["buckets"][name] = {"n": len(items), "pair_count": len(upper), "pair_cosine_mean": float(upper.mean()), "pair_cosine_median": float(np.median(upper)), "pair_cosine_q25": float(np.quantile(upper, .25)), "pair_cosine_q75": float(np.quantile(upper, .75)), "negative_pair_rate": float(np.mean(upper < 0)), "aggregation_ratio": aggregation_ratio}
            for value in upper:
                csv_rows.append({"bucket": name, "pair_cosine": float(value)})
        else:
            summary["buckets"][name] = {"n": len(items), "pair_count": 0, "pair_cosine_mean": None, "pair_cosine_median": None, "pair_cosine_q25": None, "pair_cosine_q75": None, "negative_pair_rate": None, "aggregation_ratio": None}
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bucket", "pair_cosine"])
        writer.writeheader()
        writer.writerows(csv_rows)
    cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
