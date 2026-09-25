#!/usr/bin/env python3
"""No-training proxies for scale-feature specialization and tiny-reference influence."""
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
from diagnostics.inter_instance_gradient_coherence import BUCKETS, bucket


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--reference-images", type=Path, required=True)
    p.add_argument("--reference-labels", type=Path, required=True)
    p.add_argument("--train-images", type=Path, required=True)
    p.add_argument("--train-labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference-limit", type=int, default=64)
    p.add_argument("--train-limit", type=int, default=64)
    return p.parse_args()


def load_sample(image_path, label_path, device):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    height, width = image.shape[:2]
    pad_h, pad_w = (32 - height % 32) % 32, (32 - width % 32) % 32
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    objects = []
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, values[:5])
            objects.append({"cls": int(cls), "cx": cx * width / padded_w, "cy": cy * height / padded_h, "w": bw * width / padded_w, "h": bh * height / padded_h, "sqrt_area": float(np.sqrt(bw * width * bh * height))})
    if objects:
        bboxes = torch.tensor([[o["cx"], o["cy"], o["w"], o["h"]] for o in objects], device=device)
        classes = torch.tensor([[o["cls"]] for o in objects], device=device)
        batch_idx = torch.zeros(len(objects), device=device, dtype=torch.long)
    else:
        bboxes = torch.zeros((0, 4), device=device)
        classes = torch.zeros((0, 1), device=device)
        batch_idx = torch.zeros(0, device=device, dtype=torch.long)
    return tensor, {"img": tensor, "bboxes": bboxes, "cls": classes, "batch_idx": batch_idx}, objects


def setup(weights, device):
    from ultralytics import YOLO
    net = YOLO(str(weights)).model.to(device)
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
    return net, net.init_criterion()


def gradient_for_gt(net, criterion, tensor, base_batch, objects, index):
    batch = {key: value.clone() for key, value in base_batch.items()}
    keep = torch.zeros(len(objects), device=tensor.device, dtype=torch.bool)
    keep[index] = True
    batch["bboxes"][~keep] = 0.0
    batch["cls"][~keep] = 0.0
    net.zero_grad(set_to_none=True)
    preds = net.predict(tensor)
    if isinstance(preds, tuple):
        preds = preds[1]
    p2 = preds["feats"][0]
    _, losses, _ = criterion.get_assigned_targets_and_loss(preds, batch)
    grad = torch.autograd.grad(losses[0] + losses[1] + losses[2], p2, allow_unused=True)[0]
    if grad is None:
        return None
    vector = grad.detach().float().mean(dim=(2, 3)).flatten()
    return vector / vector.norm().clamp_min(1e-12)


def gradient_for_image(net, criterion, tensor, batch):
    net.zero_grad(set_to_none=True)
    preds = net.predict(tensor)
    if isinstance(preds, tuple):
        preds = preds[1]
    p2 = preds["feats"][0]
    _, losses, _ = criterion.get_assigned_targets_and_loss(preds, batch)
    grad = torch.autograd.grad(losses[0] + losses[1] + losses[2], p2, allow_unused=True)[0]
    if grad is None:
        return None
    vector = grad.detach().float().mean(dim=(2, 3)).flatten()
    return vector / vector.norm().clamp_min(1e-12)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75))}


def main():
    cfg = parse_args()
    device = torch.device(cfg.device)
    net, criterion = setup(cfg.weights, device)
    ref_paths = sorted(p for p in cfg.reference_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: cfg.reference_limit]
    train_paths = sorted(p for p in cfg.train_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: cfg.train_limit]
    ref_items = []
    feature_rows = []
    with torch.enable_grad():
        for path in ref_paths:
            tensor, batch, objects = load_sample(path, cfg.reference_labels / f"{path.stem}.txt", device)
            for index, obj in enumerate(objects):
                vector = gradient_for_gt(net, criterion, tensor, batch, objects, index)
                if vector is not None:
                    ref_items.append({"bucket": bucket(obj["sqrt_area"]), "vector": vector.cpu().numpy(), "sqrt_area": obj["sqrt_area"]})
    tiny_vectors = [item["vector"] for item in ref_items if item["bucket"] == "tiny"]
    if not tiny_vectors:
        raise RuntimeError("No tiny reference gradients were collected")
    tiny_reference = np.mean(np.stack(tiny_vectors), axis=0)
    tiny_reference /= max(np.linalg.norm(tiny_reference), 1e-12)
    for name, _, _ in BUCKETS:
        items = [item for item in ref_items if item["bucket"] == name]
        if not items:
            continue
        matrix = np.stack([item["vector"] for item in items])
        mean_vector = matrix.mean(axis=0)
        abs_mean = np.abs(mean_vector)
        top_k = max(1, len(abs_mean) // 10)
        feature_rows.append({"bucket": name, "n": len(items), "mean_vector_norm": float(np.linalg.norm(mean_vector)), "top10_channel_share": float(np.sort(abs_mean)[-top_k:].sum() / max(abs_mean.sum(), 1e-12)), "cosine_to_tiny_mean": float(np.dot(mean_vector, tiny_reference) / max(np.linalg.norm(mean_vector), 1e-12))})
    influence_rows = []
    with torch.enable_grad():
        for path in train_paths:
            tensor, batch, objects = load_sample(path, cfg.train_labels / f"{path.stem}.txt", device)
            if not objects:
                continue
            vector = gradient_for_image(net, criterion, tensor, batch)
            if vector is None:
                continue
            tiny_count = sum(bucket(obj["sqrt_area"]) == "tiny" for obj in objects)
            score = float(np.dot(vector.cpu().numpy(), tiny_reference))
            influence_rows.append({"image": path.name, "num_gt": len(objects), "tiny_count": tiny_count, "tiny_fraction": tiny_count / len(objects), "tiny_reference_cosine": score})
    summary = {"weights": str(cfg.weights), "reference_images": len(ref_paths), "train_images": len(influence_rows), "reference_gt": len(ref_items), "feature_proxy": feature_rows, "influence": {"all": summarize([r["tiny_reference_cosine"] for r in influence_rows]), "tiny_containing": summarize([r["tiny_reference_cosine"] for r in influence_rows if r["tiny_count"] > 0]), "no_tiny": summarize([r["tiny_reference_cosine"] for r in influence_rows if r["tiny_count"] == 0])}}
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(influence_rows[0]) if influence_rows else ["image"])
        writer.writeheader()
        writer.writerows(influence_rows)
    cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
