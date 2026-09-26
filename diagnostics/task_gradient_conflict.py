#!/usr/bin/env python3
"""No-training per-GT localization-gradient probe on shared detector features."""
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
        reg_loss = losses[0] + losses[2]
        g_reg = torch.autograd.grad(reg_loss, features, retain_graph=False, allow_unused=True)[0]
        if g_reg is None:
            continue
        vector = g_reg.detach().float().mean(dim=(2, 3)).flatten()
        norm = vector.norm()
        if not torch.isfinite(norm) or norm.item() <= 1e-12:
            continue
        results.append({
            "bucket": bucket,
            "gt_index": selected_index,
            "num_gt": 1,
            "sqrt_area": batch["_rows"][selected_index]["sqrt_area"],
            "reg_loss": float(reg_loss.detach().item()),
            "grad_norm": float(norm.item()),
            "grad_vector": vector.cpu().numpy(),
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
    summary = {
        "weights": str(args.weights),
        "images": str(args.images),
        "limit": args.limit,
        "loss": "box + DFL",
        "vector": "mean over P2 spatial dimensions, raw norm preserved",
        "rows": len(rows),
        "buckets": {},
    }
    for bucket, _, _ in BUCKETS:
        values = [row for row in rows if row["bucket"] == bucket]
        vectors = [row["grad_vector"] for row in values]
        norms = np.asarray([row["grad_norm"] for row in values], dtype=np.float64)
        pair_cosines = []
        for index, vector in enumerate(vectors):
            for other in vectors[index + 1:]:
                pair_cosines.append(float(np.dot(vector, other) / max(np.linalg.norm(vector) * np.linalg.norm(other), 1e-12)))
        mean_vector = np.mean(vectors, axis=0) if vectors else None
        raw_top1_energy = None
        centered_top1_energy = None
        if len(vectors) >= 2:
            matrix = np.asarray(vectors, dtype=np.float64)
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            raw_energy = singular_values**2
            raw_top1_energy = float(raw_energy[0] / max(raw_energy.sum(), 1e-12))
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            centered_values = np.linalg.svd(centered, compute_uv=False)
            centered_energy = centered_values**2
            centered_top1_energy = float(centered_energy[0] / max(centered_energy.sum(), 1e-12))
        summary["buckets"][bucket] = {
            "n": len(values),
            "grad_norm_mean": float(np.mean(norms)) if len(values) else None,
            "grad_norm_median": float(np.median(norms)) if len(values) else None,
            "grad_norm_q25": float(np.quantile(norms, 0.25)) if len(values) else None,
            "grad_norm_q75": float(np.quantile(norms, 0.75)) if len(values) else None,
            "pair_count": len(pair_cosines),
            "pair_cosine_mean": float(np.mean(pair_cosines)) if pair_cosines else None,
            "pair_cosine_median": float(np.median(pair_cosines)) if pair_cosines else None,
            "pair_cosine_q25": float(np.quantile(pair_cosines, 0.25)) if pair_cosines else None,
            "pair_cosine_q75": float(np.quantile(pair_cosines, 0.75)) if pair_cosines else None,
            "negative_pair_rate": float(np.mean(np.asarray(pair_cosines) < 0)) if pair_cosines else None,
            "aggregation_ratio_raw": float(np.linalg.norm(mean_vector) / max(float(np.mean(norms)), 1e-12)) if values else None,
            "svd_top1_energy_raw": raw_top1_energy,
            "svd_top1_energy_centered": centered_top1_energy,
        }
        if bucket == "tiny" and len(vectors) >= 8:
            matrix = np.asarray(vectors, dtype=np.float64)
            normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            summary["gradient_clusters"] = {}
            for cluster_count in (2, 3, 4, 8):
                model = KMeans(n_clusters=cluster_count, random_state=args.seed, n_init=20)
                labels = model.fit_predict(normalized)
                summary["gradient_clusters"][f"k{cluster_count}"] = {
                    "n": len(vectors),
                    "silhouette_cosine": float(silhouette_score(normalized, labels, metric="cosine")),
                    "cluster_sizes": [int(size) for size in np.bincount(labels, minlength=cluster_count)],
                }
    for row in rows:
        del row["grad_vector"]
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
