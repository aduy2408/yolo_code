#!/usr/bin/env python3
"""Center-ring diagnostic probes split by miss mechanism cohorts."""

from __future__ import annotations

import argparse
import json
import math
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
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402


CHECKPOINTS = {
    "Plain": Path("/marimo/yolo_code/runs/checkpoints_4way/train/yolov8n_p2_baseline_seed42/weights/best.pt"),
    "Plain+FTAL": Path("/marimo/yolo_code/runs/checkpoints_4way/runs/plain_p2_factorized_k15/seed_42/weights/best.pt"),
    "GAP": Path("/marimo/yolo_code/runs/checkpoints_4way/runs/gap/seed_42/weights/best.pt"),
    "GAP+FTAL": Path("/marimo/yolo_code/runs/checkpoints_4way/runs/gap_factorized_k15/seed_42/weights/best.pt"),
}
COHORTS = ("A_invisible", "B_poor_localization", "C_low_conf_localized")


def read_labels(image: Path) -> torch.Tensor:
    label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
    if not rows:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor([[float(value) for value in row[1:5]] for row in rows], dtype=torch.float32)


def load_samples(dataset_root: Path) -> list[dict]:
    direct = dataset_root / "levir_ship_yolo_seed42/images/test"
    if direct.exists():
        return [{"image": p, "boxes": read_labels(p)} for p in sorted(direct.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    nested = dataset_root / "levir_ship_yolo_seed42/levir_ship_yolo_seed42"
    cache = nested / "labels/test.cache"
    if cache.exists():
        cache_obj = np.load(cache, allow_pickle=True).item()
        image_roots = [nested / "images/test", ROOT / "LevirShipData/All Images", Path("/marimo/LevirShipData/All Images")]
        samples = []
        for item in cache_obj["labels"]:
            name = Path(item["im_file"]).name
            image = next((root / name for root in image_roots if (root / name).exists()), None)
            if image is not None:
                samples.append({"image": image, "boxes": torch.as_tensor(item["bboxes"], dtype=torch.float32).reshape(-1, 4)})
        return samples

    raise FileNotFoundError(f"Could not find LEVIR test images/cache under {dataset_root}")


def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a) == 0 or len(b) == 0:
        return torch.zeros((len(a), len(b)), device=a.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-8)


def greedy_matched_gt(gt: torch.Tensor, pred: torch.Tensor, iou_thr: float = 0.5) -> set[int]:
    mat = iou_matrix(gt, pred)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    if mat.numel() == 0:
        return matched_gt
    for flat_idx in torch.argsort(mat.flatten(), descending=True):
        gi, pi = divmod(int(flat_idx), mat.shape[1])
        if mat[gi, pi] < iou_thr:
            break
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
    return matched_gt


def ring_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    c, _, _ = x.shape
    size = 2 * radius + 1
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist = ((xx - radius) ** 2 + (yy - radius) ** 2).float().sqrt()
    mask = (dist > 1.0) & (dist <= radius)
    kernel = (mask.float() / mask.sum()).view(1, 1, size, size).repeat(c, 1, 1, 1).to(x.device)
    return F.conv2d(F.pad(x.unsqueeze(0), (radius, radius, radius, radius), mode="replicate"), kernel, groups=c).squeeze(0)


def random_bg_vector(ring: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    bg = torch.nonzero(~gt_mask, as_tuple=False)
    if len(bg) == 0:
        return ring.mean(dim=(1, 2), keepdim=True)
    y, x = bg[random.randrange(len(bg))]
    return ring[:, y, x].view(-1, 1, 1)


def evaluate_probe(x: torch.Tensor, y: torch.Tensor, epochs: int) -> float:
    model = nn.Linear(x.shape[1], 1).to(x.device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    pos_weight = (y == 0).sum().float() / (y == 1).sum().float().clamp_min(1.0)
    opt = optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x).squeeze(1), y.float()).backward()
        opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(model(x).squeeze(1))
        order = torch.argsort(probs, descending=True)
        sorted_y = y[order]
        tp = sorted_y.cumsum(0)
        fp = (1 - sorted_y).cumsum(0)
        recall = tp / sorted_y.sum().clamp_min(1)
        precision = tp / (tp + fp).clamp_min(1)
        return float(((recall[1:] - recall[:-1]) * precision[1:]).sum().item())


def scan_cohorts(wrapper: YOLO, samples: list[dict], device: str, imgsz: int, final_conf: float) -> list[dict]:
    misses = []
    for sample in samples:
        image = sample["image"]
        original = cv2.imread(str(image))
        if original is None:
            continue
        h, w = original.shape[:2]
        boxes = sample["boxes"]
        if len(boxes) == 0:
            continue
        gt = xywh2xyxy(boxes)
        gt[:, [0, 2]] *= w
        gt[:, [1, 3]] *= h

        # Candidate pool is low-confidence NMS output; NMS IoU is explicit by repo protocol.
        low = wrapper.predict(image, conf=0.001, iou=0.5, imgsz=imgsz, device=device, verbose=False)[0].boxes
        final = wrapper.predict(image, conf=final_conf, iou=0.5, imgsz=imgsz, device=device, verbose=False)[0].boxes
        low_boxes = low.xyxy.detach().cpu()
        final_boxes = final.xyxy.detach().cpu()
        best_iou = iou_matrix(gt, low_boxes).max(dim=1).values if len(low_boxes) else torch.zeros(len(gt))
        matched = greedy_matched_gt(gt, final_boxes, 0.5)

        for gi, iou in enumerate(best_iou.tolist()):
            if gi in matched:
                continue
            if iou < 0.1:
                cohort = "A_invisible"
            elif iou < 0.5:
                cohort = "B_poor_localization"
            else:
                cohort = "C_low_conf_localized"
            g = gt[gi]
            area = float((g[2] - g[0]) * (g[3] - g[1]))
            side = math.sqrt(area)
            misses.append({"image": image, "gt": g, "gt_idx": gi, "best_iou": iou, "cohort": cohort, "side": side})
    return misses


def gt_mask_for(gt: torch.Tensor, fmap_shape: tuple[int, int], image_shape: tuple[int, int], device: str) -> torch.Tensor:
    hp, wp = fmap_shape
    h, w = image_shape
    x1, y1 = int(gt[0] * wp / w), int(gt[1] * hp / h)
    x2, y2 = max(int(gt[2] * wp / w), x1 + 1), max(int(gt[3] * hp / h), y1 + 1)
    mask = torch.zeros((hp, wp), dtype=torch.bool, device=device)
    mask[max(y1, 0):min(y2, hp), max(x1, 0):min(x2, wp)] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--final-conf", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--radii", default="5")
    parser.add_argument("--max-per-cohort", type=int, default=0)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    samples = load_samples(args.dataset_root)
    radii = [int(r) for r in args.radii.split(",") if r.strip()]

    ref = YOLO(CHECKPOINTS["GAP+FTAL"])
    misses = scan_cohorts(ref, samples, device, args.imgsz, args.final_conf)
    if args.max_per_cohort:
        kept = []
        for cohort in COHORTS:
            kept.extend([m for m in misses if m["cohort"] == cohort][: args.max_per_cohort])
        misses = kept

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
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
        for miss in misses:
            original = cv2.imread(str(miss["image"]))
            h, w = original.shape[:2]
            image = letterbox(image=original)
            tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
            with torch.no_grad():
                net(tensor)
            p2 = acts["p2"]
            c, hp, wp = p2.shape
            mask = gt_mask_for(miss["gt"].to(device), (hp, wp), (h, w), device)
            y = mask.flatten().long()
            for radius in radii:
                ring = ring_mean(p2, radius)
                rand = random_bg_vector(ring, mask)
                probes = {
                    "F": p2,
                    "Ring": ring,
                    "[F,Ring]": torch.cat([p2, ring], dim=0),
                    "F-Ring": p2 - ring,
                    "F-RandomRing": p2 - rand,
                }
                for name, feat in probes.items():
                    x = feat.permute(1, 2, 0).reshape(-1, feat.shape[0])
                    rows.append({
                        "cohort": miss["cohort"],
                        "probe": name,
                        "radius": radius,
                        "ap": evaluate_probe(x, y, args.epochs),
                    })
        handle.remove()

        by_key: dict[str, dict] = {}
        for cohort in COHORTS:
            for radius in radii:
                for probe in ("F", "Ring", "[F,Ring]", "F-Ring", "F-RandomRing"):
                    vals = [r["ap"] for r in rows if r["cohort"] == cohort and r["radius"] == radius and r["probe"] == probe]
                    if vals:
                        by_key[f"{cohort}|R{radius}|{probe}"] = {
                            "mean_ap": float(np.mean(vals)),
                            "count": len(vals),
                            "rescue_rate_ap_gt_0_5": float(np.mean([v > 0.5 for v in vals])),
                        }
        summary[variant] = by_key

    cohort_counts = {c: sum(m["cohort"] == c for m in misses) for c in COHORTS}
    output = {
        "protocol": {
            "split": "test",
            "nms_iou": 0.5,
            "candidate_conf": 0.001,
            "final_conf": args.final_conf,
            "imgsz": args.imgsz,
            "candidate_pool": "low_conf_nms_predictions",
            "reference_detector": "GAP+FTAL",
            "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
        },
            "image_count": len(samples),
        "miss_count": len(misses),
        "cohort_counts": cohort_counts,
        "summary": summary,
    }
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_center_ring_cohorts_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
