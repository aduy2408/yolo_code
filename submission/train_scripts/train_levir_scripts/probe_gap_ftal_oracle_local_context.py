#!/usr/bin/env python3
"""Oracle local-context probe for GAP+FTAL raw P2 candidates."""

from __future__ import annotations

import argparse
import csv
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
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402


DEFAULT_CKPT_CANDIDATES = (
    Path("/marimo/yolo_code/runs/checkpoints_4way/runs/gap_factorized_k15/seed_42/weights/best.pt"),
    Path("/marimo/yolo_code/runs/levir_yolov8n_p2_gap_factorized_tal/gap_factorized_k15/seed_42/weights/best.pt"),
    ROOT / "runs/levir_yolov8n_p2_gap_factorized_tal/gap_factorized_k15/seed_42/weights/best.pt",
)
HF_REPO = "duyle2408/levir-yolov8n-p2-gap-factorized-tal-seed42"
HF_FILENAME = "runs/gap_factorized_k15/seed_42/weights/best.pt"


def resolve_checkpoint(path: Path | None) -> Path:
    if path and path.exists():
        return path
    for candidate in DEFAULT_CKPT_CANDIDATES:
        if candidate.exists():
            return candidate
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError("GAP+FTAL best.pt not found locally and huggingface_hub is unavailable") from exc
    cached = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, repo_type="dataset")
    return Path(cached)


def read_labels(image: Path) -> torch.Tensor:
    label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
    if not rows:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor([[float(v) for v in row[1:5]] for row in rows], dtype=torch.float32)


def load_samples(dataset_root: Path) -> list[dict]:
    direct = dataset_root / "levir_ship_yolo_seed42/images/test"
    if direct.exists():
        return [{"image": p, "boxes": read_labels(p)} for p in sorted(direct.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    for nested in (dataset_root / "levir_ship_yolo_seed42", dataset_root / "levir_ship_yolo_seed42/levir_ship_yolo_seed42"):
        cache = nested / "labels/test.cache"
        if not cache.exists():
            continue
        cache_obj = np.load(cache, allow_pickle=True).item()
        roots = [
            nested / "images/test",
            ROOT / "LevirShipData/All Images",
            Path("/marimo/LevirShipData/All Images"),
            Path("/marimo/datasets/levir_ship_yolo_seed42/images/test"),
        ]
        samples = []
        for item in cache_obj["labels"]:
            name = Path(item["im_file"]).name
            image = next((root / name for root in roots if (root / name).exists()), None)
            if image is not None:
                samples.append({"image": image, "boxes": torch.as_tensor(item["bboxes"], dtype=torch.float32).reshape(-1, 4)})
        if samples:
            return samples
    raise FileNotFoundError(f"Could not find LEVIR test split under {dataset_root}")


def letterbox_xyxy(gt_xyxy: torch.Tensor, image_shape: tuple[int, int], imgsz: int) -> torch.Tensor:
    h, w = image_shape
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (imgsz - new_w) / 2, (imgsz - new_h) / 2
    out = gt_xyxy.clone()
    out[:, [0, 2]] = out[:, [0, 2]] * r + dw
    out[:, [1, 3]] = out[:, [1, 3]] * r + dh
    return out


def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a) == 0 or len(b) == 0:
        return torch.zeros((len(a), len(b)), device=b.device if len(b) else a.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-8)


def raw_p2(net: nn.Module, image_np: np.ndarray, letterbox: LetterBox, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = letterbox(image=image_np)
    tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad():
        decoded, preds = net(tensor)
    p2 = preds["feats"][0].squeeze(0)
    n_p2 = p2.shape[1] * p2.shape[2]
    boxes = xywh2xyxy(decoded[0, :4, :n_p2].T)
    logits = preds["scores"][0, 0, :n_p2]
    return p2, boxes, logits


def flatten_feature(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    c, h, w = feat.shape
    return feat.permute(1, 2, 0).reshape(h * w, c)[idx]


def avg_map(feat: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.avg_pool2d(feat[None], kernel, stride=1, padding=kernel // 2, count_include_pad=False).squeeze(0)


def ring_map(feat: torch.Tensor, radius: int = 5) -> torch.Tensor:
    c, _, _ = feat.shape
    size = 2 * radius + 1
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist = ((xx - radius) ** 2 + (yy - radius) ** 2).float().sqrt()
    mask = (dist > 1.0) & (dist <= radius)
    weight = (mask.float() / mask.sum()).view(1, 1, size, size).repeat(c, 1, 1, 1).to(feat.device)
    return F.conv2d(F.pad(feat[None], (radius, radius, radius, radius), mode="replicate"), weight, groups=c).squeeze(0)


def box_pool(feat: torch.Tensor, boxes: torch.Tensor, idx: torch.Tensor, imgsz: int, ring_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    c, h, w = feat.shape
    inner, ring = [], []
    for box in boxes[idx]:
        x1, y1, x2, y2 = box.tolist()
        fx1 = max(0, min(w - 1, math.floor(x1 * w / imgsz)))
        fy1 = max(0, min(h - 1, math.floor(y1 * h / imgsz)))
        fx2 = max(fx1 + 1, min(w, math.ceil(x2 * w / imgsz)))
        fy2 = max(fy1 + 1, min(h, math.ceil(y2 * h / imgsz)))
        patch = feat[:, fy1:fy2, fx1:fx2]
        inner.append(patch.mean((1, 2)))

        bw, bh = max(fx2 - fx1, 1), max(fy2 - fy1, 1)
        pad_x, pad_y = max(1, math.ceil(bw * ring_scale)), max(1, math.ceil(bh * ring_scale))
        rx1, ry1 = max(0, fx1 - pad_x), max(0, fy1 - pad_y)
        rx2, ry2 = min(w, fx2 + pad_x), min(h, fy2 + pad_y)
        outer = feat[:, ry1:ry2, rx1:rx2]
        outer_sum = outer.sum((1, 2)) - patch.sum((1, 2))
        outer_count = outer.shape[1] * outer.shape[2] - patch.shape[1] * patch.shape[2]
        ring.append(outer_sum / max(outer_count, 1))
    return torch.stack(inner), torch.stack(ring)


def _pool_tokens(patch: torch.Tensor, token_grid: int) -> torch.Tensor:
    if patch.shape[1] == 0 or patch.shape[2] == 0:
        return patch.new_zeros((token_grid * token_grid, patch.shape[0]))
    pooled = F.adaptive_avg_pool2d(patch[None], (token_grid, token_grid)).squeeze(0)
    return pooled.permute(1, 2, 0).reshape(token_grid * token_grid, patch.shape[0])


def box_tokens(
    feat: torch.Tensor, boxes: torch.Tensor, idx: torch.Tensor, imgsz: int, ring_scale: float, token_grid: int
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = feat.shape
    inner_tokens, ring_tokens = [], []
    for box in boxes[idx]:
        x1, y1, x2, y2 = box.tolist()
        fx1 = max(0, min(w - 1, math.floor(x1 * w / imgsz)))
        fy1 = max(0, min(h - 1, math.floor(y1 * h / imgsz)))
        fx2 = max(fx1 + 1, min(w, math.ceil(x2 * w / imgsz)))
        fy2 = max(fy1 + 1, min(h, math.ceil(y2 * h / imgsz)))
        patch = feat[:, fy1:fy2, fx1:fx2]
        inner_tokens.append(_pool_tokens(patch, token_grid))

        bw, bh = max(fx2 - fx1, 1), max(fy2 - fy1, 1)
        pad_x, pad_y = max(1, math.ceil(bw * ring_scale)), max(1, math.ceil(bh * ring_scale))
        rx1, ry1 = max(0, fx1 - pad_x), max(0, fy1 - pad_y)
        rx2, ry2 = min(w, fx2 + pad_x), min(h, fy2 + pad_y)
        outer = feat[:, ry1:ry2, rx1:rx2].clone()
        outer[:, fy1 - ry1 : fy2 - ry1, fx1 - rx1 : fx2 - rx1] = 0
        ring_tokens.append(_pool_tokens(outer, token_grid))
    return torch.stack(inner_tokens), torch.stack(ring_tokens)


class TinyAttentionVerifier(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.score = nn.Linear(3 * channels, 1)

    def attend(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.q(query)[:, None, :]
        k = self.k(tokens)
        v = self.v(tokens)
        weights = (q * k).sum(-1) / math.sqrt(query.shape[1])
        return (weights.softmax(-1)[..., None] * v).sum(1)

    def forward(self, f: torch.Tensor, inner: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
        return self.score(torch.cat((f, self.attend(f, inner), self.attend(f, ring)), dim=1)).squeeze(1)


def train_attention_probe(
    f: torch.Tensor, inner: torch.Tensor, ring: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, epochs: int
) -> torch.Tensor:
    train_mask = groups.remainder(5) != 0
    eval_mask = ~train_mask
    model = TinyAttentionVerifier(f.shape[1]).to(f.device)
    opt = Adam(model.parameters(), lr=0.001)
    loss = nn.BCEWithLogitsLoss(
        pos_weight=(y[train_mask] == 0).sum().float() / (y[train_mask] == 1).sum().float().clamp_min(1)
    )
    for _ in range(epochs):
        opt.zero_grad()
        loss(model(f[train_mask], inner[train_mask], ring[train_mask]), y[train_mask].float()).backward()
        opt.step()
    with torch.no_grad():
        logits = torch.empty_like(y, dtype=torch.float32)
        logits[train_mask] = model(f[train_mask], inner[train_mask], ring[train_mask])
        logits[eval_mask] = model(f[eval_mask], inner[eval_mask], ring[eval_mask])
    return torch.sigmoid(logits)


def ap_auc(scores: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    order = torch.argsort(scores, descending=True)
    sorted_y = y[order].float()
    tp = sorted_y.cumsum(0)
    fp = (1 - sorted_y).cumsum(0)
    recall = tp / sorted_y.sum().clamp_min(1)
    precision = tp / (tp + fp).clamp_min(1)
    ap = float(((recall[1:] - recall[:-1]) * precision[1:]).sum().item())
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return ap, float("nan")
    order2 = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order2] = torch.arange(1, len(scores) + 1, device=scores.device, dtype=torch.float32)
    auc = float(((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item())
    return ap, auc


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if len(a) < 2:
        return float("nan")
    ar = torch.argsort(torch.argsort(a.float())).float()
    br = torch.argsort(torch.argsort(b.float())).float()
    ar, br = ar - ar.mean(), br - br.mean()
    denom = ar.norm() * br.norm()
    return float((ar @ br / denom).item()) if denom > 0 else float("nan")


def train_probe(x: torch.Tensor, y: torch.Tensor, epochs: int, hidden: int) -> nn.Module:
    if hidden > 0:
        model = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)).to(x.device)
    else:
        model = nn.Linear(x.shape[1], 1).to(x.device)
    opt = Adam(model.parameters(), lr=0.01 if hidden else 0.1)
    loss = nn.BCEWithLogitsLoss(pos_weight=(y == 0).sum().float() / (y == 1).sum().float().clamp_min(1))
    for _ in range(epochs):
        opt.zero_grad()
        loss(model(x).squeeze(1), y.float()).backward()
        opt.step()
    return model


def train_probe_scores(x: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, epochs: int, hidden: int) -> torch.Tensor:
    train_mask = groups.remainder(5) != 0
    model = train_probe(x[train_mask], y[train_mask], epochs, hidden)
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(1))


def eval_scores(scores: torch.Tensor, ious: torch.Tensor, y: torch.Tensor, groups: torch.Tensor) -> dict[str, float]:
    ap, auc = ap_auc(scores, y)
    rows = {"ap": ap, "auc": auc, "spearman_score_iou": spearman(scores, ious)}
    pos, neg = scores[y == 1], scores[y == 0]
    rows["score_margin"] = float(pos.mean().item() - neg.mean().item()) if len(pos) and len(neg) else float("nan")
    best_ranks, recalls = [], {1: [], 5: [], 10: []}
    for gid in torch.unique(groups):
        mask = groups == gid
        local_scores, local_ious = scores[mask], ious[mask]
        best_iou_idx = int(torch.argmax(local_ious))
        order = torch.argsort(local_scores, descending=True)
        best_ranks.append(int((order == best_iou_idx).nonzero(as_tuple=False)[0].item()) + 1)
        for k in recalls:
            recalls[k].append(float((local_ious[order[: min(k, len(order))]] >= 0.5).any().item()))
    rows["best_iou_rank"] = float(np.mean(best_ranks)) if best_ranks else float("nan")
    for k, vals in recalls.items():
        rows[f"recall_at_{k}"] = float(np.mean(vals)) if vals else float("nan")
    return rows


def eval_scores_masked(
    scores: torch.Tensor, ious: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    return eval_scores(scores[mask], ious[mask], y[mask], groups[mask])


def train_fusion_scores(raw: torch.Tensor, verifier: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, epochs: int) -> torch.Tensor:
    train_mask = groups.remainder(5) != 0
    x = torch.stack((raw, verifier), dim=1)
    model = nn.Linear(2, 1).to(raw.device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = Adam(model.parameters(), lr=0.1)
    loss = nn.BCEWithLogitsLoss(
        pos_weight=(y[train_mask] == 0).sum().float() / (y[train_mask] == 1).sum().float().clamp_min(1)
    )
    for _ in range(epochs):
        opt.zero_grad()
        loss(model(x[train_mask]).squeeze(1), y[train_mask].float()).backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(1))


def pairwise_quality(
    scores: torch.Tensor, ious: torch.Tensor, groups: torch.Tensor, iou_min: float, min_delta: float
) -> dict[str, float]:
    correct = total = 0
    margins = []
    for gid in torch.unique(groups):
        mask = (groups == gid) & (ious >= iou_min)
        local_scores, local_ious = scores[mask], ious[mask]
        if len(local_scores) < 2:
            continue
        diff = local_ious[:, None] - local_ious[None]
        keep = diff > min_delta
        if not keep.any():
            continue
        score_diff = local_scores[:, None] - local_scores[None]
        correct += int((score_diff[keep] > 0).sum().item())
        total += int(keep.sum().item())
        margins.extend(score_diff[keep].detach().cpu().tolist())
    return {
        "pair_accuracy": float(correct / total) if total else float("nan"),
        "pair_count": int(total),
        "mean_score_margin": float(np.mean(margins)) if margins else float("nan"),
    }


def make_report(output: dict) -> str:
    lines = [
        "# GAP+FTAL Candidate Verifier Probe",
        "",
        "## Protocol",
        "",
        f"- Checkpoint: `{output['protocol']['checkpoint']}`",
        f"- Split: `{output['protocol']['split']}`",
        f"- Images: `{output['protocol']['image_count']}`; GT groups: `{output['protocol']['group_count']}`",
        f"- NMS IoU recorded by protocol: `{output['protocol']['nms_iou']}`",
        f"- Positives: `{output['protocol']['positive']}`",
        f"- Negatives: `{output['protocol']['negative']}`",
        "",
        "## Summary",
        "",
        "| Feature | AP | AUC | Spearman | Best-IoU rank | R@1 | R@5 | R@10 | Margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in output["summary"].items():
        lines.append(
            f"| {name} | {row['ap']:.4f} | {row['auc']:.4f} | {row['spearman_score_iou']:.4f} | "
            f"{row['best_iou_rank']:.2f} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} | "
            f"{row['recall_at_10']:.4f} | {row['score_margin']:.4f} |"
        )
    lines += [
        "",
        "## Held-Out Groups",
        "",
        "| Feature | AP | AUC | Spearman | Best-IoU rank | R@1 | R@5 | R@10 | Margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in output["heldout_summary"].items():
        lines.append(
            f"| {name} | {row['ap']:.4f} | {row['auc']:.4f} | {row['spearman_score_iou']:.4f} | "
            f"{row['best_iou_rank']:.2f} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} | "
            f"{row['recall_at_10']:.4f} | {row['score_margin']:.4f} |"
        )
    lines += [
        "",
        "## Held-Out Positive-Pair Ranking",
        "",
        "| Feature | Pair accuracy | Pair count | Mean score margin |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in output["heldout_pairwise_summary"].items():
        lines.append(
            f"| {name} | {row['pair_accuracy']:.4f} | {row['pair_count']} | {row['mean_score_margin']:.4f} |"
        )
    decision = output["decision"]
    lines += ["", "## Decision", "", decision]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "investigate_gap_ftal_candidate_verifier")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--near-cells", type=int, default=8)
    parser.add_argument("--top-neg", type=int, default=256)
    parser.add_argument("--min-pos", type=int, default=1)
    parser.add_argument("--pos-iou-min", type=float, default=0.5)
    parser.add_argument("--pos-iou-max", type=float, default=1.0)
    parser.add_argument("--pair-min-delta", type=float, default=0.1)
    parser.add_argument("--token-grid", type=int, default=2)
    parser.add_argument("--attention-epochs", type=int, default=30)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    ckpt = resolve_checkpoint(args.checkpoint)
    samples = load_samples(args.dataset_root)
    if args.max_cases:
        samples = samples[: args.max_cases]

    wrapper = YOLO(ckpt)
    net = wrapper.model.to(device).eval()
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    feature_rows: dict[str, list[torch.Tensor]] = {
        "raw_logit": [],
        "f_i": [],
        "[f_i,fixed_ring_R5]": [],
        "[f_i,AvgPool5x5]": [],
        "[f_i,inner,ring]": [],
        "[f_i,inner,ring,inner-ring]": [],
    }
    attn_rows: dict[str, list[torch.Tensor]] = {"f": [], "inner_tokens": [], "ring_tokens": []}
    y_rows, iou_rows, group_rows, csv_rows = [], [], [], []
    group_id = 0

    for sample in samples:
        original = cv2.imread(str(sample["image"]))
        if original is None or len(sample["boxes"]) == 0:
            continue
        h, w = original.shape[:2]
        gt = xywh2xyxy(sample["boxes"])
        gt[:, [0, 2]] *= w
        gt[:, [1, 3]] *= h
        gt = letterbox_xyxy(gt, (h, w), args.imgsz).to(device)
        p2, boxes, logits = raw_p2(net, original, letterbox, device)
        ious_all = iou_matrix(gt, boxes)
        c, hp, wp = p2.shape
        yy, xx = torch.meshgrid(torch.arange(hp, device=device), torch.arange(wp, device=device), indexing="ij")
        flat_xy = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()
        avg5, fixed_ring = avg_map(p2, 5), ring_map(p2, 5)

        for gt_idx, ious in enumerate(ious_all):
            pos_mask = (ious >= args.pos_iou_min) & (ious <= args.pos_iou_max)
            pos_idx = torch.nonzero(pos_mask, as_tuple=False).flatten()
            if len(pos_idx) < args.min_pos:
                pos_idx = torch.argsort(ious, descending=True)[: args.min_pos]
            if len(pos_idx) == 0 or float(ious[pos_idx].max().item()) < args.pos_iou_min:
                continue
            g = gt[gt_idx]
            gx = ((g[0] + g[2]) * 0.5 * wp / args.imgsz)
            gy = ((g[1] + g[3]) * 0.5 * hp / args.imgsz)
            dist = ((flat_xy[:, 0] - gx) ** 2 + (flat_xy[:, 1] - gy) ** 2).sqrt()
            neg_pool = (ious < 0.3) & (dist <= args.near_cells)
            if neg_pool.sum() < 10:
                neg_pool = ious < 0.3
            neg_idx = torch.nonzero(neg_pool, as_tuple=False).flatten()
            if len(neg_idx) > args.top_neg:
                neg_idx = neg_idx[torch.argsort(logits[neg_idx], descending=True)[: args.top_neg]]
            idx = torch.cat([pos_idx, neg_idx]).unique()
            if len(idx) < 2:
                continue
            y = ((ious[idx] >= args.pos_iou_min) & (ious[idx] <= args.pos_iou_max)).long()
            inner, ring = box_pool(p2, boxes, idx, args.imgsz, ring_scale=1.0)
            inner_tokens, ring_tokens = box_tokens(p2, boxes, idx, args.imgsz, ring_scale=1.0, token_grid=args.token_grid)
            f = flatten_feature(p2, idx)
            f5 = flatten_feature(avg5, idx)
            fr = flatten_feature(fixed_ring, idx)
            reps = {
                "raw_logit": logits[idx, None],
                "f_i": f,
                "[f_i,fixed_ring_R5]": torch.cat([f, fr], dim=1),
                "[f_i,AvgPool5x5]": torch.cat([f, f5], dim=1),
                "[f_i,inner,ring]": torch.cat([f, inner, ring], dim=1),
                "[f_i,inner,ring,inner-ring]": torch.cat([f, inner, ring, inner - ring], dim=1),
            }
            for name, x in reps.items():
                feature_rows[name].append(x.detach())
            attn_rows["f"].append(f.detach())
            attn_rows["inner_tokens"].append(inner_tokens.detach())
            attn_rows["ring_tokens"].append(ring_tokens.detach())
            y_rows.append(y.detach())
            iou_rows.append(ious[idx].detach())
            group_rows.append(torch.full_like(y, group_id))
            for local_i, anchor_idx in enumerate(idx.tolist()):
                csv_rows.append({
                    "group_id": group_id,
                    "image": str(sample["image"]),
                    "gt_idx": gt_idx,
                    "anchor_idx": anchor_idx,
                    "iou": float(ious[idx][local_i].item()),
                    "label": int(y[local_i].item()),
                    "raw_logit": float(logits[idx][local_i].item()),
                })
            group_id += 1
            if args.max_groups and group_id >= args.max_groups:
                break
        if args.max_groups and group_id >= args.max_groups:
            break

    if not y_rows:
        raise RuntimeError("No candidate groups with IoU >= 0.5 were found.")
    y = torch.cat(y_rows).to(device)
    ious = torch.cat(iou_rows).to(device)
    groups = torch.cat(group_rows).to(device)
    summary = {}
    heldout_summary = {}
    score_columns: dict[str, torch.Tensor] = {}
    heldout_mask = groups.remainder(5) == 0
    for name, chunks in feature_rows.items():
        x = torch.cat(chunks).to(device)
        if name == "raw_logit":
            scores = torch.sigmoid(x.squeeze(1))
        else:
            scores = train_probe_scores(x, y, groups, args.epochs, args.hidden)
        score_columns[name] = scores.detach().cpu()
        summary[name] = eval_scores(scores, ious, y, groups)
        heldout_summary[name] = eval_scores_masked(scores, ious, y, groups, heldout_mask)
    attn_f = torch.cat(attn_rows["f"]).to(device)
    attn_inner = torch.cat(attn_rows["inner_tokens"]).to(device)
    attn_ring = torch.cat(attn_rows["ring_tokens"]).to(device)
    scores = train_attention_probe(attn_f, attn_inner, attn_ring, y, groups, args.attention_epochs)
    score_columns["box_attention"] = scores.detach().cpu()
    summary["box_attention"] = eval_scores(scores, ious, y, groups)
    heldout_summary["box_attention"] = eval_scores_masked(scores, ious, y, groups, heldout_mask)
    raw_scores = score_columns["raw_logit"].to(device)
    verifier_scores = score_columns["[f_i,inner,ring,inner-ring]"].to(device)
    scores = train_fusion_scores(raw_scores, verifier_scores, y, groups, args.epochs)
    score_columns["raw_plus_inner_ring_residual"] = scores.detach().cpu()
    summary["raw_plus_inner_ring_residual"] = eval_scores(scores, ious, y, groups)
    heldout_summary["raw_plus_inner_ring_residual"] = eval_scores_masked(scores, ious, y, groups, heldout_mask)
    pairwise_summary = {
        name: pairwise_quality(scores_cpu.to(device), ious, groups, args.pos_iou_min, args.pair_min_delta)
        for name, scores_cpu in score_columns.items()
    }
    heldout_pairwise_summary = {
        name: pairwise_quality(
            scores_cpu.to(device)[heldout_mask],
            ious[heldout_mask],
            groups[heldout_mask],
            args.pos_iou_min,
            args.pair_min_delta,
        )
        for name, scores_cpu in score_columns.items()
    }

    base = heldout_summary["f_i"]
    best_context = max(
        (k for k in heldout_summary if k not in {"raw_logit", "f_i"}), key=lambda k: heldout_summary[k]["ap"] - base["ap"]
    )
    ap_gain = heldout_summary[best_context]["ap"] - base["ap"]
    auc_gain = heldout_summary[best_context]["auc"] - base["auc"]
    rank_gain = base["best_iou_rank"] - heldout_summary[best_context]["best_iou_rank"]
    if ap_gain >= 0.05 and auc_gain >= 0.03:
        decision = f"PASS: `{best_context}` improves AP by {ap_gain:.4f} and AUC by {auc_gain:.4f}; local context is plausible."
    elif ap_gain >= 0.02 or rank_gain >= 1.0:
        decision = f"MIXED: `{best_context}` has modest AP/rank gain; inspect per-candidate rows before designing a module."
    else:
        decision = "FAIL: oracle local context does not clearly improve over `f_i`; do not build a local-context architecture from this result."

    args.output.mkdir(parents=True, exist_ok=True)
    for name, scores in score_columns.items():
        for row, score in zip(csv_rows, scores.tolist()):
            row[f"score_{name}"] = score
    with (args.output / "per_candidate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    output = {
        "protocol": {
            "checkpoint": str(ckpt),
            "checkpoint_source": HF_REPO,
            "split": "test",
            "seed": 42,
            "imgsz": args.imgsz,
            "nms_iou": 0.5,
            "image_count": len(samples),
            "group_count": int(group_id),
            "candidate_count": int(len(y)),
            "positive": f"raw P2 decoded candidates with {args.pos_iou_min} <= IoU <= {args.pos_iou_max} to GT",
            "negative": f"IoU < 0.3, prefer cells within {args.near_cells} P2 cells and top {args.top_neg} raw logits",
            "probe": "linear" if args.hidden == 0 else f"MLP hidden={args.hidden}",
            "epochs": args.epochs,
            "attention_epochs": args.attention_epochs,
            "token_grid": args.token_grid,
            "pair_min_delta": args.pair_min_delta,
        },
        "summary": summary,
        "heldout_summary": heldout_summary,
        "pairwise_summary": pairwise_summary,
        "heldout_pairwise_summary": heldout_pairwise_summary,
        "decision": decision,
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2))
    (args.output / "report.md").write_text(make_report(output))
    print(json.dumps(output, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
