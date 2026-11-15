#!/usr/bin/env python3
"""GAP counterfactual geometry kill-test for LEVIR P2 candidates."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402


def parse_channels(text: str) -> list[int]:
    channels = [int(item) for item in text.replace(",", " ").split()]
    if not channels:
        raise argparse.ArgumentTypeError("--easy-channels must contain at least one index")
    if len(set(channels)) != len(channels) or min(channels) < 0:
        raise argparse.ArgumentTypeError("--easy-channels must be unique non-negative integers")
    return channels


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def load_gt(image: Path, width: int, height: int, ratio: float, pad: tuple[float, float]) -> torch.Tensor:
    boxes = []
    label = labels_for(image)
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                _, x, y, w, h = map(float, parts[:5])
                boxes.append(((x - w / 2) * width, (y - h / 2) * height,
                              (x + w / 2) * width, (y + h / 2) * height))
    out = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if out.numel():
        out[:, [0, 2]] = out[:, [0, 2]] * ratio + pad[0]
        out[:, [1, 3]] = out[:, [1, 3]] * ratio + pad[1]
    return out


def greedy_matches(gt: torch.Tensor, detections: torch.Tensor, thr: float = 0.5) -> tuple[set[int], set[int]]:
    if not len(gt) or not len(detections):
        return set(), set()
    ious = box_iou(gt, detections[:, :4])
    used_gt, used_det = set(), set()
    for flat in ious.reshape(-1).argsort(descending=True):
        gi, di = divmod(int(flat), ious.shape[1])
        if float(ious[gi, di]) < thr:
            break
        if gi not in used_gt and di not in used_det:
            used_gt.add(gi)
            used_det.add(di)
    return used_gt, used_det


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def edge_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean())


def single_box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(box_iou(a.reshape(1, 4), b.reshape(1, 4))[0, 0])


def resolve_images(data_yaml: Path, split: str) -> list[Path]:
    data = {}
    for line in data_yaml.read_text().splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    root = Path(data.get("path", data_yaml.parent))
    image_dir = Path(data[split])
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    return images


def decode_p2(net, tensor: torch.Tensor, muted: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hook = None
    if muted is not None:
        def mute(_module, _inputs, output):
            changed = output.clone()
            changed[:, muted] = 0
            return changed
        hook = net.model[18].register_forward_hook(mute)
    try:
        with torch.inference_mode():
            pred, raw = net(tensor)
            head = net.model[-1]
            p2_h, p2_w = raw["feats"][0].shape[2:]
            p2_count = p2_h * p2_w
            decoded = head._get_decode_boxes(raw)[0, :, :p2_count].T
            boxes = xywh_to_xyxy(decoded)
            scores = raw["scores"][0, 0, :p2_count].sigmoid()
            nms = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.5, nc=1, max_det=300)[0]
            return boxes, scores, nms
    finally:
        if hook is not None:
            hook.remove()


def best_anchor(gt_box: torch.Tensor, boxes: torch.Tensor) -> tuple[int, float]:
    ious = box_iou(gt_box.reshape(1, 4), boxes)[0]
    index = int(ious.argmax())
    return index, float(ious[index])


def top_anchor(gt_box: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> tuple[int, float]:
    inside = box_iou(gt_box.reshape(1, 4), boxes)[0] > 0
    pool = torch.where(inside)[0]
    if not len(pool):
        return best_anchor(gt_box, boxes)
    index = int(pool[scores[pool].argmax()])
    return index, float(box_iou(gt_box.reshape(1, 4), boxes[index].reshape(1, 4))[0, 0])


def match_c_to_a(c_rows: list[dict], a_rows: list[dict]) -> list[dict]:
    if not a_rows or len(c_rows) <= len(a_rows):
        return c_rows
    remaining = c_rows[:]
    picked = []
    for row in sorted(a_rows, key=lambda r: r["confidence_clean"]):
        idx = min(range(len(remaining)), key=lambda i: abs(remaining[i]["confidence_clean"] - row["confidence_clean"]))
        picked.append(remaining.pop(idx))
    return picked


def row_for(kind: str, image: Path, gt_id: int | None, anchor: int, gt_box: torch.Tensor | None,
            clean_boxes: torch.Tensor, clean_scores: torch.Tensor, pert_boxes: torch.Tensor, pert_scores: torch.Tensor,
            perturbation: str) -> dict:
    clean_box, pert_box = clean_boxes[anchor], pert_boxes[anchor]
    clean_gt = single_box_iou(clean_box, gt_box) if gt_box is not None else math.nan
    pert_gt = single_box_iou(pert_box, gt_box) if gt_box is not None else math.nan
    return {
        "population": kind,
        "perturbation": perturbation,
        "image": image.name,
        "gt_id": "" if gt_id is None else gt_id,
        "anchor_idx": anchor,
        "box_stability_iou": single_box_iou(clean_box, pert_box),
        "edge_displacement_px": edge_delta(clean_box, pert_box),
        "iou_clean_gt": clean_gt,
        "iou_perturbed_gt": pert_gt,
        "delta_iou_gt": pert_gt - clean_gt if math.isfinite(clean_gt) else math.nan,
        "confidence_clean": float(clean_scores[anchor]),
        "confidence_perturbed": float(pert_scores[anchor]),
        "delta_confidence": float(pert_scores[anchor] - clean_scores[anchor]),
    }


def med(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.median(finite)) if finite else None


def summarize(rows: list[dict]) -> dict:
    out = {}
    for perturb in sorted({r["perturbation"] for r in rows}):
        out[perturb] = {}
        for pop in sorted({r["population"] for r in rows}):
            chosen = [r for r in rows if r["perturbation"] == perturb and r["population"] == pop]
            out[perturb][pop] = {
                "n": len(chosen),
                "median_box_stability_iou": med([r["box_stability_iou"] for r in chosen]),
                "median_abs_delta_iou_gt": med([abs(r["delta_iou_gt"]) for r in chosen]),
                "median_edge_displacement_px": med([r["edge_displacement_px"] for r in chosen]),
                "median_confidence_clean": med([r["confidence_clean"] for r in chosen]),
            }
    return out


def verdict(summary: dict) -> str:
    easy = summary.get("easy", {})
    a = easy.get("A_bestgeom", {})
    b = easy.get("B_bestgeom", {})
    c = easy.get("C_hardneg", {})
    av, bv, cv = a.get("median_box_stability_iou"), b.get("median_box_stability_iou"), c.get("median_box_stability_iou")
    ad = a.get("median_abs_delta_iou_gt")
    if None in (av, bv, cv, ad):
        return "inconclusive_insufficient_population"
    if av < bv - 0.15:
        return "kill_immediately_a_fragile_vs_b"
    if av >= bv - 0.05 and av - cv >= 0.15 and ad < 0.05:
        random_gaps = []
        for name, block in summary.items():
            if name.startswith("random_") and "A_bestgeom" in block and "C_hardneg" in block:
                ra = block["A_bestgeom"]["median_box_stability_iou"]
                rc = block["C_hardneg"]["median_box_stability_iou"]
                if ra is not None and rc is not None:
                    random_gaps.append(ra - rc)
        easy_gap = av - cv
        if random_gaps and easy_gap <= float(np.median(random_gaps)) + 0.03:
            return "weak_ambiguous_random_control_similar"
        return "strong_pass"
    if av > cv and ad < 0.05:
        return "weak_ambiguous_small_gap"
    return "fail"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=ROOT / "docs/investigate_gap_feature_miss/hf_gap/runs/gap/seed_42/weights/best.pt")
    p.add_argument("--data-yaml", type=Path, default=ROOT / "datasets/levir_ship_yolo_seed42/levir_ship.yaml")
    p.add_argument("--easy-channels", type=parse_channels, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "diagnostics/gap_counterfactual_geometry")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--limit", type=int)
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--min-hard-conf", type=float, default=0.01)
    args = p.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    images = resolve_images(args.data_yaml, args.split)
    if args.limit:
        images = images[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    wrapper = YOLO(args.model_path)
    net = wrapper.model.to(args.device).eval()
    head = net.model[-1]
    if head.stride.tolist() != [4.0]:
        raise RuntimeError(f"Expected GAP P2-only Detect stride [4.0], got {head.stride.tolist()}")
    if max(args.easy_channels) >= net.model[18].cv2.conv.out_channels:
        raise RuntimeError("Easy channel index exceeds layer-18 channels")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    random_sets = []
    rng = random.Random(42)
    channels = list(range(net.model[18].cv2.conv.out_channels))
    for _ in range(args.random_controls):
        random_sets.append(sorted(rng.sample(channels, len(args.easy_channels))))

    primary_clean, secondary_clean, hardneg = [], [], []
    cache = []
    for image in images:
        original = cv2.imread(str(image))
        if original is None:
            raise RuntimeError(f"Could not read image: {image}")
        h, w = original.shape[:2]
        ratio = min(args.imgsz / h, args.imgsz / w)
        pad = ((args.imgsz - round(w * ratio)) / 2, (args.imgsz - round(h * ratio)) / 2)
        gt = load_gt(image, w, h, ratio, pad).to(args.device)
        tensor = torch.from_numpy(letterbox(image=original)[..., ::-1].copy()).to(args.device).permute(2, 0, 1).float()[None] / 255
        clean_boxes, clean_scores, det = decode_p2(net, tensor)
        matched_gt, matched_det = greedy_matches(gt, det, 0.5)
        iou_to_gt = box_iou(clean_boxes, gt).amax(1) if len(gt) else torch.zeros(len(clean_boxes), device=args.device)

        for gi, gt_box in enumerate(gt):
            anchor, iou = best_anchor(gt_box, clean_boxes)
            top, top_iou = top_anchor(gt_box, clean_boxes, clean_scores)
            if gi in matched_gt:
                primary_clean.append(("B_bestgeom", image, gi, anchor, gt_box.detach().clone(), float(clean_scores[anchor])))
                secondary_clean.append(("B_topscore", image, gi, top, gt_box.detach().clone(), float(clean_scores[top]), top_iou))
            elif iou >= 0.5:
                primary_clean.append(("A_bestgeom", image, gi, anchor, gt_box.detach().clone(), float(clean_scores[anchor])))
                secondary_clean.append(("A_topscore", image, gi, top, gt_box.detach().clone(), float(clean_scores[top]), top_iou))

        unmatched = (clean_scores >= args.min_hard_conf) & (iou_to_gt < 0.5)
        fp_det = det[[i for i in range(len(det)) if i not in matched_det], :4] if len(det) else det.new_zeros((0, 4))
        if len(fp_det):
            fp_near = box_iou(clean_boxes, fp_det).amax(1) >= 0.5
            primary_c = torch.where(unmatched & fp_near)[0]
        else:
            primary_c = torch.empty(0, device=args.device, dtype=torch.long)
        supplement = torch.where(unmatched)[0]
        ordered_raw = torch.cat((
            primary_c[clean_scores[primary_c].argsort(descending=True)],
            supplement[clean_scores[supplement].argsort(descending=True)],
        )).tolist()
        seen, ordered = set(), []
        for anchor in ordered_raw:
            if anchor not in seen:
                seen.add(anchor)
                ordered.append(anchor)
        for anchor in ordered[: max(10, len(gt) * 5)]:
            hardneg.append(("C_hardneg", image, None, int(anchor), None, float(clean_scores[anchor])))
        cache.append((image, tensor, clean_boxes.detach(), clean_scores.detach()))

    a_rows = [{"confidence_clean": x[5]} for x in primary_clean if x[0] == "A_bestgeom"]
    c_selected = match_c_to_a([{"item": x, "confidence_clean": x[5]} for x in hardneg], a_rows)
    frozen_primary = primary_clean + [x["item"] for x in c_selected]
    frozen_secondary = secondary_clean
    tensor_by_image = {image: tensor for image, tensor, _boxes, _scores in cache}
    clean_by_image = {image: (boxes, scores) for image, _tensor, boxes, scores in cache}

    def materialize(frozen: list[tuple], perturbation: str, muted: list[int]) -> list[dict]:
        perturbed = {}
        for image in {item[1] for item in frozen}:
            boxes, scores, _ = decode_p2(net, tensor_by_image[image], muted)
            clean_boxes, _clean_scores = clean_by_image[image]
            if len(boxes) != len(clean_boxes):
                raise RuntimeError(f"{image.name}: P2 anchor count changed under {perturbation}")
            perturbed[image] = (boxes.detach(), scores.detach())
        rows = []
        for item in frozen:
            kind, image, gt_id, anchor, gt_box, _conf = item[:6]
            clean_boxes, clean_scores = clean_by_image[image]
            pert_boxes, pert_scores = perturbed[image]
            rows.append(row_for(kind, image, gt_id, anchor, gt_box, clean_boxes, clean_scores, pert_boxes, pert_scores, perturbation))
        return rows

    primary_rows = materialize(frozen_primary, "easy", args.easy_channels)
    secondary_rows = materialize(frozen_secondary, "easy", args.easy_channels)
    for i, muted in enumerate(random_sets):
        primary_rows.extend(materialize(frozen_primary, f"random_{i}", muted))
        secondary_rows.extend(materialize(frozen_secondary, f"random_{i}", muted))

    fields = list(primary_rows[0].keys()) if primary_rows else [
        "population", "perturbation", "image", "gt_id", "anchor_idx", "box_stability_iou", "edge_displacement_px",
        "iou_clean_gt", "iou_perturbed_gt", "delta_iou_gt", "confidence_clean", "confidence_perturbed", "delta_confidence",
    ]
    for path, rows in (
        (args.output / "geometry_stability_per_gt.csv", primary_rows),
        (args.output / "geometry_stability_per_candidate_secondary.csv", secondary_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize(primary_rows)
    out = {
        "checkpoint": str(args.model_path),
        "data_yaml": str(args.data_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "nms_iou": 0.5,
        "easy_channels": args.easy_channels,
        "random_channels": random_sets,
        "clean_population_counts": {name: sum(1 for x in frozen_primary if x[0] == name) for name in ("A_bestgeom", "B_bestgeom", "C_hardneg")},
        "summary": summary,
        "verdict": verdict(summary),
        "selection_invariant": "All populations and anchor_idx are selected from clean inference only; perturbed runs reuse frozen anchors.",
    }
    (args.output / "geometry_stability_diagnostic.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "verdict": out["verdict"], "counts": out["clean_population_counts"]}, indent=2))


if __name__ == "__main__":
    main()
