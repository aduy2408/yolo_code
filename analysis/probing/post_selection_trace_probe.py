"""Trace P2 candidates that beat P3/P4 through the TAL assignment stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner

from analysis.probing.candidate_evidence_probe import _candidate_tensors, box_iou


def labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    result = []
    if not path.is_file():
        return result
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, cx, cy, bw, bh = map(float, values[:5])
        cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
        result.append((int(cls), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return result


def rank_desc(values: torch.Tensor, index: int) -> int:
    order = torch.argsort(values, descending=True)
    position = (order == index).nonzero(as_tuple=False)
    return int(position[0, 0].item() + 1) if position.numel() else 0


def assign_trace(
    assigner: TaskAlignedAssigner,
    class_scores: torch.Tensor,
    boxes: torch.Tensor,
    anchors: torch.Tensor,
    strides: torch.Tensor,
    gt_labels: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    gt_index: int,
    candidate: int,
    eligible: torch.Tensor,
) -> dict[str, Any]:
    align, overlaps = assigner.get_box_metrics(
        class_scores.T.unsqueeze(0), boxes.unsqueeze(0), gt_labels, gt.unsqueeze(0), eligible * mask
    )
    mask_topk = assigner.select_topk_candidates(
        align, topk_mask=mask.expand(-1, -1, assigner.topk).bool()
    )
    mask_pos = mask_topk * eligible * mask
    target_gt_idx, fg_mask, mask_pos = assigner.select_highest_overlaps(
        mask_pos, overlaps, gt.shape[0], align
    )
    target_labels, _, target_scores = assigner.get_targets(
        gt_labels, gt.unsqueeze(0), target_gt_idx, fg_mask
    )
    align_pos = align * mask_pos
    pos_align_metrics = align_pos.amax(dim=-1, keepdim=True)
    pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
    normalized = (align_pos * pos_overlaps / (pos_align_metrics + assigner.eps)).amax(-2).unsqueeze(-1)
    target_scores = target_scores * normalized

    pool = eligible[0, gt_index].nonzero(as_tuple=False).flatten()
    gt_scores = class_scores[gt_labels[0, gt_index, 0].long(), pool]
    gt_tal = align[0, gt_index, pool]
    gt_iou = overlaps[0, gt_index, pool]
    local_position = (pool == candidate).nonzero(as_tuple=False)
    if local_position.numel() == 0:
        raise RuntimeError("P2 candidate was not present in its eligible pool")
    local_index = int(local_position[0, 0].item())
    target_class = int(gt_labels[0, gt_index, 0].item())
    assigned = bool(fg_mask[0, candidate].item())
    assigned_gt = int(target_gt_idx[0, candidate].item()) if assigned else -1
    candidate_target_mass = float(target_scores[0, candidate, target_class].item())
    own_mass = float(target_scores[0, target_gt_idx[0] == gt_index, target_class].sum().item())
    max_own_mass = float(target_scores[0, :, target_class].max().item())
    rank_cls = rank_desc(gt_scores, local_index)
    rank_tal = rank_desc(gt_tal, local_index)
    rank_iou = rank_desc(gt_iou, local_index)
    topk_selected = bool(mask_topk[0, gt_index, candidate].item())
    final_positive_for_gt = assigned and assigned_gt == gt_index
    responsibility_share = candidate_target_mass / own_mass if own_mass > 1e-12 else 0.0
    return {
        "candidate": candidate,
        "eligible_count": int(pool.numel()),
        "iou": float(overlaps[0, gt_index, candidate].item()),
        "cls_score": float(class_scores[target_class, candidate].item()),
        "tal": float(align[0, gt_index, candidate].item()),
        "rank_iou": rank_iou,
        "rank_cls": rank_cls,
        "rank_tal": rank_tal,
        "topk_selected": topk_selected,
        "assigned": assigned,
        "assigned_gt": assigned_gt,
        "final_positive_for_gt": final_positive_for_gt,
        "target_mass": candidate_target_mass,
        "gt_target_mass": own_mass,
        "responsibility_share": responsibility_share,
        "gt_max_target_mass": max_own_mass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--max-images", type=int, default=808)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tiny-max-dim", type=float, default=16.0)
    parser.add_argument("--responsibility-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = args.dataset_yaml.parent
    images = sorted((split / "images/val").glob("*.jpg"))[: args.max_images]
    checkpoint_map = dict(item.split("=", 1) for item in args.checkpoint)
    device = torch.device(args.device)
    loaded_models: dict[str, Any] = {}
    for name, checkpoint in checkpoint_map.items():
        model = YOLO(checkpoint)
        model.model.to(device).eval()
        loaded_models[name] = model
    all_rows: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt_rows = labels(split / "labels/val" / f"{image_path.stem}.txt", width, height)
        if not gt_rows:
            continue
        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
        gt = torch.tensor(
            [[x[1] * 640 / width, x[2] * 640 / height, x[3] * 640 / width, x[4] * 640 / height] for x in gt_rows],
            device=device,
            dtype=torch.float32,
        )
        gt_labels = torch.tensor([[[x[0]] for x in gt_rows]], device=device, dtype=torch.long)
        mask = torch.ones((1, len(gt_rows), 1), device=device, dtype=torch.bool)

        current_data: dict[str, dict[str, Any]] = {}
        for name, model in loaded_models.items():
            with torch.no_grad():
                boxes, class_scores, _, anchors, strides = _candidate_tensors(model, tensor)
            head = model.model.model[-1]
            assigner = TaskAlignedAssigner(
                topk=args.topk, num_classes=head.nc, alpha=0.5, beta=6.0, stride=head.stride.tolist()
            )
            assigner.bs = 1
            assigner.n_max_boxes = len(gt_rows)
            with torch.no_grad():
                eligible = assigner.select_candidates_in_gts(anchors * strides, gt.unsqueeze(0), mask)
            ious = box_iou(gt, boxes)
            current_data[name] = {
                "model": model,
                "boxes": boxes,
                "class_scores": class_scores,
                "anchors": anchors,
                "strides": strides,
                "assigner": assigner,
                "eligible": eligible,
                "ious": ious,
                "gt": gt,
                "gt_labels": gt_labels,
                "mask": mask,
            }

        reference = current_data[next(iter(checkpoint_map))]
        stride_values = reference["strides"][:, 0]
        p2 = stride_values <= 4.0 + 1e-6
        for gt_index, gt_row in enumerate(gt_rows):
            if max(float(gt[gt_index, 2] - gt[gt_index, 0]), float(gt[gt_index, 3] - gt[gt_index, 1])) > args.tiny_max_dim:
                continue
            per_model_pools: dict[str, torch.Tensor] = {}
            best_p2: dict[str, tuple[int, float]] = {}
            best_other: dict[str, float] = {}
            for name, data in current_data.items():
                pool = data["eligible"][0, gt_index].bool()
                p2_pool = pool & p2
                other_pool = pool & ~p2
                if not p2_pool.any() or not other_pool.any():
                    continue
                p2_indices = p2_pool.nonzero(as_tuple=False).flatten()
                other_indices = other_pool.nonzero(as_tuple=False).flatten()
                p2_iou = data["ious"][gt_index, p2_indices]
                other_iou = data["ious"][gt_index, other_indices]
                best_index = int(p2_indices[p2_iou.argmax()].item())
                best_value = float(p2_iou.max().item())
                other_value = float(other_iou.max().item())
                per_model_pools[name] = pool.nonzero(as_tuple=False).flatten()
                best_p2[name] = (best_index, best_value)
                best_other[name] = other_value
            for name, data in current_data.items():
                if name not in best_p2 or best_p2[name][1] <= best_other[name] + 1e-7:
                    continue
                candidate = best_p2[name][0]
                with torch.no_grad():
                    row = assign_trace(
                        data["assigner"], data["class_scores"], data["boxes"], data["anchors"], data["strides"],
                        data["gt_labels"], data["gt"], data["mask"], gt_index, candidate,
                        data["eligible"],
                    )
                row.update({
                    "model": name,
                    "image_index": image_index,
                    "image": image_path.name,
                    "object_id": gt_index,
                    "p2_best_iou": best_p2[name][1],
                    "p3p4_best_iou": best_other[name],
                    "size_bucket": "eligible_le_10" if row["eligible_count"] <= args.topk else "eligible_gt_10",
                })
                if not row["topk_selected"]:
                    row["failure_bucket"] = "topk_exclusion"
                elif not row["final_positive_for_gt"]:
                    row["failure_bucket"] = "conflict_loss"
                elif row["target_mass"] < args.responsibility_threshold:
                    row["failure_bucket"] = "low_responsibility"
                elif row["rank_cls"] > args.topk:
                    row["failure_bucket"] = "score_learning_failure"
                else:
                    row["failure_bucket"] = "retained"
                all_rows.append(row)
    summary: dict[str, Any] = {
        "images": len(images),
        "topk": args.topk,
        "tiny_max_dim": args.tiny_max_dim,
        "responsibility_threshold": args.responsibility_threshold,
        "rows": len(all_rows),
        "models": {},
    }
    for name in checkpoint_map:
        model_rows = [row for row in all_rows if row["model"] == name]
        summary["models"][name] = {
            "qualifying_gt": len(model_rows),
            "eligible_le_10": sum(row["size_bucket"] == "eligible_le_10" for row in model_rows),
            "eligible_gt_10": sum(row["size_bucket"] == "eligible_gt_10" for row in model_rows),
            "failure_buckets": {
                bucket: sum(row["failure_bucket"] == bucket for row in model_rows)
                for bucket in ("topk_exclusion", "conflict_loss", "low_responsibility", "score_learning_failure", "retained")
            },
            "mean": {
                key: float(np.mean([row[key] for row in model_rows])) if model_rows else float("nan")
                for key in ("eligible_count", "iou", "cls_score", "tal", "rank_iou", "rank_cls", "rank_tal", "target_mass", "responsibility_share")
            },
            "by_size": {
                bucket: {
                    "count": sum(row["size_bucket"] == bucket for row in model_rows),
                    "failure_buckets": {
                        failure: sum(row["size_bucket"] == bucket and row["failure_bucket"] == failure for row in model_rows)
                        for failure in ("topk_exclusion", "conflict_loss", "low_responsibility", "score_learning_failure", "retained")
                    },
                }
                for bucket in ("eligible_le_10", "eligible_gt_10")
            },
        }
    (args.output_dir / "post_selection_trace.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "post_selection_trace_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in all_rows) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
