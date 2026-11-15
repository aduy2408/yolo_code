#!/usr/bin/env python3
"""Create two title-free donor-to-augmentation flow figures.

Figure 1: donor A/B source+zoom above positive original, CP1, CP2.
Figure 2: donor A/B source+zoom above negative original, R2, R4.
Red arrows project from each donor zoom panel to the corresponding output.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

from misc.plot_cp_variants_4x2 import boxes_from_labels, load_dataset, make_labels
from misc.plot_shared_donor_positive_negative import (
    FixedNegativeCanvas,
    FixedChoiceRNG,
    crop_with_margin,
    image_for,
)
from project_ultralytics.copy_paste import SmallObjectCopyPaste

ROOT = Path(__file__).resolve().parents[1]


def draw(ax, image, boxes=None, new_boxes=None):
    ax.imshow(image[:, :, ::-1], interpolation="nearest")
    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="#22a06b", linewidth=2.0))
    if new_boxes is not None:
        for x1, y1, x2, y2 in new_boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="#ff3b30", linewidth=2.0))
    ax.axis("off")


def donor_pair(ax_full, ax_zoom, image, box, fig):
    draw(ax_full, image, new_boxes=[box])
    crop, crop_box = crop_with_margin(image, box)
    local_box = np.asarray([[box[0] - crop_box[0], box[1] - crop_box[1],
                             box[2] - crop_box[0], box[3] - crop_box[1]]], dtype=np.float32)
    draw(ax_zoom, crop, new_boxes=local_box)
    fig.add_artist(ConnectionPatch(
        xyA=(box[2], box[1]), coordsA=ax_full.transData,
        xyB=(0, 1), coordsB=ax_zoom.transAxes,
        arrowstyle="-", linewidth=1.1, linestyle="--", color="#ff3b30",
    ))
    return local_box


def fixed_cp(dataset, target_index, donor, copies, seed):
    transform = SmallObjectCopyPaste(
        dataset,
        p=1.0,
        unit="single",
        copies=copies,
        placement="random",
        max_overlap=0.0,
        allow_empty_target=True,
        allow_same_source=True,
        rng=FixedChoiceRNG(seed, donor),
    )
    labels = make_labels(dataset, target_index)
    before = boxes_from_labels(labels)
    result = transform(labels)
    return result, before, boxes_from_labels(result)


def fixed_negative(dataset, target_index, donor, seed, degradation):
    transform = FixedNegativeCanvas(
        donor,
        dataset,
        p=1.0,
        target_policy="deficit",
        donor_policy="matched" if degradation == "none" else "larger",
        target_max_size=20.0,
        degradation=degradation,
        blur_sigma=0.5,
        rng=random.Random(seed),
        max_trials=100,
    )
    labels = make_labels(dataset, target_index)
    before = boxes_from_labels(labels)
    result = transform(labels)
    return result, before, boxes_from_labels(result)


def add_projection(fig, source_ax, source_box, target_ax, target_box):
    source_left = (float(source_box[0, 0]), float(source_box[0, 3]))
    source_right = (float(source_box[0, 2]), float(source_box[0, 3]))
    # Keep both bars attached to the target bbox. A high z-order prevents the
    # tiny red bbox from hiding either endpoint during rasterization.
    target_y = float(target_box[1])
    target_left = (float(target_box[0]), target_y)
    target_right = (float(target_box[2]), target_y)
    for source_point, target_point in ((source_left, target_left), (source_right, target_right)):
        fig.add_artist(ConnectionPatch(
            xyA=source_point, coordsA=source_ax.transData,
            xyB=target_point, coordsB=target_ax.transData,
            arrowstyle="-", linewidth=2.2, color="#ff3b30", alpha=0.95,
            zorder=30, clip_on=False,
        ))


def make_figure(dataset, donor_a, donor_b, target_index, mode, seed, output):
    donor_a_image = image_for(dataset.im_files[donor_a.image_index])
    donor_b_image = image_for(dataset.im_files[donor_b.image_index])
    donor_a_box = tuple(float(value) for value in donor_a.bbox_xyxy)
    donor_b_box = tuple(float(value) for value in donor_b.bbox_xyxy)

    if mode == "positive":
        cp1, before, cp1_boxes = fixed_cp(dataset, target_index, donor_a, 1, seed)
        cp2, _, cp2_boxes = fixed_cp(dataset, target_index, donor_a, 2, seed + 1)
        outputs = [cp1, cp2]
        output_boxes = [cp1_boxes[len(before):], cp2_boxes[len(before):]]
    else:
        r2, before, r2_boxes = fixed_negative(dataset, target_index, donor_a, seed + 2, "none")
        r4, _, r4_boxes = fixed_negative(dataset, target_index, donor_b, seed + 3, "weak_blur")
        outputs = [r2, r4]
        output_boxes = [r2_boxes[len(before):], r4_boxes[len(before):]]

    target_image = image_for(dataset.im_files[target_index])
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), dpi=180, squeeze=False)
    donor_a_zoom_box = donor_pair(axes[0, 0], axes[0, 1], donor_a_image, donor_a_box, fig)
    donor_b_zoom_box = donor_pair(axes[0, 2], axes[0, 3], donor_b_image, donor_b_box, fig)
    axes[0, 4].axis("off")

    draw(axes[1, 0], target_image, boxes=before)
    draw(axes[1, 1], outputs[0]["img"], boxes=before, new_boxes=output_boxes[0])
    draw(axes[1, 2], outputs[1]["img"], boxes=before, new_boxes=output_boxes[1])
    axes[1, 3].axis("off")
    axes[1, 4].axis("off")

    first_new_box = np.asarray(output_boxes[0], dtype=np.float32).reshape(-1, 4)
    second_new_box = np.asarray(output_boxes[1], dtype=np.float32).reshape(-1, 4)
    first_region = np.asarray([
        first_new_box[:, 0].min(), first_new_box[:, 1].min(),
        first_new_box[:, 2].max(), first_new_box[:, 3].max(),
    ], dtype=np.float32)
    second_region = np.asarray([
        second_new_box[:, 0].min(), second_new_box[:, 1].min(),
        second_new_box[:, 2].max(), second_new_box[:, 3].max(),
    ], dtype=np.float32)
    add_projection(fig, axes[0, 1], donor_a_zoom_box, axes[1, 1], first_region)
    add_projection(fig, axes[0, 3], donor_b_zoom_box, axes[1, 2], second_region)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, hspace=0.18, wspace=0.035)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "mode": mode,
        "target_image": dataset.im_files[target_index],
        "donor_a": {"image": dataset.im_files[donor_a.image_index], "bbox_xyxy": list(donor_a_box)},
        "donor_b": {"image": dataset.im_files[donor_b.image_index], "bbox_xyxy": list(donor_b_box)},
        "original_box_count": len(before),
        "new_box_counts": [len(boxes) for boxes in output_boxes],
        "panel_order": ["donor_a_full", "donor_a_zoom", "donor_b_full", "donor_b_zoom", "blank",
                         "original", "first_augmented", "second_augmented", "blank", "blank"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/LevirShipData")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/reports/augmentation_examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    positive_indices = [i for i, label in enumerate(dataset.labels) if len(label["bboxes"])]
    negative_indices = [i for i, label in enumerate(dataset.labels) if not len(label["bboxes"])]
    sparse_positive = [i for i in positive_indices if len(dataset.labels[i]["bboxes"]) <= 2]
    positive_candidates = sparse_positive or positive_indices
    positive_index = max(
        positive_candidates,
        key=lambda i: float(np.max(
            np.asarray(dataset.labels[i]["bboxes"], dtype=np.float32)[:, 2]
            * np.asarray(dataset.labels[i]["bboxes"], dtype=np.float32)[:, 3]
        )),
    )
    negative_index = negative_indices[0]

    pool_builder = SmallObjectCopyPaste(dataset, p=1.0, rng=random.Random(args.seed))
    pool_builder._build_pool()
    donor_a, donor_b = pool_builder.object_pool[:2]

    positive_output = args.output_dir / "levir_shared_donor_positive_flow.png"
    negative_output = args.output_dir / "levir_shared_donor_negative_flow.png"
    metadata = {
        "title_free": True,
        "seed": args.seed,
        "positive": make_figure(dataset, donor_a, donor_b, positive_index, "positive", args.seed, positive_output),
        "negative": make_figure(dataset, donor_a, donor_b, negative_index, "negative", args.seed, negative_output),
        "color_convention": {"green": "original target boxes", "red": "donor boxes and inserted output boxes"},
    }
    metadata_path = args.output_dir / "levir_shared_donor_flow.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {positive_output}")
    print(f"Saved {negative_output}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
