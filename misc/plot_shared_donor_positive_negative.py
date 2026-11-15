#!/usr/bin/env python3
"""Visualize shared donor objects on one positive and one negative target.

The figure intentionally has no titles or labels. Panel order is recorded in
JSON metadata. The same donor A is used for CP1, CP2, and R2. Donor B is used
for R4 because the R4 larger-donor policy requires a different source scale.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

from misc.plot_cp_variants_4x2 import boxes_from_labels, load_dataset, make_labels
from project_ultralytics.copy_paste import ObjectRecord, SmallObjectCopyPaste
from project_ultralytics.negative_canvas_copy_paste import NegativeCanvasCopyPaste

ROOT = Path(__file__).resolve().parents[1]


class FixedChoiceRNG:
    def __init__(self, seed: int, fixed_record: ObjectRecord):
        self.rng = random.Random(seed)
        self.fixed_record = fixed_record

    def choice(self, sequence):
        values = list(sequence)
        if any(item is self.fixed_record for item in values):
            return self.fixed_record
        return self.rng.choice(values)

    def random(self):
        return self.rng.random()

    def randrange(self, *args):
        return self.rng.randrange(*args)

    def randint(self, *args):
        return self.rng.randint(*args)

    def uniform(self, *args):
        return self.rng.uniform(*args)

    def choices(self, *args, **kwargs):
        return self.rng.choices(*args, **kwargs)


class FixedNegativeCanvas(NegativeCanvasCopyPaste):
    def __init__(self, fixed_record: ObjectRecord, *args, **kwargs):
        self.fixed_record = fixed_record
        super().__init__(*args, **kwargs)

    def _choose_donor(self, target_size: float, labels=None):
        return self.fixed_record, self._effective_source_size(self.fixed_record)


def image_for(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def crop_with_margin(image: np.ndarray, box, margin: float = 1.2):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max((x2 - x1) * margin, 5.0)
    pad_y = max((y2 - y1) * margin, 5.0)
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(w, int(np.ceil(x2 + pad_x)))
    bottom = min(h, int(np.ceil(y2 + pad_y)))
    return image[top:bottom, left:right].copy(), (left, top, right, bottom)


def draw(ax, image: np.ndarray, boxes=None, new_boxes=None):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="#22a06b", linewidth=2.0))
    if new_boxes is not None:
        for x1, y1, x2, y2 in new_boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="#ff3b30", linewidth=2.0))
    ax.axis("off")


def donor_panel(ax, donor_image, donor_box):
    draw(ax, donor_image, new_boxes=[donor_box])


def donor_crop_panel(ax, donor_image, donor_box):
    crop, crop_box = crop_with_margin(donor_image, donor_box)
    local = np.asarray([[donor_box[0] - crop_box[0], donor_box[1] - crop_box[1],
                         donor_box[2] - crop_box[0], donor_box[3] - crop_box[1]]], dtype=np.float32)
    draw(ax, crop, new_boxes=local)
    return crop, crop_box


def fixed_cp(dataset, target_index: int, donor: ObjectRecord, copies: int, seed: int):
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


def fixed_negative(dataset, target_index: int, donor: ObjectRecord, seed: int, degradation: str):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/LevirShipData")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/reports/augmentation_examples/levir_shared_donor_positive_negative.png",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "docs/reports/augmentation_examples/levir_shared_donor_positive_negative.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    positive_indices = [i for i, label in enumerate(dataset.labels) if len(label["bboxes"])]
    negative_indices = [i for i, label in enumerate(dataset.labels) if not len(label["bboxes"])]
    if not positive_indices or not negative_indices:
        raise RuntimeError("Need both positive and negative target images")

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
    if len(pool_builder.object_pool) < 2:
        raise RuntimeError("Need at least two donor objects")
    donor_a = pool_builder.object_pool[0]
    donor_b = pool_builder.object_pool[1]

    donor_a_path = Path(dataset.im_files[donor_a.image_index])
    donor_b_path = Path(dataset.im_files[donor_b.image_index])
    donor_a_image = image_for(donor_a_path)
    donor_b_image = image_for(donor_b_path)
    donor_a_box = tuple(float(value) for value in donor_a.bbox_xyxy)
    donor_b_box = tuple(float(value) for value in donor_b.bbox_xyxy)

    cp1, positive_before, cp1_boxes = fixed_cp(dataset, positive_index, donor_a, 1, args.seed)
    cp2, _, cp2_boxes = fixed_cp(dataset, positive_index, donor_a, 2, args.seed + 1)
    r2, negative_before, r2_boxes = fixed_negative(dataset, negative_index, donor_a, args.seed + 2, "none")
    r4, _, r4_boxes = fixed_negative(dataset, negative_index, donor_b, args.seed + 3, "weak_blur")

    fig, axes = plt.subplots(3, 5, figsize=(17, 10), dpi=180, squeeze=False)
    # Row 1: donor A, donor A crop, donor B, donor B crop, blank.
    donor_panel(axes[0, 0], donor_a_image, donor_a_box)
    donor_crop_panel(axes[0, 1], donor_a_image, donor_a_box)
    donor_panel(axes[0, 2], donor_b_image, donor_b_box)
    donor_crop_panel(axes[0, 3], donor_b_image, donor_b_box)
    axes[0, 4].axis("off")
    for source_axis, zoom_axis, box in ((axes[0, 0], axes[0, 1], donor_a_box),
                                        (axes[0, 2], axes[0, 3], donor_b_box)):
        fig.add_artist(ConnectionPatch(
            xyA=(box[2], box[1]), coordsA=source_axis.transData,
            xyB=(0, 1), coordsB=zoom_axis.transAxes,
            arrowstyle="-", linewidth=1.1, linestyle="--", color="#ff3b30",
        ))
        fig.add_artist(ConnectionPatch(
            xyA=(box[2], box[3]), coordsA=source_axis.transData,
            xyB=(0, 0), coordsB=zoom_axis.transAxes,
            arrowstyle="-", linewidth=1.1, linestyle="--", color="#ff3b30",
        ))

    # Row 2: one positive target, followed by CP1 and CP2.
    draw(axes[1, 0], image_for(dataset.im_files[positive_index]), boxes=positive_before)
    draw(axes[1, 1], cp1["img"], boxes=positive_before, new_boxes=cp1_boxes[len(positive_before):])
    draw(axes[1, 2], cp2["img"], boxes=positive_before, new_boxes=cp2_boxes[len(positive_before):])
    axes[1, 3].axis("off")
    axes[1, 4].axis("off")

    # Row 3: one negative target, followed by R2 and R4.
    draw(axes[2, 0], image_for(dataset.im_files[negative_index]), boxes=negative_before)
    draw(axes[2, 1], r2["img"], boxes=negative_before, new_boxes=r2_boxes[len(negative_before):])
    draw(axes[2, 2], r4["img"], boxes=negative_before, new_boxes=r4_boxes[len(negative_before):])
    axes[2, 3].axis("off")
    axes[2, 4].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.06, wspace=0.03)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "title_free": True,
        "seed": args.seed,
        "donor_a": {"image": str(donor_a_path), "image_index": donor_a.image_index, "bbox_xyxy": list(donor_a_box)},
        "donor_b": {"image": str(donor_b_path), "image_index": donor_b.image_index, "bbox_xyxy": list(donor_b_box)},
        "positive_target": {"image": dataset.im_files[positive_index], "index": positive_index},
        "negative_target": {"image": dataset.im_files[negative_index], "index": negative_index},
        "panel_order": {
            "row_1": ["donor_a_full", "donor_a_zoom", "donor_b_full", "donor_b_zoom", "blank"],
            "row_2": ["positive_original", "positive_cp1", "positive_cp2", "blank", "blank"],
            "row_3": ["negative_original", "negative_r2", "negative_r4", "blank", "blank"],
        },
        "note": "Donor A is reused for CP1, CP2, and R2. Donor B is used for R4 because its larger-donor policy requires a different source scale.",
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    print(f"Saved {args.metadata}")


if __name__ == "__main__":
    main()
