#!/usr/bin/env python3
"""Render Copy-Paste previews with explicit donor-image provenance.

Each row shows the same target canvas, the donor image selected by the real
transform, the selected donor object crop, and the transformed output.
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
from matplotlib.patches import Rectangle
import numpy as np

from misc.plot_cp_variants_4x2 import load_dataset, make_labels, boxes_from_labels
from project_ultralytics.copy_paste import SmallObjectCopyPaste
from project_ultralytics.negative_canvas_copy_paste import NegativeCanvasCopyPaste

ROOT = Path(__file__).resolve().parents[1]


class DonorTraceMixin:
    """Capture the exact donor record used by the project transform."""

    def __init__(self, *args, **kwargs):
        self.donor_records = []
        super().__init__(*args, **kwargs)

    def _crop_single(self, record, source_image):
        self.donor_records.append(record)
        return super()._crop_single(record, source_image)


class TraceSmallObjectCopyPaste(DonorTraceMixin, SmallObjectCopyPaste):
    pass


class TraceNegativeCanvasCopyPaste(DonorTraceMixin, NegativeCanvasCopyPaste):
    pass


def transform_for(dataset, variant: str, seed: int):
    rng = random.Random(seed)
    if variant == "cp1":
        return TraceSmallObjectCopyPaste(
            dataset,
            p=1.0,
            unit="single",
            copies=1,
            placement="random",
            max_overlap=0.0,
            allow_empty_target=True,
            allow_same_source=True,
            rng=rng,
        )
    if variant == "cp2":
        return TraceSmallObjectCopyPaste(
            dataset,
            p=1.0,
            unit="single",
            copies=2,
            placement="random",
            max_overlap=0.0,
            allow_empty_target=True,
            allow_same_source=True,
            rng=rng,
        )
    if variant == "negative_canvas_r2":
        return TraceNegativeCanvasCopyPaste(
            dataset,
            p=1.0,
            target_policy="deficit",
            donor_policy="matched",
            target_max_size=20.0,
            degradation="none",
            rng=rng,
            max_trials=100,
        )
    if variant == "negative_canvas_r4":
        return TraceNegativeCanvasCopyPaste(
            dataset,
            p=1.0,
            target_policy="deficit",
            donor_policy="larger",
            target_max_size=20.0,
            degradation="weak_blur",
            blur_sigma=0.5,
            rng=rng,
            max_trials=100,
        )
    raise ValueError(variant)


def image_for(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def crop_with_margin(image: np.ndarray, box: tuple[float, float, float, float], margin: float = 0.8):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max((x2 - x1) * margin, 4.0)
    pad_y = max((y2 - y1) * margin, 4.0)
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(w, int(np.ceil(x2 + pad_x)))
    bottom = min(h, int(np.ceil(y2 + pad_y)))
    return image[top:bottom, left:right].copy(), (left, top, right, bottom)


def show(ax, image: np.ndarray, boxes=None, color="#ef4444", title="", note=""):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.8))
    ax.set_title(title, fontsize=11, weight="bold", pad=5)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7.2, color="white",
                va="bottom", ha="left", bbox={"facecolor": "black", "alpha": 0.68, "pad": 2})
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/LevirShipData")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/reports/augmentation_examples/levir_cp_donor_provenance.png",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "docs/reports/augmentation_examples/levir_cp_donor_provenance.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    negative_indices = [
        index for index, label in enumerate(dataset.labels) if not len(label["bboxes"])
    ]
    if not negative_indices:
        raise RuntimeError("Need at least one negative target canvas")
    target_index = negative_indices[0]
    target_path = Path(dataset.im_files[target_index])
    variants = [
        ("CP1", "cp1"),
        ("CP2", "cp2"),
        ("Negative Canvas R2", "negative_canvas_r2"),
        ("Negative Canvas R4", "negative_canvas_r4"),
    ]

    target_image = image_for(target_path)
    fig, axes = plt.subplots(4, 4, figsize=(17, 15), dpi=180, squeeze=False)
    columns = ["Target canvas", "Donor image", "Donor object crop", "Augmented output"]
    for column, title in enumerate(columns):
        axes[0, column].set_title(title, fontsize=12, weight="bold", pad=7)

    metadata = {
        "target_image": str(target_path),
        "target_index": target_index,
        "seed": args.seed,
        "variants": [],
        "columns": columns,
        "provenance_note": "Donor image and donor crop are captured from the exact ObjectRecord selected by the project transform.",
    }

    for row, (display_name, variant) in enumerate(variants):
        transform = transform_for(dataset, variant, args.seed + row)
        original = make_labels(dataset, target_index)
        result = transform(original)
        if not transform.donor_records:
            raise RuntimeError(f"{variant} did not select a donor")
        record = transform.donor_records[-1]
        donor_path = Path(dataset.im_files[record.image_index])
        donor_image = image_for(donor_path)
        donor_box = tuple(float(value) for value in record.bbox_xyxy)
        donor_crop, crop_box = crop_with_margin(donor_image, donor_box)
        crop_box_local = np.asarray([
            donor_box[0] - crop_box[0],
            donor_box[1] - crop_box[1],
            donor_box[2] - crop_box[0],
            donor_box[3] - crop_box[1],
        ], dtype=np.float32).reshape(1, 4)
        output_boxes = boxes_from_labels(result)

        axes[row, 0].set_ylabel(display_name, fontsize=10, weight="bold", rotation=90, labelpad=12)
        show(axes[row, 0], target_image, title="" if row else columns[0], note=target_path.name[:34])
        show(axes[row, 1], donor_image, boxes=np.asarray([donor_box]), title="" if row else columns[1], note=donor_path.name[:34])
        show(axes[row, 2], donor_crop, boxes=crop_box_local, title="" if row else columns[2], note=f"object from {donor_path.stem[-18:]}")
        show(axes[row, 3], result["img"], boxes=output_boxes, title="" if row else columns[3], note=f"{len(output_boxes)} labels")

        metadata["variants"].append({
            "name": display_name,
            "variant": variant,
            "seed": args.seed + row,
            "donor_image": str(donor_path),
            "donor_image_index": int(record.image_index),
            "donor_bbox_xyxy": list(donor_box),
            "donor_crop_box_xyxy": list(crop_box),
            "target_output_boxes_xyxy": output_boxes.tolist(),
        })

    fig.subplots_adjust(left=0.055, right=0.995, top=0.96, bottom=0.02, hspace=0.10, wspace=0.06)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    print(f"Saved {args.metadata}")
    for item in metadata["variants"]:
        print(f"{item['name']}: {item['donor_image']}")


if __name__ == "__main__":
    main()
