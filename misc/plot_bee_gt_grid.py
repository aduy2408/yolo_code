#!/usr/bin/env python3
"""Plot a grid of bee images with GT bounding boxes overlaid."""

from __future__ import annotations

import os
import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


SPLITS = ("train", "val", "test")
GT_SOURCES = ("gt_one", "gt_filtered")


@dataclass(frozen=True)
class GtSample:
    rel_image_path: str
    class_token: str
    boxes: list[list[float]]


def resolve_repo_root(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    missing = [split for split in SPLITS if not (root_path / split / "videos").is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing split video folders under {root_path}: {', '.join(missing)}")
    return root_path


def parse_gt_csv(csv_path: Path) -> list[GtSample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    samples: list[GtSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        for row in reader:
            row = [item for item in row if item]
            if len(row) < 2:
                continue
            rel_image_path = row[0]
            class_token = row[1]
            values = [float(v) for v in row[2:]]
            boxes = [values[i : i + 4] for i in range(0, len(values) - 3, 4)]
            samples.append(GtSample(rel_image_path=rel_image_path, class_token=class_token, boxes=boxes))
    return samples


def pick_samples(samples: list[GtSample], *, count: int, seed: int, only_positive: bool) -> list[GtSample]:
    filtered = [sample for sample in samples if (not only_positive or sample.boxes)]
    if not filtered:
        raise ValueError("No samples available after filtering.")

    rng = random.Random(seed)
    if len(filtered) <= count:
        return filtered
    return rng.sample(filtered, count)


def image_path_for(root_path: Path, split: str, rel_image_path: str) -> Path:
    return root_path / split / Path(rel_image_path)


def draw_sample(ax, image_path: Path, sample: GtSample) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        ax.imshow(image)

    for box in sample.boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        left = min(x1, x2)
        top = min(y1, y2)
        box_w = abs(x2 - x1)
        box_h = abs(y2 - y1)
        ax.add_patch(
            Rectangle(
                (left, top),
                box_w,
                box_h,
                fill=False,
                linewidth=2,
                edgecolor="lime",
            )
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()


def parse_grid(grid: str) -> tuple[int, int]:
    if "x" not in grid:
        raise ValueError("--grid must be in the form ROWSxCOLS, e.g. 3x4")
    rows_s, cols_s = grid.lower().split("x", 1)
    rows = int(rows_s)
    cols = int(cols_s)
    if rows < 1 or cols < 1:
        raise ValueError("--grid must have positive dimensions")
    return rows, cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a grid of bee images with GT boxes.")
    parser.add_argument("--root", default="data", help="Dataset root containing train/val/test folders.")
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--gt-source", default="gt_filtered", choices=GT_SOURCES)
    parser.add_argument("--grid", default="3x4", help="Grid size as ROWSxCOLS, e.g. 3x4 or 3x3.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Include negative rows too. Default is positives only, which is better for bee plots.",
    )
    parser.add_argument("--save", default="", help="Optional output file path for the figure.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = resolve_repo_root(args.root)
    rows, cols = parse_grid(args.grid)
    total = rows * cols

    csv_path = root_path / args.split / f"{args.gt_source}.csv"
    samples = parse_gt_csv(csv_path)
    selected = pick_samples(samples, count=total, seed=args.seed, only_positive=not args.all_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.2))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, sample in zip(axes_list, selected):
        image_path = image_path_for(root_path, args.split, sample.rel_image_path)
        if not image_path.exists():
            ax.set_axis_off()
            ax.set_title(f"missing: {image_path.name}", fontsize=9)
            continue
        draw_sample(ax, image_path, sample)

    for ax in axes_list[len(selected) :]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.0005, hspace=0.02)

    if args.save:
        out_path = Path(args.save).expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
