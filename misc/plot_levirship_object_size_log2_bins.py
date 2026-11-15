#!/usr/bin/env python3
"""Plot LevirShip object-size frequency using copy-paste log2 bins."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "datasets/LevirShipData/All Images"
LABEL_DIR = ROOT / "datasets/LevirShipData/All Annotations"
OUT = ROOT / "docs/reports/augmentation_examples"


def sizes():
    values = []
    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in label_path.read_text().splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            _, _, _, bw, bh = map(float, fields[:5])
            object_width = bw * width
            object_height = bh * height
            values.append(math.sqrt(object_width * object_height))
    return values


def main():
    values = sizes()
    if not values:
        raise RuntimeError("No annotated objects found")
    minimum_power = int(math.floor(math.log2(min(values))))
    maximum_power = int(math.ceil(math.log2(max(values))))
    edges = [2**power for power in range(minimum_power, maximum_power + 2)]
    counts = [0] * (len(edges) - 1)
    for value in values:
        index = min(len(counts) - 1, max(0, int(math.floor(math.log2(value))) - minimum_power)
        )
        counts[index] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "levirship_object_size_log2_bins.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", "count", "small_cutoff_in_bin"])
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            writer.writerow([left, right, count, left <= 20 < right])

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=220)
    lefts = edges[:-1]
    widths = [right - left for left, right in zip(edges[:-1], edges[1:])]
    colors = ["#3b82f6" if left < 20 else "#94a3b8" for left in lefts]
    ax.bar(lefts, counts, width=widths, align="edge", color=colors, edgecolor="#0f172a", linewidth=0.8)
    ax.axvline(20, color="#dc2626", linestyle="--", linewidth=2, label="small-object cutoff s = 20")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**power for power in range(minimum_power, maximum_power + 1)])
    ax.set_xticklabels([str(2**power) for power in range(minimum_power, maximum_power + 1)])
    ax.set_xlabel("Object size s = √(w·h) pixels, log₂ bins")
    ax.set_ylabel("Object frequency")
    ax.set_title("LevirShip object-size frequency using copy-paste bins")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = OUT / "levirship_object_size_log2_bins.png"
    fig.savefig(png_path, facecolor="white")
    plt.close(fig)
    print(f"objects={len(values)}")
    print(f"png={png_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
