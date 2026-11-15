#!/usr/bin/env python3
"""Plot LevirShip object-size frequency in 4-pixel bins."""
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


def read_sizes():
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
            _, _, _, box_width, box_height = map(float, fields[:5])
            values.append(math.sqrt((box_width * width) * (box_height * height)))
    return values


def main():
    values = read_sizes()
    lower = 4
    upper = 36
    edges = list(range(lower, upper + 1, 4))
    counts = [0] * (len(edges) - 1)
    below_lower = 0
    above_upper = 0
    for value in values:
        if value < lower:
            below_lower += 1
        elif value >= edges[-1]:
            above_upper += 1
        else:
            index = int((value - lower) // 4)
            counts[index] += 1

    labels = [f"{left}–{right}" for left, right in zip(edges[:-1], edges[1:])]
    csv_path = OUT / "levirship_object_size_4px_bins.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", "count", "small_cutoff_in_bin"])
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            writer.writerow([left, right, count, left <= 32 < right])
        writer.writerow(["<4", "<4", below_lower, False])
        writer.writerow([f">={edges[-1]}", f">={edges[-1]}", above_upper, False])

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=220)
    bars = ax.bar(
        edges[:-1],
        counts,
        width=4,
        align="edge",
        color=["#ef4444" if left in {4, 8, 24, 28} else "#4f86c6" for left in edges[:-1]],
        edgecolor="#12263a",
        linewidth=0.8,
    )
    for bar, count in zip(bars, counts):
        if count:
            ax.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom", fontsize=8)
    ax.axvline(32, color="#dc2626", linestyle="--", linewidth=2, label="cutoff s = 32")
    ax.set_xticks(list(range(4, upper + 1, 4)))
    ax.set_xlabel("Object size (pixels)")
    ax.set_ylabel("Object frequency")
    ax.set_title("LevirShip object-size frequency with fine copy-paste bins")
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = OUT / "levirship_object_size_4px_bins.png"
    fig.savefig(png_path, facecolor="white")
    plt.close(fig)
    print(f"objects={len(values)}")
    print(f"covered={sum(counts)} below4={below_lower} above_or_equal_{edges[-1]}={above_upper}")
    print(f"png={png_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
