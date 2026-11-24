#!/usr/bin/env python3
"""Combine two rows from Pic 1 and two rows from Pic 2.

The top half keeps Pic 1 rows CP1 and CP2. The bottom half keeps Pic 2 rows
Negative Canvas R2 and Negative Canvas R4, preserving all four variants in a
shorter combined figure.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_PIC1 = ROOT / "docs/reports/augmentation_examples/levir_cp_section1_donor_zoom.png"
INPUT_PIC2 = ROOT / "docs/reports/augmentation_examples/levir_cp_section2_original_after.png"
OUTPUT = ROOT / "docs/reports/augmentation_examples/levir_cp_compact_combined_2rows.png"


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    pic1 = cv2.imread(str(INPUT_PIC1), cv2.IMREAD_COLOR)
    pic2 = cv2.imread(str(INPUT_PIC2), cv2.IMREAD_COLOR)
    if pic1 is None or pic2 is None:
        raise RuntimeError("Could not read the existing section figures")

    # The source figures have four evenly spaced rows. Keep two rows from each.
    pic1_top = pic1[: pic1.shape[0] // 2].copy()
    pic2_bottom = pic2[pic2.shape[0] // 2 :].copy()
    width = max(pic1_top.shape[1], pic2_bottom.shape[1])
    pic1_top = fit_width(pic1_top, width)
    pic2_bottom = fit_width(pic2_bottom, width)

    separator_height = 74
    separator = np.full((separator_height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        separator,
        "SECTION 2 / PIC 2: ORIGINAL TARGET -> AFTER DONOR INSERTION",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    output = np.vstack([pic1_top, separator, pic2_bottom])
    row_height_top = pic1_top.shape[0] // 2
    row_height_bottom = pic2_bottom.shape[0] // 2
    labels = [
        ("CP1", 100),
        ("CP2", row_height_top + 100),
        ("Negative Canvas R2", pic1_top.shape[0] + separator_height + 100),
        ("Negative Canvas R4", pic1_top.shape[0] + separator_height + row_height_bottom + 100),
    ]
    for label, y in labels:
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            output,
            (14, y - text_height - baseline - 8),
            (24 + text_width, y + 8),
            (255, 255, 255),
            thickness=-1,
        )
        cv2.putText(
            output,
            label,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), output)
    print(f"Saved {OUTPUT}")
    print(f"Dimensions: {output.shape[1]}x{output.shape[0]}")
    print("Top rows: CP1, CP2 from Section 1 / Pic 1")
    print("Bottom rows: Negative Canvas R2, Negative Canvas R4 from Section 2 / Pic 2")


if __name__ == "__main__":
    main()
