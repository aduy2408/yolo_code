"""Build cheap image-context descriptors for Context-Contrast Mosaic.

Example:
    python tools/build_mosaic_context_cache.py --image-dir path/to/images --output mosaic_context_cache.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def descriptor(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    image_f = image.astype(np.float32) / 255.0
    gx = cv2.Sobel(image_f, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1)
    gradient = np.sqrt(gx * gx + gy * gy)
    histogram, _ = np.histogram(image_f, bins=32, range=(0.0, 1.0), density=True)
    probability = histogram / max(float(histogram.sum()), 1e-12)
    entropy = float(-(probability[probability > 0] * np.log(probability[probability > 0])).sum())
    return np.asarray([image_f.mean(), image_f.std(), gradient.mean(), entropy, np.mean(image_f < 0.05)], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(p for p in args.image_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"no images found under {args.image_dir}")
    descriptors = np.stack([descriptor(path) for path in paths])
    mean = descriptors.mean(axis=0)
    std = descriptors.std(axis=0)
    descriptors = (descriptors - mean) / np.maximum(std, 1e-6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, im_file=np.asarray([str(p) for p in paths]), descriptor=descriptors)
    print(f"wrote {len(paths)} descriptors to {args.output}")


if __name__ == "__main__":
    main()
