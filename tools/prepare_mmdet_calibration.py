#!/usr/bin/env python3
"""Preprocess LEVIR-Ship images into deterministic TensorRT calibration NPZ batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Image directory or file list root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=(800, 800), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--mean", type=float, nargs=3, default=(123.675, 116.28, 103.53))
    parser.add_argument("--std", type=float, nargs=3, default=(58.395, 57.12, 57.375))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    if root.is_file():
        paths = [Path(line.strip()) for line in root.read_text().splitlines() if line.strip()]
    else:
        paths = [path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    return sorted(path.resolve() for path in paths)


def preprocess(path: Path, image_size: tuple[int, int], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    array = image.astype(np.float32) / 255.0
    array = (array - mean) / std
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def main() -> None:
    args = parse_args()
    import numpy as np
    if args.batch_size < 1 or (args.max_images is not None and args.max_images < 1):
        raise ValueError("batch size and max images must be positive")
    paths = image_paths(args.images)
    rng = np.random.default_rng(args.seed)
    if args.max_images is not None and len(paths) > args.max_images:
        paths = sorted(rng.choice(paths, size=args.max_images, replace=False).tolist())
    if not paths:
        raise FileNotFoundError(f"No images found under {args.images}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean = np.asarray(args.mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(args.std, dtype=np.float32).reshape(1, 3, 1, 1)
    samples = [preprocess(path, tuple(args.image_size), mean.reshape(3), std.reshape(3)) for path in paths]
    batches = []
    for start in range(0, len(samples), args.batch_size):
        current = samples[start : start + args.batch_size]
        valid_count = len(current)
        if valid_count < args.batch_size:
            current = current + [current[-1]] * (args.batch_size - valid_count)
        batch = np.stack(current, axis=0).astype(np.float32, copy=False)
        target = args.output_dir / f"batch_{len(batches):05d}.npz"
        np.savez(target, input=batch)
        batches.append(
            {
                "file": target.name,
                "count": valid_count,
                "sources": [str(path) for path in paths[start : start + valid_count]],
            }
        )

    manifest = {
        "format": "npz",
        "key": "input",
        "dtype": "float32",
        "shape": [args.batch_size, 3, args.image_size[0], args.image_size[1]],
        "mean_rgb": list(args.mean),
        "std_rgb": list(args.std),
        "resize": "stretch",
        "seed": args.seed,
        "images": len(paths),
        "batches": batches,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(paths)} images in {len(batches)} NPZ batches to {args.output_dir}")


if __name__ == "__main__":
    main()
