"""Generate degraded images for LEVIR-Ship split (DRENet requirement).

This keeps the same degradation rule as DegradeGenerate.py, but replaces the
per-pixel Python mean with OpenCV box filters and multiprocessing.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

SIZE = 512  # DRENet hardcodes 512

def degrade_one(args):
    img_path, lbl_path, dst_path = args
    if dst_path.exists():
        return
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    if h != SIZE or w != SIZE:
        img = cv2.resize(img, (SIZE, SIZE))

    if lbl_path.exists():
        label = np.loadtxt(str(lbl_path), ndmin=2)
        if label.size == 0:
            label = np.empty((0, 5))
    else:
        label = np.empty((0, 5))

    if label.shape[0] == 0:
        # No ships → simple blur
        dst = cv2.blur(img, (20, 20))
    else:
        # Object-aware blur with the same min-distance cap and crop-window mean
        # behavior as DegradeGenerate.py.
        centers_xy = label[:, 1:3] * SIZE  # cx, cy in pixels
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]  # (H,W)
        min_dist2 = np.full((SIZE, SIZE), 130 * 130, dtype=np.float32)
        for cx, cy in centers_xy:
            dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
            np.minimum(min_dist2, dist2, out=min_dist2)
        min_dist = np.sqrt(min_dist2)  # (H,W)
        box = (1.03 ** min_dist).astype(int) // 2  # (H,W)
        dst = img.copy().astype(np.float32)
        integral = cv2.integral(img.astype(np.float32), sdepth=cv2.CV_64F)
        unique_sizes = np.unique(box)
        for b in unique_sizes:
            if b == 0:
                continue
            mask = (box == b)
            y0 = np.maximum(ys - b, 0)
            y1 = np.minimum(ys + b + 1, SIZE)
            x0 = np.maximum(xs - b, 0)
            x1 = np.minimum(xs + b + 1, SIZE)
            area = ((y1 - y0) * (x1 - x0))[..., None]
            sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            dst[mask] = (sums / area)[mask]
        dst = dst.astype(np.uint8)

    cv2.imwrite(str(dst_path), dst)


def generate_for_split(split_dir: Path, split: str, workers: int = 8):
    img_dir = split_dir / "images" / split
    lbl_dir = split_dir / "labels" / split
    deg_dir = split_dir / "degrade" / split
    deg_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png"))
    tasks = [
        (p, lbl_dir / f"{p.stem}.txt", deg_dir / p.name)
        for p in img_paths
    ]
    already = sum(1 for *_, d in tasks if d.exists())
    print(f"  [{split}] {len(tasks)} images, {already} already done")
    if already == len(tasks):
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(degrade_one, tasks), total=len(tasks), desc=f"  {split}"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", type=Path,
                   default=Path("/mnt/data/varroa/yolo_related/datasets/levir_ship_yolo_seed42"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    for split in ("train", "val", "test"):
        print(f"\nGenerating degrade/{split}...")
        generate_for_split(args.split_dir, split, args.workers)
    print("\nDone.")
