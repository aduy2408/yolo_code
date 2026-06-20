#!/usr/bin/env python3
"""Convert the original Varroa bbox dataset to Ultralytics YOLO format.

This module is intentionally standalone. It does not import from
object_detection_related so YOLO experiments can evolve independently.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def resolve_repo_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    missing = [split for split in SPLITS if not (root / split / "videos").is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing split video folders under {root}: {', '.join(missing)}")
    return root


def parse_gt_csv(csv_path: Path) -> dict[str, list[list[float]]]:
    """Parse gt_filtered.csv into a dictionary mapping relative image path to a list of xyxy boxes."""
    if not csv_path.exists():
        return {}

    gt_dict: dict[str, list[list[float]]] = {}
    lines = [line.strip() for line in csv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        rel_img_path = parts[0]
        values = [float(value) for value in parts[2:]]
        boxes: list[list[float]] = []
        for idx in range(0, len(values) - 3, 4):
            boxes.append(values[idx : idx + 4])
        gt_dict[rel_img_path] = boxes
    return gt_dict


def clamp_xyxy_boxes(boxes: list[list[float]], width: int, height: int) -> list[list[float]]:
    clamped: list[list[float]] = []
    for x1, y1, x2, y2 in boxes:
        left, right = sorted((max(0.0, min(float(x1), width)), max(0.0, min(float(x2), width))))
        top, bottom = sorted((max(0.0, min(float(y1), height)), max(0.0, min(float(y2), height))))
        if right > left and bottom > top:
            clamped.append([left, top, right, bottom])
    return clamped


def yolo_rows_from_boxes(boxes: list[list[float]], width: int, height: int) -> list[str]:
    rows: list[str] = []
    for left, top, right, bottom in boxes:
        box_w = right - left
        box_h = bottom - top
        cx = (left + right) * 0.5 / width
        cy = (top + bottom) * 0.5 / height
        rows.append(f"0 {cx:.8f} {cy:.8f} {box_w / width:.8f} {box_h / height:.8f}")
    return rows


def flattened_name(image_path: Path, image_dir: Path) -> str:
    return "__".join(image_path.relative_to(image_dir).parts)


def iter_images(image_dir: Path) -> list[Path]:
    images: list[Path] = []
    for suffix in IMAGE_EXTENSIONS:
        images.extend(image_dir.rglob(f"*{suffix}"))
    return sorted(images)


def write_data_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "varroa.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names: [varroa]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare_dataset(
    root: str | Path = ".",
    out_dir: str | Path = "yolo_related/datasets/varroa_yolo",
    *,
    overwrite_labels: bool = True,
    copy_images: bool = True,
    limit: int = 0,
) -> Path:
    root_path = resolve_repo_root(root)
    out_path = Path(out_dir).expanduser()
    if not out_path.is_absolute():
        out_path = (root_path / out_path).resolve()

    for split in SPLITS:
        image_dir = root_path / split / "videos"
        gt_csv_path = root_path / split / "gt_filtered.csv"
        gt_dict = parse_gt_csv(gt_csv_path)

        yolo_image_dir = out_path / "images" / split
        yolo_label_dir = out_path / "labels" / split
        yolo_image_dir.mkdir(parents=True, exist_ok=True)
        yolo_label_dir.mkdir(parents=True, exist_ok=True)

        images = iter_images(image_dir)
        
        # Chỉ giữ lại các ảnh có tên trong file gt_filtered.csv
        if gt_dict:
            images = [img for img in images if str(img.relative_to(root_path / split)).replace("\\", "/") in gt_dict]

        if limit > 0:
            images = images[:limit]

        for image_path in images:
            dst_name = flattened_name(image_path, image_dir)
            dst_image = yolo_image_dir / dst_name
            dst_label = yolo_label_dir / Path(dst_name).with_suffix(".txt").name

            if copy_images and not dst_image.exists():
                shutil.copy2(image_path, dst_image)

            if overwrite_labels or not dst_label.exists():
                with Image.open(image_path) as image:
                    width, height = image.size
                
                rel_path = str(image_path.relative_to(root_path / split)).replace("\\", "/")
                boxes = clamp_xyxy_boxes(gt_dict.get(rel_path, []), width, height)
                dst_label.write_text("\n".join(yolo_rows_from_boxes(boxes, width, height)), encoding="utf-8")

    return write_data_yaml(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Varroa labels to Ultralytics YOLO format.")
    parser.add_argument("--root", default=".", help="Repo root containing train/val/test folders.")
    parser.add_argument("--out-dir", default="yolo_related/datasets/varroa_yolo")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images per split for smoke tests.")
    parser.add_argument("--no-copy-images", action="store_true", help="Only write labels/YAML; do not copy images.")
    parser.add_argument("--keep-labels", action="store_true", help="Do not overwrite existing YOLO label files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_path = prepare_dataset(
        args.root,
        args.out_dir,
        overwrite_labels=not args.keep_labels,
        copy_images=not args.no_copy_images,
        limit=args.limit,
    )
    print(f"YOLO dataset ready: {yaml_path}")


if __name__ == "__main__":
    main()
