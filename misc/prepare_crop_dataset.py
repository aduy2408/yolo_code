#!/usr/bin/env python3
"""Build a crop-around-GT Varroa YOLO dataset.

This is a data-level localization experiment: make mites larger in the input
by cropping around each GT box, then resize the crop to YOLO's usual image
size. By default train/val/test are cropped so validation measures the crop
pipeline. Use --no-crop-eval to keep val/test full-image for a distribution
shift sanity check.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from misc.prepare_dataset import (
    CLASS_POLICIES,
    GT_SOURCES,
    SPLITS,
    SampleRecord,
    clamp_xyxy_boxes,
    filter_records,
    flattened_name,
    load_records,
    resolve_repo_root,
    source_image_path,
    yolo_rows_from_boxes,
)


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def intersect_area(a: list[float], b: CropBox) -> float:
    left = max(a[0], b.left)
    top = max(a[1], b.top)
    right = min(a[2], b.right)
    bottom = min(a[3], b.bottom)
    return max(0.0, right - left) * max(0.0, bottom - top)


def crop_box_from_center(cx: float, cy: float, size: int, width: int, height: int) -> CropBox:
    size = min(size, width, height)
    left = round(cx - size * 0.5)
    top = round(cy - size * 0.5)
    left = max(0, min(left, width - size))
    top = max(0, min(top, height - size))
    return CropBox(left=left, top=top, right=left + size, bottom=top + size)


def boxes_for_crop(
    boxes: list[list[float]],
    crop: CropBox,
    *,
    min_visible: float,
) -> list[list[float]]:
    kept: list[list[float]] = []
    for box in boxes:
        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if box_area <= 0:
            continue
        if intersect_area(box, crop) / box_area < min_visible:
            continue
        left = max(box[0], crop.left) - crop.left
        top = max(box[1], crop.top) - crop.top
        right = min(box[2], crop.right) - crop.left
        bottom = min(box[3], crop.bottom) - crop.top
        if right > left and bottom > top:
            kept.append([left, top, right, bottom])
    return kept


def output_stem(root_path: Path, record: SampleRecord) -> str:
    image_path = source_image_path(root_path, record)
    image_dir = root_path / record.source_split / "videos"
    return f"{record.source_split}__{Path(flattened_name(image_path, image_dir)).stem}"


def save_crop(
    image: Image.Image,
    crop: CropBox,
    boxes: list[list[float]],
    image_out: Path,
    label_out: Path,
    *,
    resize: int,
) -> None:
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    cropped = image.crop((crop.left, crop.top, crop.right, crop.bottom)).resize((resize, resize), Image.BILINEAR)
    cropped.save(image_out)
    scale_x = resize / crop.width
    scale_y = resize / crop.height
    scaled = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in boxes]
    label_out.write_text("\n".join(yolo_rows_from_boxes(scaled, resize, resize)), encoding="utf-8")


def random_background_crop(
    rng: random.Random,
    boxes: list[list[float]],
    width: int,
    height: int,
    sizes: list[int],
    *,
    max_overlap: float,
    attempts: int = 30,
) -> CropBox | None:
    for _ in range(attempts):
        size = min(rng.choice(sizes), width, height)
        left = rng.randint(0, max(0, width - size))
        top = rng.randint(0, max(0, height - size))
        crop = CropBox(left, top, left + size, top + size)
        if all(intersect_area(box, crop) / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9) <= max_overlap for box in boxes):
            return crop
    return None


def write_data_yaml(out_dir: Path, counts: dict[str, int], args: argparse.Namespace) -> Path:
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
                f"split_counts: {{train: {counts['train']}, val: {counts['val']}, test: {counts['test']}}}",
                f"crop_sizes: {args.crop_sizes}",
                f"crops_per_gt: {args.crops_per_gt}",
                f"jitter: {args.jitter}",
                f"min_visible: {args.min_visible}",
                f"resize: {args.resize}",
                f"background_per_image: {args.background_per_image}",
                f"crop_eval: {str(args.crop_eval).lower()}",
                f"gt_source: {args.gt_source}",
                f"class_policy: {args.class_policy}",
                f"seed: {args.seed}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare_crop_dataset(args: argparse.Namespace) -> Path:
    root_path = resolve_repo_root(args.root)
    out_path = Path(args.out_dir).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    if out_path.exists() and args.overwrite:
        shutil.rmtree(out_path)

    records = load_records(root_path, args.gt_source)
    records = filter_records(records, only_positives=False, class_policy=args.class_policy)
    by_split = {split: [r for r in records if r.source_split == split] for split in SPLITS}
    rng = random.Random(args.seed)
    counts = {split: 0 for split in SPLITS}
    sizes = [int(x) for x in args.crop_sizes.split(",") if x.strip()]

    for split in SPLITS:
        split_records = by_split[split]
        if args.limit > 0:
            split_records = split_records[: args.limit]
        for record in split_records:
            image_path = source_image_path(root_path, record)
            if not image_path.exists():
                continue
            with Image.open(image_path) as image_raw:
                image = image_raw.convert("RGB")
                width, height = image.size
                boxes = clamp_xyxy_boxes(record.boxes, width, height)
                stem = output_stem(root_path, record)

                crop_this_split = split == "train" or args.crop_eval
                if crop_this_split and boxes:
                    for box_idx, box in enumerate(boxes):
                        box_cx = (box[0] + box[2]) * 0.5
                        box_cy = (box[1] + box[3]) * 0.5
                        for crop_idx in range(args.crops_per_gt):
                            size = rng.choice(sizes)
                            jitter_px = size * args.jitter
                            cx = box_cx + rng.uniform(-jitter_px, jitter_px)
                            cy = box_cy + rng.uniform(-jitter_px, jitter_px)
                            crop = crop_box_from_center(cx, cy, size, width, height)
                            crop_boxes = boxes_for_crop(boxes, crop, min_visible=args.min_visible)
                            if not crop_boxes:
                                continue
                            name = f"{stem}__gt{box_idx:02d}_crop{crop_idx:02d}_s{crop.width}.png"
                            save_crop(
                                image,
                                crop,
                                crop_boxes,
                                out_path / "images" / split / name,
                                out_path / "labels" / split / Path(name).with_suffix(".txt").name,
                                resize=args.resize,
                            )
                            counts[split] += 1

                    for bg_idx in range(args.background_per_image if split == "train" else 0):
                        crop = random_background_crop(
                            rng, boxes, width, height, sizes, max_overlap=args.background_max_overlap
                        )
                        if crop is None:
                            continue
                        name = f"{stem}__bg{bg_idx:02d}_s{crop.width}.png"
                        save_crop(
                            image,
                            crop,
                            [],
                            out_path / "images" / split / name,
                            out_path / "labels" / split / Path(name).with_suffix(".txt").name,
                            resize=args.resize,
                        )
                        counts[split] += 1
                elif not crop_this_split:
                    image_name = f"{stem}{image_path.suffix}"
                    image_out = out_path / "images" / split / image_name
                    label_out = out_path / "labels" / split / Path(image_name).with_suffix(".txt").name
                    image_out.parent.mkdir(parents=True, exist_ok=True)
                    label_out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, image_out)
                    label_out.write_text("\n".join(yolo_rows_from_boxes(boxes, width, height)), encoding="utf-8")
                    counts[split] += 1

    return write_data_yaml(out_path, counts, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Varroa crop-around-GT YOLO dataset.")
    parser.add_argument("--root", default="data", help="Root containing train/val/test folders.")
    parser.add_argument("--out-dir", default="datasets/varroa_yolo_crop_around_gt")
    parser.add_argument("--gt-source", default="gt_one", choices=GT_SOURCES)
    parser.add_argument("--class-policy", default="map-3-to-1", choices=CLASS_POLICIES)
    parser.add_argument("--crop-sizes", default="96,128,160,192", help="Comma-separated square crop sizes.")
    parser.add_argument("--crops-per-gt", type=int, default=3)
    parser.add_argument("--jitter", type=float, default=0.25, help="Center jitter as fraction of crop size.")
    parser.add_argument("--min-visible", type=float, default=0.65, help="Minimum original box area visible in crop.")
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--background-per-image", type=int, default=1)
    parser.add_argument("--background-max-overlap", type=float, default=0.05)
    parser.add_argument(
        "--crop-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop val/test too. Use --no-crop-eval to keep val/test full-image.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max source records per split for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    yaml_path = prepare_crop_dataset(parse_args())
    print(f"Crop YOLO dataset ready: {yaml_path}")
    print(yaml_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
