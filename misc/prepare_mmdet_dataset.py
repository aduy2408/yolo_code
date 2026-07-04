#!/usr/bin/env python3
"""Convert the original Varroa bbox dataset to MMDetection COCO format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from prepare_dataset import (
    CLASS_POLICIES,
    GT_SOURCES,
    SPLITS,
    assign_output_splits,
    clamp_xyxy_boxes,
    dedupe_records,
    filter_records,
    load_records,
    output_name_for_record,
    resolve_repo_root,
    source_image_path,
)


def xyxy_to_coco_bbox(box: list[float]) -> list[float]:
    left, top, right, bottom = box
    return [left, top, right - left, bottom - top]


def build_coco_for_split(
    records: list,
    *,
    root_path: Path,
    image_dir: Path,
    copy_images: bool,
    start_image_id: int = 1,
    start_annotation_id: int = 1,
) -> dict:
    images = []
    annotations = []
    image_id = start_image_id
    annotation_id = start_annotation_id

    image_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        source_path = source_image_path(root_path, record)
        file_name = output_name_for_record(root_path, record)
        target_path = image_dir / file_name

        if copy_images:
            shutil.copy2(source_path, target_path)

        with Image.open(source_path) as image:
            width, height = image.size

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for box in clamp_xyxy_boxes(record.boxes, width, height):
            bbox = xyxy_to_coco_bbox(box)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "varroa"}],
    }


def prepare_mmdet_dataset(
    root: str | Path = "..",
    out_dir: str | Path = "datasets/varroa_coco",
    *,
    copy_images: bool = True,
    limit: int = 0,
    gt_source: str = "gt_one",
    only_positives: bool = True,
    class_policy: str = "map-3-to-1",
    seed: int = 42,
) -> Path:
    if gt_source not in GT_SOURCES:
        raise ValueError(f"Unsupported gt_source: {gt_source}")
    if class_policy not in CLASS_POLICIES:
        raise ValueError(f"Unsupported class_policy: {class_policy}")

    root_path = resolve_repo_root(root)
    out_path = Path(out_dir).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    records = load_records(root_path, gt_source)
    records = filter_records(records, only_positives=only_positives, class_policy=class_policy)
    records = dedupe_records(records)
    split_records = assign_output_splits(records, seed=seed)

    annotations_dir = out_path / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        records_for_split = split_records[split]
        if limit > 0:
            records_for_split = records_for_split[:limit]

        image_dir = out_path / "images" / split
        if copy_images and image_dir.exists():
            shutil.rmtree(image_dir)

        coco = build_coco_for_split(
            records_for_split,
            root_path=root_path,
            image_dir=image_dir,
            copy_images=copy_images,
        )

        annotation_path = annotations_dir / f"instances_{split}.json"
        annotation_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Varroa labels to MMDetection COCO format.")
    parser.add_argument("--root", default="..", help="Dataset root containing train/val/test folders.")
    parser.add_argument("--out-dir", default="datasets/varroa_coco")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images per split for smoke tests.")
    parser.add_argument("--no-copy-images", action="store_true", help="Only write COCO JSON; do not copy images.")
    parser.add_argument("--gt-source", default="gt_one", choices=GT_SOURCES)
    parser.add_argument(
        "--only-positives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only positive rows. Use --no-only-positives to keep token 0 rows.",
    )
    parser.add_argument("--class-policy", default="map-3-to-1", choices=CLASS_POLICIES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = prepare_mmdet_dataset(
        args.root,
        args.out_dir,
        copy_images=not args.no_copy_images,
        limit=args.limit,
        gt_source=args.gt_source,
        only_positives=args.only_positives,
        class_policy=args.class_policy,
        seed=args.seed,
    )
    print(f"MMDetection COCO dataset ready: {out_path}")


if __name__ == "__main__":
    main()
