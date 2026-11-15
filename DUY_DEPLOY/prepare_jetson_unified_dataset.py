from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATASETS = {
    "levirship": "levir_ship",
    "tinyperson": "tinyperson",
    "varroa": "varroa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def prepare_dataset(data_root: Path, output_root: Path, name: str, source_name: str) -> Path:
    source = data_root / source_name
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    category_ids: set[int] = set()
    for split in ("val", "test"):
        annotation_path = source / "annotations" / "seed_42" / f"{split}.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_by_id = {int(row["id"]): row for row in payload["images"]}
        annotations: dict[int, list[dict]] = {}
        for row in payload["annotations"]:
            annotations.setdefault(int(row["image_id"]), []).append(row)
            category_ids.add(int(row["category_id"]))
        for image_id, image in image_by_id.items():
            relative = Path(image["file_name"])
            source_image = source / relative
            if not source_image.is_file():
                source_image = source / "images" / "all" / relative.name
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            image_path = destination / "images" / split / relative.name
            label_path = destination / "labels" / split / relative.with_suffix(".txt").name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists():
                os.symlink(source_image, image_path)
            width = float(image["width"])
            height = float(image["height"])
            lines = []
            for annotation in annotations.get(image_id, []):
                x, y, box_width, box_height = map(float, annotation["bbox"])
                cx = (x + box_width / 2.0) / width
                cy = (y + box_height / 2.0) / height
                lines.append(f"{int(annotation['category_id']) - 1} {cx:.10f} {cy:.10f} {box_width / width:.10f} {box_height / height:.10f}")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    yaml_path = destination / f"{name}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {destination}",
                "train: images/val",
                "val: images/val",
                "test: images/test",
                f"nc: {len(category_ids)}",
                "names: [object]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = [prepare_dataset(args.data_root, args.output_root, name, source) for name, source in DATASETS.items()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
