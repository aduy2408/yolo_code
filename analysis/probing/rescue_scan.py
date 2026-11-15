"""Full-test rescue scan for baseline vs OACP and context swaps.

Uses the declared COCO test manifest, matches predictions to GT boxes, and
renders only cases selected by fixed rules. It does not rank images by visual
appeal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import load_yolo


def box_iou(a: np.ndarray, b: list[float]) -> float:
    x1, y1, x2, y2 = a.tolist()
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-8)


def match(result: Any, gt: list[float], threshold: float) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return {"hit": False, "iou": 0.0, "score": 0.0, "box": None}
    coords = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    ious = [box_iou(coord, gt) for coord in coords]
    index = int(np.argmax(ious))
    return {
        "hit": bool(ious[index] >= threshold),
        "iou": float(ious[index]),
        "score": float(scores[index]),
        "box": coords[index].round(2).tolist(),
    }


def clean_background(image: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    output = image.copy()
    fill = np.median(image.reshape(-1, 3), axis=0).astype(image.dtype)
    for x1, y1, x2, y2 in boxes:
        output[max(0, int(y1)) : min(image.shape[0], int(y2)), max(0, int(x1)) : min(image.shape[1], int(x2))] = fill
    return output


def context_swap(image: np.ndarray, donor: np.ndarray, gt: list[float], donor_boxes: list[list[float]], margin: int) -> np.ndarray:
    height, width = image.shape[:2]
    background = clean_background(donor, donor_boxes)
    x1, y1, x2, y2 = [int(value) for value in gt]
    cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
    cx2, cy2 = min(width, x2 + margin), min(height, y2 + margin)
    output = background.copy()
    output[cy1:cy2, cx1:cx2] = image[cy1:cy2, cx1:cx2]
    return output


def draw(image: np.ndarray, gt: list[float], baseline: dict[str, Any], oacp: dict[str, Any], title: str) -> np.ndarray:
    output = image.copy()
    x1, y1, x2, y2 = [int(value) for value in gt]
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for record, color in ((baseline, (0, 0, 255)), (oacp, (255, 0, 0))):
        if record["box"] is not None:
            bx1, by1, bx2, by2 = [int(value) for value in record["box"]]
            cv2.rectangle(output, (bx1, by1), (bx2, by2), color, 1)
    label = f"{title} | B {baseline['score']:.2f}/{baseline['iou']:.2f} | O {oacp['score']:.2f}/{oacp['iou']:.2f}"
    cv2.putText(output, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def render_case(record: dict[str, Any], output: Path) -> None:
    original = cv2.imread(record["image"])
    swapped = cv2.imread(record["swapped_image"])
    if original is None or swapped is None:
        return
    gt = record["gt"]
    panels = [
        draw(original, gt, record["baseline_original"], record["oacp_original"], "original"),
        draw(swapped, gt, record["baseline_swapped"], record["oacp_swapped"], "context swap"),
    ]
    canvas = np.concatenate(panels, axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oacp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--swap-margin", type=int, default=2)
    parser.add_argument("--max-render", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.test_json.read_text())
    images = {item["id"]: item for item in manifest["images"]}
    annotations: dict[int, list[list[float]]] = {}
    for item in manifest["annotations"]:
        annotations.setdefault(item["image_id"], []).append([float(item["bbox"][0]), float(item["bbox"][1]), float(item["bbox"][0] + item["bbox"][2]), float(item["bbox"][1] + item["bbox"][3])])
    paths = [args.data_root / item["file_name"] for item in images.values()]
    paths = [path if path.is_file() else Path("datasets/LevirShipData/All Images") / path.name for path in paths]
    image_ids = list(images)
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()][:5]
        raise FileNotFoundError(f"Missing test images: {missing}")
    baseline = load_yolo(args.baseline, args.device)
    oacp = load_yolo(args.oacp, args.device)
    original_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    swap_rescues: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_count = 0
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        batch_ids = image_ids[start : start + args.batch_size]
        frames = [cv2.imread(str(path)) for path in batch_paths]
        if any(frame is None for frame in frames):
            raise FileNotFoundError(f"Could not decode batch starting at {start}")
        base_results = baseline.predict(frames, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, batch=args.batch_size)
        oacp_results = oacp.predict(frames, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, batch=args.batch_size)
        swapped_images: list[np.ndarray] = []
        swap_rows: list[dict[str, Any]] = []
        for local_index, image_id in enumerate(batch_ids):
            for object_id, gt in enumerate(annotations.get(image_id, [])):
                object_count += 1
                row = {
                    "image": str(batch_paths[local_index]), "image_id": image_id, "object_id": object_id, "gt": gt,
                    "baseline_original": match(base_results[local_index], gt, args.match_iou),
                    "oacp_original": match(oacp_results[local_index], gt, args.match_iou),
                }
                donor_index = (start + local_index + 1) % len(image_ids)
                donor_id = image_ids[donor_index]
                donor_path = paths[donor_index]
                donor = cv2.imread(str(donor_path))
                swapped_images.append(context_swap(frames[local_index], donor, gt, annotations.get(donor_id, []), args.swap_margin))
                swap_rows.append(row | {"donor": str(donor_path)})
        if not swapped_images:
            continue
        base_swapped = baseline.predict(swapped_images, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, batch=args.batch_size)
        oacp_swapped = oacp.predict(swapped_images, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, batch=args.batch_size)
        for local_index, row in enumerate(swap_rows):
            row["baseline_swapped"] = match(base_swapped[local_index], row["gt"], args.match_iou)
            row["oacp_swapped"] = match(oacp_swapped[local_index], row["gt"], args.match_iou)
            if not row["baseline_original"]["hit"] and row["oacp_original"]["hit"]:
                rescue_rows.append(row)
            if row["baseline_original"]["hit"] and row["oacp_original"]["hit"] and not row["baseline_swapped"]["hit"] and row["oacp_swapped"]["hit"]:
                swap_rescues.append(row)
        print(f"scanned {min(start + args.batch_size, len(paths))}/{len(paths)} images", flush=True)
    rescue_rows.sort(key=lambda row: row["oacp_original"]["iou"], reverse=True)
    swap_rescues.sort(key=lambda row: row["oacp_swapped"]["iou"], reverse=True)
    for name, rows in (("baseline_miss_oacp_rescue", rescue_rows), ("context_swap_oacp_rescue", swap_rescues)):
        selected = rows[: args.max_render]
        for index, row in enumerate(selected):
            image = cv2.imread(row["image"])
            donor = cv2.imread(row["donor"])
            swapped = context_swap(image, donor, row["gt"], [], args.swap_margin)
            swapped_path = args.output_dir / "render_inputs" / name / f"{index:03d}.png"
            swapped_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(swapped_path), swapped)
            row["swapped_image"] = str(swapped_path)
            render_case(row, args.output_dir / name / f"{index:03d}.jpg")
    summary = {
        "protocol": {"test_json": str(args.test_json), "image_count": len(paths), "object_count": object_count, "match_iou": args.match_iou, "conf": args.conf, "imgsz": args.imgsz, "swap_margin": args.swap_margin, "batch_size": args.batch_size},
        "baseline_miss_oacp_rescue_count": len(rescue_rows),
        "context_swap_oacp_rescue_count": len(swap_rescues),
        "baseline_miss_oacp_rescue": rescue_rows,
        "context_swap_oacp_rescue": swap_rescues,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("protocol", "baseline_miss_oacp_rescue_count", "context_swap_oacp_rescue_count")}, indent=2))


if __name__ == "__main__":
    main()
