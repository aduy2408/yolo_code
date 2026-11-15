"""Object-preserving context counterfactuals for paired detector checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import load_yolo


def gt_boxes(label_path: Path, shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    height, width = shape
    boxes = []
    for line in label_path.read_text().splitlines():
        values = [float(item) for item in line.split()]
        if len(values) < 5:
            continue
        _, cx, cy, bw, bh = values[:5]
        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)
        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)
        boxes.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    return boxes


def expand(box: tuple[int, int, int, int], shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = box
    return max(0, x1 - margin), max(0, y1 - margin), min(width, x2 + margin), min(height, y2 + margin)


def variants(image: np.ndarray, box: tuple[int, int, int, int], donor: np.ndarray, margin: int) -> dict[str, np.ndarray]:
    x1, y1, x2, y2 = box
    protected = expand(box, image.shape[:2], margin)
    px1, py1, px2, py2 = protected
    fill = np.median(image.reshape(-1, 3), axis=0).astype(image.dtype)
    object_only = np.full_like(image, fill)
    object_only[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    context_only = image.copy()
    context_only[y1:y2, x1:x2] = fill
    removed = context_only.copy()
    removed[py1:py2, px1:px2] = fill
    degraded = image.copy()
    blurred = cv2.GaussianBlur(image, (0, 0), 12)
    degraded[py1:py2, px1:px2] = image[py1:py2, px1:px2]
    degraded[:py1] = blurred[:py1]
    degraded[py2:] = blurred[py2:]
    degraded[py1:py2, :px1] = blurred[py1:py2, :px1]
    degraded[py1:py2, px2:] = blurred[py1:py2, px2:]
    swapped = donor.copy()
    swapped[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return {"original": image, "object_only": object_only, "context_only": context_only, "object_removed": removed, "context_degraded": degraded, "background_swapped": swapped}


def iou(a: np.ndarray, b: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, x2 - x1) * max(0, y2 - y1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-8)


def matched_score(model: Any, image: np.ndarray, target: tuple[int, int, int, int], device: str, conf: float) -> tuple[float, np.ndarray | None]:
    result = model.predict(image, imgsz=max(image.shape[:2]), device=device, conf=conf, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0, None
    coords = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    index = int(np.argmax([iou(coords[i], target) for i in range(len(coords))]))
    overlap = iou(coords[index], target)
    return (float(scores[index]), coords[index]) if overlap >= 0.1 else (0.0, None)


def occlusion(model: Any, image: np.ndarray, target: tuple[int, int, int, int], device: str, conf: float, grid: int) -> np.ndarray:
    height, width = image.shape[:2]
    baseline, _ = matched_score(model, image, target, device, conf)
    heat = np.zeros((grid, grid), dtype=np.float32)
    for row in range(grid):
        for col in range(grid):
            y1, y2 = row * height // grid, (row + 1) * height // grid
            x1, x2 = col * width // grid, (col + 1) * width // grid
            masked = image.copy()
            fill = np.median(image.reshape(-1, 3), axis=0).astype(image.dtype)
            masked[y1:y2, x1:x2] = fill
            score, _ = matched_score(model, masked, target, device, conf)
            heat[row, col] = max(0.0, baseline - score)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_CUBIC)
    return heat / max(float(heat.max()), 1e-8)


def draw(image: np.ndarray, title: str, target: tuple[int, int, int, int], score: float | None = None) -> np.ndarray:
    output = image.copy()
    x1, y1, x2, y2 = target
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 1)
    label = title if score is None else f"{title}  matched={score:.2f}"
    cv2.putText(output, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def heat_overlay(image: np.ndarray, heat: np.ndarray, title: str, target: tuple[int, int, int, int]) -> np.ndarray:
    color = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_JET)
    return draw(cv2.addWeighted(image, 0.5, color, 0.5, 0), title, target)


def montage(images: list[np.ndarray], columns: int = 3) -> np.ndarray:
    h, w = images[0].shape[:2]
    rows = (len(images) + columns - 1) // columns
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    return canvas


def run(args: argparse.Namespace) -> None:
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Cannot read image: {args.image}")
    labels = args.labels or args.image.with_suffix(".txt")
    boxes = gt_boxes(labels, image.shape[:2])
    if not boxes:
        raise ValueError(f"No YOLO boxes found in {labels}")
    donor = cv2.imread(str(args.donor)) if args.donor else image.copy()
    baseline = load_yolo(args.baseline, args.device)
    oacp = load_yolo(args.oacp, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for object_id, target in enumerate(boxes):
        cases = variants(image, target, donor, args.protection)
        panels = []
        for name, case in cases.items():
            score_b, _ = matched_score(baseline, case, target, args.device, args.confidence)
            score_o, _ = matched_score(oacp, case, target, args.device, args.confidence)
            panels.extend([draw(case, f"{name} | B={score_b:.2f}", target), draw(case, f"{name} | O={score_o:.2f}", target)])
        panels.extend([
            heat_overlay(image, occlusion(baseline, image, target, args.device, args.confidence, args.grid), "baseline object-conditioned occlusion", target),
            heat_overlay(image, occlusion(oacp, image, target, args.device, args.confidence, args.grid), "OACP object-conditioned occlusion", target),
        ])
        output = args.output_dir / f"{args.image.stem}_object_{object_id}.jpg"
        cv2.imwrite(str(output), montage(panels, columns=4))
        original_b, _ = matched_score(baseline, image, target, args.device, args.confidence)
        original_o, _ = matched_score(oacp, image, target, args.device, args.confidence)
        degraded_b, _ = matched_score(baseline, cases["context_degraded"], target, args.device, args.confidence)
        degraded_o, _ = matched_score(oacp, cases["context_degraded"], target, args.device, args.confidence)
        summary.append({"object_id": object_id, "output": str(output), "baseline_context_drop": original_b - degraded_b, "oacp_context_drop": original_o - degraded_o})
        print(output)
    (args.output_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oacp", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/oacp_context_counterfactuals"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--protection", type=int, default=12)
    parser.add_argument("--grid", type=int, default=6)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
