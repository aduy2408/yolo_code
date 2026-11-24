#!/usr/bin/env python3
"""Run tiled detection on a video, with optional temporal ByteTrack association.

The detector sees native-resolution overlapping tiles. Predictions are translated
back to full-frame coordinates, merged with class-aware NMS, then optionally
tracked. Use ``--detect-every 3`` for a speed-oriented run; ``1`` preserves recall.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, stride))
    if values[-1] != length - tile_size:
        values.append(length - tile_size)
    return values


def make_tiles(frame: np.ndarray, tile_size: int, overlap: float) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, round(tile_size * (1.0 - overlap)))
    height, width = frame.shape[:2]
    tiles: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for top in tile_starts(height, tile_size, stride):
        for left in tile_starts(width, tile_size, stride):
            crop = frame[top : min(top + tile_size, height), left : min(left + tile_size, width)]
            pad_bottom = tile_size - crop.shape[0]
            pad_right = tile_size - crop.shape[1]
            if pad_bottom or pad_right:
                crop = cv2.copyMakeBorder(
                    crop,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(114, 114, 114),
                )
            tiles.append(crop)
            origins.append((left, top))
    return tiles, origins


def class_aware_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float
) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    # Offset classes into disjoint coordinate ranges, then use one vectorized NMS.
    max_coordinate = float(max(boxes.max(), 1.0) * 2.0)
    shifted = boxes + classes[:, None] * max_coordinate
    return nms(torch.from_numpy(shifted).float(), torch.from_numpy(scores).float(), iou).numpy()


def empty_detections() -> "DetectionResults":
    return DetectionResults(
        np.empty((0, 4), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )


class DetectionResults:
    """Small adapter for BYTETracker's documented result fields."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy.astype(np.float32, copy=False)
        self.conf = conf.astype(np.float32, copy=False)
        self.cls = cls.astype(np.float32, copy=False)

    @property
    def xywh(self) -> np.ndarray:
        output = self.xyxy.copy()
        output[:, 2] -= output[:, 0]
        output[:, 3] -= output[:, 1]
        output[:, 0] += output[:, 2] / 2
        output[:, 1] += output[:, 3] / 2
        return output

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, index: np.ndarray) -> "DetectionResults":
        return DetectionResults(self.xyxy[index], self.conf[index], self.cls[index])


def tiled_detect(
    model: YOLO,
    frame: np.ndarray,
    tile_size: int,
    overlap: float,
    conf: float,
    nms_iou: float,
    batch: int,
    device: str,
    half: bool,
) -> tuple[DetectionResults, int]:
    tiles, origins = make_tiles(frame, tile_size, overlap)
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    for start in range(0, len(tiles), batch):
        results = model.predict(
            source=tiles[start : start + batch],
            imgsz=tile_size,
            conf=conf,
            iou=nms_iou,
            device=device,
            half=half,
            verbose=False,
        )
        for result, (left, top) in zip(results, origins[start : start + batch]):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            current = result.boxes.xyxy.cpu().numpy()
            current[:, [0, 2]] += left
            current[:, [1, 3]] += top
            boxes.append(current)
            scores.append(result.boxes.conf.cpu().numpy())
            classes.append(result.boxes.cls.cpu().numpy())
    if not boxes:
        return empty_detections(), len(tiles)
    merged_boxes = np.concatenate(boxes)
    merged_scores = np.concatenate(scores)
    merged_classes = np.concatenate(classes)
    keep = class_aware_nms(merged_boxes, merged_scores, merged_classes, nms_iou)
    return DetectionResults(merged_boxes[keep], merged_scores[keep], merged_classes[keep]), len(tiles)


def tracker_args() -> Namespace:
    return Namespace(
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.80,
        fuse_score=True,
        gmc_method="sparseOptFlow",
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        with_reid=False,
        model="auto",
    )


def draw_tracks(frame: np.ndarray, tracks: np.ndarray, detections: DetectionResults) -> None:
    for row in tracks:
        x1, y1, x2, y2, track_id, score, class_id = row[:7]
        label = f"{int(track_id)} {float(score):.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, label, (int(x1), max(18, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def draw_detections(frame: np.ndarray, detections: DetectionResults) -> None:
    for (x1, y1, x2, y2), score in zip(detections.xyxy, detections.conf):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, f"person {score:.2f}", (int(x1), max(18, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Video file or image directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detect-every", type=int, default=1, help="Run tiled detection every N frames")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort", "none"), default="bytetrack")
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def self_test() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tiles, origins = make_tiles(frame, 640, 0.25)
    assert len(tiles) == 8
    assert origins[0] == (0, 0)
    assert origins[-1] == (1280, 440)
    detections = DetectionResults(np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32), np.array([.9, .8]), np.zeros(2))
    assert class_aware_nms(detections.xyxy, detections.conf, detections.cls, .5).tolist() == [0]
    print("self-test: OK")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.detect_every < 1 or args.batch < 1 or args.tile_size < 32:
        raise ValueError("detect-every, batch, and tile-size must be positive")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (round(width * args.output_scale), round(height * args.output_scale))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output: {args.output}")
    model = YOLO(str(args.weights))
    if args.tracker == "bytetrack":
        tracker = BYTETracker(tracker_args(), frame_rate=round(fps))
    elif args.tracker == "botsort":
        tracker = BOTSORT(tracker_args())
    else:
        tracker = None
    frame_index = 0
    detection_frames = 0
    total_tiles = 0
    total_detections = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.max_frames and frame_index >= args.max_frames:
            break
        if frame_index % args.detect_every == 0:
            detections, tiles = tiled_detect(
                model, frame, args.tile_size, args.overlap, args.conf, args.nms_iou,
                args.batch, args.device, args.half and args.device != "cpu",
            )
            detection_frames += 1
            total_tiles += tiles
            total_detections += len(detections)
        else:
            detections, tiles = empty_detections(), 0
        if tracker is not None:
            tracks = tracker.update(detections, frame)
            draw_tracks(frame, tracks, detections)
        else:
            draw_detections(frame, detections)
        if args.output_scale != 1.0:
            frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()
    summary = {
        "source": str(args.source), "output": str(args.output), "frames": frame_index,
        "detection_frames": detection_frames, "tile_size": args.tile_size,
        "overlap": args.overlap, "tiles_per_detection_frame": total_tiles / detection_frames if detection_frames else 0,
        "mean_detections_per_detection_frame": total_detections / detection_frames if detection_frames else 0,
        "detect_every": args.detect_every, "tracker": args.tracker, "nms_iou": args.nms_iou,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
