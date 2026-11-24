#!/usr/bin/env python3
"""Optimized tiled video inference with safe temporal and I/O shortcuts.

Defaults favor recall: every frame, 25% tile overlap, no gating, and ByteTrack.
Use ``--detect-every``, ``--gate roi``/``motion``, ``--encoder nvenc``, and
``--async-io`` only after measuring the quality/speed tradeoff on the target video.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from evaluate_test.tiled_video_inference import DetectionResults, empty_detections, tile_starts


def crop_tile(frame: np.ndarray, left: int, top: int, tile_size: int) -> np.ndarray:
    crop = frame[top : min(top + tile_size, frame.shape[0]), left : min(left + tile_size, frame.shape[1])]
    bottom = tile_size - crop.shape[0]
    right = tile_size - crop.shape[1]
    if bottom or right:
        crop = cv2.copyMakeBorder(crop, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return crop


def all_origins(frame: np.ndarray, tile_size: int, overlap: float) -> list[tuple[int, int]]:
    stride = max(1, round(tile_size * (1.0 - overlap)))
    height, width = frame.shape[:2]
    return [
        (left, top)
        for top in tile_starts(height, tile_size, stride)
        for left in tile_starts(width, tile_size, stride)
    ]


def selected_origins(
    frame: np.ndarray,
    previous: np.ndarray | None,
    tracks: np.ndarray,
    tile_size: int,
    overlap: float,
    gate: str,
    roi_pad: int,
    motion_threshold: int,
    motion_ratio: float,
) -> list[tuple[int, int]]:
    origins = all_origins(frame, tile_size, overlap)
    if gate == "none" or previous is None:
        return origins
    if gate == "roi":
        if not len(tracks):
            return origins
        height, width = frame.shape[:2]
        selected = []
        for left, top in origins:
            right, bottom = left + tile_size, top + tile_size
            if any(
                max(left, x1 - roi_pad) < min(right, x2 + roi_pad)
                and max(top, y1 - roi_pad) < min(bottom, y2 + roi_pad)
                for x1, y1, x2, y2 in tracks[:, :4]
            ):
                selected.append((left, top))
        return selected or origins
    if gate == "motion":
        old = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(old, new)
        mask = cv2.dilate((diff >= motion_threshold).astype(np.uint8), np.ones((9, 9), np.uint8))
        selected = []
        for left, top in origins:
            region = mask[top : min(top + tile_size, mask.shape[0]), left : min(left + tile_size, mask.shape[1])]
            if float(np.mean(region)) >= motion_ratio:
                selected.append((left, top))
        return selected or origins
    raise ValueError(f"unsupported gate: {gate}")


def selected_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    offset = float(max(boxes.max(), 1.0) * 2.0)
    shifted = boxes + classes[:, None] * offset
    return nms(torch.from_numpy(shifted).float(), torch.from_numpy(scores).float(), threshold).numpy()


def tiled_detect_selected(
    model: YOLO,
    frame: np.ndarray,
    origins: list[tuple[int, int]],
    tile_size: int,
    conf: float,
    nms_iou: float,
    batch: int,
    device: str,
    half: bool,
) -> tuple[DetectionResults, int]:
    if not origins:
        return empty_detections(), 0
    tiles = [crop_tile(frame, left, top, tile_size) for left, top in origins]
    boxes, scores, classes = [], [], []
    for start in range(0, len(tiles), batch):
        results = model.predict(
            source=tiles[start : start + batch], imgsz=tile_size, conf=conf,
            iou=nms_iou, device=device, half=half, verbose=False,
        )
        for result, (left, top) in zip(results, origins[start : start + batch]):
            if result.boxes is None or not len(result.boxes):
                continue
            current = result.boxes.xyxy.cpu().numpy()
            current[:, [0, 2]] += left
            current[:, [1, 3]] += top
            boxes.append(current)
            scores.append(result.boxes.conf.cpu().numpy())
            classes.append(result.boxes.cls.cpu().numpy())
    if not boxes:
        return empty_detections(), len(origins)
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    classes = np.concatenate(classes)
    keep = selected_nms(boxes, scores, classes, nms_iou)
    return DetectionResults(boxes[keep], scores[keep], classes[keep]), len(origins)


def tracker_args() -> Namespace:
    return Namespace(
        track_high_thresh=0.25, track_low_thresh=0.10, new_track_thresh=0.25,
        track_buffer=30, match_thresh=0.80, fuse_score=True,
    )


def draw(frame: np.ndarray, tracks: np.ndarray, detections: DetectionResults, tracking: bool) -> None:
    if tracking:
        rows = tracks
        for x1, y1, x2, y2, track_id, score, _class_id, *_ in rows:
            label = f"{int(track_id)} {float(score):.2f}"
            color = (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), max(18, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    else:
        for (x1, y1, x2, y2), score in zip(detections.xyxy, detections.conf):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"person {score:.2f}", (int(x1), max(18, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, encoder: str) -> None:
        codec = "h264_nvenc" if encoder == "nvenc" else "libx264"
        self.process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", codec,
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def release(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg encoder exited with {code}")


def open_writer(path: Path, width: int, height: int, fps: float, encoder: str):
    if encoder == "opencv":
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open output: {path}")
        return writer
    return FfmpegWriter(path, width, height, fps, encoder)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--burst-size", type=int, default=0, help="detect this many consecutive frames, then skip --burst-skip frames")
    parser.add_argument("--burst-skip", type=int, default=1, help="frames skipped between detection bursts")
    parser.add_argument("--tracker", choices=("bytetrack", "none"), default="bytetrack")
    parser.add_argument("--gate", choices=("none", "motion", "roi"), default="none")
    parser.add_argument("--roi-pad", type=int, default=160)
    parser.add_argument("--motion-threshold", type=int, default=18)
    parser.add_argument("--motion-ratio", type=float, default=0.002)
    parser.add_argument("--gate-refresh", type=int, default=10)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--encoder", choices=("opencv", "x264", "nvenc"), default="opencv")
    parser.add_argument("--async-io", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def self_test() -> None:
    frame = np.zeros((1080, 1920, 3), np.uint8)
    origins = all_origins(frame, 640, 0.25)
    assert len(origins) == 8
    assert selected_origins(frame, None, np.empty((0, 8)), 640, .25, "roi", 160, 18, .002) == origins
    print("self-test: OK")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.burst_size < 0 or args.burst_skip < 0:
        raise ValueError("burst-size and burst-skip must be non-negative")
    if args.burst_size and args.detect_every != 1:
        raise ValueError("use either --detect-every or --burst-size, not both")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (round(width * args.output_scale), round(height * args.output_scale))
    writer = open_writer(args.output, *output_size, fps, args.encoder)
    model = YOLO(str(args.weights))
    tracker = BYTETracker(tracker_args(), frame_rate=round(fps)) if args.tracker == "bytetrack" else None
    previous = None
    active_tracks = np.empty((0, 8), dtype=np.float32)
    frame_index = detection_frames = total_tiles = total_detections = 0
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=2) if args.async_io else None
    read_future: Future | None = executor.submit(capture.read) if executor else None
    write_future: Future | None = None
    try:
        while True:
            if read_future is not None:
                ok, frame = read_future.result()
                read_future = executor.submit(capture.read)
            else:
                ok, frame = capture.read()
            if not ok or (args.max_frames and frame_index >= args.max_frames):
                break
            raw_frame = frame.copy()
            if args.burst_size:
                period = args.burst_size + args.burst_skip
                should_detect = (frame_index % period) < args.burst_size
            else:
                should_detect = frame_index % args.detect_every == 0
            refresh = args.gate_refresh > 0 and frame_index % args.gate_refresh == 0
            if should_detect:
                gate = "none" if refresh else args.gate
                origins = selected_origins(frame, previous, active_tracks, args.tile_size, args.overlap, gate, args.roi_pad, args.motion_threshold, args.motion_ratio)
                detections, tiles = tiled_detect_selected(model, frame, origins, args.tile_size, args.conf, args.nms_iou, args.batch, args.device, args.half and args.device != "cpu")
                detection_frames += 1
                total_tiles += tiles
                total_detections += len(detections)
            else:
                detections, tiles = empty_detections(), 0
            if tracker is not None:
                active_tracks = tracker.update(detections, frame)
                draw(frame, active_tracks, detections, True)
            else:
                draw(frame, np.empty((0, 8)), detections, False)
            if args.output_scale != 1.0:
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            if write_future is not None:
                write_future.result()
            if executor:
                write_future = executor.submit(writer.write, frame)
            else:
                writer.write(frame)
            previous = raw_frame
            frame_index += 1
    finally:
        if read_future is not None:
            read_future.cancel()
        if write_future is not None:
            write_future.result()
        capture.release()
        if executor:
            executor.shutdown(wait=True)
        writer.release()
    elapsed = time.perf_counter() - started
    summary = {
        "source": str(args.source), "output": str(args.output), "frames": frame_index,
        "elapsed_seconds": elapsed, "output_fps": frame_index / elapsed if elapsed else 0,
        "detection_frames": detection_frames, "tile_size": args.tile_size, "overlap": args.overlap,
        "tiles_per_detection_frame": total_tiles / detection_frames if detection_frames else 0,
        "mean_detections_per_detection_frame": total_detections / detection_frames if detection_frames else 0,
        "detect_every": args.detect_every, "tracker": args.tracker, "gate": args.gate,
        "burst_size": args.burst_size, "burst_skip": args.burst_skip,
        "encoder": args.encoder, "async_io": args.async_io, "nms_iou": args.nms_iou,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
