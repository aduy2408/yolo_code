#!/usr/bin/env python3
"""Train VisDrone MMDetection baselines with one shared YOLO-style pipeline.

This runner deliberately keeps the detector config and optimizer independent
from the augmentation policy.  FCOS, RetinaNet, and RTMDet all receive the
same train pipeline:

    Mosaic(p=1.0)
    -> RandomAffine(scale/translate, no rotate/shear/perspective)
    -> YOLOXHSVRandomAug
    -> RandomFlip(p=0.5)
    -> Resize/Pad to the requested square size

The implementation uses MMDetection transforms, so it is not byte-identical
to Ultralytics Mosaic.  It is the explicit cross-detector approximation used
by this project and, unlike the old runner, is identical across the three
MMDetection models.  MixUp, CachedMixUp, RandomCrop, and detector-specific
multi-scale policies are intentionally disabled.

Expected dataset layout::

    <data-root>/annotations/train.json
    <data-root>/annotations/val.json
    <data-root>/annotations/test.json
    <data-root>/train/<images>
    <data-root>/val/<images>
    <data-root>/test/<images>

The annotation paths can also be supplied explicitly.  Use this runner from
the MMDetection environment, for example ``/marimo/mmdet-venv/bin/python``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any


MODELS = {
    "faster_rcnn": "/marimo/mmdet_code/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "cascade_rcnn": "/marimo/mmdet_code/mmdetection/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
    "fcos": "/marimo/mmdet_code/mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
    "retinanet": "/marimo/mmdet_code/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py",
    "rtmdet": "/marimo/mmdet_code/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
}

CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def shared_yolo_train_pipeline(imgsz: int) -> list[dict[str, Any]]:
    """Return the common augmentation pipeline for every detector.

    The values mirror the YOLO baseline controls: Mosaic probability 1.0,
    scale ratio 0.5..1.5, translate ratio 0.1, HSV 0.015/0.7/0.4, and
    horizontal flip probability 0.5.  Rotation, shear, perspective, and
    MixUp remain disabled.
    """

    img_scale = (imgsz, imgsz)
    pre_transform = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True),
    ]
    return [
        dict(
            type="Mosaic",
            img_scale=img_scale,
            pad_val=114.0,
            prob=1.0,
            pre_transform=pre_transform,
        ),
        dict(
            type="RandomAffine",
            scaling_ratio_range=(0.5, 1.5),
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            max_translate_ratio=0.1,
            border=(-imgsz // 2, -imgsz // 2),
            border_val=(114.0, 114.0, 114.0),
        ),
        dict(type="YOLOXHSVRandomAug"),
        dict(type="RandomFlip", prob=0.5),
        dict(type="Resize", scale=img_scale, keep_ratio=False),
        dict(type="Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
        dict(type="PackDetInputs"),
    ]


def shared_eval_pipeline(imgsz: int) -> list[dict[str, Any]]:
    return [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(imgsz, imgsz), keep_ratio=False),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="PackDetInputs"),
    ]


def resolve_annotation(path: Path, data_root: Path) -> str:
    candidate = path if path.is_absolute() else data_root / path
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing annotation file: {candidate}")
    return str(candidate)


def resolve_image_dir(path: Path, data_root: Path) -> str:
    candidate = path if path.is_absolute() else data_root / path
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Missing image directory: {candidate}")
    return str(candidate)


def build_config(args: argparse.Namespace):
    from mmengine.config import Config

    config_path = Path(args.config or MODELS[args.model]).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"MMDetection config not found: {config_path}")
    cfg = Config.fromfile(str(config_path))

    data_root = args.data_root.resolve()
    train_ann = resolve_annotation(args.train_ann, data_root)
    val_ann = resolve_annotation(args.val_ann, data_root)
    test_ann = resolve_annotation(args.test_ann, data_root)
    train_images = resolve_image_dir(args.train_images, data_root)
    val_images = resolve_image_dir(args.val_images, data_root)
    test_images = resolve_image_dir(args.test_images, data_root)

    metainfo = dict(classes=CLASSES)
    train_dataset = dict(
        type="CocoDataset",
        ann_file=train_ann,
        data_root="",
        data_prefix=dict(img=train_images + os.sep),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=shared_yolo_train_pipeline(args.imgsz),
    )
    eval_pipeline = shared_eval_pipeline(args.imgsz)
    val_dataset = dict(
        type="CocoDataset",
        ann_file=val_ann,
        data_root="",
        data_prefix=dict(img=val_images + os.sep),
        metainfo=metainfo,
        test_mode=True,
        pipeline=eval_pipeline,
    )
    test_dataset = dict(
        type="CocoDataset",
        ann_file=test_ann,
        data_root="",
        data_prefix=dict(img=test_images + os.sep),
        metainfo=metainfo,
        test_mode=True,
        pipeline=eval_pipeline,
    )

    train_dataloader = copy.deepcopy(dict(cfg.train_dataloader))
    train_dataloader.update(
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        sampler=dict(type="DefaultSampler", shuffle=True),
        batch_sampler=dict(type="AspectRatioBatchSampler"),
        dataset=train_dataset,
    )
    cfg.train_dataloader = train_dataloader
    val_dataloader = copy.deepcopy(dict(cfg.val_dataloader))
    val_dataloader.update(
        batch_size=1,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=val_dataset,
    )
    cfg.val_dataloader = val_dataloader
    test_dataloader = copy.deepcopy(dict(cfg.test_dataloader))
    test_dataloader.update(
        batch_size=1,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=test_dataset,
    )
    cfg.test_dataloader = test_dataloader
    cfg.val_evaluator = dict(type="CocoMetric", ann_file=val_ann, metric="bbox")
    cfg.test_evaluator = dict(type="CocoMetric", ann_file=test_ann, metric="bbox")
    cfg.val_cfg = dict(type="ValLoop")
    cfg.test_cfg = dict(type="TestLoop")
    train_cfg = copy.deepcopy(dict(cfg.train_cfg))
    train_cfg.update(type="EpochBasedTrainLoop", max_epochs=args.epochs, val_interval=1)
    cfg.train_cfg = train_cfg
    cfg.default_hooks = cfg.get("default_hooks", {})
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=1,
        save_last=True,
    )
    cfg.default_hooks.logger = dict(type="LoggerHook", interval=50)
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.randomness = dict(seed=args.seed, deterministic=True)
    cfg.env_cfg = cfg.get("env_cfg", {})
    cfg.env_cfg.setdefault("cudnn_benchmark", False)
    return cfg, config_path


def write_manifest(args: argparse.Namespace, config_path: Path, pipeline: list[dict[str, Any]]) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    manifest = {
        "runner": str(Path(__file__).resolve()),
        "model": args.model,
        "config": str(config_path),
        "git_sha": git_sha,
        "dataset": "VisDrone2019-DET",
        "data_root": str(args.data_root.resolve()),
        "image_size": [args.imgsz, args.imgsz],
        "batch_size": args.batch_size,
        "workers": args.workers,
        "epochs": args.epochs,
        "seed": args.seed,
        "split": "official VisDrone2019-DET train/val/test-dev",
        "augmentation_policy": "shared_yolo_style_mosaic",
        "augmentation": pipeline,
        "disabled_detector_specific_augmentations": [
            "CachedMixUp",
            "MixUp",
            "RandomCrop",
            "detector-specific multi-scale policy",
        ],
        "nms_iou": 0.5,
    }
    (args.work_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--config")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-ann", type=Path, default=Path("annotations/train.json"))
    parser.add_argument("--val-ann", type=Path, default=Path("annotations/val.json"))
    parser.add_argument("--test-ann", type=Path, default=Path("annotations/test.json"))
    parser.add_argument("--train-images", type=Path, default=Path("train"))
    parser.add_argument("--val-images", type=Path, default=Path("val"))
    parser.add_argument("--test-images", type=Path, default=Path("test"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.work_dir = args.work_dir.resolve()
    seed_everything(args.seed)
    cfg, config_path = build_config(args)
    pipeline = shared_yolo_train_pipeline(args.imgsz)
    write_manifest(args, config_path, pipeline)
    cfg.dump(str(args.work_dir / "resolved_config.py"))
    if args.dry_run:
        print(json.dumps({"work_dir": str(args.work_dir), "dry_run": True}, sort_keys=True))
        return

    from mmengine.runner import Runner

    runner = Runner.from_cfg(cfg)
    runner.train()
    runner.test()
    print(json.dumps({"work_dir": str(args.work_dir), "model": args.model}, sort_keys=True))


if __name__ == "__main__":
    main()
