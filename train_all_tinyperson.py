#!/usr/bin/env python3
"""Train/evaluate/upload TinyPerson YOLOv8n base and P2/P3 plain neck with GAP attention and FTAL."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIGS = {
    "yolov8n_base": ROOT / "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_base.yaml",
    "yolov8n_p2p3_plain_gap": ROOT / "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_p2p3_plain_gap.yaml",
}

VARIANTS = {
    "yolov8n_base": {
        "factorized_tal_target": False,
    },
    "yolov8n_p2p3_plain_gap": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
}

REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
)


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Uploader:
    def __init__(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required before this upload-required workflow starts")
        if not repo_id.strip():
            raise ValueError("--hf-repo-id is required before this upload-required workflow starts")
        from huggingface_hub import HfApi

        self.repo_id, self.api = repo_id, HfApi(token=token)
        self.api.whoami()
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    @staticmethod
    def retry(operation):
        for attempt in range(3):
            try:
                return operation()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def upload_run(self, variant: str, seed: int, run_dir: Path) -> None:
        missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
        if missing:
            raise RuntimeError(f"{variant}: refusing incomplete upload: {missing}")
        remote = f"runs/{variant}/seed_{seed}"
        self.retry(lambda: self.api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=self.repo_id, repo_type="dataset"))
        expected = {f"{remote}/{path}" for path in REQUIRED}
        uploaded = set(self.retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset")))
        missing = sorted(expected - uploaded)
        if missing:
            raise RuntimeError(f"{variant}: Hugging Face verification failed: {missing}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(json.dumps({"repo_id": self.repo_id, "variant": variant, "seed": seed, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
        self.retry(lambda: self.api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/{marker.name}", repo_id=self.repo_id, repo_type="dataset"))


def prepare_test_set(data_root: Path, output_dir: Path) -> Path:
    """Prepares the test set and converts COCO annotations to YOLO format."""
    test_out_dir = output_dir / "tinyperson_test"
    test_images_dir = test_out_dir / "images"
    test_labels_dir = test_out_dir / "labels"

    # If already prepared, skip
    if test_images_dir.exists() and test_labels_dir.exists() and list(test_labels_dir.glob("*.txt")):
        print("Test set already prepared.", flush=True)
        return test_out_dir

    print("Preparing test set...", flush=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    test_json_path = data_root / "annotations/task/tiny_set_test_all.json"
    if not test_json_path.is_file():
        raise FileNotFoundError(f"Test annotation file not found: {test_json_path}")

    with open(test_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert annotations
    img_id_to_file = {}
    for img in data["images"]:
        img_id_to_file[img["id"]] = img

    # Group annotations by image
    ann_by_image = {}
    for ann in data["annotations"]:
        # Only keep non-ignored, non-uncertain annotations
        if ann.get("ignore", False) or ann.get("uncertain", False):
            continue
        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann)

    for img_id, img_info in img_id_to_file.items():
        src_path = data_root / "test" / img_info["file_name"]
        if not src_path.is_file():
            # Try plain filename or search in labeled_images/pure_bg_images
            src_path = data_root / "test" / "labeled_images" / Path(img_info["file_name"]).name
            if not src_path.is_file():
                continue

        dest_name = f"test_{img_id}.jpg"
        dest_img_path = test_images_dir / dest_name
        dest_lbl_path = test_labels_dir / f"test_{img_id}.txt"

        # Symlink or copy image
        if not dest_img_path.exists():
            try:
                dest_img_path.symlink_to(src_path)
            except OSError:
                shutil.copy2(src_path, dest_img_path)

        # Write labels
        w, h = img_info["width"], img_info["height"]
        lines = []
        for ann in ann_by_image.get(img_id, []):
            bx, by, bw, bh = ann["bbox"]
            # Convert to YOLO format
            cx = (bx + bw / 2.0) / w
            cy = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        with open(dest_lbl_path, "w", encoding="utf-8") as lf:
            lf.writelines(lines)

    return test_out_dir


def prepare_seed_dataset(data_root: Path, output_dir: Path, test_out_dir: Path, seed: int) -> Path:
    """Prepares the dataset split for a specific seed by cropping training images."""
    seed_dir = output_dir / f"tinyperson_seed_{seed}"
    if (seed_dir / "images/train").exists() and (seed_dir / "labels/train").exists():
        print(f"Dataset for seed {seed} already prepared.", flush=True)
        return seed_dir

    print(f"Preparing dataset split for seed {seed}...", flush=True)
    for split in ("train", "val"):
        (seed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (seed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Link test set
    os.symlink(test_out_dir / "images", seed_dir / "images/test")
    os.symlink(test_out_dir / "labels", seed_dir / "labels/test")

    train_json_path = data_root / "erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json"
    if not train_json_path.is_file():
        raise FileNotFoundError(f"Train annotation file not found: {train_json_path}")

    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group annotations by image_id
    ann_by_image = {}
    for ann in data["annotations"]:
        # train set in erase_with_uncertain_dataset already has ignores removed, but double check
        if ann.get("ignore", False) or ann.get("uncertain", False):
            continue
        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann)

    # Split train/val by shuffling images
    images = data["images"].copy()
    random.Random(seed).shuffle(images)

    val_count = int(len(images) * 0.1)
    splits = {
        "val": images[:val_count],
        "train": images[val_count:],
    }

    for split_name, split_images in splits.items():
        print(f"  Cropping {len(split_images)} images for {split_name} split...", flush=True)
        for img_info in split_images:
            img_id = img_info["id"]
            src_img_path = data_root / "erase_with_uncertain_dataset/train" / img_info["file_name"]
            if not src_img_path.is_file():
                # Try fallback search
                src_img_path = data_root / "erase_with_uncertain_dataset/train/labeled_images" / Path(img_info["file_name"]).name
                if not src_img_path.is_file():
                    continue

            # Crop the image piece
            x1, y1, x2, y2 = img_info["corner"]
            dest_img_path = seed_dir / "images" / split_name / f"img_{img_id}.jpg"
            
            # Crop image on the fly
            with Image.open(src_img_path) as im:
                cropped = im.crop((x1, y1, x2, y2))
                cropped.convert("RGB").save(dest_img_path, "JPEG", quality=95)

            # Write YOLO labels (annotations coordinates are already cropped-relative)
            dest_lbl_path = seed_dir / "labels" / split_name / f"img_{img_id}.txt"
            lines = []
            for ann in ann_by_image.get(img_id, []):
                bx, by, bw, bh = ann["bbox"]
                # size of cut image piece is always 640x512
                cx = (bx + bw / 2.0) / 640.0
                cy = (by + bh / 2.0) / 512.0
                nw = bw / 640.0
                nh = bh / 512.0
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            with open(dest_lbl_path, "w", encoding="utf-8") as lf:
                lf.writelines(lines)

    # Create dataset yaml file
    yaml_path = seed_dir / "tinyperson.yaml"
    yaml_content = f"""path: {seed_dir}
train: images/train
val: images/val
test: images/test

names:
  0: person
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Dataset YAML generated for seed {seed} at: {yaml_path}", flush=True)
    return seed_dir


def training_complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    return (
        (run_dir / "weights/best.pt").is_file()
        and (run_dir / "weights/last.pt").is_file()
        and results.is_file()
        and sum(1 for _ in results.open(encoding="utf-8")) > 1
    )


def train(variant: str, seed: int, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    if training_complete(run_dir, args.epochs):
        print(f"Reusing completed training: {run_dir}", flush=True)
        return run_dir
    seed_everything(seed)
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(CONFIGS[variant])
    if args.pretrained and variant == "yolov8n_base":
        model.load(args.pretrained, smart_transfer=True)

    # Train model
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=seed,
        deterministic=True,
        amp=True,
        plots=False,
        project=str(args.project / variant),
        name=f"seed_{seed}",
        exist_ok=True,
        **VARIANTS[variant],
    )
    if not training_complete(run_dir, args.epochs):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            iou=0.5,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_metadata(variant: str, run_dir: Path, seed: int, data_yaml: Path, args: argparse.Namespace) -> None:
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    shutil.copy2(CONFIGS[variant], run_dir / "config.yaml")
    model = YOLO(run_dir / "weights/best.pt")
    head = model.model.model[-1]
    manifest = {
        "variant": variant,
        "seed": seed,
        "config": CONFIGS[variant].name,
        "data_yaml": str(data_yaml),
        "topology": "P2/P3 plain RepC2f -> GAP ChannelAttention -> shared Detect" if "gap" in variant else "Standard YOLOv8n Head",
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "factorized_tal": VARIANTS[variant],
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "model_gflops_thop": get_flops(model.model, imgsz=args.imgsz),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summaries(args: argparse.Namespace) -> None:
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            path = args.project / variant / f"seed_{seed}" / "evaluation_metrics.json"
            if path.is_file():
                rows.append({"variant": variant, "seed": seed, **json.loads(path.read_text(encoding="utf-8"))})
    if not rows:
        return
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed"}, key))
    with (args.project / "summary_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for variant in args.variants:
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        record = {"variant": variant, "runs": len(group)}
        for key in sorted(set.intersection(*(set(row) for row in group)) - {"variant", "seed", "checkpoint"}):
            values = [float(row[key]) for row in group]
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(record)
    (args.project / "summary_aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def complete(run_dir: Path) -> bool:
    return all((run_dir / path).is_file() for path in REQUIRED)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "TinyPerson")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/tinyperson_yolov8n_baselines")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hf-repo-id", default="duyle2408/tinyperson-yolov8n-baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (
        args.data_root.resolve(),
        args.dataset_root.resolve(),
        args.project.resolve(),
    )
    uploader = Uploader(args.hf_repo_id)

    # 1. Prepare test set (shared across seeds)
    test_out_dir = prepare_test_set(args.data_root, args.dataset_root)

    # 2. Run sequential seed experiments
    for seed in args.seeds:
        # Prepare dataset split for this seed
        seed_dir = prepare_seed_dataset(args.data_root, args.dataset_root, test_out_dir, seed)
        data_yaml = seed_dir / "tinyperson.yaml"

        for variant in args.variants:
            run_dir = train(variant, seed, data_yaml, args)
            evaluate(run_dir, data_yaml, args)
            write_metadata(variant, run_dir, seed, data_yaml, args)
            write_summaries(args)
            if not complete(run_dir):
                raise RuntimeError(f"Required post-evaluation artifacts are incomplete: {run_dir}")
            uploader.upload_run(variant, seed, run_dir)

    write_summaries(args)
    print("TinyPerson training matrix complete!", flush=True)


if __name__ == "__main__":
    main()
