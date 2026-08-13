#!/usr/bin/env python3
"""Train/evaluate/upload Varroa YOLOv8n P2/P3 plain GAP target variants."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

import train_all_levir_yolov8n_p2_gap_scale_temper as base


ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIGS = {
    "varroa_p2p3_plain": ROOT / "models_related/models_config/yolov8/varroa/yolov8n_varroa_p2p3_plain.yaml",
    "varroa_p2p3_plain_gap": ROOT / "models_related/models_config/yolov8/varroa/yolov8n_varroa_p2p3_plain_gap.yaml",
    "varroa_p2p3_plain_gap_factorized_k15": ROOT / "models_related/models_config/yolov8/varroa/yolov8n_varroa_p2p3_plain_gap.yaml",
}
VARIANTS = {
    "varroa_p2p3_plain": {
        "factorized_tal_target": False,
    },
    "varroa_p2p3_plain_gap": {
        "factorized_tal_target": False,
    },
    "varroa_p2p3_plain_gap_factorized_k15": {
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


def validate_dataset(data_yaml: Path) -> None:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    for split in ("train", "val", "test"):
        rel = payload.get(split)
        if not rel:
            raise ValueError(f"Missing {split!r} in {data_yaml}")
        image_dir = root / rel
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        count = sum(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} for path in image_dir.iterdir())
        if count <= 0:
            raise ValueError(f"No images found for {split}: {image_dir}")


def ensure_dataset(args: argparse.Namespace) -> None:
    try:
        validate_dataset(args.data_yaml)
        return
    except (FileNotFoundError, ValueError):
        from misc.prepare_dataset import prepare_dataset

        args.data_yaml = prepare_dataset(
            args.data_root,
            args.dataset_root / "varroa_yolo_seed42",
            gt_source="gt_one",
            only_positives=True,
            class_policy="map-3-to-1",
            seed=42,
        ).resolve()
        validate_dataset(args.data_yaml)


def model_for(variant: str, pretrained: str):
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(CONFIGS[variant])
    model.load(pretrained, smart_transfer=True)
    return model


def training_complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    return (
        (run_dir / "weights/best.pt").is_file()
        and (run_dir / "weights/last.pt").is_file()
        and results.is_file()
        and sum(1 for _ in results.open(encoding="utf-8")) - 1 == epochs
    )


def train(variant: str, seed: int, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    if training_complete(run_dir, args.epochs):
        print(f"Reusing completed training: {run_dir}", flush=True)
        return run_dir
    seed_everything(seed)
    model_for(variant, args.pretrained).train(
        data=str(args.data_yaml),
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


def evaluate(run_dir: Path, args: argparse.Namespace) -> dict:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(args.data_yaml),
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


def write_metadata(variant: str, run_dir: Path, seed: int, args: argparse.Namespace) -> None:
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
        "data_yaml": str(args.data_yaml),
        "topology": "P2/P3 plain RepC2f -> GAP ChannelAttention -> shared Detect",
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


class Uploader(base.Uploader):
    def upload_run(self, variant: str, seed: int, run_dir: Path) -> None:
        old = base.REQUIRED
        try:
            base.REQUIRED = REQUIRED
            super().upload_run(variant, seed, run_dir)
        finally:
            base.REQUIRED = old


def complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    return all((run_dir / path).is_file() for path in REQUIRED) and sum(1 for _ in results.open(encoding="utf-8")) - 1 == epochs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "datasets/varroa_yolo/varroa.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "varroa_data")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/varroa_yolov8n_p2_gap_factorized_tal")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hf-repo-id", default="duyle2408/varroa-yolov8n-p2p3-gap-factorized-tal")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_yaml, args.data_root, args.dataset_root, args.project = (
        args.data_yaml.resolve(),
        args.data_root.resolve(),
        args.dataset_root.resolve(),
        args.project.resolve(),
    )
    ensure_dataset(args)
    uploader = Uploader(args.hf_repo_id)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, seed, args)
            evaluate(run_dir, args)
            write_metadata(variant, run_dir, seed, args)
            write_summaries(args)
            if not complete(run_dir, args.epochs):
                raise RuntimeError(f"Required post-evaluation artifacts are incomplete: {run_dir}")
            uploader.upload_run(variant, seed, run_dir)
    write_summaries(args)


if __name__ == "__main__":
    main()
