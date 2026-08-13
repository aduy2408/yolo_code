#!/usr/bin/env python3
"""Train/evaluate/upload seed-42 GAP P2 scale-tempered TAL target variants."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_channel_only.yaml"
VARIANTS = {
    "gap": {"scale_temper_target": False},
    "gap_temper_mild": {"scale_temper_target": True, "scale_temper_tau_min": 0.75, "scale_temper_lambda": 0.5},
    "gap_temper_medium": {"scale_temper_target": True, "scale_temper_tau_min": 0.50, "scale_temper_lambda": 0.5},
    "gap_temper_strong": {"scale_temper_target": True, "scale_temper_tau_min": 0.50, "scale_temper_lambda": 1.0},
}
DEFAULT_VARIANTS = ("gap_temper_mild", "gap_temper_medium", "gap_temper_strong")
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
    "scale_temper_diagnostic.json",
    "ranking_summary.json",
)


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int = 42) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_split(args: argparse.Namespace) -> Path:
    from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

    data_yaml = workflow.prepare_fixed_split(
        argparse.Namespace(data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42)
    )
    workflow.validate_split(data_yaml)
    return data_yaml


def model_for(pretrained: str):
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import ChannelAttention

    model = YOLO(CONFIG)
    model.load(pretrained, smart_transfer=True)
    attention, head = model.model.model[19], model.model.model[-1]
    if not isinstance(attention, ChannelAttention) or getattr(attention, "descriptor", "avg") != "avg":
        raise TypeError("GAP ChannelAttention did not resolve at layer 19")
    if head.f != [19] or head.stride.tolist() != [4.0]:
        raise ValueError(f"expected ChannelAttention -> Detect stride [4.0], got {head.f}, {head.stride.tolist()}")
    return model


def complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    return all((run_dir / path).is_file() for path in REQUIRED) and sum(1 for _ in results.open(encoding="utf-8")) - 1 == epochs


def training_complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    return (
        all((run_dir / path).is_file() for path in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"))
        and sum(1 for _ in results.open(encoding="utf-8")) - 1 == epochs
    )


def train(variant: str, data_yaml: Path, seed: int, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    if training_complete(run_dir, args.epochs):
        return run_dir
    seed_everything(seed)
    model_for(args.pretrained).train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        patience=0,
        seed=seed,
        deterministic=True,
        amp=True,
        plots=False,
        project=str(args.project / variant),
        name=f"seed_{seed}",
        exist_ok=True,
        scale_temper_s1=16.0,
        scale_temper_s2=32.0,
        scale_temper_warmup_start=5,
        scale_temper_warmup_end=15,
        scale_temper_p2_only=True,
        **VARIANTS[variant],
    )
    missing = [path for path in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml") if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"{variant}: training ended without artifacts: {missing}")
    if sum(1 for _ in (run_dir / "results.csv").open(encoding="utf-8")) - 1 != args.epochs:
        raise RuntimeError(f"{variant}: results.csv does not contain {args.epochs} completed epochs")
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float | str]:
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
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def raw_p2_rows(run_dir: Path, args: argparse.Namespace) -> list[dict]:
    local_ultralytics()
    from train_levir_scripts import analyze_p2_cbam_ranking as ranking

    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/test"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if args.ranking_limit:
        images = images[: args.ranking_limit]
    ns = argparse.Namespace(imgsz=args.imgsz, device=args.device, expected_seed=int(run_dir.name.rsplit("_", 1)[1]))
    ranking.EXPECTED_LEVELS["gap_scale_temper"] = 1
    return ranking.inspect_model("gap_scale_temper", run_dir / "weights/best.pt", images, ns)


def diagnose_from_raw(rows: list[dict]) -> dict[str, int | float | str]:
    good = [row for row in rows if row["candidate_count"] and row["iou_best"] >= 0.5]
    conf = sorted(float(row["best_iou_confidence"]) for row in good)
    return {
        "checkpoint": "best.pt",
        "candidate_source": "raw decoded P2 before threshold and NMS",
        "gt": len(rows),
        "raw_p2_no_candidate": sum(row["candidate_count"] == 0 for row in rows),
        "raw_p2_best_iou_ge_0_5": len(good),
        "raw_p2_best_iou_ge_0_5_conf_lt_0_25": sum(float(row["best_iou_confidence"]) < 0.25 for row in good),
        "raw_p2_best_iou_ge_0_5_conf_lt_0_10": sum(float(row["best_iou_confidence"]) < 0.10 for row in good),
        "good_candidate_conf_median": float(conf[len(conf) // 2]) if conf else 0.0,
        "good_candidate_conf_max": float(max(conf)) if conf else 0.0,
    }


def ranking_summary(rows: list[dict], run_dir: Path, args: argparse.Namespace) -> dict:
    from train_levir_scripts import analyze_p2_cbam_ranking as ranking

    summary = {
        "protocol": {
            "seed": 42,
            "split": "test",
            "images": len({row["image"] for row in rows}),
            "nms_iou": 0.5,
            "candidate_rule": "decoded P2 anchor center inside GT; overlapping anchors assigned to highest-IoU GT",
            "prediction_stage": "decoded P2 boxes and sigmoid class scores before threshold and NMS",
        },
        "raw_p2": ranking.descriptive_summary(rows),
    }
    (run_dir / "ranking_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def write_metadata(variant: str, run_dir: Path, seed: int, args: argparse.Namespace) -> None:
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    shutil.copy2(CONFIG, run_dir / "config.yaml")
    model = YOLO(run_dir / "weights/best.pt")
    head = model.model.model[-1]
    manifest = {
        "variant": variant,
        "seed": seed,
        "split_seed": 42,
        "config": CONFIG.name,
        "topology": "P2 -> GAP ChannelAttention -> shared Detect",
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "scale_temper": VARIANTS[variant],
        "scale_temper_s1": 16.0,
        "scale_temper_s2": 32.0,
        "scale_temper_warmup_start": 5,
        "scale_temper_warmup_end": 15,
        "scale_temper_p2_only": True,
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "model_gflops_thop": get_flops(model.model, imgsz=args.imgsz),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        self.retry(lambda: self.api.upload_folder(folder_path=run_dir, path_in_repo=remote, repo_id=self.repo_id, repo_type="dataset"))
        expected = {f"{remote}/{path}" for path in REQUIRED}
        uploaded = set(self.retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset")))
        missing = sorted(expected - uploaded)
        if missing:
            raise RuntimeError(f"{variant}: Hugging Face verification failed: {missing}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(json.dumps({"repo_id": self.repo_id, "variant": variant, "seed": seed, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
        self.retry(lambda: self.api.upload_file(path_or_fileobj=marker, path_in_repo=f"{remote}/{marker.name}", repo_id=self.repo_id, repo_type="dataset"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_gap_scale_temper")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-gap-scale-temper-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(DEFAULT_VARIANTS))
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = Uploader(args.hf_repo_id)
    data_yaml = prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, data_yaml, seed, args)
            evaluate(run_dir, data_yaml, args)
            rows = raw_p2_rows(run_dir, args)
            diag = diagnose_from_raw(rows)
            (run_dir / "scale_temper_diagnostic.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            ranking_summary(rows, run_dir, args)
            write_metadata(variant, run_dir, seed, args)
            if not complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
