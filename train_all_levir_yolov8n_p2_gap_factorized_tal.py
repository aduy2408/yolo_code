#!/usr/bin/env python3
"""Train/evaluate/upload seed-42 GAP P2 factorized TAL target variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_all_levir_yolov8n_p2_gap_scale_temper as base


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "gap_factorized_ceiling": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.0,
        "factorized_tal_lambda": 0.5,
    },
    "gap_factorized_separation": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 1.0,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
    },
    "gap_factorized_k15": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
    },
    "gap_factorized_k20": {
        "factorized_tal_target": True,
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 2.0,
        "factorized_tal_lambda": 0.5,
    },
}
DEFAULT_VARIANTS = ("gap_factorized_ceiling", "gap_factorized_separation")
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
    "factorized_tal_diagnostic.json",
    "ranking_summary.json",
)


def train(variant: str, data_yaml: Path, seed: int, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    if base.training_complete(run_dir, args.epochs):
        return run_dir
    base.seed_everything(seed)
    base.model_for(args.pretrained).train(
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
        factorized_tal_s_max=32.0,
        factorized_tal_warmup_start=5,
        factorized_tal_warmup_end=15,
        factorized_tal_p2_only=True,
        **VARIANTS[variant],
    )
    if not base.training_complete(run_dir, args.epochs):
        raise RuntimeError(f"{variant}: required training artifacts are incomplete")
    return run_dir


def write_metadata(variant: str, run_dir: Path, seed: int, args: argparse.Namespace) -> None:
    import shutil

    base.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    shutil.copy2(base.CONFIG, run_dir / "config.yaml")
    model = YOLO(run_dir / "weights/best.pt")
    head = model.model.model[-1]
    manifest = {
        "variant": variant,
        "seed": seed,
        "split_seed": 42,
        "config": base.CONFIG.name,
        "topology": "P2 -> GAP ChannelAttention -> shared Detect",
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "factorized_tal": VARIANTS[variant],
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "model_gflops_thop": get_flops(model.model, imgsz=args.imgsz),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_gap_factorized_tal")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-gap-factorized-tal-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(DEFAULT_VARIANTS))
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = Uploader(args.hf_repo_id)
    data_yaml = base.prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, data_yaml, seed, args)
            base.evaluate(run_dir, data_yaml, args)
            rows = base.raw_p2_rows(run_dir, args)
            diag = base.diagnose_from_raw(rows)
            (run_dir / "factorized_tal_diagnostic.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            base.ranking_summary(rows, run_dir, args)
            write_metadata(variant, run_dir, seed, args)
            if not complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
