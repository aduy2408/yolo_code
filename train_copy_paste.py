#!/usr/bin/env python3
"""Run the matched CP0-CP3 screening matrix on LEVIR-Ship or TinyPerson.

This runner intentionally defaults to printing the effective protocol. Full
training requires the Marimo workflow marker and HF authentication so every
completed run is uploaded before the next one starts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from copy_paste_protocol import VARIANTS, effective_settings, variant_overrides

ROOT = Path(__file__).resolve().parent


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _prepare(dataset: str, data_root: Path, dataset_root: Path, split_seed: int) -> Path:
    if dataset == "levir":
        from misc.prepare_levir_ship import prepare

        return prepare(data_root, dataset_root / f"levir_ship_copy_paste_split_{split_seed}", split_seed)
    if dataset == "tinyperson":
        import train_all_tinyperson as workflow

        test_dir = workflow.prepare_test_set(data_root, dataset_root)
        seed_dir = workflow.prepare_seed_dataset(data_root, dataset_root, test_dir, split_seed)
        return seed_dir / "tinyperson.yaml"
    raise ValueError(f"unsupported dataset: {dataset}")


def _upload(run_dir: Path, repo_id: str, dataset: str, variant: str, seed: int) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir), repo_id=repo_id, repo_type="dataset",
        path_in_repo=f"copy_paste/{dataset}/{variant}/seed_{seed}",
    )
    (run_dir / "upload_complete.json").write_text(
        json.dumps({"repo_id": repo_id, "dataset": dataset, "variant": variant, "seed": seed}, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_one(args: argparse.Namespace, data_yaml: Path, variant: str, seed: int) -> Path:
    from ultralytics import YOLO

    run_dir = args.project / args.dataset / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    required = [run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv"]
    if all(path.is_file() for path in required):
        return run_dir
    settings = variant_overrides(variant)
    if args.mosaic_interaction:
        settings["mosaic"] = args.mosaic
        settings["close_mosaic"] = args.close_mosaic
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
        device=args.device, workers=args.workers, patience=0, seed=seed,
        deterministic=True, amp=True, plots=False, project=str(run_dir.parent),
        name=run_dir.name, exist_ok=True, val=True, iou=0.5, **settings,
    )
    model = YOLO(run_dir / "weights/best.pt")
    metrics = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch_size,
                        device=args.device, workers=args.workers, plots=False, iou=0.5,
                        project=str(run_dir / "evaluation"), name="val", exist_ok=True)
    manifest = effective_settings(
        args.dataset, variant, seed, args.split_seed, commit_sha=_git_sha(),
        data_yaml=str(data_yaml), epochs=args.epochs, patience=0, imgsz=args.imgsz,
        batch_size=args.batch_size, device=args.device, workers=args.workers,
        hf_repo_id=args.hf_repo_id, upload_required=True,
        augmentation=settings,
        mosaic_interaction=args.mosaic_interaction,
        metrics={key: float(value) for key, value in metrics.results_dict.items()},
    )
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("levir", "tinyperson"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--mosaic-interaction", action="store_true",
                        help="Enable Mosaic while retaining the matched Copy-Paste settings")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    configs = [effective_settings(args.dataset, variant, seed, args.split_seed, epochs=args.epochs, patience=0,
                                  imgsz=args.imgsz, batch_size=args.batch_size, device=args.device,
                                  workers=args.workers, hf_repo_id=args.hf_repo_id,
                                  augmentation={**variant_overrides(variant),
                                                **({"mosaic": args.mosaic, "close_mosaic": args.close_mosaic}
                                                   if args.mosaic_interaction else {})},
                                  mosaic_interaction=args.mosaic_interaction)
               for seed in args.seeds for variant in args.variants]
    if args.print_effective_config:
        print(json.dumps({"runs": configs}, indent=2, sort_keys=True))
        return
    data_yaml = _prepare(args.dataset, args.data_root, args.dataset_root, args.split_seed)
    if args.prepare_only:
        print(data_yaml)
        return
    if args.epochs != 100:
        raise ValueError("full Copy-Paste screening requires epochs=100")
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("refusing to train outside the Marimo training workflow")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for upload-required Copy-Paste screening")
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = _run_one(args, data_yaml, variant, seed)
            _upload(run_dir, args.hf_repo_id, args.dataset, variant, seed)


if __name__ == "__main__":
    main()
