#!/usr/bin/env python3
"""Post-hoc val/test evaluation for uploaded LEVIR Copy-Paste checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def split_metrics(result, split: str) -> dict[str, float]:
    values = {key: float(value) for key, value in result.results_dict.items()}
    return {
        **{f"{split}/{key}": value for key, value in values.items()},
        f"{split}/AP50": values["metrics/mAP50(B)"],
        f"{split}/mAP50-95": values["metrics/mAP50-95(B)"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("post-hoc evaluation must run through utils.marimo_ops launch")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for post-hoc evaluation uploads")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "models_related" / "ultralytics"))
    from huggingface_hub import HfApi, hf_hub_download
    from ultralytics import YOLO
    from misc.prepare_levir_ship import prepare

    api = HfApi(token=token)
    api.create_repo(repo_id=args.output_repo, repo_type="dataset", exist_ok=True)
    data_yaml = prepare(args.data_root, args.dataset_root / "levir_ship_posthoc_split_42", 42)

    for variant in args.variants:
        for seed in args.seeds:
            relative = f"copy_paste/levir/{variant}/seed_{seed}"
            out_dir = args.work_dir / variant / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = out_dir / "weights" / "best.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.is_file():
                downloaded = hf_hub_download(
                    repo_id=args.source_repo,
                    filename=f"{relative}/weights/best.pt",
                    repo_type="dataset",
                    local_dir=str(args.work_dir / "hf_cache"),
                )
                shutil.copy2(downloaded, checkpoint)

            model = YOLO(str(checkpoint))
            metrics: dict[str, float | str] = {
                "source_repo": args.source_repo,
                "source_prefix": relative,
                "data_yaml": str(data_yaml),
                "split_seed": 42,
                "nms_iou": 0.5,
            }
            for split in ("val", "test"):
                result = model.val(
                    data=str(data_yaml), split=split, imgsz=args.imgsz,
                    batch=args.batch_size, device=args.device, workers=args.workers,
                    plots=False, iou=0.5, project=str(out_dir / "evaluation"),
                    name=split, exist_ok=True,
                )
                metrics.update(split_metrics(result, split))

            output = out_dir / "posthoc_test_metrics.json"
            output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            remote_path = f"{relative}/posthoc_test_metrics.json"
            api.upload_file(
                path_or_fileobj=str(output), path_in_repo=remote_path,
                repo_id=args.output_repo, repo_type="dataset",
            )
            remote_files = set(api.list_repo_files(repo_id=args.output_repo, repo_type="dataset"))
            if remote_path not in remote_files:
                raise RuntimeError(f"Missing uploaded post-hoc artifact: {remote_path}")
            print(
                json.dumps({
                    "variant": variant,
                    "seed": seed,
                    "val/AP50": metrics["val/AP50"],
                    "val/mAP50-95": metrics["val/mAP50-95"],
                    "test/AP50": metrics["test/AP50"],
                    "test/mAP50-95": metrics["test/mAP50-95"],
                    "remote_repo": args.output_repo,
                    "remote_path": remote_path,
                }, sort_keys=True),
                flush=True,
            )


if __name__ == "__main__":
    main()
