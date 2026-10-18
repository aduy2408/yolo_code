#!/usr/bin/env python3
"""Evaluate the uploaded CACS checkpoints on LEVIR-Ship and TinyPerson.

The source CACS repositories use prefixes of the form
``copy_paste/<dataset>/<variant>/no_mosaic/seed_<seed>``.  This runner keeps
that source layout, evaluates split-qualified standard metrics plus native
size metrics, and uploads a separate verified evaluation bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from evaluate_test.evaluate_augmentation_size_metrics import (
    evaluate_size_metrics,
    evaluate_standard_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("levir", "tinyperson")
VARIANTS = ("negative_canvas_r2", "negative_canvas_r4")
SEEDS = (42, 44)
MODELS = ("yolov8", "yolov9", "yolov11")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", action="append", required=True, metavar="MODEL=DATASET=REPO")
    parser.add_argument("--data-root-levir", type=Path, required=True)
    parser.add_argument("--data-root-tinyperson", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_map(values: list[str]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for value in values:
        model, separator, rest = value.partition("=")
        dataset, separator2, repo = rest.partition("=")
        if not separator or not separator2 or model not in MODELS or dataset not in DATASETS or "/" not in repo:
            raise ValueError(f"Invalid --source-repo {value!r}; expected MODEL=DATASET=ORG/REPO")
        result[(model, dataset)] = repo
    expected = {(model, dataset) for model in MODELS for dataset in DATASETS}
    missing = sorted(expected - result.keys())
    if missing:
        raise ValueError(f"Missing source repositories: {missing}")
    return result


def require_context(repo_id: str) -> str:
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required evaluation")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if "/" not in repo_id:
        raise RuntimeError("--hf-repo-id must be task-specific")
    return token


def prepare_datasets(args: argparse.Namespace) -> dict[tuple[str, str], tuple[Path, Path]]:
    from misc.prepare_levir_ship import prepare as prepare_levir
    from train_scripts import train_all_tinyperson

    levir_out = args.dataset_root / f"levir_ship_cacs_eval_split_{args.split_seed}"
    levir_yaml = prepare_levir(args.data_root_levir, levir_out, args.split_seed)
    tiny_test = train_all_tinyperson.prepare_test_set(args.data_root_tinyperson, args.dataset_root)
    tiny_split = train_all_tinyperson.prepare_seed_dataset(
        args.data_root_tinyperson,
        args.dataset_root,
        tiny_test,
        args.split_seed,
    )
    tiny_yaml = tiny_split / "tinyperson.yaml"
    return {
        ("levir", "copy_paste"): (levir_yaml, levir_out),
        ("tinyperson", "copy_paste"): (tiny_yaml, tiny_test),
    }


def main() -> None:
    args = parse_args()
    token = require_context(args.hf_repo_id)
    sources = source_map(args.source_repo)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)
    prepared = prepare_datasets(args)
    remote_files = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
    eval_args = SimpleNamespace(
        batch_size=args.batch_size,
        image_size=512,
        device=args.device,
        workers=args.workers,
    )
    pending: list[CommitOperationAdd] = []
    complete = 0
    for model in MODELS:
        for dataset in DATASETS:
            source_repo = sources[(model, dataset)]
            data_yaml, test_root = prepared[(dataset, "copy_paste")]
            data_root = args.data_root_levir if dataset == "levir" else args.data_root_tinyperson
            eval_args.image_size = 512 if dataset == "levir" else 640
            for variant in VARIANTS:
                for seed in SEEDS:
                    source_prefix = f"copy_paste/{dataset}/{variant}/no_mosaic/seed_{seed}"
                    out_prefix = f"cacs/{dataset}/{model}/{variant}/seed_{seed}"
                    marker_remote = f"{out_prefix}/cacs_evaluation_complete.json"
                    if marker_remote in remote_files and not args.force:
                        print(json.dumps({"prefix": out_prefix, "status": "SKIP_VERIFIED"}), flush=True)
                        complete += 1
                        continue
                    out_dir = args.project / out_prefix
                    checkpoint = out_dir / "weights" / "best.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    if not checkpoint.is_file():
                        downloaded = hf_hub_download(
                            repo_id=source_repo,
                            filename=f"{source_prefix}/weights/best.pt",
                            repo_type="dataset",
                            local_dir=str(args.project / "hf_cache" / f"{model}_{dataset}"),
                        )
                        shutil.copy2(downloaded, checkpoint)
                    source_metrics_path = hf_hub_download(
                        repo_id=source_repo,
                        filename=f"{source_prefix}/evaluation_metrics.json",
                        repo_type="dataset",
                        local_dir=str(out_dir / "source_metrics"),
                    )
                    source_metrics = json.loads(Path(source_metrics_path).read_text())
                    size_metrics, artifacts, protocol = evaluate_size_metrics(
                        out_dir,
                        dataset,
                        data_yaml,
                        test_root,
                        data_root,
                        eval_args,
                    )
                    metrics = {**source_metrics, **size_metrics}
                    required = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")
                    if any(key not in metrics for key in required):
                        metrics.update(evaluate_standard_metrics(out_dir, data_yaml, dataset, eval_args))
                    if any(key not in metrics for key in required):
                        raise RuntimeError(f"Missing split-qualified metrics for {out_prefix}")
                    metrics.update({
                        "source_repo": source_repo,
                        "source_prefix": source_prefix,
                        "output_repo": args.hf_repo_id,
                        "output_prefix": out_prefix,
                        "model": model,
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "split_seed": args.split_seed,
                        "nms_iou": 0.5,
                        "test_protocol": "TinyPerson official corner-window merged evaluator" if dataset == "tinyperson" else "LEVIR-Ship standard held-out test split",
                        "size_metric_protocol": protocol,
                    })
                    metrics_path = out_dir / "evaluation_metrics.json"
                    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
                    manifest = out_dir / "cacs_evaluation_manifest.json"
                    manifest.write_text(json.dumps({"source_repo": source_repo, "source_prefix": source_prefix, "output_repo": args.hf_repo_id, "output_prefix": out_prefix, "dataset": dataset, "model": model, "variant": variant, "seed": seed, "split_seed": args.split_seed, "nms_iou": 0.5, "metric_keys": sorted(metrics), "source_artifacts": [str(path.relative_to(out_dir)) for path in artifacts if path.is_file()]}, indent=2, sort_keys=True) + "\n")
                    marker = out_dir / "cacs_evaluation_complete.json"
                    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "remote_prefix": out_prefix, "verified": True}, indent=2, sort_keys=True) + "\n")
                    uploads = [(metrics_path, f"{out_prefix}/evaluation_metrics.json"), (manifest, f"{out_prefix}/cacs_evaluation_manifest.json"), (marker, marker_remote)]
                    uploads.extend((artifact, f"{out_prefix}/{artifact.relative_to(out_dir)}") for artifact in artifacts if artifact.is_file())
                    pending.extend(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in uploads)
                    complete += 1
                    print(json.dumps({"prefix": out_prefix, "status": "LOCAL_COMPLETE", **{key: metrics.get(key) for key in required}, "val_size/AP50-Small": metrics.get("val_size/AP50-Small"), "test_size/AP50-Small": metrics.get("test_size/AP50-Small"), "test_merged/AP50-Small": metrics.get("test_merged/AP50-Small")}, sort_keys=True), flush=True)
    if pending:
        api.create_commit(repo_id=args.hf_repo_id, repo_type="dataset", operations=pending, commit_message="Upload CACS val test and size metrics")
        remote_files = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
        missing = sorted({operation.path_in_repo for operation in pending} - remote_files)
        if missing:
            raise RuntimeError(f"Remote verification failed: {missing}")
    print(json.dumps({"status": "COMPLETE", "jobs": complete, "uploaded_files": len(pending), "hf_repo_id": args.hf_repo_id}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
