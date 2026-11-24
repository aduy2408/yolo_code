#!/usr/bin/env python3
"""Backfill split-qualified and protocol-specific metrics from augmentation checkpoints.

This runner downloads the already completed augmentation checkpoints from the three
source repositories, evaluates them on the canonical split protocols, and uploads
separate evaluation-backfill artifacts. It never trains and never mutates the
original augmentation repositories.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "oacp": "duyle2408/augmentation-oacp-runs",
    "mosaic": "duyle2408/augmentation-mosaic-runs",
    "copy_paste": "duyle2408/augmentation-copy-paste-runs",
}
DATASETS = ("levir", "varroa", "tinyperson")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-levir", type=Path, required=True)
    parser.add_argument("--data-root-varroa", type=Path, required=True)
    parser.add_argument("--data-root-tinyperson", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True, help="Task-specific evaluation output repository")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100, help="Contract metadata; evaluation does not train")
    parser.add_argument("--patience", type=int, default=0, help="Contract metadata; evaluation does not train")
    parser.add_argument("--seed", type=int, default=42, help="Contract metadata")
    parser.add_argument("--model-yaml", default="matrix", help="Contract metadata")
    parser.add_argument("--data-root", default="matrix", help="Contract metadata")
    parser.add_argument("--jobs", nargs="*", help="Exact source prefixes; default evaluates all 38 checkpoints")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_upload_context(repo_id: str) -> str:
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required evaluation")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for evaluation uploads")
    if not repo_id.strip() or "/" not in repo_id:
        raise RuntimeError("--hf-repo-id must be a task-specific Hugging Face repository")
    return token


def split_metrics(result, split: str) -> dict[str, float]:
    values = {key: float(value) for key, value in result.results_dict.items()}
    return {
        **{f"{split}/{key}": value for key, value in values.items()},
        f"{split}/AP50": values["metrics/mAP50(B)"],
        f"{split}/mAP50-95": values["metrics/mAP50-95(B)"],
    }


def prepare_dataset(dataset: str, args: argparse.Namespace, method: str):
    data_roots = {
        "levir": args.data_root_levir,
        "varroa": args.data_root_varroa,
        "tinyperson": args.data_root_tinyperson,
    }
    root = data_roots[dataset]
    if dataset == "levir":
        from misc.prepare_levir_ship import prepare
        out = args.dataset_root / f"levir_ship_augmentation_matrix_split_{args.split_seed}"
        return prepare(root, out, args.split_seed), out
    if dataset == "tinyperson":
        from train_scripts import train_all_tinyperson as workflow
        test_out = workflow.prepare_test_set(root, args.dataset_root)
        split_root = workflow.prepare_seed_dataset(root, args.dataset_root, test_out, args.split_seed)
        return split_root / "tinyperson.yaml", test_out
    from misc.prepare_dataset import prepare_dataset as prepare_varroa
    only_positives = method != "copy_paste"
    suffix = "copy_paste" if method == "copy_paste" else "augmentation"
    out = args.dataset_root / f"varroa_{suffix}_matrix_split_{args.split_seed}"
    yaml_path = prepare_varroa(
        root, out, gt_source="gt_one", only_positives=only_positives,
        class_policy="map-3-to-1", seed=args.split_seed,
    )
    return yaml_path, out


def source_prefixes(api) -> list[tuple[str, str]]:
    output = []
    for method, repo in REPOS.items():
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        for path in files:
            if path.endswith("/weights/best.pt"):
                output.append((repo, path[: -len("/weights/best.pt")]))
    return sorted(output)


def evaluate_one(api, args: argparse.Namespace, repo: str, prefix: str, prepared: dict):
    dataset, method, variant, mosaic_mode, seed_name = prefix.split("/")
    seed = int(seed_name.removeprefix("seed_"))
    slug = repo.split("/", 1)[1].replace("/", "_")
    out_dir = args.project / slug / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_base = f"{slug}/{prefix}"
    marker_remote = f"{remote_base}/evaluation_backfill_complete.json"
    if not args.force and marker_remote in set(api.list_repo_files(args.hf_repo_id, repo_type="dataset")):
        print(f"SKIP {prefix}", flush=True)
        return

    import huggingface_hub
    checkpoint = out_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=repo, filename=f"{prefix}/weights/best.pt", repo_type="dataset",
            local_dir=str(args.project / "hf_cache" / slug),
        )
        shutil.copy2(downloaded, checkpoint)

    data_yaml, test_root = prepared[dataset, method]
    from train_scripts.train_all_augmentation_matrix import _evaluate
    eval_args = SimpleNamespace(
        data_roots={
            "levir": args.data_root_levir,
            "varroa": args.data_root_varroa,
            "tinyperson": args.data_root_tinyperson,
        },
        dataset_root=args.dataset_root,
        imgsz={"levir": args.image_size, "varroa": args.image_size, "tinyperson": args.image_size},
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
        nms_iou=args.nms_iou,
    )
    spec = {"dataset": dataset, "method": method, "variant": variant, "mosaic_mode": mosaic_mode}
    metrics = _evaluate(out_dir, data_yaml, spec, eval_args, test_root)
    metrics.update({
        "source_repo": repo,
        "source_prefix": prefix,
        "evaluation_commit": subprocess_git_sha(),
        "split_seed": args.split_seed,
        "nms_iou": args.nms_iou,
        "test_protocol": (
            "TinyPerson standard corner-window test plus merged evaluator"
            if dataset == "tinyperson"
            else "LEVIR-Ship standard held-out test split"
            if dataset == "levir"
            else "Varroa standard held-out test split"
        ),
    })
    metrics_path = out_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "source_repo": repo,
        "source_prefix": prefix,
        "output_repo": args.hf_repo_id,
        "output_prefix": remote_base,
        "dataset": dataset,
        "method": method,
        "variant": variant,
        "mosaic_mode": mosaic_mode,
        "seed": seed,
        "split_seed": args.split_seed,
        "nms_iou": args.nms_iou,
        "checkpoint": f"{prefix}/weights/best.pt",
        "test_protocol": metrics["test_protocol"],
        "metrics": {key: metrics[key] for key in ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")},
    }
    manifest_path = out_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = out_dir / "evaluation_backfill_complete.json"
    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "remote_prefix": remote_base, "verified": True}, indent=2) + "\n", encoding="utf-8")
    for local, remote in ((metrics_path, f"{remote_base}/evaluation_metrics.json"), (manifest_path, f"{remote_base}/evaluation_manifest.json"), (marker, marker_remote)):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=args.hf_repo_id, repo_type="dataset")
    remote = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
    expected = {f"{remote_base}/{name}" for name in ("evaluation_metrics.json", "evaluation_manifest.json", "evaluation_backfill_complete.json")}
    missing = sorted(expected - remote)
    if missing:
        raise RuntimeError(f"Remote verification failed for {prefix}: {missing}")
    print(json.dumps({"prefix": prefix, "val/AP50": metrics["val/AP50"], "val/mAP50-95": metrics["val/mAP50-95"], "test/AP50": metrics["test/AP50"], "test/mAP50-95": metrics["test/mAP50-95"]}, sort_keys=True), flush=True)


def subprocess_git_sha() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    token = require_upload_context(args.hf_repo_id)
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    jobs = source_prefixes(api)
    if args.jobs:
        wanted = set(args.jobs)
        jobs = [(repo, prefix) for repo, prefix in jobs if prefix in wanted]
        missing = sorted(wanted - {prefix for _, prefix in jobs})
        if missing:
            raise RuntimeError(f"Requested prefixes not found: {missing}")
    required_pairs = {(prefix.split("/")[0], prefix.split("/")[1]) for _, prefix in jobs}
    prepared = {
        pair: prepare_dataset(pair[0], args, pair[1])
        for pair in sorted(required_pairs)
    }
    print(json.dumps({"jobs": len(jobs), "output_repo": args.hf_repo_id}, sort_keys=True), flush=True)
    for repo, prefix in jobs:
        evaluate_one(api, args, repo, prefix, prepared)


if __name__ == "__main__":
    main()
