#!/usr/bin/env python3
"""Backfill AP50-Small and area-bucket metrics for every YOLO augmentation checkpoint.

This runner is evaluation-only. It downloads completed checkpoints from the three
augmentation repositories, reuses the existing native TinyBenchmark bucket
 evaluator for LEVIR/Varroa, reuses the official TinyPerson merged evaluator for
TinyPerson, and uploads separate size-metric artifacts. Original training repos
are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import CommitOperationAdd

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
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100, help="Contract metadata; evaluation does not train")
    parser.add_argument("--patience", type=int, default=0, help="Contract metadata; evaluation does not train")
    parser.add_argument("--seed", type=int, default=42, help="Contract metadata")
    parser.add_argument("--model-yaml", default="source checkpoint weights/best.pt", help="Contract metadata")
    parser.add_argument("--data-root", default="matrix", help="Contract metadata")
    parser.add_argument("--defer-upload", action="store_true", help="Write local artifacts but defer all HF commits")
    parser.add_argument("--upload-only", action="store_true", help="Upload existing local artifacts without evaluating")
    parser.add_argument("--jobs", nargs="*", help="Exact source prefixes; default evaluates every uploaded checkpoint")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_upload_context(repo_id: str) -> str:
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required evaluation")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not repo_id.strip() or "/" not in repo_id:
        raise RuntimeError("--hf-repo-id must be task-specific")
    return token


def source_prefixes(api) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for repo in REPOS.values():
        for path in api.list_repo_files(repo_id=repo, repo_type="dataset"):
            if path.endswith("/weights/best.pt"):
                output.append((repo, path[: -len("/weights/best.pt")]))
    return sorted(output)


def prepare_dataset(dataset: str, args: argparse.Namespace, method: str):
    roots = {
        "levir": args.data_root_levir,
        "varroa": args.data_root_varroa,
        "tinyperson": args.data_root_tinyperson,
    }
    root = roots[dataset]
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
        root,
        out,
        gt_source="gt_one",
        only_positives=only_positives,
        class_policy="map-3-to-1",
        seed=args.split_seed,
    )
    return yaml_path, out


def download_json(huggingface_hub, repo: str, path: str, destination: Path) -> dict:
    try:
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=repo,
            filename=path,
            repo_type="dataset",
            local_dir=str(destination.parent),
        )
    except Exception:
        return {}
    try:
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def evaluate_standard_metrics(out_dir: Path, data_yaml: Path, dataset: str, args: argparse.Namespace) -> dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(out_dir / "weights/best.pt")
    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            iou=0.5,
            plots=False,
            project=str(out_dir / "evaluation"),
            name=f"standard_{split}",
            exist_ok=True,
        )
        metrics[f"{split}/AP50"] = float(result.results_dict["metrics/mAP50(B)"])
        metrics[f"{split}/mAP50-95"] = float(result.results_dict["metrics/mAP50-95(B)"])
    return metrics


def evaluate_size_metrics(out_dir: Path, dataset: str, data_yaml: Path, test_root: Path, data_root: Path, args: argparse.Namespace) -> tuple[dict, list[Path], str]:
    from train_scripts.train_all_yolo_baselines_no_mosaic import local_ultralytics

    local_ultralytics()
    eval_args = SimpleNamespace(
        batch_size=args.batch_size,
        imgsz=args.image_size,
        device=args.device,
        workers=args.workers,
    )
    from evaluate_test.size_bucket_evaluator import evaluate_native_size_buckets

    metrics: dict[str, float | str] = {}
    for split in ("val", "test"):
        metrics.update(
            evaluate_native_size_buckets(
                out_dir,
                data_yaml,
                split=split,
                imgsz=args.image_size,
                batch=args.batch_size,
                device=args.device,
                workers=args.workers,
            )
        )
    artifact_paths = [
        out_dir / "evaluation" / "val_size_ground_truth.json",
        out_dir / "evaluation" / "val_size_predictions.json",
        out_dir / "evaluation" / "test_size_ground_truth.json",
        out_dir / "evaluation" / "test_size_predictions.json",
    ]
    protocol_family = "native_test_size_and_val_size"
    if dataset == "tinyperson":
        from train_scripts.train_all_tinyperson import evaluate_merged_test

        metrics.update(evaluate_merged_test(out_dir, test_root, data_root, eval_args))
        metrics["test_merged/protocol"] = "TinyPerson official corner-window merged TinyBenchmark evaluator; IoU=0.50:0.05:0.75"
        artifact_paths.append(out_dir / "evaluation" / "test_merged_predictions.json")
        protocol_family = "native_val_test_size_and_tinyperson_test_merged"
    return metrics, artifact_paths, protocol_family


def evaluate_one(api, huggingface_hub, args: argparse.Namespace, repo: str, prefix: str, prepared: dict[tuple[str, str], tuple[Path, Path]], remote_files: set[str]) -> None:
    dataset, method, variant, mosaic_mode, seed_name = prefix.split("/")
    seed = int(seed_name.removeprefix("seed_"))
    source_slug = repo.split("/", 1)[1].replace("/", "_")
    out_dir = args.project / source_slug / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = f"{source_slug}/{prefix}"
    marker_remote = f"{remote_prefix}/size_metrics_complete.json"
    if not args.force and marker_remote in remote_files:
        print(f"SKIP_SIZE_VERIFIED {prefix}", flush=True)
        return

    checkpoint = out_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=repo,
            filename=f"{prefix}/weights/best.pt",
            repo_type="dataset",
            local_dir=str(args.project / "hf_cache" / source_slug),
        )
        shutil.copy2(downloaded, checkpoint)

    data_yaml, test_root = prepared[dataset, method]
    data_root = {
        "levir": args.data_root_levir,
        "varroa": args.data_root_varroa,
        "tinyperson": args.data_root_tinyperson,
    }[dataset]
    source_metrics = download_json(
        huggingface_hub,
        repo,
        f"{prefix}/evaluation_metrics.json",
        out_dir / "source_metrics",
    )
    size_metrics, artifacts, protocol_family = evaluate_size_metrics(
        out_dir, dataset, data_yaml, test_root, data_root, args
    )
    metrics = {
        **source_metrics,
        **size_metrics,
        "source_repo": repo,
        "source_prefix": prefix,
        "output_repo": args.hf_repo_id,
        "output_prefix": remote_prefix,
        "dataset": dataset,
        "method": method,
        "variant": variant,
        "mosaic_mode": mosaic_mode,
        "seed": seed,
        "split_seed": args.split_seed,
        "nms_iou": 0.5,
        "size_metric_family": protocol_family,
    }
    required_standard = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")
    missing_standard = [key for key in required_standard if key not in metrics]
    if missing_standard:
        metrics.update(evaluate_standard_metrics(out_dir, data_yaml, dataset, args))
    missing_standard = [key for key in required_standard if key not in metrics]
    if missing_standard:
        raise RuntimeError(f"Missing split-qualified standard metrics for {prefix}: {missing_standard}")
    metrics_path = out_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "source_repo": repo,
        "source_prefix": prefix,
        "output_repo": args.hf_repo_id,
        "output_prefix": remote_prefix,
        "dataset": dataset,
        "method": method,
        "variant": variant,
        "mosaic_mode": mosaic_mode,
        "seed": seed,
        "split_seed": args.split_seed,
        "nms_iou": 0.5,
        "checkpoint": f"{prefix}/weights/best.pt",
        "size_metric_family": protocol_family,
        "size_metric_keys": sorted(key for key in size_metrics if key.startswith(("val_size/", "test_size/", "test_merged/"))),
        "size_metric_protocols": {
            key: value for key, value in size_metrics.items()
            if key.endswith("/protocol")
        },
        "size_metric_source_artifacts": [str(path.relative_to(out_dir)) for path in artifacts if path.is_file()],
    }
    manifest_path = out_dir / "size_metrics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker_path = out_dir / "size_metrics_complete.json"
    marker_path.write_text(
        json.dumps(
            {
                "repo_id": args.hf_repo_id,
                "remote_prefix": remote_prefix,
                "protocols": {
                    key: value for key, value in size_metrics.items()
                    if key.endswith("/protocol")
                },
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.defer_upload:
        print(f"LOCAL_SIZE_COMPLETE {prefix}", flush=True)
        return
    uploads = [
        (metrics_path, f"{remote_prefix}/evaluation_metrics.json"),
        (manifest_path, f"{remote_prefix}/size_metrics_manifest.json"),
        (marker_path, marker_remote),
    ]
    for artifact in artifacts:
        if artifact.is_file():
            uploads.append((artifact, f"{remote_prefix}/{artifact.relative_to(out_dir)}"))
    api.create_commit(
        repo_id=args.hf_repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for local, remote in uploads
        ],
        commit_message=f"Add size metrics for {prefix}",
    )
    print(
        json.dumps(
            {
                "prefix": prefix,
                "size_metric_family": protocol_family,
                "size_metric_keys": sorted(key for key in size_metrics if key.startswith(("val_size/", "test_size/", "test_merged/"))),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def upload_local_artifacts(api, args: argparse.Namespace, remote_files: set[str]) -> None:
    groups: dict[str, list[CommitOperationAdd]] = {}
    for marker_path in sorted(args.project.rglob("size_metrics_complete.json")):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            remote_prefix = str(marker["remote_prefix"])
            manifest_path = marker_path.parent / "size_metrics_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            relative_paths = [
                "evaluation_metrics.json",
                "size_metrics_manifest.json",
                "size_metrics_complete.json",
                *manifest.get("size_metric_source_artifacts", []),
            ]
            if f"{remote_prefix}/size_metrics_complete.json" in remote_files:
                continue
            operations = []
            for relative in sorted(set(relative_paths)):
                local = marker_path.parent / relative
                if local.is_file():
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=f"{remote_prefix}/{relative}",
                            path_or_fileobj=str(local),
                        )
                    )
            if operations:
                groups[remote_prefix] = operations
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    prefixes = sorted(groups)
    for start in range(0, len(prefixes), 8):
        chunk = prefixes[start : start + 8]
        api.create_commit(
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            operations=[operation for prefix in chunk for operation in groups[prefix]],
            commit_message=f"Add deferred size metrics {start + 1}-{start + len(chunk)}",
        )
        print(f"UPLOAD_BATCH_COMPLETE {len(chunk)}", flush=True)
    print(f"UPLOAD_ONLY_COMPLETE {len(prefixes)}", flush=True)


def main() -> None:
    args = parse_args()
    token = require_upload_context(args.hf_repo_id)
    from huggingface_hub import HfApi
    import huggingface_hub

    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    remote_files = set(api.list_repo_files(repo_id=args.hf_repo_id, repo_type="dataset"))
    if args.upload_only:
        upload_local_artifacts(api, args, remote_files)
        return
    jobs = source_prefixes(api)
    if args.jobs:
        wanted = set(args.jobs)
        jobs = [(repo, prefix) for repo, prefix in jobs if prefix in wanted]
        missing = sorted(wanted - {prefix for _, prefix in jobs})
        if missing:
            raise RuntimeError(f"Requested prefixes not found: {missing}")
    if not jobs:
        raise RuntimeError("No augmentation checkpoints found")
    required_pairs = {(prefix.split("/")[0], prefix.split("/")[1]) for _, prefix in jobs}
    prepared = {
        pair: prepare_dataset(pair[0], args, pair[1])
        for pair in sorted(required_pairs)
    }
    print(json.dumps({"jobs": len(jobs), "output_repo": args.hf_repo_id}, sort_keys=True), flush=True)
    for repo, prefix in jobs:
        evaluate_one(api, huggingface_hub, args, repo, prefix, prepared, remote_files)
        remote_files.add(f"{repo.split('/', 1)[1].replace('/', '_')}/{prefix}/size_metrics_complete.json")


if __name__ == "__main__":
    main()
