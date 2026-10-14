"""Run the complete LevirShip probe matrix through the Marimo launcher.

The compatibility arguments are part of the shared Marimo run contract. They
are recorded for provenance but do not alter the diagnostic probe protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import causal_intervention, gradient_probe, information_probe, object_survival_probe
from .common import DATASETS, ProbeConfig, file_digest, git_commit, set_seed, write_json


def _parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        name, separator, checkpoint = value.partition("=")
        if not separator or not name or not checkpoint:
            raise ValueError(f"Checkpoint must use name=path syntax: {value!r}")
        checkpoints[name] = Path(checkpoint)
    if not checkpoints:
        raise ValueError("At least one --checkpoint name=path is required")
    return checkpoints


def _run_one(name: str, checkpoint: Path, output: Path, config: ProbeConfig) -> dict[str, Any]:
    spec = DATASETS["levir-ship"]
    model_output = output / name
    model_output.mkdir(parents=True, exist_ok=True)
    summary = {
        "causal": causal_intervention.run(spec, checkpoint, model_output, config),
        "information": information_probe.run(spec, checkpoint, model_output, config),
        "survival": object_survival_probe.run(spec, checkpoint, model_output, config),
        "gradient": gradient_probe.run(spec, checkpoint, model_output, config),
    }
    write_json(
        model_output / "probe_complete.json",
        {
            "model": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_digest(checkpoint),
            "dataset": spec.name,
            "split_protocol": spec.split,
            "max_images": config.max_images,
            "source_commit": git_commit(),
            "summary": summary,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--datasets", nargs="+", default=["levir-ship"])
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--device", default="0")

    # Shared Marimo contract fields. They are intentionally recorded and
    # validated by marimo_ops, while this probe uses its own parameters.
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--model-yaml", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--nms-iou", type=float, required=True)
    args = parser.parse_args()

    if args.datasets != ["levir-ship"]:
        raise ValueError("The current full probe is intentionally limited to LevirShip")
    checkpoints = _parse_checkpoints(args.checkpoint)
    config = ProbeConfig(seed=args.seed, imgsz=args.image_size, max_images=args.max_images, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    summaries: dict[str, Any] = {}
    for name, checkpoint in checkpoints.items():
        summaries[name] = _run_one(name, checkpoint, args.output, config)

    write_json(
        args.output / "full_probe_complete.json",
        {
            "models": list(checkpoints),
            "dataset": "LEVIR-Ship",
            "split_protocol": "all_source",
            "max_images": args.max_images,
            "device": args.device,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "source_commit": git_commit(),
            "contract": {
                "epochs": args.epochs,
                "patience": args.patience,
                "workers": args.workers,
                "image_size": args.image_size,
                "model_yaml": args.model_yaml,
                "data_root": args.data_root,
                "dataset_yaml": args.dataset_yaml,
                "hf_repo_id": args.hf_repo_id,
                "nms_iou": args.nms_iou,
            },
            "summaries": summaries,
        },
    )


if __name__ == "__main__":
    main()
