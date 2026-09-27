"""Run all four TOD probes on the available local dataset/checkpoint matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import causal_intervention, gradient_probe, information_probe, object_survival_probe
from .common import DATASETS, ProbeConfig, set_seed, write_json


DEFAULT_CHECKPOINTS = {
    # The report-listed trained artifacts include custom Detect variants that
    # are not importable by every historical fork in this checkout. The
    # canonical local YOLOv8n baseline is the compatible control for probing.
    "levir-ship": Path("yolov8n.pt"),
    "visdrone": Path("yolov8n.pt"),
    "tinyperson": Path("yolov8n.pt"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/tod_probes"))
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--checkpoint", action="append", default=[], help="dataset=checkpoint override")
    args = parser.parse_args()
    overrides = dict(value.split("=", 1) for value in args.checkpoint)
    config = ProbeConfig(max_images=args.max_images, device=args.device)
    set_seed(config.seed)
    all_summaries = {}
    for key in args.datasets:
        checkpoint = Path(overrides.get(key, DEFAULT_CHECKPOINTS[key]))
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[2] / checkpoint
        dataset_output = args.output / key
        dataset_output.mkdir(parents=True, exist_ok=True)
        spec = DATASETS[key]
        all_summaries[key] = {
            "causal": causal_intervention.run(spec, checkpoint, dataset_output, config),
            "information": information_probe.run(spec, checkpoint, dataset_output, config),
            "survival": object_survival_probe.run(spec, checkpoint, dataset_output, config),
            "gradient": gradient_probe.run(spec, checkpoint, dataset_output, config),
        }
    write_json(args.output / "manifest.json", {"config": vars(args), "datasets": all_summaries})


if __name__ == "__main__":
    main()
