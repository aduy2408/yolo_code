#!/usr/bin/env python3
"""Collect per-transition band diagnostics from a project frequency model.

Example:
    PYTHONPATH=vendor/ultralytics_upstream:. conda run -n ml2 python \
      diagnostics/analyze_frequency_sampling.py \
      --checkpoint runs/frequency_sampling_v1/levir_p3p5/seed_42/weights/best.pt \
      --source datasets/levir_ship_frequency_split_42/images/val
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _frequency_modules(model) -> list[tuple[str, object]]:
    from project_ultralytics.modules.frequency_sampling import IBSDown, IBSUp, FreqDown, FreqDownV2, FreqUp, FreqUpV2

    kinds = (FreqDown, FreqDownV2, FreqUp, FreqUpV2, IBSDown, IBSUp)
    return [(name, module) for name, module in model.named_modules() if isinstance(module, kinds)]


def collect(model, predictions: Iterable[object], limit: int | None = None) -> dict[str, dict[str, float]]:
    modules = _frequency_modules(model.model)
    values: dict[str, list[float]] = defaultdict(list)
    count = 0
    for _ in predictions:
        for name, module in modules:
            for key, value in module.last_stats.items():
                values[f"{name}/{key}"].append(float(value))
        count += 1
        if limit is not None and count >= limit:
            break
    result = {}
    for name, module in modules:
        prefix = f"{name}/"
        stats = {
            key[len(prefix):]: sum(items) / len(items)
            for key, items in values.items()
            if key.startswith(prefix) and items
        }
        result[name] = {"type": module.__class__.__name__, **stats}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Project frequency model checkpoint or YAML")
    parser.add_argument("--source", required=True, help="Validation/test image directory or dataset source")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    from project_ultralytics import load_project_model

    model = load_project_model(args.checkpoint, task="detect", verbose=False)
    for _, module in _frequency_modules(model.model):
        module.record_stats = True
        module.last_stats = {}
    predictions = model.predict(source=args.source, imgsz=args.imgsz, stream=True, verbose=False)
    result = collect(model, predictions, args.limit)
    print("stage/type                 " + "  ".join(sorted({key for stats in result.values() for key in stats if key != "type"})))
    for name, stats in result.items():
        metrics = "  ".join(f"{key}={value:.5g}" for key, value in stats.items() if key != "type")
        print(f"{name}/{stats['type']:<12} {metrics}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
