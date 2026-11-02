from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "dataset",
    "model",
    "provider",
    "batch_size",
    "image_size",
    "warmup",
    "iterations",
    "mean_preprocess_ms",
    "median_preprocess_ms",
    "p95_preprocess_ms",
    "mean_inference_ms",
    "median_inference_ms",
    "p95_inference_ms",
    "mean_postprocess_ms",
    "median_postprocess_ms",
    "p95_postprocess_ms",
    "mean_total_ms",
    "median_total_ms",
    "p95_total_ms",
    "images_per_second_end_to_end",
    "images_per_second_inference_only",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results_root.rglob("*.json")):
        if path.name == "failure.json" or path.name in {"summary.json"}:
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("provider") != "TensorrtExecutionProvider":
            continue
        row["result_file"] = str(path)
        rows.append(row)
    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("model", "")))
    failures = []
    for path in sorted(args.results_root.rglob("failure.json")):
        try:
            failures.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    summary = {
        "success_count": len(rows),
        "failure_count": len(failures),
        "successes": rows,
        "failures": failures,
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.results_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + ["result_file"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"success_count": len(rows), "failure_count": len(failures), "summary": str(args.results_root / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
