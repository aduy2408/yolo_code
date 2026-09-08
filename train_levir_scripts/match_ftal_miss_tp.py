#!/usr/bin/env python3
"""Match good-geometry misses to TP objects by size, oracle IoU, and support."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    return statistics.mean(float(row[key]) for row in rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-tolerance", type=float, default=0.05)
    parser.add_argument("--support-tolerance", type=int, default=2)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(newline="", encoding="utf-8")))
    misses = [
        row for row in rows
        if row["size_group"] in {"small", "medium"}
        and row["failure_group"] == "miss"
        and float(row["oracle_iou"]) >= 0.5
    ]
    candidates = [row for row in rows if row["failure_group"] == "tp75"]
    pairs = []
    used: set[int] = set()
    for miss in sorted(misses, key=lambda row: float(row["oracle_iou"])):
        options = []
        for index, tp in enumerate(candidates):
            if index in used or tp["size_group"] != miss["size_group"]:
                continue
            iou_gap = abs(float(tp["oracle_iou"]) - float(miss["oracle_iou"]))
            support_gap = abs(int(float(tp["support_count"])) - int(float(miss["support_count"])))
            if iou_gap <= args.iou_tolerance and support_gap <= args.support_tolerance:
                options.append((iou_gap + 0.02 * support_gap, index, tp))
        if not options:
            continue
        _, index, tp = min(options, key=lambda item: item[0])
        used.add(index)
        pairs.append({
            "size_group": miss["size_group"],
            "miss_image": miss["image"], "miss_gt_index": miss["gt_index"],
            "tp_image": tp["image"], "tp_gt_index": tp["gt_index"],
            "miss_oracle_iou": float(miss["oracle_iou"]), "tp_oracle_iou": float(tp["oracle_iou"]),
            "miss_support": int(float(miss["support_count"])), "tp_support": int(float(tp["support_count"])),
            "miss_target_mass": float(miss["target_mass"]), "tp_target_mass": float(tp["target_mass"]),
            "miss_q_max": float(miss["q_max"]), "tp_q_max": float(tp["q_max"]),
            "miss_q_mean": float(miss["q_mean"]), "tp_q_mean": float(tp["q_mean"]),
        })
    def summary(group: str) -> dict[str, float | int | None]:
        selected = [pair for pair in pairs if group == "all" or pair["size_group"] == group]
        return {
            "pairs": len(selected),
            "miss_oracle_iou": mean(selected, "miss_oracle_iou"), "tp_oracle_iou": mean(selected, "tp_oracle_iou"),
            "miss_support": mean(selected, "miss_support"), "tp_support": mean(selected, "tp_support"),
            "miss_target_mass": mean(selected, "miss_target_mass"), "tp_target_mass": mean(selected, "tp_target_mass"),
            "miss_q_max": mean(selected, "miss_q_max"), "tp_q_max": mean(selected, "tp_q_max"),
            "miss_q_mean": mean(selected, "miss_q_mean"), "tp_q_mean": mean(selected, "tp_q_mean"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "iou_tolerance": args.iou_tolerance, "support_tolerance": args.support_tolerance,
        "candidate_misses": len(misses), "matched_pairs": len(pairs),
        "summary": {group: summary(group) for group in ("all", "small", "medium")},
        "pairs": pairs,
    }, indent=2, sort_keys=True) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
