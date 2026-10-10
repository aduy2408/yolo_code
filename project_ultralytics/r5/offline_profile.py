"""Offline profile construction for R5-A.

The input JSON contains prediction records with boxes, scores, classes, and GT
boxes/classes in the same coordinate system. This keeps profile provenance
explicit and makes the evaluator reusable for R5-E.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .recall_evaluator import RecallEvaluator, RecallProfile
from .scale_bins import ScaleBinSpec


def load_records(path: str | Path) -> list[Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records")
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("records JSON must be a list or an object with a records list")
    return records


def build_offline_profile(
    records: Iterable[Mapping[str, Any]],
    *,
    bin_edges: tuple[float, ...] = (0.0, 8.0, 12.0, 16.0, 20.0),
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.01,
    source_split: str = "train",
    kappa: float = 10.0,
    gamma: float = 1.0,
) -> dict[str, object]:
    evaluator = RecallEvaluator(
        ScaleBinSpec.from_edges(bin_edges),
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )
    profile = evaluator.evaluate_records(records, method="offline_recall", source_split=source_split)
    result = profile.to_dict()
    result["kappa"] = float(kappa)
    result["gamma"] = float(gamma)
    # Persist raw smoothed miss-rate. The controller owns gamma so an
    # experiment applies the exponent exactly once when deriving P(bin).
    result["difficulty"] = profile.difficulty(kappa=kappa, gamma=1.0).tolist()
    result["difficulty_gamma"] = 1.0
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_offline_profile(
        load_records(args.records),
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        source_split=args.source_split,
        kappa=args.kappa,
        gamma=args.gamma,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["build_offline_profile", "load_records", "main", "parse_args"]

if __name__ == "__main__":
    main()
