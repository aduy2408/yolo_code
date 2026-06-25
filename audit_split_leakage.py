#!/usr/bin/env python3
"""Audit how many source test records land in generated training splits."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from prepare_dataset import (
    CLASS_POLICIES,
    GT_SOURCES,
    SPLITS,
    assign_output_splits,
    dedupe_records,
    filter_records,
    load_records,
    resolve_repo_root,
)


@dataclass(frozen=True)
class SeedAudit:
    seed: int
    train_sources: Counter[str]
    val_sources: Counter[str]
    test_sources: Counter[str]
    train_total: int
    test_in_train: int
    pct_test_in_train: float


def format_sources(counts: Counter[str]) -> str:
    return ",".join(f"{split}={counts.get(split, 0)}" for split in SPLITS)


def audit_seed(records, *, seed: int, total_source_test: int) -> SeedAudit:
    split_records = assign_output_splits(records, seed=seed)
    train_sources = Counter(record.source_split for record in split_records["train"])
    val_sources = Counter(record.source_split for record in split_records["val"])
    test_sources = Counter(record.source_split for record in split_records["test"])
    test_in_train = train_sources.get("test", 0)
    pct_test_in_train = (test_in_train / total_source_test * 100.0) if total_source_test else 0.0

    return SeedAudit(
        seed=seed,
        train_sources=train_sources,
        val_sources=val_sources,
        test_sources=test_sources,
        train_total=len(split_records["train"]),
        test_in_train=test_in_train,
        pct_test_in_train=pct_test_in_train,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find seeds that place the most original test records in the generated train split."
    )
    parser.add_argument("--root", default="data", help="Dataset root containing train/val/test folders.")
    parser.add_argument("--gt-source", default="gt_one", choices=GT_SOURCES)
    parser.add_argument(
        "--only-positives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only positive rows. Use --no-only-positives to keep token 0 rows.",
    )
    parser.add_argument("--class-policy", default="map-3-to-1", choices=CLASS_POLICIES)
    parser.add_argument("--max-seed", type=int, default=10000, help="Scan seeds from 0 to max-seed - 1.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of best seeds to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_seed < 1:
        raise ValueError("--max-seed must be at least 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    root_path = resolve_repo_root(args.root)
    records = load_records(root_path, args.gt_source)
    records = filter_records(records, only_positives=args.only_positives, class_policy=args.class_policy)
    records = dedupe_records(records)

    source_counts = Counter(record.source_split for record in records)
    total_source_test = source_counts.get("test", 0)

    print(f"root: {root_path}")
    print(f"gt_source: {args.gt_source}")
    print(f"only_positives: {str(args.only_positives).lower()}")
    print(f"class_policy: {args.class_policy}")
    print(f"records: total={len(records)} {format_sources(source_counts)}")
    print(f"scanned_seeds: 0..{args.max_seed - 1}")
    print()

    audits = [audit_seed(records, seed=seed, total_source_test=total_source_test) for seed in range(args.max_seed)]
    audits.sort(key=lambda item: (item.test_in_train, item.pct_test_in_train, -item.seed), reverse=True)

    print(
        "rank seed test_in_train pct_test_in_train train_total "
        "train_sources val_sources test_sources"
    )
    for rank, audit in enumerate(audits[: args.top_k], start=1):
        print(
            f"{rank:<4} "
            f"{audit.seed:<5} "
            f"{audit.test_in_train:<13} "
            f"{audit.pct_test_in_train:>16.2f}% "
            f"{audit.train_total:<11} "
            f"{format_sources(audit.train_sources)} "
            f"{format_sources(audit.val_sources)} "
            f"{format_sources(audit.test_sources)}"
        )


if __name__ == "__main__":
    main()
