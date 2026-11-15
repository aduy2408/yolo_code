#!/usr/bin/env python3
"""Small, dependency-free controls for bounded research experiments.

This module validates experiment identity before launch and records lifecycle
state separately from the immutable manifest. It does not launch training.
Remote execution remains the responsibility of ``utils.marimo_ops``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DECISIONS = {
    "reject",
    "revise",
    "replicate",
    "promote",
    "infrastructure-investigation",
}
STATUSES = {
    "planned",
    "preflight_failed",
    "smoke_failed",
    "queued",
    "running",
    "complete_unverified",
    "complete",
    "failed",
    "abandoned",
    "rejected",
    "promoted",
}


class ResearchLoopError(ValueError):
    """An invalid manifest or ledger transition."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ResearchLoopError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResearchLoopError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResearchLoopError(f"Expected a JSON object: {path}")
    return value


def _require(mapping: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ResearchLoopError(f"{label} is missing: {', '.join(missing)}")


def _ledger_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResearchLoopError(f"Invalid ledger JSON at {path}:{number}") from exc
        if not isinstance(value, dict):
            raise ResearchLoopError(f"Ledger record at {path}:{number} is not an object")
        records.append(value)
    return records


def validate_manifest(manifest_path: Path, ledger_path: Path | None = None, max_variants: int = 3) -> dict[str, Any]:
    """Validate one-factor identity and bounded branch reuse before launch."""
    manifest = _read_json(manifest_path)
    _require(
        manifest,
        (
            "hypothesis_id",
            "experiment_id",
            "purpose",
            "baseline",
            "treatment",
            "changed_factors",
            "expected_diagnostics",
            "promotion_rule",
            "artifact_contract",
        ),
        "manifest",
    )
    hypothesis_id = manifest["hypothesis_id"]
    experiment_id = manifest["experiment_id"]
    if not isinstance(hypothesis_id, str) or not re.fullmatch(r"H\d{3,}", hypothesis_id):
        raise ResearchLoopError("hypothesis_id must match H###")
    if not isinstance(experiment_id, str) or not re.fullmatch(r"H\d{3,}-E\d{3,}", experiment_id):
        raise ResearchLoopError("experiment_id must match H###-E###")
    if not experiment_id.startswith(hypothesis_id + "-"):
        raise ResearchLoopError("experiment_id must belong to hypothesis_id")
    if not isinstance(manifest["purpose"], str) or not manifest["purpose"].strip():
        raise ResearchLoopError("purpose must be non-empty")
    baseline = manifest["baseline"]
    if not isinstance(baseline, dict):
        raise ResearchLoopError("baseline must be an object")
    _require(
        baseline,
        (
            "model_yaml",
            "provenance_class",
            "source_sha_status",
            "dataset",
            "dataset_yaml",
            "split_seed",
            "training_seed",
            "image_size",
            "workers",
            "epochs",
            "nms_iou",
        ),
        "baseline",
    )
    provenance_class = baseline["provenance_class"]
    source_sha_status = baseline["source_sha_status"]
    if provenance_class not in {"new", "historical", "recovered"}:
        raise ResearchLoopError("baseline.provenance_class must be new, historical, or recovered")
    if source_sha_status not in {"verified", "partial", "unavailable"}:
        raise ResearchLoopError("baseline.source_sha_status must be verified, partial, or unavailable")
    source_sha = baseline.get("source_sha")
    if provenance_class == "new":
        if not isinstance(source_sha, str) or len(source_sha.strip()) < 7:
            raise ResearchLoopError("new baseline requires source_sha")
        if source_sha_status != "verified":
            raise ResearchLoopError("new baseline requires source_sha_status=verified")
    elif source_sha is not None and (not isinstance(source_sha, str) or len(source_sha.strip()) < 7):
        raise ResearchLoopError("historical/recovered source_sha must be absent or at least 7 characters")
    if provenance_class in {"historical", "recovered"} and source_sha is None and source_sha_status == "verified":
        raise ResearchLoopError("source_sha_status=verified requires source_sha")
    if baseline["nms_iou"] != 0.5:
        raise ResearchLoopError("baseline.nms_iou must be exactly 0.5")
    changed = manifest["changed_factors"]
    if not isinstance(changed, list) or len(changed) != 1 or not isinstance(changed[0], str) or not changed[0].strip():
        raise ResearchLoopError("changed_factors must contain exactly one non-empty factor")
    for key in ("treatment", "promotion_rule"):
        if not isinstance(manifest[key], dict):
            raise ResearchLoopError(f"{key} must be an object")
    for key in ("expected_diagnostics", "artifact_contract"):
        value = manifest[key]
        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
            raise ResearchLoopError(f"{key} must be a non-empty list of strings")

    result = {
        "manifest": str(manifest_path),
        "hypothesis_id": hypothesis_id,
        "experiment_id": experiment_id,
        "status": "valid",
    }
    if ledger_path is None:
        return result
    records = _ledger_records(ledger_path)
    if any(record.get("experiment_id") == experiment_id for record in records):
        raise ResearchLoopError(f"experiment_id already exists in ledger: {experiment_id}")
    variants = {record.get("experiment_id") for record in records if record.get("hypothesis_id") == hypothesis_id}
    variants.discard(None)
    if len(variants) >= max_variants:
        raise ResearchLoopError(f"hypothesis {hypothesis_id} already has {len(variants)} variants; limit is {max_variants}")
    return result


def append_record(ledger_path: Path, record_path: Path) -> dict[str, Any]:
    """Append one lifecycle record, refusing duplicate run identities."""
    record = _read_json(record_path)
    _require(record, ("run_id", "hypothesis_id", "experiment_id", "status", "artifacts_verified"), "ledger record")
    if not isinstance(record["run_id"], str) or not record["run_id"].strip():
        raise ResearchLoopError("run_id must be non-empty")
    if record["status"] not in STATUSES:
        raise ResearchLoopError(f"unknown status: {record['status']}")
    if not isinstance(record["artifacts_verified"], bool):
        raise ResearchLoopError("artifacts_verified must be boolean")
    if record["status"] in {"complete", "promoted", "rejected"} and not isinstance(record.get("decision"), str):
        raise ResearchLoopError("terminal records require decision")
    if "decision" in record and record["decision"] not in DECISIONS:
        raise ResearchLoopError(f"unknown decision: {record['decision']}")
    records = _ledger_records(ledger_path)
    if any(item.get("run_id") == record["run_id"] for item in records):
        raise ResearchLoopError(f"run_id already exists in ledger: {record['run_id']}")
    record = {"recorded_at": datetime.now(timezone.utc).isoformat(), **record}
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate bounded research contracts and append ledger records")
    sub = parser.add_subparsers(dest="action", required=True)
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--ledger", type=Path)
    validate.add_argument("--max-variants", type=int, default=3)
    append = sub.add_parser("append-record")
    append.add_argument("--ledger", type=Path, required=True)
    append.add_argument("--record", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "validate-manifest":
            result = validate_manifest(args.manifest, args.ledger, args.max_variants)
        else:
            result = append_record(args.ledger, args.record)
    except ResearchLoopError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
