from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    queue = json.loads((args.root / "queue_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for job in queue["jobs"]:
        path = args.root / "checkpoints" / job["job_id"] / "best.pt"
        rows.append({**job, "local_path": str(path), "status": "ready" if path.is_file() else "missing", "size": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None})
    for job in queue.get("skipped_jobs", []):
        path = args.root / "checkpoints" / job["job_id"] / "best.pt"
        rows.append({**job, "local_path": str(path), "status": "ready_reused" if path.is_file() else "unavailable", "size": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None})
    rows.sort(key=lambda row: row["job_id"])
    (args.root / "checkpoint_inventory.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "image": args.image,
        "queue_manifest": str(args.root / "queue_manifest.json"),
        "checkpoint_inventory": str(args.root / "checkpoint_inventory.json"),
        "protocol": {"rect": False, "batch_size": 1, "half": False, "augment": False, "conf": 0.001, "nms_iou": 0.5, "max_det": 300, "splits": ["val", "test"], "image_sizes": {"levirship": 512, "tinyperson": 640, "varroa": 640}},
        "dataset_yaml": {name: str(args.root / "tools" / f"{name}_jetson.yaml") for name in ("levirship", "tinyperson", "varroa")},
        "evaluator": str(args.root / "tools" / "evaluate.py"),
        "evaluator_sha256": sha256(args.root / "tools" / "evaluate.py"),
        "wrapper": str(args.root / "tools" / "jetson_unified_evaluate.py"),
        "wrapper_sha256": sha256(args.root / "tools" / "jetson_unified_evaluate.py"),
        "job_count": len(rows),
        "ready_count": sum(row["status"] in {"ready", "ready_reused"} for row in rows),
        "unavailable_count": sum(row["status"] == "unavailable" for row in rows),
    }
    (args.root / "run_manifest.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: run[key] for key in ("host", "image", "job_count", "ready_count", "unavailable_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
