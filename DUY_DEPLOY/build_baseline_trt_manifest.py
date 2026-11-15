from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the verified 38-job baseline TensorRT manifest.")
    parser.add_argument("--inventory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jobs: dict[str, dict[str, object]] = {}
    for inventory_path in args.inventory:
        for row in json.loads(inventory_path.read_text(encoding="utf-8")):
            if row.get("kind") != "baseline" or row.get("status") != "ready":
                continue
            dataset = str(row.get("dataset_name", row["dataset"]))
            model = str(row["model"])
            seed = int(row["seed"])
            if dataset == "varroa" and seed == 44:
                continue
            if dataset == "levirship" and model == "yolo11n" and seed == 43:
                continue
            job_id = str(row["job_id"])
            if job_id in jobs:
                raise SystemExit(f"DUPLICATE_JOB {job_id}")
            jobs[job_id] = {
                "job_id": job_id,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "image_size": row["image_size"],
                "repo_id": row["repo_id"],
                "revision": row.get("revision"),
                "hf_path": row["hf_path"],
                "expected_sha256": row.get("sha256") or row.get("expected_sha256"),
                "checkpoint_rel": f"checkpoints/{job_id}/best.pt",
            }

    ordered = [jobs[key] for key in sorted(jobs)]
    if len(ordered) != 38:
        raise SystemExit(f"BASELINE_COUNT_MISMATCH {len(ordered)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "job_count": len(ordered),
                "source_inventories": [str(path) for path in args.inventory],
                "protocol": {"batch_size": 1, "conf": 0.001, "nms_iou": 0.5, "max_det": 300},
                "jobs": ordered,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"MANIFEST_OK jobs={len(ordered)} output={args.output}")


if __name__ == "__main__":
    main()
