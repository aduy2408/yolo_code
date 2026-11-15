from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for job in manifest["jobs"]:
        checkpoint = args.checkpoint_root / job["checkpoint_rel"]
        print("\t".join((str(job["job_id"]), str(job["dataset"]), str(job["image_size"]), str(checkpoint))))


if __name__ == "__main__":
    main()
