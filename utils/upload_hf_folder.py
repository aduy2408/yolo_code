#!/usr/bin/env python3
"""Upload and verify a completed workflow artifact folder on Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--prefix", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not args.folder.is_dir():
        raise RuntimeError(f"Artifact folder does not exist: {args.folder}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(args.folder),
        path_in_repo=args.prefix.strip("/"),
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    remote_files = {
        item.rfilename
        for item in api.list_repo_tree(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo=args.prefix.strip("/"),
            recursive=True,
        )
        if getattr(item, "rfilename", None)
    }
    expected = {
        f"{args.prefix.strip('/')}/{path.relative_to(args.folder).as_posix()}"
        for path in args.folder.rglob("*")
        if path.is_file()
    }
    missing = sorted(expected - remote_files)
    if missing:
        raise RuntimeError(f"Hugging Face upload verification failed: {missing}")

    marker = args.folder / "hf_upload_complete.json"
    marker.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "remote_prefix": args.prefix.strip("/"),
                "verified_files": sorted(expected),
                "remote_verified": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{args.prefix.strip('/')}/hf_upload_complete.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(json.dumps({"repo_id": args.repo_id, "remote_prefix": args.prefix, "files": len(expected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
