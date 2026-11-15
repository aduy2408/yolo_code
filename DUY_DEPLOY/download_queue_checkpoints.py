from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for job in manifest["jobs"]:
        destination = args.output_root / "checkpoints" / job["job_id"] / "best.pt"
        url = f"https://huggingface.co/datasets/{job['repo_id']}/resolve/main/{job['hf_path']}"
        expected = job.get("expected_sha256")
        action = "reuse"
        if destination.is_file() and destination.stat().st_size > 0:
            actual = sha256(destination)
            if expected and actual != expected:
                raise RuntimeError(f"SHA mismatch for existing {destination}: {actual} != {expected}")
        else:
            action = "download"
            download(url, destination)
            actual = sha256(destination)
            if expected and actual != expected:
                raise RuntimeError(f"SHA mismatch for {destination}: {actual} != {expected}")
        rows.append({**job, "url": url, "local_path": str(destination), "sha256": actual, "action": action, "size": destination.stat().st_size})
        print(action, job["job_id"], destination.stat().st_size, actual)
    (args.output_root / "checkpoint_inventory.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
