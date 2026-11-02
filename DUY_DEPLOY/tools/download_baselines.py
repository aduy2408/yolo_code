#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checkpoints"
MODELS = ("yolov5nu", "yolov8n", "yolov9t", "yolov10n", "yolo11n")
SOURCES = {
    "levirship": ("duyle2408/levir-ship-yolo-baselines", "train/{model}_seed42/weights/best.pt"),
    "tinyperson": ("duyle2408/tinyperson-yolo-baselines", "runs/{model_short}/seed_42/weights/best.pt"),
    "varroa": ("duyle2408/varroa-yolo-baselines-mosaic-two-seed-runs", "runs/varroa/{model_short}/seed_42/weights/best.pt"),
}
MODEL_SHORT = {"yolov5nu": "yolov5", "yolov8n": "yolov8", "yolov9t": "yolov9", "yolov10n": "yolov10", "yolo11n": "yolov11"}

def request_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "DUY_DEPLOY-baseline-audit/1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)

def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "DUY_DEPLOY-baseline-audit/1"})
    with urllib.request.urlopen(req, timeout=120) as response, target.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> None:
    token = os.environ.get("HF_TOKEN")
    headers = {"User-Agent": "DUY_DEPLOY-baseline-audit/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    records = []
    for dataset, (repo, pattern) in SOURCES.items():
        api = f"https://huggingface.co/api/datasets/{repo}"
        req = urllib.request.Request(api, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            info = json.load(response)
        revision = info["sha"]
        for model in MODELS:
            path = pattern.format(model=model, model_short=MODEL_SHORT[model])
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
            target = OUT / dataset / model / "seed_42" / "best.pt"
            print(f"[{dataset}/{model}] {repo}/{path}", flush=True)
            download(url, target)
            digest = sha256(target)
            records.append({
                "dataset": dataset,
                "model": model,
                "seed": 42,
                "repo_id": repo,
                "revision": revision,
                "hf_path": path,
                "local_path": str(target.relative_to(ROOT)),
                "size_bytes": target.stat().st_size,
                "sha256": digest,
            })
    manifest = {
        "purpose": "PyTorch Ultralytics Jetson baseline benchmark",
        "models": list(MODELS),
        "datasets": list(SOURCES),
        "seed": 42,
        "records": records,
    }
    manifest_path = ROOT / "manifests" / "checkpoints_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")

if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.url}", file=sys.stderr)
        raise
