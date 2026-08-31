#!/usr/bin/env python3
"""Pull LEVIR-Ship MMDetection baselines and measure parameters/GFLOPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


MODELS = ("atss", "cascade_rcnn", "faster_rcnn", "fcos", "retinanet", "rtmdet")
SEEDS = (42, 43, 44)
SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)\s*$", re.IGNORECASE)


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_size(value: str, target_unit: str) -> float:
    match = SIZE_RE.match(value)
    if not match:
        raise ValueError(f"Cannot parse formatted size: {value!r}")
    number = float(match.group(1))
    source = match.group(2).upper()
    powers = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}
    return number * (1000 ** (powers[source] - powers[target_unit.upper()]))


def parse_complexity_output(output: str) -> dict[str, float]:
    flops = re.search(r"^Flops:\s*(.+?)\s*$", output, re.MULTILINE)
    params = re.search(r"^Params:\s*(.+?)\s*$", output, re.MULTILINE)
    if not flops or not params:
        raise ValueError(f"Missing Flops/Params in output:\n{output[-4000:]}")
    return {
        "gflops": parse_size(flops.group(1), "G"),
        "parameters_m": parse_size(params.group(1), "M"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_files(api: Any, repo_id: str, models: list[str]) -> tuple[str, dict[str, dict[str, Any]]]:
    info = api.dataset_info(repo_id, files_metadata=True)
    siblings = {item.rfilename: item for item in info.siblings}
    discovered: dict[str, dict[str, Any]] = {}
    for model in models:
        checkpoints = [
            item for path, item in siblings.items()
            if path.startswith(f"{model}/best_") and path.endswith(".pth")
        ]
        if len(checkpoints) != 1:
            raise RuntimeError(f"{repo_id}/{model}: expected one best checkpoint, got {len(checkpoints)}")
        config = f"{model}/patched_config.py"
        if config not in siblings:
            raise RuntimeError(f"{repo_id}: missing {config}")
        checkpoint = checkpoints[0]
        discovered[model] = {
            "checkpoint": checkpoint.rfilename,
            "checkpoint_size": checkpoint.size,
            "config": config,
        }
    return info.sha, discovered


def download_file(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    destination: Path,
    token: str,
) -> Path:
    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    )
    if not destination.is_file() or destination.stat().st_size != cached.stat().st_size:
        shutil.copy2(cached, destination)
    return destination


def model_fingerprint(python: str, config: Path, mmdet_root: Path) -> str:
    code = (
        "import hashlib,json; from mmengine.config import Config; "
        f"c=Config.fromfile({str(config)!r}); "
        "x=c.model.to_dict() if hasattr(c.model,'to_dict') else dict(c.model); "
        "print(hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest())"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(mmdet_root), env.get("PYTHONPATH", "")))
    return subprocess.check_output([python, "-c", code], env=env, text=True).strip()


def measure_complexity(
    python: str,
    config: Path,
    mmdet_root: Path,
    ann_file: Path,
    image_root: Path,
) -> tuple[dict[str, float], str]:
    command = [
        python,
        str(mmdet_root / "tools/analysis_tools/get_flops.py"),
        str(config),
        "--num-images",
        "1",
        "--cfg-options",
        f"val_dataloader.dataset.ann_file={ann_file}",
        f"val_dataloader.dataset.data_prefix.img={image_root}/",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(mmdet_root), env.get("PYTHONPATH", "")))
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        command,
        cwd=mmdet_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"MMDetection FLOPs command failed ({result.returncode}):\n{result.stdout}")
    return parse_complexity_output(result.stdout), result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-owner", default="duyle2408")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmdet-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import HfApi
    from utils.marimo_ops import require_training_context

    require_training_context()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    models = comma_list(args.models)
    seeds = [int(seed) for seed in comma_list(args.seeds)]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if not args.mmdet_root.is_dir():
        raise FileNotFoundError(args.mmdet_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    rows: list[dict[str, Any]] = []
    inventories: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        repo_id = f"{args.hf_owner}/levir_ship_mmdet_runs_seed{seed}"
        revision, files = discover_files(api, repo_id, models)
        inventories[seed] = {"repo_id": repo_id, "revision": revision, "files": files}
        for model in models:
            item = files[model]
            target = args.output_dir / f"seed_{seed}" / model
            checkpoint = download_file(
                repo_id=repo_id,
                revision=revision,
                filename=item["checkpoint"],
                destination=target / Path(item["checkpoint"]).name,
                token=token,
            )
            config = download_file(
                repo_id=repo_id,
                revision=revision,
                filename=item["config"],
                destination=target / "patched_config.py",
                token=token,
            )
            expected_size = item["checkpoint_size"]
            if expected_size is not None and checkpoint.stat().st_size != expected_size:
                raise RuntimeError(f"Size mismatch: {checkpoint}")
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "repo_id": repo_id,
                    "revision": revision,
                    "checkpoint_source": item["checkpoint"],
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": sha256(checkpoint),
                    "config_path": str(config),
                    "model_fingerprint": model_fingerprint(args.python, config, args.mmdet_root),
                }
            )
            (target / "download_manifest.json").write_text(json.dumps(rows[-1], indent=2) + "\n")
            print(f"downloaded {model} seed {seed}: {checkpoint.stat().st_size / 2**20:.1f} MiB", flush=True)

    complexities: dict[str, dict[str, Any]] = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        fingerprints = {row["model_fingerprint"] for row in model_rows}
        if len(fingerprints) != 1:
            raise RuntimeError(f"{model}: model configs differ across seeds: {fingerprints}")
        config = Path(model_rows[0]["config_path"])
        metrics, raw = measure_complexity(
            args.python,
            config,
            args.mmdet_root,
            args.ann_file,
            args.image_root,
        )
        complexities[model] = {
            **metrics,
            "input_size": [512, 512],
            "model_fingerprint": next(iter(fingerprints)),
            "applies_to_seeds": seeds,
        }
        model_dir = args.output_dir / "complexity" / model
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "get_flops_output.txt").write_text(raw, encoding="utf-8")
        (model_dir / "complexity.json").write_text(json.dumps(complexities[model], indent=2) + "\n")
        print(f"{model}: {metrics['parameters_m']:.3f} M params, {metrics['gflops']:.3f} GFLOPs", flush=True)

    manifest = {
        "models": models,
        "seeds": seeds,
        "mmdet_root": str(args.mmdet_root.resolve()),
        "python": str(Path(args.python).resolve()),
        "complexity_ann_file": str(args.ann_file.resolve()),
        "complexity_image_root": str(args.image_root.resolve()),
        "inventories": inventories,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "downloads.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "complexity.json").write_text(json.dumps(complexities, indent=2) + "\n")


if __name__ == "__main__":
    main()
