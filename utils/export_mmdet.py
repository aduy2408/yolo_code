#!/usr/bin/env python3
"""Export an MMDetection checkpoint through MMDeploy to ONNX or TensorRT.

The target interpreter must have MMDetection, MMDeploy, and the selected backend
installed. This wrapper also downloads a checkpoint/config from a Hugging Face
dataset repository when requested.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_CONFIG_NAMES = ("patched_config.py", "config.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Local MMDetection .pth checkpoint.")
    source.add_argument("--repo-id", help="Hugging Face dataset/model repository containing the run artifacts.")
    parser.add_argument("--config", type=Path, help="Local MMDetection config.py.")
    parser.add_argument("--hf-checkpoint-path", help="Exact .pth path in the Hugging Face repository.")
    parser.add_argument("--hf-config-path", help="Exact .py config path in the Hugging Face repository.")
    parser.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    parser.add_argument("--deploy-config", type=Path, required=True, help="MMDeploy deploy config for this backend.")
    parser.add_argument("--mmdet-root", type=Path, help="MMDetection checkout containing custom project modules.")
    parser.add_argument("--backend", choices=("onnx", "tensorrt"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Representative image used by MMDeploy export.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with MMDetection/MMDeploy/backend installed.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def download_hf(repo_id: str, repo_type: str, checkpoint_path: str | None, config_path: str | None, cache: Path) -> tuple[Path, Path]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for --repo-id") from exc
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = [
        item.rfilename
        for item in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True)
        if getattr(item, "rfilename", "")
    ]
    checkpoints = sorted(path for path in files if path.lower().endswith((".pth", ".pt")))
    if checkpoint_path:
        if checkpoint_path not in files:
            raise FileNotFoundError(f"Checkpoint is not in {repo_id}: {checkpoint_path}")
    elif checkpoints:
        checkpoint_path = next((path for path in checkpoints if path.lower().endswith("/best.pth")), checkpoints[0])
    else:
        raise FileNotFoundError(f"No .pth/.pt checkpoint found in {repo_id}")
    configs = sorted(path for path in files if path.lower().endswith(".py"))
    if config_path:
        if config_path not in files:
            raise FileNotFoundError(f"Config is not in {repo_id}: {config_path}")
    else:
        config_path = next((path for name in REMOTE_CONFIG_NAMES for path in configs if path.lower().endswith(name)), None)
        if config_path is None:
            raise FileNotFoundError(f"No patched_config.py or config.py found in {repo_id}; pass --hf-config-path")
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(hf_hub_download(repo_id, checkpoint_path, repo_type=repo_type, token=token, local_dir=cache))
    config = Path(hf_hub_download(repo_id, config_path, repo_type=repo_type, token=token, local_dir=cache))
    print(f"Resolved HF artifacts: {repo_id}/{config_path}, {repo_id}/{checkpoint_path}")
    return config, checkpoint


def main() -> None:
    args = parse_args()
    if args.work_dir.exists() and any(args.work_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Work directory is not empty: {args.work_dir}; use --overwrite or choose another directory")
    if args.repo_id and args.config:
        raise ValueError("Use either --config or --repo-id; --config is downloaded automatically with --repo-id")
    if args.repo_id:
        config, checkpoint = download_hf(
            args.repo_id, args.repo_type, args.hf_checkpoint_path, args.hf_config_path, args.work_dir / "hf_cache"
        )
    else:
        config = args.config
        checkpoint = args.checkpoint
        if config is None:
            raise ValueError("--config is required with --checkpoint")
        if not config.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("Both --config and --checkpoint must point to existing files")
    deploy = args.mmdeploy_root / "tools" / "deploy.py"
    if not deploy.is_file():
        raise FileNotFoundError(f"MMDeploy deploy.py not found: {deploy}")
    if not args.deploy_config.is_file():
        raise FileNotFoundError(f"Deploy config not found: {args.deploy_config}")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(deploy),
        str(args.deploy_config),
        str(config),
        str(checkpoint),
        str(args.input),
        "--work-dir",
        str(args.work_dir),
        "--device",
        args.device,
        "--dump-info",
    ]
    env = os.environ.copy()
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if args.mmdet_root:
        if not args.mmdet_root.is_dir():
            raise FileNotFoundError(f"MMDetection root not found: {args.mmdet_root}")
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(args.mmdet_root), env.get("PYTHONPATH", "")) if item
        )
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=args.mmdeploy_root, check=True, env=env)
    suffix = ".onnx" if args.backend == "onnx" else ".engine"
    outputs = sorted(args.work_dir.rglob(f"*{suffix}"))
    if not outputs:
        raise RuntimeError(f"MMDeploy completed but produced no {suffix} file under {args.work_dir}")
    manifest = {
        "backend": args.backend,
        "config": str(config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "deploy_config": str(args.deploy_config.resolve()),
        "input": str(args.input.resolve()),
        "device": args.device,
        "outputs": [str(path.resolve()) for path in outputs],
    }
    (args.work_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
