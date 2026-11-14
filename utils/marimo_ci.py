"""Single remote entrypoint for the manual Marimo MLOps workflow."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from utils.marimo_ops import launch_detached, preflight, status, write_run_contract


def run(config: dict[str, Any]) -> None:
    repo = Path(config["repo"]).resolve()
    operation = config["operation"]
    python = config.get("python") or os.environ.get("PYTHON", "python")
    data_root = Path(config["data_root"]).resolve()
    dataset_root = Path(config["dataset_root"]).resolve()
    dataset_yaml_value = str(config.get("dataset_yaml") or "").strip()
    if operation == "train" and config["dataset"] == "levirship" and not dataset_yaml_value:
        from misc.prepare_levir_ship import prepare

        dataset_yaml_value = str(
            prepare(data_root, dataset_root / f"levir_ship_ci_split_{config['split_seed']}", int(config["split_seed"]))
        )
        config["dataset_yaml"] = dataset_yaml_value
    if operation == "evaluate" and not dataset_yaml_value:
        raise ValueError("dataset_yaml is required for evaluate")
    run_dir_value = str(config.get("run_dir") or "").strip()
    run_dir = Path(run_dir_value or repo / "runs/ci" / f"{operation}-{time.strftime('%Y%m%dT%H%M%SZ')}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    command = command_for(config, python, run_dir)
    is_train = operation == "train"
    artifact_root = (
        run_dir / "project" / f"yolov9t_oacp_seed{config['seed']}"
        if is_train
        else run_dir
    )
    required_paths = [
        "utils/marimo_ops.py",
        command[1],
    ]
    dataset_validation = operation in {"train", "evaluate"}
    contract = {
        "dataset": config["dataset"],
        "data_root": str(data_root) if dataset_validation else "matrix",
        "dataset_yaml": dataset_yaml_value if dataset_validation else "matrix",
        "model_yaml": config.get("model_yaml", "matrix"),
        "seed": config.get("seed", "matrix") if is_train else "matrix",
        "split_seed": config.get("split_seed", "matrix") if is_train else "matrix",
        "workers": config.get("workers", "matrix") if is_train else "matrix",
        "image_size": config.get("image_size", "matrix") if is_train else "matrix",
        "epochs": config.get("epochs", "matrix") if is_train else "matrix",
        "patience": config.get("patience", "matrix") if is_train else "matrix",
        "nms_iou": 0.5,
        "min_test_map50": config.get("min_test_map50", ""),
        "min_test_map50_95": config.get("min_test_map50_95", ""),
        "hf_repo_id": config.get("hf_repo_id", "matrix") if is_train else "matrix",
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        raise RuntimeError(f"Refusing to reuse run directory: {run_dir}")
    write_run_contract(run_dir, contract)

    preflight(
        repo=repo,
        expected_sha=config.get("expected_sha"),
        python=python,
        required_paths=required_paths,
        epochs=int(config["epochs"]) if is_train else None,
        patience=int(config["patience"]) if is_train else None,
        upload_required=is_train,
        hf_repo_id=config.get("hf_repo_id") if is_train else None,
        data_root=data_root if dataset_validation else None,
        dataset_yaml=Path(dataset_yaml_value) if dataset_validation else None,
        contract_json=contract_path,
    )

    environment = os.environ.copy()
    ultralytics_path = repo / "models_related/ultralytics"
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(ultralytics_path), environment.get("PYTHONPATH", "")) if item
    )
    environment["MARIMO_TRAIN_WORKFLOW"] = "1"
    launch_detached(
        command,
        cwd=repo,
        log_path=run_dir / "train.log",
        pid_path=run_dir / "train.pid",
        state_path=run_dir / "state.json",
        artifact_root=artifact_root,
        contract_path=contract_path,
        env=environment,
    )
    total_epochs = int(config["epochs"]) if operation == "train" and str(config.get("epochs", "")).isdigit() else None

    while True:
        report = status(run_dir, emit=False)
        report["training"] = training_progress(run_dir / "train.log", total_epochs)
        print(json.dumps(report, sort_keys=True), flush=True)
        if not report["process_active"]:
            state = report.get("state") or {}
            if state.get("returncode") != 0:
                raise RuntimeError(f"Remote {operation} failed: returncode={state.get('returncode')}")
            if is_train:
                quality_gate(artifact_root, config)
            break
        time.sleep(int(config.get("poll_seconds", 30)))

def quality_gate(artifact_root: Path, config: dict[str, Any]) -> None:
    for name in ("evaluation_metrics.json", "manifest.json"):
        path = artifact_root / name
        if not path.is_file():
            raise RuntimeError(f"Missing required quality-gate artifact: {path}")
    metrics_path = artifact_root / "evaluation_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if float(metrics.get("nms_iou", -1)) != 0.5:
        raise RuntimeError(f"Expected nms_iou=0.5, got {metrics.get('nms_iou')}")
    values = {
        "min_test_map50": metrics.get("test/metrics/mAP50(B)"),
        "min_test_map50_95": metrics.get("test/metrics/mAP50-95(B)"),
    }
    for name, value in values.items():
        threshold = str(config.get(name) or "").strip()
        if threshold and (value is None or float(value) < float(threshold)):
            raise RuntimeError(f"Quality gate failed: {name}={value}, required>={threshold}")

def training_progress(log_path: Path, total_epochs: int | None) -> dict[str, int | float | None]:
    if not log_path.is_file():
        return {"epoch": None, "epochs_total": total_epochs, "progress": None}
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        match = re.search(r"(?:^|\s)(\d+)\s*/\s*(\d+)(?:\s|$)", line)
        if match:
            epoch = int(match.group(1))
            epochs_total = int(match.group(2))
            return {"epoch": epoch, "epochs_total": epochs_total, "progress": epoch / epochs_total}
    return {"epoch": None, "epochs_total": total_epochs, "progress": None}


def command_for(config: dict[str, Any], python: str, run_dir: Path) -> list[str]:
    operation = config["operation"]
    if operation == "train":
        return [
            python,
            config.get("train_script", "train_scripts/train_levir_scripts/train_all_levir_yolov9t_oacp.py"),
            "--data-root", config["data_root"],
            "--dataset-root", config["dataset_root"],
            "--project", str(run_dir / "project"),
            "--hf-repo-id", config["hf_repo_id"],
            "--seed", str(config["seed"]),
            "--split-seed", str(config["split_seed"]),
            "--epochs", str(config["epochs"]),
            "--patience", str(config["patience"]),
            "--imgsz", str(config["image_size"]),
            "--batch-size", str(config["batch_size"]),
            "--device", str(config["device"]),
            "--workers", str(config["workers"]),
        ]
    if operation == "evaluate":
        return [
            python, "evaluate_test/evaluate_stw_tph_checkpoints.py",
            "--checkpoint", f"model={config['checkpoint']}",
            "--backend", config.get("backend", "stw"),
            "--datasets", config["dataset"],
            "--data-yaml", f"{config['dataset']}={config['dataset_yaml']}",
            "--output", str(run_dir), "--device", str(config["device"]),
            "--batch-size", str(config["batch_size"]), "--workers", str(config["workers"]),
        ]
    if operation == "export":
        return [
            python, "utils/export_yolo.py", "--checkpoint", config["checkpoint"],
            "--output-dir", str(run_dir / "exports"), "--format", *config.get("formats", ["onnx"]),
            "--imgsz", str(config["image_size"]), "--batch", "1", "--device", str(config["device"]),
            "--overwrite",
        ]
    raise ValueError(f"Unsupported operation: {operation}")

