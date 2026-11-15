#!/usr/bin/env python3
"""Queue baseline .pt and upstream Ultralytics .engine evaluation on Jetson."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


DEFAULT_PROTOCOL = {"batch_size": 1, "conf": 0.001, "nms_iou": 0.5, "max_det": 300}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--image", default="duy-yolo-pytorch:jetpack6")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--max-seeds", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-pt", action="store_true")
    parser.add_argument("--job", action="append")
    parser.add_argument("--tinyperson-save-only", action="store_true")
    parser.add_argument("--pt-only", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_jobs(manifest: dict, max_seeds: int, requested: set[str] | None) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for job in manifest["jobs"]:
        if requested and job["job_id"] not in requested:
            continue
        groups[(job["dataset"], job["model"])].append(job)
    jobs = []
    for key in sorted(groups):
        jobs.extend(sorted(groups[key], key=lambda row: (row["seed"], row["job_id"]))[:max_seeds])
    return jobs


def run_stage(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def docker_base(args: argparse.Namespace) -> list[str]:
    return ["docker", "run", "--rm", "--runtime=nvidia", "--ipc=host", "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics"]


def evaluate_command(args: argparse.Namespace, job: dict, model_path: str, result_dir: Path, engine_dir: Path | None = None) -> list[str]:
    checkpoint_root = args.checkpoint_root.resolve()
    data_root = args.data_root.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    command = docker_base(args)
    command += ["-v", f"{checkpoint_root}:/checkpoints:ro", "-v", f"{data_root}:/datasets:ro", "-v", f"{result_dir.resolve()}:/results"]
    if engine_dir is not None:
        command += ["-v", f"{engine_dir.resolve()}:/engine:ro"]
    command += ["--entrypoint", "python3", args.image]
    if job["dataset"] == "tinyperson":
        command += [
            "/app/evaluate_ultralytics_tinyperson.py",
            "--model", model_path,
            "--data-root", "/datasets",
            "--results-root", "/results",
            "--precision", args.precision,
            "--conf", str(DEFAULT_PROTOCOL["conf"]),
            "--iou", str(DEFAULT_PROTOCOL["nms_iou"]),
            "--max-det", str(DEFAULT_PROTOCOL["max_det"]),
        ]
        if args.tinyperson_save_only:
            command.append("--save-only")
    else:
        command += [
            "/app/evaluate_ultralytics.py",
            "--model", model_path,
            "--data-root", "/datasets",
            "--dataset", job["dataset"],
            "--split", "val", "test",
            "--results-root", "/results",
            "--precision", args.precision,
            "--conf", str(DEFAULT_PROTOCOL["conf"]),
            "--iou", str(DEFAULT_PROTOCOL["nms_iou"]),
            "--max-det", str(DEFAULT_PROTOCOL["max_det"]),
        ]
    return command


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = selected_jobs(manifest, args.max_seeds, set(args.job) if args.job else None)
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("No jobs selected")
    args.results_root.mkdir(parents=True, exist_ok=True)
    selected_path = args.results_root / "queue_manifest.json"
    selected_path.write_text(json.dumps({"created_at": now(), "protocol": DEFAULT_PROTOCOL, "jobs": jobs}, indent=2) + "\n", encoding="utf-8")
    if args.pt_only and args.skip_pt:
        raise SystemExit("--pt-only cannot be combined with --skip-pt")
    total_steps = len(jobs) if args.pt_only else len(jobs) * (2 if args.skip_pt else 3)
    description = "augmentation pt-only" if args.pt_only else ("augmentation pt+engine" if not args.skip_pt else "augmentation engine")
    with tqdm(total=total_steps, desc=description, unit="step") as progress:
        for job in jobs:
            job_id = job["job_id"]
            checkpoint = args.checkpoint_root / job["checkpoint_rel"]
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            job_root = args.results_root / job_id
            engine_dir = job_root / "engine"
            logs = job_root / "logs"
            status_path = job_root / "status.json"
            status = {"job": job, "protocol": DEFAULT_PROTOCOL, "started_at": now(), "stages": {}}
            checkpoint_rel = "/checkpoints/" + str(Path(job["checkpoint_rel"]))
            engine_path = engine_dir / "best.engine"
            if not args.pt_only:
                if not engine_path.is_file():
                    export_cmd = docker_base(args) + [
                        "-v", f"{checkpoint.parent.resolve()}:/input:ro",
                        "-v", f"{engine_dir.resolve()}:/out",
                        "--entrypoint", "python3", args.image,
                        "/app/export_engine.py",
                        "--checkpoint", f"/input/{checkpoint.name}",
                        "--output", "/out/best.engine",
                        "--imgsz", str(job["image_size"]),
                        "--precision", args.precision,
                    ]
                    run_stage(export_cmd, logs / "export.log")
                    status["stages"]["export"] = {"status": "ok", "engine": str(engine_path)}
                else:
                    status["stages"]["export"] = {"status": "existing", "engine": str(engine_path)}
            if not args.skip_pt:
                pt_result = job_root / "pt"
                pt_metrics = list(pt_result.glob("*_corner_predictions.json" if args.tinyperson_save_only and job["dataset"] == "tinyperson" else "*_metrics.json"))
                if not pt_metrics:
                    run_stage(evaluate_command(args, job, checkpoint_rel, pt_result), logs / "pt_evaluate.log")
                    status["stages"]["pt"] = {"status": "ok", "results": str(pt_result)}
                else:
                    status["stages"]["pt"] = {"status": "existing", "results": str(pt_result)}
                progress.update(1)
            if args.pt_only:
                status["finished_at"] = now()
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                continue
            engine_result = job_root / "engine_eval"
            engine_metrics = list(engine_result.glob("*_corner_predictions.json" if args.tinyperson_save_only and job["dataset"] == "tinyperson" else "*_metrics.json"))
            if args.tinyperson_save_only and job["dataset"] == "tinyperson":
                engine_metrics += list(engine_result.glob("*_metrics.json"))
            if not engine_metrics:
                run_stage(evaluate_command(args, job, f"/engine/best.engine", engine_result, engine_dir), logs / "engine_evaluate.log")
                status["stages"]["engine"] = {"status": "ok", "results": str(engine_result)}
            else:
                status["stages"]["engine"] = {"status": "existing", "results": str(engine_result)}
            progress.update(1)
            status["finished_at"] = now()
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"QUEUE_COMPLETE jobs={len(jobs)} results={args.results_root}", flush=True)


if __name__ == "__main__":
    main()
