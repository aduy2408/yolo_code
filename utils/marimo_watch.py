#!/usr/bin/env python3
"""Monitor detached Marimo run supervisors without storing credentials."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from utils.marimo_client import MarimoClient, MarimoConfig

DEFAULT_PATHS = (
    "runs/stw_yolo_oacp_corrected_20261026/levirship_supervisor",
    "runs/stw_yolo_oacp_corrected_20261026/tinyperson_supervisor",
    "runs/stw_yolo_oacp_corrected_20261026/tinyperson_seed43_supervisor",
)
REMOTE_CODE = """
import json
import os
import csv
import re
from pathlib import Path
from utils.marimo_ops import status
os.chdir(os.environ["MARIMO_WATCH_REPO"])
result = []
required = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
            "evaluation_metrics.json", "experiment_manifest.json",
            "upload_complete.json")

def progress(seed_root):
    epochs = None
    args_path = seed_root / "args.yaml"
    if args_path.is_file():
        match = re.search(r"(?m)^epochs:\\s*([0-9]+)", args_path.read_text(errors="replace"))
        if match:
            epochs = int(match.group(1))
    completed = None
    results_path = seed_root / "results.csv"
    if results_path.is_file():
        with results_path.open(newline="", errors="replace") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            value = rows[-1].get("epoch")
            if value not in (None, ""):
                completed = int(float(value)) + 1
                if epochs is not None:
                    completed = min(completed, epochs)
    return {"epochs_completed": completed, "epochs_requested": epochs}
result = []
required = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
            "evaluation_metrics.json", "experiment_manifest.json",
            "upload_complete.json")
for path in json.loads(os.environ["MARIMO_WATCH_PATHS"]):
    try:
        value = status(Path(path), emit=False)
        artifact_root = Path(str(value.get("artifact_root", "")))
        dataset = str(value.get("dataset", ""))
        children = []
        candidate_roots = sorted((artifact_root / dataset / "oacp").glob("seed_*"))
        if not candidate_roots:
            candidate_roots = sorted(artifact_root.glob("seed*/s*_x*"))
        for seed_root in candidate_roots:
            required_for_run = (
                ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
                 "evaluation_metrics.json", "manifest.json", "upload_complete.json")
                if re.fullmatch(r"s\d+_x\d+", seed_root.name)
                else required
            )
            present = {item: (seed_root / item).is_file() for item in required_for_run}
            children.append({"run_dir": str(seed_root), "complete": all(present.values()),
                             "progress": progress(seed_root), "required_artifacts": present})
        result.append({key: value.get(key) for key in (
            "run_dir", "pid", "process_alive", "process_identity",
            "observed_status", "continuation_state", "upload_verified")}
                      | {"artifacts": children})
    except Exception as exc:
        result.append({"run_dir": path, "error": f"{type(exc).__name__}: {exc}"})
print(json.dumps(result, sort_keys=True))
"""


def configs_from_environment() -> list[MarimoConfig]:
    configs = []
    for index in range(1, 33):
        url = os.environ.get(f"MARIMO_WATCH_SERVER_{index}_URL")
        token = os.environ.get(f"MARIMO_WATCH_SERVER_{index}_TOKEN")
        if not url and not token:
            continue
        if not url or not token:
            raise SystemExit(f"MARIMO_WATCH_SERVER_{index}_URL and _TOKEN are both required")
        configs.append(MarimoConfig(url, token))
    if not configs:
        raise SystemExit("Set MARIMO_WATCH_SERVER_N_URL and MARIMO_WATCH_SERVER_N_TOKEN")
    return configs


def check_server(config: MarimoConfig, repo: str, paths: tuple[str, ...]) -> dict[str, object]:
    code_env = {
        "MARIMO_WATCH_REPO": repo,
        "MARIMO_WATCH_PATHS": json.dumps(paths),
    }
    code = f"import os\nos.environ.update({code_env!r})\n{REMOTE_CODE}"
    output = ""
    for event, payload in MarimoClient(config, timeout=60).execute_events(code):
        if event in ("stdout", "stderr"):
            output += str(payload.get("data", ""))
    return json.loads(output.strip().splitlines()[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Remote repository checkout")
    parser.add_argument("--path", action="append", dest="paths", default=None)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    configs = configs_from_environment()
    paths = tuple(args.paths or DEFAULT_PATHS)
    while True:
        timestamp = datetime.now(timezone.utc).isoformat()
        for index, config in enumerate(configs, 1):
            try:
                result = check_server(config, args.repo, paths)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps({"at": timestamp, "server": index, "result": result}, sort_keys=True), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
