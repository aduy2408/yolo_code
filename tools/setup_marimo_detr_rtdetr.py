#!/usr/bin/env python3
"""Prepare Marimo servers for native Hugging Face DETR and RT-DETR runs.

This is a setup and validation tool, not a training launcher. It connects to
one or more live Marimo kernels, creates an isolated remote workspace, checks
the Python/runtime/model imports and canonical dataset mounts, and writes a
secret-free setup report. Training must still be launched through the project's
``python -m utils.marimo_ops launch`` workflow.

Credentials are accepted only from a local, owner-readable server file or the
environment. They are never written into the remote workspace, command line,
JSON reports, or normal output.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.marimo_client import MarimoClient, MarimoClientError, MarimoConfig


DEFAULT_REMOTE_ROOT = "/marimo/hf_detr_rtdetr"
DEFAULT_PYTHON = "/tmp/uv-venv/bin/python"
DEFAULT_DATASETS = ("varroa", "levirship", "tinyperson", "visdrone")

MODEL_SPECS = {
    "detr": {
        "checkpoint": "facebook/detr-resnet-50",
        "class": "DetrForObjectDetection",
        "processor": "DetrImageProcessor",
        "role": "DETR ResNet-50",
    },
    "rtdetr": {
        "checkpoint": "PekingU/rtdetr_r18vd",
        "class": "RTDetrForObjectDetection",
        "processor": "RTDetrImageProcessor",
        "role": "RT-DETR ResNet-18",
    },
}

DATASET_MOUNTS = {
    "varroa": ["/marimo/Varroa"],
    "levirship": ["/marimo/LevirShip", "/marimo/LevirShip/LevirShipData"],
    "tinyperson": ["/marimo/TinyPerson", "/marimo/TinyPerson/TinyPersonData"],
    "visdrone": ["/marimo/VisDrone2019", "/marimo/VisDrone2019/VisDrone2019-DET-train"],
}


@dataclass(frozen=True)
class Server:
    host: str
    token: str

    @property
    def url(self) -> str:
        return self.host if self.host.startswith("http") else f"https://{self.host}"


class SetupError(RuntimeError):
    """A local parsing or remote setup error."""


def _parse_assignments(text: str, *, source: str) -> list[Server]:
    """Parse HOST=/ACCESS_TOKEN= blocks without echoing token values."""
    servers: list[Server] = []
    host = ""
    token = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("HOST="):
            if host and token:
                servers.append(Server(host, token))
            host = line.split("=", 1)[1].strip().strip("\"'")
            token = ""
        elif line.startswith("ACCESS_TOKEN=") or line.startswith("MARIMO_ACCESS_TOKEN="):
            token = line.split("=", 1)[1].strip().strip("\"'")
    if host and token:
        servers.append(Server(host, token))
    if not servers:
        raise SetupError(f"No HOST=/ACCESS_TOKEN= pair found in {source}")
    return servers


def load_servers(path: Path | None) -> list[Server]:
    if path is not None:
        try:
            mode = path.stat().st_mode & 0o777
        except OSError as exc:
            raise SetupError(f"Cannot read server file {path}: {exc}") from exc
        if mode & 0o077:
            raise SetupError(f"Refusing server file {path}: permissions must be 0600 or stricter")
        return _deduplicate(_parse_assignments(path.read_text(), source=str(path)))

    inline = os.environ.get("MARIMO_SERVERS_FILE")
    if inline:
        return load_servers(Path(inline))

    host = os.environ.get("MARIMO_URL", "").strip()
    token = (os.environ.get("MARIMO_ACCESS_TOKEN") or os.environ.get("MARIMO_TOKEN", "")).strip()
    if host and token:
        return [Server(host, token)]
    raise SetupError("Provide --servers-file, MARIMO_SERVERS_FILE, or MARIMO_URL plus MARIMO_ACCESS_TOKEN")


def _deduplicate(servers: Iterable[Server]) -> list[Server]:
    seen: set[str] = set()
    result: list[Server] = []
    for server in servers:
        key = server.url.rstrip("/")
        if key not in seen:
            seen.add(key)
            result.append(Server(key, server.token))
    return result


def _git_value(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def local_provenance() -> dict[str, str]:
    try:
        commit = _git_value(["git", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SetupError(f"Not a Git checkout or Git is unavailable: {exc}") from exc
    try:
        remote = _git_value(["git", "config", "--get", "remote.origin.url"])
    except subprocess.CalledProcessError:
        remote = ""
    return {"commit": commit, "remote": remote}


def _json_b64(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(payload).decode("ascii")


def build_remote_code(
    *,
    remote_root: str,
    python: str,
    datasets: list[str],
    models: list[str],
    provenance: Mapping[str, str],
    runner: str,
) -> str:
    config = {
        "schema": 1,
        "models": {name: MODEL_SPECS[name] for name in models},
        "datasets": {name: DATASET_MOUNTS[name] for name in datasets},
        "runtime": {
            "python": python,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "backbone_learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "image_size": 512,
            "batch_size": 2,
            "workers": 8,
            "epochs": 100,
            "patience": 15,
            "split_seed": 42,
            "training_seeds": [42, 43],
            "nms_iou": 0.5,
        },
        "runner": runner,
        "source": dict(provenance),
    }
    config_b64 = _json_b64(config)
    models_json = json.dumps({name: MODEL_SPECS[name] for name in models}, sort_keys=True)
    datasets_json = json.dumps({name: DATASET_MOUNTS[name] for name in datasets}, sort_keys=True)
    return f'''\
import base64, importlib, json, os, pathlib, subprocess, sys, time

root = pathlib.Path({remote_root!r})
root.mkdir(parents=True, exist_ok=True)
python = pathlib.Path({python!r})
models = json.loads({models_json!r})
datasets = json.loads({datasets_json!r})
config = json.loads(base64.b64decode({config_b64!r}).decode())
report = {{
    "schema": 1,
    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "workspace": str(root),
    "models": {{}},
    "datasets": {{}},
    "runtime": {{}},
    "runner": config["runner"],
    "source": config["source"],
}}

if not python.is_file() or not os.access(python, os.X_OK):
    raise RuntimeError(f"Python executable is not executable: {{python}}")
probe = subprocess.run([str(python), "-c", "import sys; print(sys.executable); print(sys.version)"], capture_output=True, text=True)
if probe.returncode:
    raise RuntimeError(probe.stderr.strip() or "Python probe failed")
report["runtime"]["python"] = probe.stdout.splitlines()[0]
report["runtime"]["version"] = probe.stdout.splitlines()[1] if len(probe.stdout.splitlines()) > 1 else ""

imports = "from transformers import DetrForObjectDetection, DetrImageProcessor, RTDetrForObjectDetection, RTDetrImageProcessor"
for dependency, probe_code in (("timm", "import timm; print(timm.__version__)"), ("pycocotools", "from pycocotools.coco import COCO; print('ok')")):
    dependency_probe = subprocess.run([str(python), "-c", probe_code], capture_output=True, text=True)
    if dependency_probe.returncode:
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", dependency], capture_output=True, text=True)
        dependency_probe = subprocess.run([str(python), "-c", probe_code], capture_output=True, text=True)
    report["runtime"][f"{{dependency}}_import"] = dependency_probe.returncode == 0
    if dependency_probe.returncode == 0:
        report["runtime"][f"{{dependency}}_version"] = dependency_probe.stdout.strip()
    else:
        report["runtime"][f"{{dependency}}_error"] = dependency_probe.stderr[-2000:]
transformers_probe = subprocess.run([str(python), "-c", imports], capture_output=True, text=True)
report["runtime"]["transformers_import"] = transformers_probe.returncode == 0
if transformers_probe.returncode:
    report["runtime"]["transformers_error"] = transformers_probe.stderr[-2000:]

for name, spec in models.items():
    entry = dict(spec)
    entry["import_ok"] = transformers_probe.returncode == 0
    entry["checkpoint_probe"] = "not_run"
    if entry["import_ok"]:
        code = "from transformers import AutoConfig; AutoConfig.from_pretrained(%r); print('ok')" % spec["checkpoint"]
        checkpoint = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
        entry["checkpoint_probe"] = checkpoint.returncode == 0
        if checkpoint.returncode:
            entry["checkpoint_error"] = checkpoint.stderr[-2000:]
    report["models"][name] = entry

for name, candidates in datasets.items():
    existing = [path for path in candidates if pathlib.Path(path).exists()]
    discovered = []
    if not existing:
        for search_root in (pathlib.Path("/marimo"), pathlib.Path("/data"), pathlib.Path("/mnt/data")):
            if not search_root.is_dir():
                continue
            for candidate in search_root.iterdir():
                if "visdrone" in candidate.name.lower():
                    discovered.append(str(candidate))
                if candidate.is_dir():
                    try:
                        for nested in candidate.iterdir():
                            if "visdrone" in nested.name.lower():
                                discovered.append(str(nested))
                    except OSError:
                        pass
    report["datasets"][name] = {{
        "candidates": candidates,
        "existing": existing,
        "discovered": sorted(set(discovered)),
        "ready": bool(existing),
    }}

config_path = root / "detr_rtdetr_setup.json"
report_path = root / "setup_report.json"
config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\\n")
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
print(json.dumps({{"workspace": str(root), "report": str(report_path), "models": report["models"], "datasets": report["datasets"], "transformers_import": report["runtime"]["transformers_import"]}}, sort_keys=True))
'''


def run_server(server: Server, code: str) -> dict[str, object]:
    client = MarimoClient(MarimoConfig(server.url, server.token))
    output: list[str] = []
    for event, payload in client.execute_events(code):
        if event == "stdout":
            output.append(str(payload.get("data", "")))
        elif event == "stderr":
            output.append(str(payload.get("data", "")))
        elif event == "done" and payload.get("success") is False:
            error = payload.get("error", {})
            message = error.get("msg", "remote execution failed") if isinstance(error, dict) else str(error)
            raise SetupError(message)
    text = "".join(output).strip()
    if not text:
        raise SetupError("Remote setup returned no output")
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise SetupError(f"Remote setup returned no JSON report: {text[-1000:]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servers-file", type=Path, help="0600 file containing HOST=/ACCESS_TOKEN= blocks")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Remote Python executable")
    parser.add_argument("--runner", default="train_hf_detection_matrix.py", help="Expected HF matrix runner name")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--dry-run", action="store_true", help="Validate input and print host names only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        servers = load_servers(args.servers_file)
        provenance = local_provenance()
        if args.dry_run:
            for server in servers:
                print(server.url)
            return 0
        code = build_remote_code(
            remote_root=args.remote_root,
            python=args.python,
            datasets=args.datasets,
            models=args.models,
            provenance=provenance,
            runner=args.runner,
        )
        failures = 0
        for server in servers:
            try:
                result = run_server(server, code)
                print(json.dumps({"server": server.url, "status": "ok", "result": result}, sort_keys=True))
            except (SetupError, MarimoClientError) as exc:
                failures += 1
                print(json.dumps({"server": server.url, "status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1 if failures else 0
    except (OSError, SetupError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
