#!/usr/bin/env python3
"""Small local tests for the dependency-free Marimo orchestration helper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from utils.marimo_ops import (
    MarimoOpsError,
    artifacts,
    complete_verified,
    is_pid_alive,
    launch_detached,
    preflight,
    require_training_context,
    status,
    write_run_contract,
)


class MarimoOpsTests(unittest.TestCase):
    def test_training_context_is_fail_closed(self) -> None:
        old_marker = os.environ.pop("MARIMO_TRAIN_WORKFLOW", None)
        old_token = os.environ.pop("HF_TOKEN", None)
        old_repo = os.environ.pop("MARIMO_HF_REPO_ID", None)
        try:
            with self.assertRaises(MarimoOpsError):
                require_training_context(hf_repo_id="test/repo")
            os.environ["MARIMO_TRAIN_WORKFLOW"] = "1"
            with self.assertRaises(MarimoOpsError):
                require_training_context(hf_repo_id="test/repo")
            os.environ["HF_TOKEN"] = "test-token"
            os.environ["MARIMO_HF_REPO_ID"] = "other/repo"
            with self.assertRaises(MarimoOpsError):
                require_training_context(hf_repo_id="test/repo")
            os.environ["MARIMO_HF_REPO_ID"] = "test/repo"
            require_training_context(hf_repo_id="test/repo")
        finally:
            for key, value in {
                "MARIMO_TRAIN_WORKFLOW": old_marker,
                "HF_TOKEN": old_token,
                "MARIMO_HF_REPO_ID": old_repo,
            }.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_training_context_rejects_missing_task_specific_repository(self) -> None:
        from utils.marimo_ops import resolve_hf_repo_id

        old_marker = os.environ.get("MARIMO_TRAIN_WORKFLOW")
        old_token = os.environ.get("HF_TOKEN")
        old_repo = os.environ.get("MARIMO_HF_REPO_ID")
        old_task = os.environ.get("MARIMO_TASK_NAME")
        old_hf_repo = os.environ.get("HF_REPO_ID")
        old_experiment = os.environ.get("MARIMO_EXPERIMENT_NAME")
        try:
            os.environ["MARIMO_TRAIN_WORKFLOW"] = "1"
            os.environ["HF_TOKEN"] = "test-token"
            for key in ("MARIMO_HF_REPO_ID", "HF_REPO_ID", "MARIMO_TASK_NAME", "MARIMO_EXPERIMENT_NAME"):
                os.environ.pop(key, None)
            with self.assertRaises(MarimoOpsError):
                resolve_hf_repo_id(token="test-token")
        finally:
            for key, value in {
                "MARIMO_TRAIN_WORKFLOW": old_marker,
                "HF_TOKEN": old_token,
                "MARIMO_HF_REPO_ID": old_repo,
                "MARIMO_TASK_NAME": old_task,
                "HF_REPO_ID": old_hf_repo,
                "MARIMO_EXPERIMENT_NAME": old_experiment,
            }.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    def test_current_process_is_alive(self) -> None:
        self.assertTrue(is_pid_alive(__import__("os").getpid()))
        self.assertFalse(is_pid_alive(-1))

    def test_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaises(MarimoOpsError):
                artifacts(run_dir, ["results.csv"])

    def test_artifacts_success_requires_the_declared_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for relative in ("weights/best.pt", "weights/last.pt", "results.csv"):
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact")
            self.assertEqual(
                artifacts(run_dir, ["weights/best.pt", "weights/last.pt", "results.csv"])
                ["required_artifacts"],
                ["weights/best.pt", "weights/last.pt", "results.csv"],
            )

    def test_run_contract_is_required_and_immutable(self) -> None:
        contract = {
            "dataset": "levir",
            "data_root": "/data/levir",
            "dataset_yaml": "/data/levir/data.yaml",
            "model_yaml": "models/yolov8.yaml",
            "seed": 42,
            "split_seed": 42,
            "workers": 8,
            "epochs": 100,
            "patience": 0,
            "nms_iou": 0.5,
            "hf_repo_id": "user/task-runs",
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            payload = write_run_contract(run_dir, contract)
            self.assertEqual(payload["dataset"], "levir")
            with self.assertRaises(MarimoOpsError):
                write_run_contract(run_dir, contract)

    def test_complete_verified_requires_all_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaises(MarimoOpsError):
                complete_verified(run_dir)
            for relative in (
                "weights/best.pt",
                "weights/last.pt",
                "results.csv",
                "experiment_manifest.json",
            ):
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact")
            write_run_contract(
                run_dir,
                {
                    "dataset": "levir",
                    "data_root": "/data/levir",
                    "dataset_yaml": "/data/levir/data.yaml",
                    "model_yaml": "models/yolov8.yaml",
                    "seed": 42,
                    "split_seed": 42,
                    "workers": 8,
                    "epochs": 100,
                    "patience": 0,
                    "nms_iou": 0.5,
                    "hf_repo_id": "user/task-runs",
                },
            )
            (run_dir / "evaluation_metrics.json").write_text(json.dumps({
                "val/AP50": 0.8,
                "val/mAP50-95": 0.3,
                "test/AP50": 0.7,
                "test/mAP50-95": 0.25,
            }))
            (run_dir / "upload_complete.json").write_text(json.dumps({
                "repo_id": "user/task-runs",
                "remote_prefix": "levir/seed_42",
                "verified": ["weights/best.pt"],
            }))
            result = complete_verified(run_dir)
            self.assertEqual(result["status"], "complete_verified")

    def test_preflight_checks_exact_sha_clean_tree_and_upload_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "ready.txt").write_text("ready")
            data_root = repo / "dataset"
            for split in ("train", "val", "test"):
                (data_root / "images" / split).mkdir(parents=True, exist_ok=True)
            dataset_yaml = data_root / "data.yaml"
            dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\n")
            subprocess.run(["git", "add", "ready.txt", "dataset/data.yaml"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

            with patch("utils.marimo_ops.ensure_hf_repo", return_value="test/repo"):
                result = preflight(
                    repo=repo,
                    expected_sha=sha,
                    python=sys.executable,
                    required_paths=["ready.txt"],
                    epochs=100,
                    patience=0,
                    upload_required=True,
                    hf_repo_id="test/repo",
                    data_root=data_root,
                    dataset_yaml=dataset_yaml,
                )
            self.assertEqual(result["git_sha"], sha)
            self.assertEqual(result["python"], sys.executable)

            (repo / "dirty.txt").write_text("dirty")
            with self.assertRaises(MarimoOpsError):
                preflight(repo=repo, expected_sha=sha)
            with self.assertRaises(MarimoOpsError):
                preflight(repo=repo, expected_sha="0" * 40, allow_dirty=True)

    def test_duplicate_launch_is_rejected_while_pid_is_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result = launch_detached(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                cwd=run_dir,
                log_path=run_dir / "train.log",
                pid_path=run_dir / "train.pid",
                state_path=run_dir / "state.json",
            )
            try:
                with self.assertRaises(MarimoOpsError):
                    launch_detached(
                        [sys.executable, "-c", "print('duplicate')"],
                        cwd=run_dir,
                        log_path=run_dir / "train.log",
                        pid_path=run_dir / "train.pid",
                        state_path=run_dir / "state.json",
                    )
            finally:
                if is_pid_alive(result.pid):
                    os.kill(result.pid, 15)

    def test_cli_status_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "utils.marimo_ops",
                    "status",
                    "--run-dir",
                    str(run_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["run_dir"], str(run_dir))
            self.assertFalse(payload["process_alive"])

    def test_cli_launch_creates_durable_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "utils.marimo_ops",
                    "launch",
                    "--cwd",
                    tmp,
                    "--run-dir",
                    str(run_dir),
                    "--",
                    sys.executable,
                    "-c",
                    "print('cli-launch-ok')",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertTrue((run_dir / "train.pid").is_file())
            self.assertTrue((run_dir / "train.log").is_file())
            self.assertTrue((run_dir / "state.json").is_file())
            self.assertGreater(payload["pid"], 0)

    def test_launch_and_status_record_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result = launch_detached(
                [sys.executable, "-c", "print('smoke-ok')"],
                cwd=run_dir,
                log_path=run_dir / "train.log",
                pid_path=run_dir / "train.pid",
                state_path=run_dir / "state.json",
            )
            self.assertGreater(result.pid, 0)
            for _ in range(50):
                if not is_pid_alive(result.pid):
                    break
                time.sleep(0.02)
            report = status(run_dir)
            self.assertIn("process_alive", report)
            self.assertEqual(report["observed_status"], "not_running_unverified")
            self.assertTrue((run_dir / "state.json").is_file())
            self.assertTrue((run_dir / "train.log").is_file())
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["pid"], result.pid)
            self.assertEqual(state["status"], "running")


if __name__ == "__main__":
    unittest.main()
