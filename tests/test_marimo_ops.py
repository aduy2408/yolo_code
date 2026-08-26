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
from pathlib import Path

from utils.marimo_ops import (
    MarimoOpsError,
    artifacts,
    is_pid_alive,
    launch_detached,
    preflight,
    require_training_context,
    status,
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

    def test_preflight_checks_exact_sha_clean_tree_and_upload_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "ready.txt").write_text("ready")
            subprocess.run(["git", "add", "ready.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

            result = preflight(
                repo=repo,
                expected_sha=sha,
                python=sys.executable,
                required_paths=["ready.txt"],
                epochs=100,
                patience=0,
                upload_required=True,
                hf_repo_id="test/repo",
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
