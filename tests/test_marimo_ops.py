#!/usr/bin/env python3
"""Small local tests for the dependency-free Marimo orchestration helper."""

from __future__ import annotations

import json
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
    status,
)


class MarimoOpsTests(unittest.TestCase):
    def test_current_process_is_alive(self) -> None:
        self.assertTrue(is_pid_alive(__import__("os").getpid()))
        self.assertFalse(is_pid_alive(-1))

    def test_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaises(MarimoOpsError):
                artifacts(run_dir, ["results.csv"])

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
            self.assertTrue((run_dir / "state.json").is_file())
            self.assertTrue((run_dir / "train.log").is_file())
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["pid"], result.pid)
            self.assertEqual(state["status"], "running")


if __name__ == "__main__":
    unittest.main()
