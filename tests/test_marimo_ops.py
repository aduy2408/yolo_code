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
    file_sha256,
    is_pid_alive,
    launch_detached,
    preflight,
    require_training_context,
    resolve_marimo_dataset_root,
    status,
    validate_command_contract,
    write_run_contract,
)
from utils.marimo.contracts import (
    DatasetSpec,
    ExecutionSpec,
    ExperimentContract,
    OutputSpec,
    SourceSpec,
)

COMPLETION_ARTIFACTS_FOR_TEST = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "experiment_manifest.json",
    "upload_complete.json",
)


class MarimoOpsTests(unittest.TestCase):
    def test_generic_contract_supports_baseline_and_external_sources(self) -> None:
        for kind, source_kind in (("baseline", "git"), ("external_repo", "external_git")):
            contract = ExperimentContract(
                run_id=f"run-{kind}",
                experiment_kind=kind,
                source=SourceSpec(
                    kind=source_kind,
                    repo="/marimo/sources/project",
                    commit="abc123",
                    remote="https://example.invalid/project.git",
                ),
                dataset=DatasetSpec(
                    name="levir",
                    root="/marimo/LevirShip",
                    yaml="/marimo/experiment/data.yaml",
                    split_seed=42,
                ),
                execution=ExecutionSpec(
                    command=("python", "train.py"),
                    cwd="/marimo/experiment",
                    seed=43,
                ),
                outputs=OutputSpec(
                    artifact_root="/marimo/runs/run-1",
                    hf_repo_id="user/task-runs",
                ),
                extra={"architecture": "baseline"},
            )
            restored = ExperimentContract.from_mapping(contract.to_dict())
            self.assertEqual(restored.experiment_kind, kind)
            self.assertEqual(restored.source.kind, source_kind)
            self.assertEqual(restored.execution.command, ("python", "train.py"))

    def test_generic_contract_rejects_non_marimo_backend(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentContract(
                run_id="run",
                experiment_kind="baseline",
                source=SourceSpec(kind="git", repo="/repo", commit="abc"),
                dataset=DatasetSpec(name="data", root="/data", yaml="/data.yaml"),
                execution=ExecutionSpec(command=("python", "train.py"), cwd="/repo"),
                outputs=OutputSpec(artifact_root="/runs/run"),
                backend="local",
            )

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

    def test_marimo_dataset_mounts_are_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marimo = Path(tmp) / "marimo"
            (marimo / "LevirShip").mkdir(parents=True)
            (marimo / "Varroa").mkdir()
            (marimo / "TinyPerson").mkdir()
            self.assertEqual(resolve_marimo_dataset_root("levir", marimo), (marimo / "LevirShip").resolve())
            self.assertEqual(resolve_marimo_dataset_root("varroa", marimo), (marimo / "Varroa").resolve())
            self.assertEqual(resolve_marimo_dataset_root("tinyperson", marimo), (marimo / "TinyPerson").resolve())
            with self.assertRaises(MarimoOpsError):
                resolve_marimo_dataset_root("unknown", marimo)

    def test_status_classifies_checkpoint_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "weights").mkdir()
            (run_dir / "weights/last.pt").write_text("checkpoint")
            report = status(run_dir, emit=False)
            self.assertEqual(report["continuation_state"], "checkpoint_present_evaluation_pending")

    def test_launch_command_must_match_contract(self) -> None:
        contract = {
            "dataset": "levir",
            "data_root": "/marimo/LevirShip/LevirShipData",
            "dataset_yaml": "/marimo/yolo_code/datasets/levir.yaml",
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
            write_run_contract(run_dir, contract)
            validate_command_contract(
                run_dir,
                [
                    "runner.py", "--epochs", "100", "--patience", "0", "--workers", "8",
                    "--seed", "42", "--split-seed", "42", "--hf-repo-id", "user/task-runs",
                    "--model-yaml", "models/yolov8.yaml", "--data-root", "/marimo/LevirShip/LevirShipData",
                ],
            )
            with self.assertRaises(MarimoOpsError):
                validate_command_contract(run_dir, ["runner.py", "--epochs", "400"])

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
            (run_dir / "state.json").write_text(json.dumps({
                "status": "exited",
                "returncode": 0,
                "contract_sha256": file_sha256(run_dir / "run_contract.json"),
            }))
            (run_dir / "evaluation_metrics.json").write_text(json.dumps({
                "val/AP50": 0.8,
                "val/mAP50-95": 0.3,
                "test/AP50": 0.7,
                "test/mAP50-95": 0.25,
            }))
            (run_dir / "upload_complete.json").write_text(json.dumps({
                "repo_id": "user/task-runs",
                "remote_prefix": "levir/seed_42",
                "verified": list(COMPLETION_ARTIFACTS_FOR_TEST),
            }))
            result = complete_verified(run_dir)
            self.assertEqual(result["status"], "complete_verified")

    def test_artifact_root_is_used_for_status_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "wrapper"
            artifact_root = root / "outputs" / "run-1"
            write_run_contract(
                run_dir,
                {
                    "dataset": "levir", "data_root": "/data/levir",
                    "dataset_yaml": "/data/levir/data.yaml", "model_yaml": "models/yolov8.yaml",
                    "seed": 42, "split_seed": 42, "workers": 8, "epochs": 100,
                    "patience": 0, "nms_iou": 0.5, "hf_repo_id": "user/task-runs",
                },
            )
            (run_dir / "state.json").write_text(json.dumps({
                "status": "exited",
                "returncode": 0,
                "contract_sha256": file_sha256(run_dir / "run_contract.json"),
                "cwd": str(root),
                "command": ["runner.py", "--project", "outputs/run-1"],
            }))
            for relative in COMPLETION_ARTIFACTS_FOR_TEST:
                path = artifact_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}" if path.suffix == ".json" else "artifact")
            (artifact_root / "evaluation_metrics.json").write_text(json.dumps({
                "val/AP50": 0.8, "val/mAP50-95": 0.3,
                "test/AP50": 0.7, "test/mAP50-95": 0.25,
            }))
            (artifact_root / "upload_complete.json").write_text(json.dumps({
                "repo_id": "user/task-runs", "remote_prefix": "run-1",
                "verified": list(COMPLETION_ARTIFACTS_FOR_TEST),
            }))
            report = status(run_dir, emit=False)
            self.assertEqual(report["artifact_root"], str(artifact_root.resolve()))
            self.assertEqual(report["observed_status"], "artifacts_present")
            verified = complete_verified(run_dir)
            self.assertEqual(verified["artifact_root"], str(artifact_root.resolve()))

    def test_custom_state_file_and_upload_marker_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "wrapper"
            artifact_root = root / "outputs"
            write_run_contract(
                run_dir,
                {
                    "dataset": "levir", "data_root": "/data/levir",
                    "dataset_yaml": "/data/levir/data.yaml", "model_yaml": "models/yolov8.yaml",
                    "seed": 42, "split_seed": 42, "workers": 8, "epochs": 100,
                    "patience": 0, "nms_iou": 0.5, "hf_repo_id": "user/task-runs",
                },
            )
            (run_dir / "custom-state.json").parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "custom-state.json").write_text(json.dumps({
                "cwd": str(root), "artifact_root": str(artifact_root),
                "command": ["runner.py", "--resume=/checkpoints/last.pt"],
                "log_path": "custom.log",
            }))
            (run_dir / "custom.log").write_text("resume failed before loading checkpoint\n")
            (artifact_root / "upload_complete.json").parent.mkdir(parents=True, exist_ok=True)
            (artifact_root / "upload_complete.json").write_text(json.dumps({
                "repo_id": "wrong/repo", "remote_prefix": "run", "verified": ["x"],
            }))
            report = status(run_dir, state_file="custom-state.json", emit=False)
            self.assertEqual(report["artifact_root"], str(artifact_root.resolve()))
            self.assertTrue(report["upload_marker_present"])
            self.assertFalse(report["upload_verified"])
            self.assertEqual(report["resume_evidence"], "pending")

    def test_require_files_preserves_generator_paths(self) -> None:
        from utils.marimo_ops import require_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifact.txt").write_text("ok")
            self.assertEqual(require_files(root, (item for item in ["artifact.txt"])), ["artifact.txt"])

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

    def test_preflight_matrix_contract_requires_and_validates_explicit_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            data_root = repo / "dataset"
            for split in ("train", "val", "test"):
                (data_root / split).mkdir(parents=True)
            dataset_yaml = data_root / "data.yaml"
            dataset_yaml.write_text("path: .\ntrain: train\nval: val\ntest: test\n")
            (repo / "ready.txt").write_text("ready")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            contract = repo / "run_contract.json"
            contract.write_text(json.dumps({
                "dataset": "matrix", "data_root": "matrix", "dataset_yaml": "matrix",
                "model_yaml": "matrix", "seed": "matrix", "split_seed": 42,
                "workers": 8, "epochs": 100, "patience": 0, "nms_iou": 0.5,
                "hf_repo_id": "user/task-runs",
            }))
            subprocess.run(["git", "add", "run_contract.json"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "contract"], cwd=repo, check=True)
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            result = preflight(
                repo=repo, expected_sha=sha, python=sys.executable,
                required_paths=["ready.txt"], epochs=100, patience=0,
                hf_repo_id="user/task-runs", data_root=data_root,
                dataset_yaml=dataset_yaml, contract_json=contract,
            )
            self.assertEqual(result["dataset_yaml"], str(dataset_yaml.resolve()))

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
            write_run_contract(
                run_dir,
                {
                    "dataset": "levir",
                    "data_root": "/marimo/LevirShip/LevirShipData",
                    "dataset_yaml": "/marimo/yolo_code/datasets/levir.yaml",
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
                    "--epochs", "100", "--patience", "0", "--workers", "8",
                    "--seed", "42", "--split-seed", "42", "--hf-repo-id", "user/task-runs",
                    "--model-yaml", "models/yolov8.yaml", "--data-root", "/marimo/LevirShip/LevirShipData",
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
            self.assertEqual(state["status"], "exited")
            self.assertEqual(state["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
