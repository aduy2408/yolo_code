import json
import tempfile
import unittest
from pathlib import Path

from utils.research_loop import ResearchLoopError, append_record, validate_manifest


class ResearchLoopTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.manifest = self.root / "manifest.json"
        self.ledger = self.root / "ledger.jsonl"
        self.manifest.write_text(
            json.dumps(
                {
                    "hypothesis_id": "H001",
                    "experiment_id": "H001-E001",
                    "purpose": "Test one mechanism",
                    "baseline": {
                        "model_yaml": "model.yaml",
                        "provenance_class": "new",
                        "source_sha": "abcdef1234567",
                        "source_sha_status": "verified",
                        "dataset": "LEVIR-Ship",
                        "dataset_yaml": "data.yaml",
                        "split_seed": 42,
                        "training_seed": 42,
                        "image_size": 512,
                        "workers": 2,
                        "epochs": 1,
                        "nms_iou": 0.5,
                    },
                    "treatment": {"method": "one-change"},
                    "changed_factors": ["method"],
                    "expected_diagnostics": ["oracle_gap"],
                    "promotion_rule": {"must_improve": ["oracle_gap"], "must_not_reduce": ["test/AP50"]},
                    "artifact_contract": ["results.csv"],
                }
            )
        )

    def test_validates_one_factor_manifest(self):
        result = validate_manifest(self.manifest, self.ledger)
        self.assertEqual(result["status"], "valid")

    def test_allows_recovered_baseline_without_source_sha(self):
        payload = json.loads(self.manifest.read_text())
        payload["baseline"]["provenance_class"] = "recovered"
        payload["baseline"]["source_sha_status"] = "unavailable"
        payload["baseline"].pop("source_sha")
        self.manifest.write_text(json.dumps(payload))
        result = validate_manifest(self.manifest)
        self.assertEqual(result["status"], "valid")

    def test_rejects_verified_recovered_baseline_without_source_sha(self):
        payload = json.loads(self.manifest.read_text())
        payload["baseline"]["provenance_class"] = "recovered"
        payload["baseline"]["source_sha_status"] = "verified"
        payload["baseline"].pop("source_sha")
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ResearchLoopError, "verified requires source_sha"):
            validate_manifest(self.manifest)

    def test_rejects_new_baseline_without_source_sha(self):
        payload = json.loads(self.manifest.read_text())
        payload["baseline"].pop("source_sha")
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ResearchLoopError, "new baseline requires source_sha"):
            validate_manifest(self.manifest)

    def test_rejects_nonstandard_nms(self):
        payload = json.loads(self.manifest.read_text())
        payload["baseline"]["nms_iou"] = 0.7
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ResearchLoopError, "nms_iou"):
            validate_manifest(self.manifest)

    def test_rejects_duplicate_run_id(self):
        record = self.root / "record.json"
        record.write_text(
            json.dumps(
                {
                    "run_id": "H001-E001-R001",
                    "hypothesis_id": "H001",
                    "experiment_id": "H001-E001",
                    "status": "planned",
                    "artifacts_verified": False,
                }
            )
        )
        append_record(self.ledger, record)
        with self.assertRaisesRegex(ResearchLoopError, "already exists"):
            append_record(self.ledger, record)


if __name__ == "__main__":
    unittest.main()
