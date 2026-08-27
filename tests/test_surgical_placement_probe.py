import unittest
from pathlib import Path

from train_levir_scripts import train_kvca_surgical_placement_probe as probe


class SurgicalPlacementConfigTests(unittest.TestCase):
    def test_all_configs_exist_and_keep_single_p2_detect(self):
        for placement, path in probe.CONFIGS.items():
            self.assertTrue(path.is_file(), placement)
            text = path.read_text()
            self.assertIn("p2_offset_regression: false", text)
            self.assertEqual(text.count("Detect, [nc]"), 1)
            self.assertIn("ChannelAttention, []", text)

    def test_expected_probe_contract(self):
        self.assertEqual(probe.KVCA_LAYERS, {"A": 16, "B": 18, "C": 19})
        self.assertEqual(probe.EXPECTED_KVCA, {"A": (64, 4), "B": (96, 8), "C": (32, 8)})
        for placement, path in probe.CONFIGS.items():
            text = path.read_text()
            channels, sr = probe.EXPECTED_KVCA[placement]
            nominal = channels * 4
            self.assertIn(f"KVCompressedAttention, [{nominal}, 4, {sr}, group_weight, 0.0]", text)

    def test_runner_defaults_match_protocol(self):
        args = probe.parse_args(["--canonical-checkpoint", "checkpoint.pt"])
        self.assertEqual(args.epochs, 15)
        self.assertEqual(args.patience, 0)
        self.assertEqual(args.imgsz, 512)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.seed, 42)

    def test_probe_uses_full_ftal_and_val_only_screen(self):
        text = Path(probe.__file__).read_text()
        self.assertIn('factorized_tal_mode="legacy"', text)
        self.assertIn("factorized_tal_warmup_start=0", text)
        self.assertIn("factorized_tal_warmup_end=0", text)
        self.assertIn('for split in ("val",):', text)
        self.assertIn("unmapped_source", text)


if __name__ == "__main__":
    unittest.main()
