import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "utils/pull_mmdet_baselines_complexity.py"
SPEC = importlib.util.spec_from_file_location("pull_mmdet_baselines_complexity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_size_units():
    assert MODULE.parse_size("123.4 G", "G") == 123.4
    assert MODULE.parse_size("25.6 M", "M") == 25.6
    assert MODULE.parse_size("1250 K", "M") == 1.25


def test_parse_complexity_output():
    parsed = MODULE.parse_complexity_output("Input shape: (512, 512)\nFlops: 239.1 G\nParams: 36.4 M\n")
    assert parsed == {"gflops": 239.1, "parameters_m": 36.4}
