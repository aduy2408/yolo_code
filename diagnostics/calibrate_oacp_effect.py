#!/usr/bin/env python3
"""Calibrate effect-adaptive OACP target from fixed-R2 JSONL diagnostics.

Run a diagnostic pass first with the R2 control environment and
``OACP_DIAGNOSTICS_PATH`` pointing at a JSONL file, then run this script.  It
writes a JSON artifact that can be supplied as ``OACP_TARGET_EFFECT`` using the
reported ``target_effect`` value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "models_related/ultralytics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from project_ultralytics.context_augment import calibrate_effect_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--quantile", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate_effect_target(str(args.diagnostics), args.quantile)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
