#!/usr/bin/env python3
"""Marimo contract adapter for periodic TinyPerson stability retraining."""
from __future__ import annotations

import sys
from pathlib import Path

from train_scripts import train_all_tinyperson as runner


EXPECTED_MODEL = Path(
    "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_base.yaml"
)


def main() -> None:
    args = list(sys.argv[1:])
    translated: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item == "--seed":
            translated.append("--seeds")
            translated.append(args[index + 1])
            index += 2
            continue
        if item == "--model-yaml":
            model = Path(args[index + 1])
            expected = (Path.cwd() / EXPECTED_MODEL).resolve()
            if model.resolve() != expected:
                raise SystemExit(f"Unexpected model YAML: {model} != {expected}")
            index += 2
            continue
        translated.append(item)
        index += 1
    sys.argv = [str(Path(runner.__file__).resolve()), *translated]
    runner.main()


if __name__ == "__main__":
    main()
