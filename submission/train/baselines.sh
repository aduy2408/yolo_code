#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${PYTHON:-python}" "$ROOT/train_scripts/train_all_yolo_baselines_no_mosaic.py" "$@"
