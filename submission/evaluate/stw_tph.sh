#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${PYTHON:-python}" "$ROOT/evaluate_test/evaluate_stw_tph_checkpoints.py" "$@"
