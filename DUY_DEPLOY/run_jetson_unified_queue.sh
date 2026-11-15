#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/unified_pt_jetson_seed42_43_20261103}
IMAGE=${IMAGE:-duy-yolo-pytorch:jetpack6}
RUNTIME_DEPS=${RUNTIME_DEPS:-/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/runtime_deps}
DATA_SOURCE=${DATA_SOURCE:-/home/ait01/jetson_seed42_benchmark/data}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-3600}

mkdir -p "$ROOT"/logs "$ROOT"/results "$ROOT"/status

python3 - "$ROOT/checkpoint_inventory.json" "$ROOT/jobs.tsv" <<'PY'
import json
import sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text())
with Path(sys.argv[2]).open("w", encoding="utf-8") as stream:
    for row in rows:
        if row["status"] in {"ready", "ready_reused"}:
            stream.write("\t".join((row["job_id"], row["dataset"], row["local_path"])) + "\n")
PY

while IFS=$'\t' read -r job_id dataset checkpoint; do
  result="$ROOT/results/$job_id"
  log="$ROOT/logs/eval_${job_id}.log"
  status="$ROOT/status/${job_id}.ok"
  if [[ -s "$result/ultralytics_metrics.json" && -f "$status" ]]; then
    echo "REUSE $job_id"
    continue
  fi
  case "$dataset" in
    levirship|tinyperson|varroa) ;;
    *) echo "UNKNOWN_DATASET $dataset" >&2; exit 2 ;;
  esac
  echo "EVALUATE $job_id"
  mkdir -p "$result"
  timeout --signal=TERM --kill-after=30 "$TIMEOUT_SECONDS" docker run --rm --runtime=nvidia --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream:/app \
    -v "$ROOT/checkpoints:/checkpoints:ro" \
    -v "$ROOT/datasets:/datasets:ro" \
    -v "$DATA_SOURCE:$DATA_SOURCE:ro" \
    -v "$ROOT/results:/results" \
    -v "$ROOT/tools/evaluate.py:/app/evaluate.py:ro" \
    -v "$ROOT/tools/jetson_unified_evaluate.py:/app/jetson_unified_evaluate.py:ro" \
    -v "$ROOT/tools/${dataset}_jetson.yaml:/app/${dataset}.yaml:ro" \
    -v "$RUNTIME_DEPS:/opt/runtime_deps:ro" \
    --entrypoint python3 "$IMAGE" /app/jetson_unified_evaluate.py \
    --model "/checkpoints/$job_id/best.pt" \
    --dataset "$dataset" \
    --dataset-yaml "/app/${dataset}.yaml" \
    --split val test \
    --results-root "/results/$job_id" \
    --batch-size 1 --conf 0.001 --nms-iou 0.5 --max-det 300 \
    >"$log" 2>&1
  test -s "$result/ultralytics_metrics.json"
  touch "$status"
done < "$ROOT/jobs.tsv"

python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted((root / "results").glob("*/ultralytics_metrics.json")):
    if not (root / "status" / f"{path.parent.name}.ok").is_file():
        continue
    rows.extend(json.loads(path.read_text()))
(root / "metrics_all.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"QUEUE_COMPLETE jobs={len({row['model'] for row in rows})} split_rows={len(rows)}")
PY
