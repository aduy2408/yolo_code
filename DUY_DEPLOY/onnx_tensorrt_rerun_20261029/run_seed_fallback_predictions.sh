#!/usr/bin/env bash
set -euo pipefail

# Run one seed-42/43 matrix with seed 44 as the per-model fallback when seed 43 is absent.
# LEVIR uses no-Mosaic ONNX, TinyPerson/Varroa use Mosaic ONNX.
# TinyPerson is prediction-only: corner-window predictions are saved with --save-only.

bundle_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$bundle_root/.." && pwd)
models_root=${MODELS_ROOT:-"$repo_root/DUY_DEPLOY/models"}
data_root=${DATA_ROOT:-/home/ait01/jetson_seed42_benchmark/data}
results_root=${RESULTS_ROOT:-"$repo_root/DUY_DEPLOY/results/seed_fallback_predictions_$(date +%Y%m%d_%H%M%S)"}
image=${IMAGE:-yolo-onnx-jetson:jp6}
batch_size=${BATCH_SIZE:-4}
model_timeout=${MODEL_TIMEOUT:-900}
conf=${CONF:-0.001}
nms_iou=${NMS_IOU:-0.5}
max_det=${MAX_DET:-300}

if [[ ${1:-} == "--dry-run" ]]; then
  dry_run=1
else
  dry_run=0
fi

models=(yolov5 yolov8 yolov9 yolov10 yolo11)
datasets=(levirship tinyperson varroa)

select_seed() {
  local dataset=$1 model=$2 seed
  local mode=no_mosaic
  [[ "$dataset" == "tinyperson" || "$dataset" == "varroa" ]] && mode=mosaic
  for seed in 42 43 44; do
    if [[ -f "$models_root/onnx_cuda_fp16/$dataset/$mode/$model/seed_${seed}/best_fp16_batch1.onnx" ]]; then
      if [[ "$seed" == 44 && -f "$models_root/onnx_cuda_fp16/$dataset/$mode/$model/seed_43/best_fp16_batch1.onnx" ]]; then
        continue
      fi
      printf '%s\n' "$seed"
      return 0
    fi
  done
  return 1
}

run_one() {
  local dataset=$1 model=$2 seed=$3 mode=$4 relative=$5 job=$6
  local model_dir="$models_root/onnx_cuda_fp16/$dataset/$mode/$model/seed_${seed}"
  local job_root="$results_root/$dataset/$model/seed_${seed}"
  local cache_root="$results_root/engines_by_model/$dataset/$model/seed_${seed}"
  mkdir -p "$job_root/metrics" "$job_root/predictions" "$cache_root"
  echo "$job"
  if ((dry_run)); then
    return 0
  fi
  local cidfile status=0 cid
  cidfile=$(mktemp)
  rm -f "$cidfile"
  if [[ "$dataset" == "tinyperson" ]]; then
    timeout --signal=TERM --kill-after=20 "$model_timeout" docker run --cidfile "$cidfile" --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
      -v "$model_dir:/models:ro" -v "$data_root:/datasets:ro" \
      -v "$bundle_root/source/evaluate.py:/app/evaluate.py:ro" \
      -v "$bundle_root/source/evaluate_tinyperson_merged.py:/app/evaluate_tinyperson_merged.py:ro" \
      -v "$job_root:/results" -v "$cache_root:/cache" --entrypoint python3 "$image" \
      /app/evaluate_tinyperson_merged.py --models-root /models --model-path /models/best_fp16_batch1.onnx \
      --data-root /datasets --results-root /results/metrics --predictions-root /results/predictions \
      --trt-cache-root /cache --batch-size "$batch_size" --conf "$conf" --nms-iou "$nms_iou" --max-det "$max_det" --save-only \
      >"$job_root.log" 2>&1 || status=$?
  else
    timeout --signal=TERM --kill-after=20 "$model_timeout" docker run --cidfile "$cidfile" --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
      -v "$model_dir:/models:ro" -v "$data_root:/datasets:ro" \
      -v "$bundle_root/source/evaluate.py:/app/evaluate.py:ro" \
      -v "$job_root:/results" -v "$cache_root:/cache" --entrypoint python3 "$image" \
      /app/evaluate.py --models-root /models --data-root /datasets --dataset "$dataset" --split val test \
      --results-root /results/metrics --predictions-root /results/predictions --provider tensorrt \
      --trt-cache-root /cache --batch-size "$batch_size" --conf "$conf" --nms-iou "$nms_iou" --max-det "$max_det" \
      >"$job_root.log" 2>&1 || status=$?
  if [[ -s "$cidfile" ]]; then
    cid=$(<"$cidfile")
    docker rm -f "$cid" >/dev/null 2>&1 || true
  fi
  rm -f "$cidfile"
  fi
  if ((status != 0)); then
    printf '{"dataset":"%s","model":"%s","seed":%s,"status":"failed","log":"%s"}\n' "$dataset" "$model" "$seed" "$job_root.log" >"$job_root/failure.json"
  fi
}

mkdir -p "$results_root"
printf '{\n  "split_seed": 42,\n  "datasets": {"levirship": "no_mosaic", "tinyperson": "mosaic", "varroa": "mosaic"},\n  "seed_policy": "include seed 42 and seed 43; use seed 44 only when seed 43 is absent",\n  "jobs": [\n' >"$results_root/matrix.json"
first=1
for dataset in "${datasets[@]}"; do
  mode=no_mosaic
  [[ "$dataset" == "tinyperson" || "$dataset" == "varroa" ]] && mode=mosaic
  for model in "${models[@]}"; do
    for seed in 42 43; do
      if [[ "$seed" == 43 && ! -f "$models_root/onnx_cuda_fp16/$dataset/$mode/$model/seed_43/best_fp16_batch1.onnx" ]]; then
        seed=44
      fi
      model_path="$models_root/onnx_cuda_fp16/$dataset/$mode/$model/seed_${seed}/best_fp16_batch1.onnx"
      if [[ ! -f "$model_path" ]]; then
        echo "MISSING $dataset/$model/seed_$seed" >&2
        exit 1
      fi
      job="$dataset/$model/seed_$seed"
      ((first)) || printf ',\n' >>"$results_root/matrix.json"
      first=0
      printf '    {"dataset":"%s","model":"%s","seed":%s,"augmentation":"%s","model_path":"%s"}' "$dataset" "$model" "$seed" "$mode" "$model_path" >>"$results_root/matrix.json"
      run_one "$dataset" "$model" "$seed" "$mode" "$model_path" "$job"
    done
  done
done
printf '\n  ]\n}\n' >>"$results_root/matrix.json"
echo "Results: $results_root"
