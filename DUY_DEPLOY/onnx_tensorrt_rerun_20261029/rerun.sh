#!/usr/bin/env bash
set -euo pipefail

# Run on the Jetson host from the DUY_DEPLOY checkout.
# Each model gets its own TensorRT cache to prevent cross-seed/cross-shape reuse.

usage() {
  cat <<'EOF'
Usage:
  ./rerun.sh benchmark [--dataset DATASET]
  ./rerun.sh e2e      [--dataset DATASET]
  ./rerun.sh accuracy [--dataset DATASET]
  ./rerun.sh all      [--dataset DATASET]

DATASET: levirship, tinyperson, varroa, or all (default: all); add --dry-run to print Docker commands only.
Environment overrides:
  DEPLOY_ROOT  DUY_DEPLOY root (default: script parent)
  IMAGE        Docker image (default: yolo-onnx-jetson:jp6)
  DATA_ROOT    dataset mount (default: /home/ait01/jetson_seed42_benchmark/data)
  BATCH_SIZE   default 4
  WARMUP       default 50
  ITERATIONS   default 200
  MODEL_TIMEOUT seconds per model (default 120)
  SEEDS        comma-separated seed filter (default: 42,43)
  REUSE_TRT_CACHE_ROOT  existing TensorRT cache to reuse instead of building a new cache
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -eq 0 ]]; then
  usage
  [[ $# -eq 0 ]] && exit 2 || exit 0
fi
phase=$1
shift
[[ "$phase" =~ ^(benchmark|e2e|accuracy|all)$ ]] || { usage >&2; exit 2; }

dataset_filter=all
dry_run=0
while (($#)); do
  case "$1" in
    --dataset) dataset_filter=${2:?missing dataset}; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$dataset_filter" =~ ^(all|levirship|tinyperson|varroa)$ ]] || { echo "invalid dataset: $dataset_filter" >&2; exit 2; }

bundle_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$bundle_root/.." && pwd)
models_root=${MODELS_ROOT:-"$repo_root/DUY_DEPLOY/models"}
image=${IMAGE:-yolo-onnx-jetson:jp6}
data_root=${DATA_ROOT:-/home/ait01/jetson_seed42_benchmark/data}
batch_size=${BATCH_SIZE:-4}
warmup=${WARMUP:-50}
iterations=${ITERATIONS:-200}
model_timeout=${MODEL_TIMEOUT:-120}
seed_filter=${SEEDS:-42,43}
want_seed() { [[ -z "$1" || ",$seed_filter," == *,"$1",* ]]; }
run_root=${RESULTS_ROOT:-"$repo_root/DUY_DEPLOY/results/tensorrt/rerun_20261029"}
predictions_root=${PREDICTIONS_ROOT:-"$run_root/predictions"}
resume_existing=${RESUME_EXISTING:-0}
cache_root="$run_root/engines_by_model"
reuse_cache_root=${REUSE_TRT_CACHE_ROOT:-}
mkdir -p "$run_root/benchmark" "$run_root/accuracy" "$cache_root"

container_args=(
  --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all
  -v "$models_root:/models:ro"
  -v "$data_root:/datasets:ro"
  -v "$bundle_root/source/benchmark.py:/app/benchmark.py:ro"
  -v "$bundle_root/source/benchmark_end_to_end.py:/app/benchmark_end_to_end.py:ro"
  -v "$bundle_root/source/evaluate.py:/app/evaluate.py:ro"
  -v "$bundle_root/source/summarize_e2e.py:/app/summarize_e2e.py:ro"
  -v "$run_root:/results"
)
if [[ -n "$reuse_cache_root" ]]; then
  container_args+=( -v "$reuse_cache_root:/reuse_cache:ro" )
fi
container_args+=( --entrypoint python3 "$image" )

docker_run() {
  if ((dry_run)); then
    printf 'DRY-RUN docker run'
    printf ' %q' "$@"
    printf '\n'
  else
    docker run "$@"
  fi
}
docker_run_timed() {
  if ((dry_run)); then
    docker_run "$@"
    return
  fi
  local cidfile status=0 cid
  cidfile=$(mktemp)
  rm -f "$cidfile"
  timeout --signal=TERM --kill-after=10 "$model_timeout" \
    docker run --cidfile "$cidfile" "$@" || status=$?
  if [[ -s "$cidfile" ]]; then
    cid=$(<"$cidfile")
    docker rm -f "$cid" >/dev/null 2>&1 || true
  fi
  rm -f "$cidfile"
  return "$status"
}

want_dataset() { [[ "$dataset_filter" == all || "$dataset_filter" == "$1" ]]; }
want_model() {
  [[ -z "$seed_filter" ]] && return 0
  [[ "$1" =~ seed_([0-9]+) ]] || return 1
  want_seed "${BASH_REMATCH[1]}"
}

run_benchmark() {
  local dataset model relative key model_cache output_dir
  while IFS= read -r model; do
    relative=${model#"$models_root/onnx_fp16_dynamic/"}
    dataset=${relative%%/*}
    want_dataset "$dataset" || continue
    want_model "$relative" || continue
    key=${relative%.onnx}
    model_cache="$cache_root/benchmark/$key"
    output_dir="$run_root/benchmark/$key"
    mkdir -p "$model_cache" "$output_dir"
    echo "BENCHMARK $relative"
    docker_run "${container_args[@]}" \
      /app/benchmark.py \
      --models-root "/models/onnx_fp16_dynamic/${relative%/*}" \
      --results-root /results/benchmark/$key \
      --provider tensorrt \
      --trt-cache-root "/results/engines_by_model/benchmark/$key" \
      --batch-size "$batch_size" --warmup "$warmup" --iterations "$iterations"
  done < <(find "$models_root/onnx_fp16_dynamic" -type f -name '*.onnx' | sort)
}

run_e2e() {
  local dataset model relative key model_cache log_path failure_path trt_cache_path output_dir
  while IFS= read -r model; do
    relative=${model#"$models_root/onnx_cuda_fp16/"}
    dataset=${relative%%/*}
    want_dataset "$dataset" || continue
    want_model "$relative" || continue
    key=${relative%.onnx}
    output_dir="$run_root/e2e/$key"
    if [[ "$resume_existing" == "1" ]] && find "$output_dir" -maxdepth 1 -type f -name '*.json' ! -name failure.json -print -quit 2>/dev/null | grep -q .; then
      echo "SKIP existing e2e $relative"
      continue
    fi
    model_cache="$cache_root/e2e/$key"
    log_path="$run_root/logs/e2e/$key.log"
    failure_path="$run_root/e2e/$key/failure.json"
    trt_cache_path="/results/engines_by_model/e2e/$key"
    if [[ -n "$reuse_cache_root" ]]; then
      trt_cache_path="/reuse_cache"
    fi
    mkdir -p "$model_cache" "$(dirname "$log_path")" "$(dirname "$failure_path")"
    echo "E2E $relative"
    if ! docker_run_timed "${container_args[@]}" \
      /app/benchmark_end_to_end.py \
      --models-root "/models/onnx_cuda_fp16/${relative%/*}" \
      --data-root /datasets --dataset "$dataset" \
      --results-root /results/e2e/$key \
      --trt-cache-root "$trt_cache_path" \
      --batch-size "$batch_size" --warmup "$warmup" --iterations "$iterations" \
      --conf 0.001 --nms-iou 0.5 --max-det 300 >"$log_path" 2>&1; then
      printf '{\n  "dataset": "%s",\n  "model": "%s",\n  "status": "failed",\n  "log": "%s"\n}\n' "$dataset" "$relative" "$log_path" >"$failure_path"
      echo "FAILED $relative; see $log_path" >&2
    fi
  done < <(find "$models_root/onnx_cuda_fp16" -type f -name '*.onnx' | sort)
  docker_run "${container_args[@]}" /app/summarize_e2e.py --results-root /results/e2e
}

run_accuracy() {
  local dataset model relative key model_cache trt_cache_path log_path failure_path prediction_marker
  while IFS= read -r model; do
    relative=${model#"$models_root/onnx_cuda_fp16/"}
    dataset=${relative%%/*}
    want_dataset "$dataset" || continue
    want_model "$relative" || continue
    key=${relative%.onnx}
    prediction_marker="$predictions_root/$key"
    if [[ "$resume_existing" == "1" ]] && find "$prediction_marker" -type f -name '*.json' -print -quit 2>/dev/null | grep -q .; then
      echo "SKIP existing predictions $relative"
      continue
    fi
    model_cache="$cache_root/accuracy/$key"
    trt_cache_path="/results/engines_by_model/accuracy/$key"
    if [[ -n "$reuse_cache_root" ]]; then
      trt_cache_path="/reuse_cache"
    fi
    log_path="$run_root/logs/accuracy/$key.log"
    failure_path="$run_root/accuracy/$key/failure.json"
    mkdir -p "$model_cache" "$(dirname "$log_path")" "$(dirname "$failure_path")"
    echo "ACCURACY $relative"
    if ! docker_run_timed "${container_args[@]}" \
      /app/evaluate.py \
      --models-root "/models/onnx_cuda_fp16/${relative%/*}" \
      --data-root /datasets --dataset "$dataset" --split test \
      --results-root /results/accuracy/$key \
      --predictions-root /results/predictions/$key \
      --provider tensorrt \
      --trt-cache-root "$trt_cache_path" \
      --batch-size "$batch_size" --conf 0.001 --nms-iou 0.5 --max-det 300 >"$log_path" 2>&1; then
      printf '{\n  "dataset": "%s",\n  "model": "%s",\n  "status": "failed",\n  "log": "%s"\n}\n' "$dataset" "$relative" "$log_path" >"$failure_path"
      echo "FAILED $relative; see $log_path" >&2
    fi
  done < <(find "$models_root/onnx_cuda_fp16" -type f -name '*.onnx' | sort)
}

case "$phase" in
  benchmark) run_benchmark ;;
  e2e) run_e2e ;;
  accuracy) run_accuracy ;;
  all) run_benchmark; run_e2e; run_accuracy ;;
esac

echo "Results: $run_root"
