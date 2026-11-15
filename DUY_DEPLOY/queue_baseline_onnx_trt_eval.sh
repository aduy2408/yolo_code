#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY}
MANIFEST=${MANIFEST:-$ROOT/manifests/baseline_trt_38_manifest.json}
DATA_ROOT=${DATA_ROOT:-/home/ait01/jetson_seed42_benchmark/data}
OUT=${OUT:-$ROOT/results/baseline_onnx_trt_eval_20261104}
PYTORCH_IMAGE=${PYTORCH_IMAGE:-duy-yolo-pytorch:jetpack6}
ONNX_IMAGE=${ONNX_IMAGE:-yolo-onnx-jetson:jp6}
EXPORT_TIMEOUT=${EXPORT_TIMEOUT:-1800}
BUILD_TIMEOUT=${BUILD_TIMEOUT:-1800}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-7200}

mkdir -p "$OUT"/{onnx,engines,results,logs,status}
python3 "$ROOT/download_queue_checkpoints.py" --manifest "$MANIFEST" --output-root "$OUT"
[[ "${DOWNLOAD_ONLY:-0}" == "1" ]] && exit 0

mapfile -t JOBS < <(python3 "$ROOT/iter_baseline_trt_jobs.py" --manifest "$MANIFEST" --checkpoint-root "$OUT")
[[ "${#JOBS[@]}" -eq 38 ]] || { echo "BASELINE_JOB_COUNT_MISMATCH ${#JOBS[@]}" >&2; exit 1; }

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
for precision in fp32 fp16; do
  while IFS=$'\t' read -r job dataset imgsz ckpt; do
    onnx="$OUT/onnx/$precision/$job/best_${precision}_batch1.onnx"
    log="$OUT/logs/$job/${precision}_export.log"
    mkdir -p "$(dirname "$onnx")" "$(dirname "$log")"
    echo "EXPORT precision=$precision job=$job"
    timeout --signal=TERM --kill-after=30 "$EXPORT_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
      -v "$ckpt:/models/best.pt:ro" -v "$(dirname "$onnx"):/out" -v "$ROOT/docker/pytorch_jetson/export_onnx.py:/app/export_onnx.py:ro" \
      -v "$ROOT/runtime_deps:/opt/runtime_deps:ro" -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream \
      --entrypoint python3 "$PYTORCH_IMAGE" /app/export_onnx.py --checkpoint /models/best.pt --output "/out/best_${precision}_batch1.onnx" --imgsz "$imgsz" --precision "$precision" >"$log" 2>&1
    touch "$OUT/status/${precision}_${job}.onnx_ok"
  done < <(printf '%s\n' "${JOBS[@]}")
done
  fi

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
for precision in fp32 fp16; do
  while IFS=$'\t' read -r job dataset imgsz ckpt; do
    onnx="$OUT/onnx/$precision/$job/best_${precision}_batch1.onnx"
    cache="$OUT/engines/$precision/$job"
    log="$OUT/logs/$job/${precision}_build.log"
    mkdir -p "$cache"
    echo "BUILD precision=$precision job=$job"
    timeout --signal=TERM --kill-after=30 "$BUILD_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
      -v "$onnx:/models/best_${precision}_batch1.onnx:ro" -v "$cache:/cache" -v "$DATA_ROOT:/datasets:ro" \
      -v "$ROOT/docker/jetson_onnx/evaluate.py:/app/evaluate.py:ro" -v "$ROOT/docker/jetson_onnx/build_engine_dataset.py:/app/build_engine_dataset.py:ro" \
      -v "$ROOT/runtime_deps:/opt/runtime_deps:ro" -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream \
      --entrypoint python3 "$ONNX_IMAGE" /app/build_engine_dataset.py --model "/models/best_${precision}_batch1.onnx" --cache /cache --data-root /datasets --dataset "$dataset" --precision "$precision" >"$log" 2>&1
    touch "$OUT/status/${precision}_${job}.engine_ok"
  done < <(printf '%s\n' "${JOBS[@]}")
done
  fi

python3 "$ROOT/check_dual_precision_artifacts.py" --root "$OUT" --expected-count 38 --manifest "$MANIFEST" --engines-root "$OUT/engines"

for precision in fp32 fp16; do
  while IFS=$'\t' read -r job dataset imgsz ckpt; do
    onnx="$OUT/onnx/$precision/$job/best_${precision}_batch1.onnx"
    cache="$OUT/engines/$precision/$job"
    result="$OUT/results/$precision/$job"
    log="$OUT/logs/$job/${precision}_evaluate.log"
    mkdir -p "$result/metrics" "$result/predictions"
    echo "EVALUATE precision=$precision job=$job"
    if [[ "$dataset" == "tinyperson" ]]; then
      timeout --signal=TERM --kill-after=30 "$EVAL_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
        -v "$onnx:/models/best_${precision}_batch1.onnx:ro" -v "$cache:/trt_cache" -v "$DATA_ROOT:/datasets:ro" -v "$result:/results" \
        -v "$ROOT/docker/jetson_onnx/evaluate.py:/app/evaluate.py:ro" -v "$ROOT/docker/jetson_onnx/evaluate_tinyperson_merged.py:/app/evaluate_tinyperson_merged.py:ro" \
        -v "$ROOT/runtime_deps:/opt/runtime_deps:ro" -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream \
        --entrypoint python3 "$ONNX_IMAGE" /app/evaluate_tinyperson_merged.py --models-root /models --model-path "/models/best_${precision}_batch1.onnx" \
        --data-root /datasets --results-root /results/metrics --predictions-root /results/predictions --trt-cache-root /trt_cache \
        --precision "$precision" --batch-size 1 --conf 0.001 --nms-iou 0.5 --max-det 300 >"$log" 2>&1
    else
      timeout --signal=TERM --kill-after=30 "$EVAL_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
        -v "$onnx:/models/best_${precision}_batch1.onnx:ro" -v "$cache:/trt_cache" -v "$DATA_ROOT:/datasets:ro" -v "$result:/results" \
        -v "$ROOT/docker/jetson_onnx/evaluate.py:/app/evaluate.py:ro" \
        -v "$ROOT/runtime_deps:/opt/runtime_deps:ro" -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream \
        --entrypoint python3 "$ONNX_IMAGE" /app/evaluate.py --models-root /models --data-root /datasets --dataset "$dataset" --split val test --results-root /results/metrics \
        --provider tensorrt --precision "$precision" --trt-cache-root /trt_cache --batch-size 1 --conf 0.001 --nms-iou 0.5 --max-det 300 >"$log" 2>&1
    fi
    touch "$OUT/status/${precision}_${job}.eval_ok"
  done < <(printf '%s\n' "${JOBS[@]}")
done

echo "QUEUE_COMPLETE out=$OUT jobs=38 precisions=fp32,fp16"
