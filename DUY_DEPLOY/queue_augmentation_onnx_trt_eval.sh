#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY}
DATA_ROOT=${DATA_ROOT:-/home/ait01/jetson_seed42_benchmark/data}
OUT=${OUT:-$ROOT/results/augmentation_onnx_trt_eval_20261103}
PYTORCH_IMAGE=${PYTORCH_IMAGE:-duy-yolo-pytorch:jetpack6}
ONNX_IMAGE=${ONNX_IMAGE:-yolo-onnx-jetson:jp6}
HF_REPO_OACP=${HF_REPO_OACP:-duyle2408/model-augmentation-seed42-oacp-runs}
HF_REPO_CP=${HF_REPO_CP:-duyle2408/model-augmentation-seed42-copy-paste-runs}
EXPORT_TIMEOUT=${EXPORT_TIMEOUT:-1800}
BUILD_TIMEOUT=${BUILD_TIMEOUT:-1800}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-7200}

mkdir -p "$OUT"/{checkpoints,onnx,engines,results,logs,status}

# dataset model augmentation source-repo/path mapping is fixed by aug_3yolo_main.md.
for spec in \
  "levir yolov8 oacp r2 no_mosaic 512" "levir yolov8 nc2 negative_canvas_r2 no_mosaic 512" "levir yolov8 nc4 negative_canvas_r4 no_mosaic 512" \
  "levir yolov11 oacp r2 no_mosaic 512" "levir yolov11 nc2 negative_canvas_r2 no_mosaic 512" "levir yolov11 nc4 negative_canvas_r4 no_mosaic 512" \
  "tinyperson yolov8 oacp r2 mosaic 640" "tinyperson yolov8 nc2 negative_canvas_r2 mosaic 640" "tinyperson yolov8 nc4 negative_canvas_r4 mosaic 640" \
  "tinyperson yolov11 oacp r2 mosaic 640" "tinyperson yolov11 nc2 negative_canvas_r2 mosaic 640" "tinyperson yolov11 nc4 negative_canvas_r4 mosaic 640"; do
  read -r dataset model aug variant mosaic imgsz <<< "$spec"
  repo="$HF_REPO_CP"; method="copy_paste"
  [[ "$aug" == "oacp" ]] && repo="$HF_REPO_OACP" && method="oacp"
  job="$dataset/$model/$aug"
  ckpt="$OUT/checkpoints/$job/best.pt"
  url="https://huggingface.co/datasets/$repo/resolve/main/$model/$dataset/$method/$variant/$mosaic/seed_42/weights/best.pt"
  log="$OUT/logs/${dataset}_${model}_${aug}"
  mkdir -p "$(dirname "$ckpt")" "$log"
  if [[ ! -s "$ckpt" ]]; then
    echo "DOWNLOAD $job"
    curl_args=(--fail --location --retry 3 --retry-delay 5)
    [[ -n "${HF_TOKEN:-}" ]] && curl_args+=(-H "Authorization: Bearer $HF_TOKEN")
    curl "${curl_args[@]}" "$url" -o "$ckpt" >"$log/download.log" 2>&1
  fi
  [[ -s "$ckpt" ]] || { echo "MISSING_CHECKPOINT $job" >&2; exit 1; }
done

if [[ "${DOWNLOAD_ONLY:-0}" == "1" ]]; then
  echo "DOWNLOAD_COMPLETE out=$OUT"
  exit 0
fi
# Phase 1: export every PyTorch checkpoint to a static FP16 ONNX model.
for spec in \
  "levir yolov8 oacp 512" "levir yolov8 nc2 512" "levir yolov8 nc4 512" "levir yolov11 oacp 512" "levir yolov11 nc2 512" "levir yolov11 nc4 512" \
  "tinyperson yolov8 oacp 640" "tinyperson yolov8 nc2 640" "tinyperson yolov8 nc4 640" "tinyperson yolov11 oacp 640" "tinyperson yolov11 nc2 640" "tinyperson yolov11 nc4 640"; do
  read -r dataset model aug imgsz <<< "$spec"
  job="$dataset/$model/$aug"; ckpt="$OUT/checkpoints/$job/best.pt"; onnx="$OUT/onnx/$job/best_fp16_batch1.onnx"; log="$OUT/logs/${dataset}_${model}_${aug}/export.log"
  mkdir -p "$(dirname "$onnx")"
  echo "EXPORT $job"
  timeout --signal=TERM --kill-after=30 "$EXPORT_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$ckpt:/models/best.pt:ro" -v "$(dirname "$onnx"):/out" -v "$ROOT/docker/pytorch_jetson/export_onnx.py:/app/export_onnx.py:ro" \
    -v "$ROOT/runtime_deps:/opt/runtime_deps:ro" -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream \
    --entrypoint python3 "$PYTORCH_IMAGE" /app/export_onnx.py --checkpoint /models/best.pt --output /out/best_fp16_batch1.onnx --imgsz "$imgsz" >"$log" 2>&1
  touch "$OUT/status/${dataset}_${model}_${aug}.onnx_ok"
done

# Phase 2: build all device-specific TensorRT engines before any full evaluation.
for spec in \
  "levir yolov8 oacp 512" "levir yolov8 nc2 512" "levir yolov8 nc4 512" "levir yolov11 oacp 512" "levir yolov11 nc2 512" "levir yolov11 nc4 512" \
  "tinyperson yolov8 oacp 640" "tinyperson yolov8 nc2 640" "tinyperson yolov8 nc4 640" "tinyperson yolov11 oacp 640" "tinyperson yolov11 nc2 640" "tinyperson yolov11 nc4 640"; do
  read -r dataset model aug imgsz <<< "$spec"
  build_dataset="$dataset"
  [[ "$build_dataset" == "levir" ]] && build_dataset="levirship"
  job="$dataset/$model/$aug"; onnx="$OUT/onnx/$job/best_fp16_batch1.onnx"; cache="$OUT/engines/$job"; log="$OUT/logs/${dataset}_${model}_${aug}/build.log"
  mkdir -p "$cache"
  echo "BUILD $job"
  timeout --signal=TERM --kill-after=30 "$BUILD_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$onnx:/models/best_fp16_batch1.onnx:ro" -v "$cache:/cache" -v "$DATA_ROOT:/datasets:ro" \
    -v "$ROOT/docker/jetson_onnx/evaluate.py:/app/evaluate.py:ro" -v "$ROOT/docker/jetson_onnx/build_engine_dataset.py:/app/build_engine_dataset.py:ro" \
    -v "$ROOT/runtime_deps:/opt/runtime_deps:ro" -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream \
    --entrypoint python3 "$ONNX_IMAGE" /app/build_engine_dataset.py --model /models/best_fp16_batch1.onnx --cache /cache --data-root /datasets --dataset "$build_dataset" >"$log" 2>&1
  touch "$OUT/status/${dataset}_${model}_${aug}.engine_ok"
done

# Phase 3: evaluate each completed TensorRT engine. TinyPerson uses merged evaluation.
for spec in \
  "levir yolov8 oacp" "levir yolov8 nc2" "levir yolov8 nc4" "levir yolov11 oacp" "levir yolov11 nc2" "levir yolov11 nc4" \
  "tinyperson yolov8 oacp" "tinyperson yolov8 nc2" "tinyperson yolov8 nc4" "tinyperson yolov11 oacp" "tinyperson yolov11 nc2" "tinyperson yolov11 nc4"; do
  read -r dataset model aug <<< "$spec"
  job="$dataset/$model/$aug"; onnx="$OUT/onnx/$job/best_fp16_batch1.onnx"; cache="$OUT/engines/$job"; result="$OUT/results/$job"; log="$OUT/logs/${dataset}_${model}_${aug}/evaluate.log"
  mkdir -p "$result/metrics" "$result/predictions"
  echo "EVALUATE $job"
  if [[ "$dataset" == "levir" ]]; then
    timeout --signal=TERM --kill-after=30 "$EVAL_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
      -v "$onnx:/models/best_fp16_batch1.onnx:ro" -v "$cache:/trt_cache" -v "$DATA_ROOT:/datasets:ro" -v "$result:/results" \
      -v "$ROOT/docker/jetson_onnx/evaluate.py:/app/evaluate.py:ro" \
      -v "$ROOT/runtime_deps:/opt/runtime_deps:ro" -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream \
      --entrypoint python3 "$ONNX_IMAGE" /app/evaluate.py \
      --models-root /models --data-root /datasets --dataset levirship --split val test --results-root /results/metrics \
      --provider tensorrt --trt-cache-root /trt_cache --batch-size 1 --conf 0.001 --nms-iou 0.5 --max-det 300 >"$log" 2>&1
  else
    timeout --signal=TERM --kill-after=30 "$EVAL_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
      -v "$onnx:/models/best_fp16_batch1.onnx:ro" -v "$cache:/cache" -v "$DATA_ROOT:/datasets:ro" -v "$result:/results" \
      -v "$ROOT/docker/jetson_onnx/evaluate.py:/app/evaluate.py:ro" -v "$ROOT/docker/jetson_onnx/evaluate_tinyperson_merged.py:/app/evaluate_tinyperson_merged.py:ro" \
      -v "$ROOT/runtime_deps:/opt/runtime_deps:ro" -e PYTHONPATH=/opt/runtime_deps:/opt:/opt/project_root:/opt/ultralytics_src_upstream \
      --entrypoint python3 "$ONNX_IMAGE" /app/evaluate_tinyperson_merged.py --models-root /models --model-path /models/best_fp16_batch1.onnx \
      --data-root /datasets --results-root /results/metrics --predictions-root /results/predictions --trt-cache-root /cache \
      --batch-size 1 --conf 0.001 --nms-iou 0.5 --max-det 300 >"$log" 2>&1
  fi
  touch "$OUT/status/${dataset}_${model}_${aug}.eval_ok"
done

echo "QUEUE_COMPLETE out=$OUT"
