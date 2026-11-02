#!/usr/bin/env bash
set -euo pipefail

shopt -s globstar nullglob
models_root=${1:?models root required}
engines_root=${2:?engines root required}
trtexec_bin=${TRTEXEC_BIN:-}
if [[ -z "$trtexec_bin" ]]; then
  trtexec_bin=$(command -v trtexec || true)
fi
if [[ -z "$trtexec_bin" && -x /usr/src/tensorrt/bin/trtexec ]]; then
  trtexec_bin=/usr/src/tensorrt/bin/trtexec
fi
if [[ -z "$trtexec_bin" ]]; then
  echo "trtexec not found; set TRTEXEC_BIN" >&2
  exit 2
fi

for model in "$models_root"/**/*.onnx; do
  [[ -f "$model" ]] || continue
  relative=${model#"$models_root"/}
  dataset=${relative%%/*}
  size=640
  [[ "$dataset" == levirship ]] && size=512
  engine="$engines_root/${relative%.onnx}.fp16.b1-8.engine"
  mkdir -p "$(dirname "$engine")"
  if [[ -f "$engine" ]]; then
    echo "SKIP $engine"
    continue
  fi
  echo "BUILD $model -> $engine"
  "$trtexec_bin" \
    --onnx="$model" \
    --saveEngine="$engine" \
    --fp16 \
    --minShapes=images:1x3x${size}x${size} \
    --optShapes=images:4x3x${size}x${size} \
    --maxShapes=images:8x3x${size}x${size} \
    --workspace=2048 \
    --skipInference
 done
