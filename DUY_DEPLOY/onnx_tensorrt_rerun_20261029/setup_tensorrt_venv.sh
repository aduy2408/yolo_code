#!/usr/bin/env bash
set -euo pipefail

# Creates a clean Python environment for the no-Docker ONNX/TensorRT notebook.
# The active Marimo kernel must be restarted with this venv afterward.

BASE_DIR=${BASE_DIR:-$(pwd)}
VENV=${VENV:-$BASE_DIR/.venv-tensorrt}
PYTHON_BIN=${PYTHON_BIN:-}
unset PYTHONPATH
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN=$(command -v "$candidate")
      break
    fi
  done
fi
[[ -n "$PYTHON_BIN" ]] || { echo "No supported Python interpreter found" >&2; exit 1; }
"$PYTHON_BIN" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# Match this host's CUDA 13 runtime instead of mixing ORT 1.22/CUDA 12.
"$VENV/bin/python" -m pip install ultralytics onnx
"$VENV/bin/python" -m pip install --force-reinstall \
  "onnxruntime-gpu==1.27.0" \
  "flatbuffers" "protobuf>=4.25.8" "packaging" \
  --no-deps
"$VENV/bin/python" -m pip install \
  "tensorrt==10.13.3.9" \
  "tensorrt-cu13-bindings==10.13.3.9" \
  "tensorrt-cu13-libs==10.13.3.9" \
  --no-deps

SITE=$($VENV/bin/python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
mapfile -t LIB_DIRS < <(
  find "$SITE" /usr/local/lib/python3.13/site-packages/nvidia \
    -type d \( -name lib -o -name lib64 -o -name tensorrt_libs \) -print 2>/dev/null | sort -u
)
LD_LIBRARY_PATH=$(IFS=:; echo "${LIB_DIRS[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH

"$VENV/bin/python" - <<'PY'
import ctypes
import importlib.util
import pathlib
import subprocess
import torch
import onnxruntime as ort
import tensorrt as trt

print("python:", __import__("sys").executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torch.cuda:", torch.cuda.is_available())
print("tensorrt:", getattr(trt, "__version__", "bindings"))
print("onnxruntime:", ort.__version__)
print("providers:", ort.get_available_providers())
assert torch.cuda.is_available(), "CUDA is unavailable"
assert "TensorrtExecutionProvider" in ort.get_available_providers(), "ORT TensorRT provider is unavailable"

ort_root = pathlib.Path(ort.__file__).parent
provider = next(ort_root.rglob("libonnxruntime_providers_tensorrt.so"))
ldd = subprocess.run(["ldd", str(provider)], check=True, capture_output=True, text=True).stdout
missing = [line.strip() for line in ldd.splitlines() if "=> not found" in line]
if missing:
    raise SystemExit("Missing TensorRT/CUDA libraries:\n" + "\n".join(missing))
print("TensorRT provider libraries: OK")
PY

echo
echo "TensorRT venv ready: $VENV"
echo "Use it with:"
echo "  source $VENV/bin/activate"
echo "  export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""
echo "Then restart Marimo from that shell; an existing kernel cannot change Python environments."
