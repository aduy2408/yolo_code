FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ORT_TENSORRT_ENGINE_CACHE_ENABLE=1 \
    ORT_TENSORRT_CACHE_PATH=/results/ort_engine_cache

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libjpeg-turbo8 \
        zlib1g && \
    rm -rf /var/lib/apt/lists/*

ADD https://bootstrap.pypa.io/get-pip.py /tmp/get-pip.py
RUN python3 /tmp/get-pip.py --disable-pip-version-check && \
    rm -f /tmp/get-pip.py

WORKDIR /app

# Jetson ARM64 wheel; the host JetPack runtime supplies CUDA/TensorRT libraries.
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl /tmp/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir \
        /tmp/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl \
        -r /tmp/requirements.txt && \
    rm -f /tmp/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

COPY preflight.py /app/preflight.py
COPY benchmark.py /app/benchmark.py
COPY evaluate.py /app/evaluate.py
COPY prepare_cuda_models.py /app/prepare_cuda_models.py
COPY evaluate_tinyperson_merged.py /app/evaluate_tinyperson_merged.py
COPY merge_tinyperson_predictions.py /app/merge_tinyperson_predictions.py

ENTRYPOINT ["python3", "/app/preflight.py"]
