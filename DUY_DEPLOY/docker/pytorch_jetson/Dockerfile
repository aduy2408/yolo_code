FROM yolo-onnx-jetson:jp6

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt:/opt/project_root:/opt/ultralytics_src_upstream

COPY libcusparselt0-cuda-12_0.8.0.4-1_arm64.deb /tmp/libcusparselt0.deb
RUN dpkg -i /tmp/libcusparselt0.deb && ln -sf /usr/lib/aarch64-linux-gnu/libcusparseLt/12/libcusparseLt.so.0 /usr/local/lib/libcusparseLt.so.0 && ldconfig && rm -f /tmp/libcusparselt0.deb
COPY torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl /tmp/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
COPY torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl /tmp/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
RUN python3 -m pip install --no-cache-dir --no-deps --break-system-packages /tmp/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl /tmp/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl && rm -f /tmp/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl /tmp/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
COPY project_ultralytics/ /opt/project_root/project_ultralytics/
COPY ultralytics_src_upstream/ /opt/ultralytics/
COPY benchmark_ultralytics.py /app/benchmark_ultralytics.py
COPY benchmark_ultralytics_forward.py /app/benchmark_ultralytics_forward.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/benchmark_ultralytics.py"]
