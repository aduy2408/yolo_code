ARG BASE_IMAGE=duy-yolo-pytorch:jetpack6
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DUY_WEB_REGISTRY=/config/models.json \
    DUY_WEB_MODEL_ROOT=/models \
    DUY_WEB_OUTPUT_ROOT=/tmp/duy_deploy_outputs

WORKDIR /app
COPY requirements-web.txt /tmp/requirements-web.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-web.txt
COPY app /app/app
COPY web /app/web
COPY models.json /app/models.json
RUN mkdir -p /config && cp /app/models.json /config/models.json

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
ENTRYPOINT ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
