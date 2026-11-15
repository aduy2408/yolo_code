# DUY_DEPLOY_DOCKER_WEB

Dockerized single-image inference UI for target-compatible Ultralytics TensorRT engines.

## Boundary

This service **does not** train, download checkpoints, export models, rebuild TensorRT engines, or change precision during a request. Prepare engines offline, validate them on the target Jetson, then mount them read-only.

Inference uses `YOLO(engine).predict(...)`. FP16 and FP32 are separate pre-exported artifacts selected by the API/UI; they are never silently substituted.

## Offline preparation

Use the repository's `ml2` environment for export, artifact inspection, and host-side smoke checks:

```bash
source /home/duylearch/miniconda3/bin/activate ml2
python DUY_DEPLOY_DOCKER_WEB/tools/export_ultralytics_engine.py \
  --checkpoint /path/to/parity/best.pt \
  --output /path/to/models/oacp-r2-yolov8-levirship-s43-fp16.engine \
  --imgsz 512 \
  --precision fp16
```

Run the command again with `--precision fp32` and a separate output path. The utility writes a sidecar manifest with engine and checkpoint SHA-256 values. Export must happen on a TensorRT-compatible target; an engine is not portable across arbitrary GPU/JetPack/TensorRT combinations.

Copy `models.example.json` only when creating a new operator registry. This checkout currently contains a local FP16/FP32 registry for the OACP R2 YOLOv8 LEVIR-Ship seed-43 engines. Those engines were validated on the current x86_64 CUDA/TensorRT runtime, not on Jetson; re-export/revalidate them on the deployment Jetson before using the Jetson Docker image. Do not put generated engines or checkpoints in this source folder.

## Docker runtime

The runtime image is parameterized by target:

### Current workstation: x86_64/CUDA 12.4

`Dockerfile.workstation` uses the pinned `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` base, installs Ultralytics `8.4.45`, TensorRT CUDA12 `10.14.1.48.post1`, and the web dependencies. This is the reproducible image for the current machine and the checked-in x86 TensorRT engines.

```bash
cd /path/to/yolo_related/DUY_DEPLOY_DOCKER_WEB
docker buildx bake --file docker-bake.hcl workstation --load
docker compose -f docker-compose.workstation.yml up -d
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

The workstation compose file enables NVIDIA runtime, mounts `./models` read-only, and uses the checked-in registry. It requires a working NVIDIA Container Toolkit installation on the host.

The image contains only the API/frontend and a fallback copy of `models.json`; it does not contain the TensorRT engines. Engine files are imported at runtime through a read-only bind mount. Changing models or the registry therefore does not require rebuilding the image:

```bash
DUY_WEB_MODELS_DIR=/absolute/path/to/engines \
DUY_WEB_REGISTRY_FILE=/absolute/path/to/models.json \
DUY_WEB_OUTPUT_DIR=/absolute/path/to/runtime_outputs \
docker compose -f docker-compose.workstation.yml up -d
```

The registry paths inside the container remain `/models` and `/config/models.json`. Registry `artifact` paths are resolved under `/models`, verified by SHA-256, and loaded once into the in-process model cache at startup.

### Jetson target: ARM64/JetPack

Jetson uses a different base image and must not reuse the workstation image or its TensorRT engines. Build on the Jetson, or on an ARM64 builder with the matching JetPack/TensorRT base:

```bash
cd /path/to/yolo_related/DUY_DEPLOY_DOCKER_WEB
BASE_IMAGE=duy-yolo-pytorch:jetpack6 \
DOCKER_PLATFORM=linux/arm64 \
RUNTIME_TAG=duy-deploy-web:jetpack6 \
docker buildx bake --file docker-bake.hcl runtime
```

If the Jetson base has a different tag, replace only `BASE_IMAGE` and record the exact base digest in the deployment log. Export the engines separately on the Jetson-compatible environment; do not copy x86 TensorRT engines to Jetson.

### Other target bases

The generic `Dockerfile` only adds the web layer to an existing Ultralytics/TensorRT base:

```bash
BASE_IMAGE=your-registry/your-ultralytics-tensorrt-base:tag \
DOCKER_PLATFORM=linux/amd64 \
RUNTIME_TAG=your-registry/duy-deploy-web:tag \
docker buildx bake --file docker-bake.hcl runtime
```

For the current x86 workstation, the portable smoke image validates the web/API layer without TensorRT:

```bash
docker buildx bake --file docker-bake.hcl portable-smoke --load
docker compose -f docker-compose.smoke.yml up -d
curl http://127.0.0.1:18000/healthz
docker compose -f docker-compose.smoke.yml down
```

The smoke image deliberately has an empty registry and is not an inference deployment. Real inference requires a target-compatible runtime base.

Run the generic runtime compose service after building it:

```bash
mkdir -p runtime_outputs models
test -f models.json || cp models.example.json models.json
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

Run with GPU access:

```bash
docker compose up -d
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

The compose service mounts:

- `./models:/models:ro` — validated `.engine` artifacts;
- `./models.json:/config/models.json:ro` — registry/provenance;
- `./runtime_outputs:/tmp/duy_deploy_outputs` — short-TTL annotated outputs.

The service preloads and warms all registered FP16/FP32 engines once at startup. `ModelCache` keeps loaded `YOLO(engine)` instances in process memory, so requests do not reload models. Readiness fails closed if the registry, checksum, engine load/warmup, backend identity, or provider assertion fails; startup records load/warmup timing and request telemetry records available host memory snapshots. A target-specific GPU memory/OOM failure during preload also keeps readiness failed.

## API

```text
GET  /healthz
GET  /readyz
GET  /v1/models
GET  /v1/metrics
POST /v1/models/{model_id}/predict?precision=fp16|fp32
GET  /v1/requests/{request_id}/output.png
```

Example request:

```bash
curl -X POST \
  -F file=@/path/to/image.jpg \
  'http://127.0.0.1:8000/v1/models/oacp-r2-yolov8-levirship-s43/predict?precision=fp16'
```

The response includes detections, original-image pixel coordinates, model provenance, resource snapshot, sequential FPS, and timing for queue, decode, preprocess, inference, postprocess, rendering, encoding, and total latency. TensorRT inference keeps the parity protocol at `conf: 0.001`, `nms_iou: 0.5`, and `max_det: 300`; product/UI detections are filtered at `display_conf: 0.3` after inference.

Open `http://127.0.0.1:8000/` for the browser UI. It exposes model/precision selectors, input/output images, detection table, latency bars, CPU/GPU metrics, provenance, and bounded request history.

## Verification

The repository tests that do not require a GPU can run with:

```bash
PYTHONPATH=DUY_DEPLOY_DOCKER_WEB python -m unittest discover -s DUY_DEPLOY_DOCKER_WEB/tests -v
```

The target-dependent gate still must run on the Jetson with both engines and ten fixed golden images. Record exact engine SHA-256, runtime identity, observed latency/FPS, and `nms_iou: 0.5` before claiming readiness.
