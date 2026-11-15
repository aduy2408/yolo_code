# DUY_DEPLOY_DOCKER_WEB — kế hoạch triển khai inference web

## 1. Goal

Xây dựng một web app nội bộ, chạy bằng Docker trên Jetson có GPU, để:

1. Chọn model detection đã export/validate cho TensorRT.
2. Upload một ảnh và nhận ảnh output có bounding boxes + detections.
3. Hiển thị metrics inference và host theo thời gian thực.
4. Giữ provenance của model, backend, precision, input size và protocol.
5. Không silent fallback từ TensorRT sang CUDA/CPU.

**MVP goal tự đặt:** trong một container inference duy nhất, người dùng mở browser, chọn một model OACP/NegativeCanvas đã đăng ký, upload ảnh, và nhận kết quả trong một request với output image, JSON detections, latency breakdown, FPS, CPU/GPU/memory snapshot và model metadata. Có smoke test chạy được không cần training/evaluation dataset.

## 2. Evidence và các quyết định đã chốt

- Mọi bước offline export, artifact inspection và host-side smoke phải chạy trong environment `ml2`:

```bash
source /home/duylearch/miniconda3/bin/activate ml2
```

- Docker runtime vẫn phải pin dependencies tương ứng với target Jetson; không giả định environment host `ml2` tự động tồn tại bên trong container.
- `DUY_DEPLOY/` đã có pipeline ONNX/TensorRT, benchmark, evaluation và Docker image Jetson.
- TensorRT engine phụ thuộc device, JetPack, TensorRT version và GPU. Export phải chạy ngoài project này trên target tương thích; không build engine trong web image hoặc request.
- Hiện checkout không có standalone `.engine` đã được track đầy đủ; đây là prerequisite artifact preparation, không phải lý do để dùng checkpoint `.pt` trong API MVP.
- `DUY_DEPLOY/manifests/augmentation_trt_12_manifest.json` chỉ là logical job list, thiếu engine path/hash/runtime identity. Cần tạo deployment registry riêng.
- OACP parity gần đây: HF source `duyle2408/oacp-r2-parity-rerun-seed43-44`; seed 43/44; split seed 42.
- NegativeCanvas R2/R4 có nhiều artifact family; phải chọn canonical source trước khi đăng ký model, không trộn duplicate artifact.
- Protocol inference mặc định: batch `1`, `conf=0.001`, NMS IoU `0.5`, `max_det=300`. TinyPerson dùng input `640`; LEVIR-Ship dùng `512`; TinyPerson merged/corner evaluation không phải behavior của single-image inference.
- Không đưa accuracy offline vào UI như thể đó là per-request runtime metric. UI chỉ hiển thị accuracy/provenance khi registry có artifact đã audit.

## 3. Kiến trúc MVP

### Backend

- FastAPI + Uvicorn.
- Một process, một worker, batch 1; model load một lần lúc startup.
- Static model registry (`models.json`) với artifact path read-only.
- `ModelCache` giữ loaded `YOLO(engine)` instances theo key `(model_id, precision)` trong process memory; không load lại engine mỗi request.
- MVP preload và warmup cả FP16/FP32 artifacts đã đăng ký lúc startup. Request chỉ lookup cache và predict.
- Nếu GPU memory không đủ để preload cả hai precision, readiness fail rõ ràng; không tự động unload/reload hoặc fallback. Chỉ chuyển sang lazy-load sau khi có số đo memory thực tế.
- `InferenceService` giữ bounded semaphore để tránh nhiều request tranh chấp GPU.
- Adapter TensorRT duy nhất trong MVP: Ultralytics `YOLO(<validated>.engine)` và `model.predict(...)`.

### Frontend

- Vanilla HTML/CSS/JS static files phục vụ trực tiếp từ FastAPI.
- Không thêm React/Next/Vite ở MVP; giảm build chain và giữ deployment một container.
- Một màn hình:
  - model selector + precision selector (`fp16` / `fp32`) + input size;
  - upload/dropzone và input preview;
  - output preview có boxes, class, confidence;
  - KPI cards;
  - request details/provenance;
  - bounded recent-request table.

### Container

- Base image kế thừa hoặc tương thích với JetPack 6 / L4T runtime hiện có.
- Image chỉ chứa app, Python dependencies, frontend và registry schema.
- `docker-compose.yml` cho local Jetson usage, `--runtime=nvidia`, GPU visibility, mounts và healthcheck.
- Không dùng Docker build hoặc web service để export/build TensorRT engine. Export là **offline preparation phase bên ngoài project này**, chạy thủ công bằng pipeline hiện có; web project chỉ nhận artifact đã export và validate.

## 4. API contract dự kiến

### `GET /healthz`

Liveness, không yêu cầu model loaded.

```json
{"status":"ok","service":"duy-deploy-web"}
```

### `GET /readyz`

Readiness chỉ `ready` khi registry hợp lệ, model load thành công và provider assertion pass.

```json
{
  "status": "ready",
  "models_loaded": 1,
  "provider": "TensorrtExecutionProvider"
}
```

### `GET /v1/models`

Trả model registry đã sanitize, gồm `id`, dataset, family, variant, seed, các precision khả dụng (`fp16`, `fp32`), input size, artifact kind, SHA-256, HF source/revision/path và validation status.

### `POST /v1/models/{model_id}/predict`

- Query parameter bắt buộc: `precision=fp16|fp32`; đây là lựa chọn artifact đã export sẵn, không phải yêu cầu API export runtime.
- Content type: `multipart/form-data`.
- Field: `file`.
- Optional query/body: `conf`, bounded bởi server policy; `nms_iou` cố định `0.5` trong MVP.
- Nếu precision được chọn chưa có engine trong registry, trả lỗi rõ ràng (`409 artifact_unavailable` hoặc readiness degraded), không tự export/fallback.
- Response JSON:

```json
{
  "request_id": "uuid",
  "model": {"id":"...","sha256":"...","backend":"tensorrt","precision":"fp16","requested_precision":"fp16"},
  "image": {"width": 0,"height": 0,"content_type":"image/jpeg","sha256":"..."},
  "detections": [
    {"class_id": 0,"class_name":"ship","confidence":0.91,"xyxy":[0,0,0,0]}
  ],
  "timing_ms": {
    "queue_wait": 0.0,
    "decode": 0.0,
    "preprocess": 0.0,
    "inference": 0.0,
    "postprocess": 0.0,
    "render_output": 0.0,
    "encode_output": 0.0,
    "total": 0.0
  },
  "throughput": {"fps_sequential": 0.0},
  "resources": {
    "cpu_percent": 0.0,
    "gpu_percent": null,
    "gpu_memory_used_bytes": null,
    "sampled_at": "...",
    "sampling_note":"host snapshot near request; not device-time attribution"
  },
  "output_image_url": "/v1/requests/uuid/output.png",
  "protocol": {"batch_size":1,"conf":0.001,"nms_iou":0.5,"max_det":300}
}
```

`fps_sequential = 1000 / total_ms` cho request đơn; không gọi đây là sustained FPS. Tất cả stage dùng monotonic wall-clock ở API boundary. Nếu Ultralytics trả `result.speed`, response ghi thêm `source: "ultralytics"`; nếu không, dùng wrapper measurement và ghi rõ `source: "service_wall_clock"`. Tổng stage chỉ được cộng khi các stage không overlap.

### `GET /v1/requests/{request_id}/output.png`

Trả annotated output image với TTL ngắn. JSON không nhúng base64 ảnh lớn; raw input không lưu mặc định.

### `GET /v1/metrics`

Runtime-only metrics: request count, error count, rolling latency p50/p95, rolling sequential FPS, last resource snapshot, loaded model/provider. Giữ tối đa 50 request records trong memory; output artifacts có TTL; không cần Prometheus ở MVP.

## 5. Metric taxonomy cho UI

| Nhóm | Metrics | Semantics |
|---|---|---|
| Request | queue wait, decode, preprocess, inference, postprocess, render output, encode output, total | monotonic wall-clock, milliseconds/request; UI hiển thị breakdown + stacked bar |
| Throughput | sequential FPS, optional rolling FPS | không claim video FPS nếu không có stream benchmark |
| Detection | count, confidence min/mean/max, class counts | từ response hiện tại |
| Host snapshot | CPU %, RAM used, GPU %, GPU memory used/free, temperature nếu đọc được | sample gần request, ghi rõ approximate |
| Model | model id, family, variant, seed, precision, input size, backend/provider, artifact SHA | provenance bắt buộc |
| Protocol | conf, NMS IoU, max_det, batch | NMS IoU cố định `0.5` |
| Reliability | readiness, provider exactness, request/error count, last error | không che lỗi fallback |
| Startup | model load, engine initialization, warmup | hiển thị riêng dưới readiness; không cộng vào request latency sau warmup |

Không hiển thị AP/mAP như runtime metric. Nếu thêm accuracy tab sau này, chỉ đọc artifact evaluation có provenance và tách native/merged protocol.

## 6. Model registry và artifact policy

Tạo `models.json` trong folder mới. Mỗi entry REQUIRED có:

- stable `id`;
- `dataset`, `model_family`, `variant`, `seed`, `image_size`;
- `artifact_kind` (`ultralytics_engine` trong MVP; `onnx_raw_head` không thuộc runtime MVP);
- absolute/container-relative artifact path;
- SHA-256;
- precision (`fp16` hoặc `fp32`);
- expected provider;
- source HF repo, revision, path;
- source checkpoint SHA-256 nếu có;
- labels/classes;
- `conf`, `nms_iou`, `max_det`;
- validation status + timestamp + target runtime identity.

### Artifact gate experiment

Trước khi viết đầy đủ UI, chạy controlled smoke riêng cho **cả FP16 và FP32** trên target Jetson, mỗi precision một Ultralytics `.engine`, một input shape, batch `1` và 10 golden images. Gate kiểm tra `YOLO(...).predict(...)`, backend identity, engine load, output shape, box-coordinate mapping và một request concurrency. Nếu một precision fail, precision đó không được đưa vào UI/registry; không fallback sang precision còn lại.

### Ranh giới với pipeline export

`DUY_DEPLOY_DOCKER_WEB` **không export model, không download checkpoint, không train, không rebuild model và không gọi pipeline export trong startup/request**.

Luồng artifact tách biệt:

```text
HF/checkpoint
  -> offline Ultralytics export pipeline bên ngoài
  -> Ultralytics `.engine` + checksum + deployment manifest
  -> target Jetson engine load/validation
  -> mount engine read-only vào DUY_DEPLOY_DOCKER_WEB
  -> Ultralytics `YOLO(engine).predict(...)`
  -> API inference
```

Web service chỉ kiểm tra artifact đã mount, checksum, input contract, runtime/provider và readiness. Nếu artifact chưa được chuẩn bị, service fail readiness với lỗi rõ ràng.

### Quyết định artifact và precision

Không mặc định re-export toàn bộ parity matrix. Quy trình cho mỗi precision sẽ là:

1. Kiểm tra target Jetson có `.engine` hợp lệ tương ứng với `fp16` hoặc `fp32` hay không.
2. Nếu có artifact hợp lệ, dùng trực tiếp artifact đó.
3. Nếu thiếu, export riêng precision đó từ cùng checkpoint parity bằng Ultralytics:
   - FP16: `half=True`;
   - FP32: `half=False`.
4. Validate engine trên target, ghi SHA/runtime fingerprint và đăng ký artifact.

`precision` là một parameter thật của API và một control thật trên UI, nhưng chỉ chọn giữa artifact đã export sẵn. API **không** export hoặc chuyển đổi precision trong lúc request.

Theo evidence hiện tại, checkout có checkpoint parity `.pt` và export scripts nhưng chưa có standalone `.engine` được track đầy đủ. Vì vậy khả năng cao Phase 0 sẽ cần export offline hai artifact FP16/FP32 từ cùng một parity checkpoint; việc này nằm ngoài web startup/request.

Model bootstrap đề xuất: OACP R2, YOLOv8, LEVIR-Ship, seed 43, input `512`, export cả FP16 và FP32. Sau khi end-to-end pass mới thêm YOLO11, seed 44, NegativeCanvas R2/R4 và dataset khác.

MVP sẽ đăng ký một model logical với hai artifact precision:

1. `oacp-r2-yolov8-levirship-s43-fp16`
2. `oacp-r2-yolov8-levirship-s43-fp32`

TinyPerson chỉ thêm khi quyết định rõ single-image input protocol; không giả vờ rằng merged corner evaluator là API behavior.

Không tải `.pt` tự động trong container serving. Nếu thiếu một precision, người vận hành chạy offline Ultralytics export từ parity checkpoint rồi mount engine vào service.

### Offline export utility

Thêm utility độc lập trong folder này:

`tools/export_ultralytics_engine.py`

Utility này:

- nhận `--checkpoint`, `--output`, `--imgsz`, `--precision fp16|fp32`;
- gọi `YOLO(checkpoint).export(format="engine", batch=1, imgsz=..., half=...)`;
- không được import bởi FastAPI;
- không được gọi từ Docker entrypoint, startup hoặc API;
- ghi SHA-256, precision, input size và source checkpoint vào deployment manifest;
- fail nếu output không tồn tại hoặc precision không đúng.

Ví dụ:

```bash
source /home/duylearch/miniconda3/bin/activate ml2
python DUY_DEPLOY_DOCKER_WEB/tools/export_ultralytics_engine.py \
  --checkpoint /path/to/parity/best.pt \
  --output /path/to/deploy/models/oacp-r2-yolov8-levirship-s43-fp16.engine \
  --imgsz 512 \
  --precision fp16
```

MVP sẽ đăng ký **một model logical** để chứng minh end-to-end, sau đó mở rộng matrix:

1. OACP R2, YOLOv8 hoặc YOLO11, LEVIR-Ship, seed 43/44 canonical.
2. NegativeCanvas R2/R4 sau khi canonical source được chọn.
3. TinyPerson chỉ khi quyết định rõ single-image input protocol; không giả vờ rằng merged corner evaluator là API behavior.

## 7. Phases thực hiện

### Phase 0 — Offline artifact preparation và gate

- Chọn canonical parity checkpoint từ HF/manifests.
- Activate environment `ml2`.
- Chạy `tools/export_ultralytics_engine.py` riêng cho FP16 và FP32 nếu target chưa có artifact hợp lệ.
- Kiểm tra local artifact, SHA-256, input shape, labels, precision.
- Load/warm-up cả hai engine bằng Ultralytics trên target Jetson.
- Preload cả hai engine vào `ModelCache`; ghi memory trước/sau preload.
- Assert backend/provider exactness.
- Viết deployment manifest có runtime fingerprint.

**Gate:** FP16 và FP32 đều chạy batch 1, output boxes đúng kích thước ảnh, không fallback, warm-up thành công, và không vượt GPU memory budget; precision nào fail thì chưa expose precision đó trên UI/API.


### Phase 1 — Backend core

- Tạo app layout tối thiểu: `app/main.py`, `app/config.py`, `app/registry.py`, `app/inference.py`, `app/telemetry.py`.
- Implement health/readiness/models/predict/metrics.
- Validate upload, bounded payload, decode errors, inference errors.
- Render output image server-side để frontend không tự suy diễn box coordinates.

**Gate:** curl smoke test upload ảnh và nhận detections/timing/output image; negative tests cho file hỏng, model id sai và provider not-ready.

### Phase 2 — Frontend

- Tạo `web/index.html`, `web/app.js`, `web/styles.css`.
- Model selector, upload, input/output preview, detection overlay, KPI/resource/provenance cards.
- Hiển thị loading/error/empty states; không hiển thị stale result khi request fail.

**Gate:** browser smoke test với ảnh thật và ảnh không hợp lệ; visual boxes khớp JSON.

### Phase 3 — Docker/runtime

- Tạo `Dockerfile`, `docker-compose.yml`, `.dockerignore`, requirements lock/minimum.
- GPU runtime, mounts, healthcheck, non-root where compatible, bounded shared memory.
- Add startup logs: registry, model SHA, provider/backend, runtime versions, engine path và engine identity.

**Gate:** clean `docker build`; target Jetson `docker compose up`; `/readyz` ready và predict thành công.

### Phase 4 — Verification and handoff

- Permanent tests cho registry validation, timing/FPS semantics, provider/fallback rejection, box serialization and error responses.
- Throwaway smoke script cho actual TensorRT inference.
- Record result in `DUY_DEPLOY_DOCKER_WEB/README.md`: exact command, target runtime, artifact SHA, NMS IoU `0.5`, observed latency/FPS, limitations.

**Gate:** reproducible one-command startup + one-command smoke test; no unsupported parity claim.

## 8. Explicit non-goals của MVP

- Video/RTSP/webcam streaming.
- Multi-user auth, persistent database, job queue, autoscaling.
- Runtime model upload/download hoặc HF token trong server.
- Hot reload model registry.
- Prometheus/Grafana/OpenTelemetry stack.
- Online AP/mAP evaluation.
- Cross-device TensorRT engine portability.
- Train/export tự động trong web request.

Các phần này chỉ thêm sau khi single-image path có số đo thực tế và cần thiết.

## 9. Risks và cách xử lý

- **Engine không portable:** build/validate trên target Jetson; registry ghi runtime identity.
- **Silent provider fallback:** provider exact assertion + readiness fail-closed.
- **Pre/postprocess mismatch:** reuse existing deployment decoder/Ultralytics adapter, compare one image against saved reference output.
- **Engine không tương thích target:** kiểm tra engine identity/runtime fingerprint trước readiness; không rebuild hoặc fallback trong web service.
- **Resource metrics không chính xác theo request:** label là sampled host snapshot; không dùng để tính inference GPU time.
- **Artifact provenance mơ hồ:** không register model thiếu SHA/source/protocol.
- **TinyPerson protocol lẫn native/merged:** tách UI label và docs; MVP single image không claim merged benchmark.

## 10. Definition of Done

- `DUY_DEPLOY_DOCKER_WEB/` có app, web, registry, Docker config và README vận hành.
- Docker image chạy được trên target Jetson GPU với mounted TensorRT artifact.
- `/healthz`, `/readyz`, `/v1/models`, `/v1/models/{id}/predict`, `/v1/metrics` hoạt động.
- UI hiển thị input/output, detections, latency breakdown, sequential FPS, CPU/GPU/memory snapshot, model/provider/provenance.
- TensorRT provider được xác minh exact; không fallback.
- Output boxes và JSON cùng một coordinate contract.
- Smoke test thực tế pass; permanent tests pass; mọi skipped check và runtime limitation được ghi rõ.
- Artifact và report đều ghi `nms_iou: 0.5`.
