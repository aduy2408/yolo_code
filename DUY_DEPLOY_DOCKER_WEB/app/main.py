from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .inference import InferenceService
from .output_store import OutputStore
from .registry import ModelRegistry
from .telemetry import Telemetry

settings = Settings()
registry = ModelRegistry(settings.registry_path, settings.model_root)
telemetry = Telemetry(settings.max_history)
outputs = OutputStore(settings.output_root, settings.output_ttl_seconds)
service = InferenceService(settings, registry, telemetry, outputs)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_output_root()
    await asyncio.to_thread(service.startup)
    yield


app = FastAPI(title="DUY Deploy Web", version="0.1.0", lifespan=lifespan)
@app.middleware("http")
async def reject_oversized_requests(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > settings.max_upload_bytes + 1024 * 1024:
                return JSONResponse({"detail": {"error": "request_too_large", "max_bytes": settings.max_upload_bytes}}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": {"error": "invalid_content_length"}}, status_code=400)
    return await call_next(request)


def safe_error(value: str) -> str:
    return value.split(":", 1)[0]




@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}


@app.get("/readyz")
def readyz() -> JSONResponse:
    ready = service.ready()
    providers = sorted({loaded.backend_identity for loaded in service.cache.loaded.values()})
    body = {
        "status": "ready" if ready else "not_ready",
        "models_loaded": len(service.cache.loaded),
        "provider": providers,
        "errors": [safe_error(error) for error in registry.errors] + [safe_error(f"{key[0]}:{key[1]}:{value}") for key, value in service.cache.errors.items()],
        "startup": service.cache.startup,
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.get("/v1/models")
def models() -> dict:
    return {"models": registry.public_models(service.cache.keys()), "registry_errors": [safe_error(error) for error in registry.errors]}


@app.get("/v1/metrics")
def metrics() -> dict:
    return telemetry.metrics()


@app.post("/v1/models/{model_id}/predict")
async def predict(
    model_id: str,
    file: UploadFile = File(...),
    precision: Literal["fp16", "fp32"] = Query(...),
) -> dict:
    artifact = registry.get(model_id, precision)
    if artifact is None:
        raise HTTPException(status_code=409, detail={"error": "artifact_unavailable", "model_id": model_id, "precision": precision})
    if not service.ready():
        raise HTTPException(status_code=503, detail={"error": "service_not_ready", "errors": registry.errors})
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail={"error": "unsupported_image_type"})
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={"error": "image_too_large", "max_bytes": settings.max_upload_bytes})
    if not data:
        raise HTTPException(status_code=400, detail={"error": "empty_image"})
    request_id = str(uuid.uuid4())
    try:
        return await asyncio.to_thread(service.predict, request_id, artifact, data, file.content_type)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail={"error": "inference_queue_timeout"}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"error": "model_unavailable", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail={"error": "inference_failed", "message": str(exc)}) from exc


@app.get("/v1/requests/{request_id}/output.png")
def output(request_id: str) -> FileResponse:
    path = outputs.get(request_id)
    if path is None:
        raise HTTPException(status_code=404, detail={"error": "output_expired_or_missing"})
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent.parent / "web" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "web"), name="static")
