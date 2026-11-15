from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    registry_path: Path = Path(os.getenv("DUY_WEB_REGISTRY", "/config/models.json"))
    model_root: Path = Path(os.getenv("DUY_WEB_MODEL_ROOT", "/models"))
    output_root: Path = Path(os.getenv("DUY_WEB_OUTPUT_ROOT", "/tmp/duy_deploy_outputs"))
    max_upload_bytes: int = int(os.getenv("DUY_WEB_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    output_ttl_seconds: int = int(os.getenv("DUY_WEB_OUTPUT_TTL_SECONDS", "900"))
    max_image_pixels: int = int(os.getenv("DUY_WEB_MAX_IMAGE_PIXELS", str(40_000_000)))
    max_image_dimension: int = int(os.getenv("DUY_WEB_MAX_IMAGE_DIMENSION", "10000"))
    max_history: int = int(os.getenv("DUY_WEB_MAX_HISTORY", "50"))
    preload_models: bool = os.getenv("DUY_WEB_PRELOAD_MODELS", "1").lower() not in {"0", "false", "no"}
    device: str = os.getenv("DUY_WEB_DEVICE", "0")
    conf: float = float(os.getenv("DUY_WEB_CONF", "0.001"))
    display_conf: float = float(os.getenv("DUY_WEB_DISPLAY_CONF", "0.3"))
    nms_iou: float = float(os.getenv("DUY_WEB_NMS_IOU", "0.5"))
    max_det: int = int(os.getenv("DUY_WEB_MAX_DET", "300"))
    service_name: str = os.getenv("DUY_WEB_SERVICE_NAME", "duy-deploy-web")

    @property
    def model_root_resolved(self) -> Path:
        return self.model_root.resolve()

    def ensure_output_root(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
