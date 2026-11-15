from __future__ import annotations

import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - optional on minimal images
    psutil = None


@dataclass
class RequestRecord:
    request_id: str
    timestamp: str
    model_id: str
    precision: str
    detection_count: int
    total_ms: float
    ok: bool
    error_code: str | None = None


class Telemetry:
    def __init__(self, max_history: int):
        self.history: deque[RequestRecord] = deque(maxlen=max_history)
        self.request_count = 0
        self.error_count = 0
        self.latencies: deque[float] = deque(maxlen=200)
        self.last_resources: dict[str, Any] = {}
        self.startup: dict[str, Any] = {}

    def resources(self) -> dict[str, Any]:
        sampled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(interval=None) if psutil else None,
            "ram_used_bytes": psutil.virtual_memory().used if psutil else None,
            "ram_total_bytes": psutil.virtual_memory().total if psutil else None,
            "gpu_percent": None,
            "gpu_memory_used_bytes": None,
            "gpu_memory_total_bytes": None,
            "temperature_c": None,
            "sampled_at": sampled_at,
            "sampling_note": "host snapshot near request; not device-time attribution",
        }
        if shutil.which("nvidia-smi"):
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                gpu, used, total, temp = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
                result.update({
                    "gpu_percent": float(gpu),
                    "gpu_memory_used_bytes": int(float(used) * 1024 * 1024),
                    "gpu_memory_total_bytes": int(float(total) * 1024 * 1024),
                    "temperature_c": float(temp),
                })
            except (OSError, subprocess.SubprocessError, IndexError, ValueError):
                pass
        self.last_resources = result
        return result

    def record(self, record: RequestRecord) -> None:
        self.request_count += 1
        if not record.ok:
            self.error_count += 1
        self.latencies.append(record.total_ms)
        self.history.append(record)

    def metrics(self) -> dict[str, Any]:
        values = sorted(self.latencies)
        p50 = percentile(values, 0.50)
        p95 = percentile(values, 0.95)
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "latency_ms": {"p50": p50, "p95": p95},
            "sequential_fps": (1000.0 / (sum(values) / len(values))) if values else None,
            "last_resources": self.last_resources,
            "startup": self.startup,
            "history": [record.__dict__ for record in reversed(self.history)],
        }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, int((len(values) - 1) * quantile)))
    return round(values[index], 4)
