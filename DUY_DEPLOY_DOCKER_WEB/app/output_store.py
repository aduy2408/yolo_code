from __future__ import annotations

import time
import uuid
from pathlib import Path


class OutputStore:
    def __init__(self, root: Path, ttl_seconds: int):
        self.root = root
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, request_id: str, data: bytes) -> Path:
        self.cleanup()
        path = self.root / f"{request_id}.png"
        path.write_bytes(data)
        return path

    def get(self, request_id: str) -> Path | None:
        try:
            canonical_id = str(uuid.UUID(request_id))
        except ValueError:
            return None
        path = (self.root / f"{canonical_id}.png").resolve()
        if path.parent != self.root.resolve():
            return None
        return path if path.is_file() else None

    def cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for path in self.root.glob("*.png"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
