#!/usr/bin/env python3
"""Reusable Marimo HTTP client for local-to-kernel execution.

Keep connection/configuration logic here.  Experiment-specific orchestration
belongs in ``utils.marimo_ops`` and runners should not reimplement this client.
Credentials are read from the environment first and are never included in
normal output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import requests

DEFAULT_CONFIG_PATH = Path(__file__).with_name("marimo_config.json")
DEFAULT_TIMEOUT_SECONDS = 30


class MarimoClientError(RuntimeError):
    """A connection, API, or remote execution failure."""


@dataclass(frozen=True)
class MarimoConfig:
    url: str
    token: str

    def validate(self) -> "MarimoConfig":
        if not self.url.strip():
            raise MarimoClientError("Marimo server URL is not configured")
        if not self.token.strip():
            raise MarimoClientError("Marimo access token is not configured")
        return MarimoConfig(self.url.rstrip("/"), self.token)


def load_config(path: Path = DEFAULT_CONFIG_PATH, environ: Mapping[str, str] | None = None) -> MarimoConfig:
    """Load environment credentials first, then the ignored local config."""
    env = os.environ if environ is None else environ
    url = env.get("MARIMO_URL", "").strip()
    token = (env.get("MARIMO_TOKEN") or env.get("MARIMO_ACCESS_TOKEN") or "").strip()
    if url and token:
        return MarimoConfig(url, token).validate()

    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return MarimoConfig("", "")
    except (OSError, json.JSONDecodeError) as exc:
        raise MarimoClientError(f"Invalid Marimo config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MarimoClientError(f"Marimo config must be a JSON object: {path}")
    return MarimoConfig(str(payload.get("url", "")), str(payload.get("token", ""))).validate()


def save_config(url: str, token: str, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Save a local config with owner-only permissions.

    Prefer environment variables for shared machines.  This function exists for
    the legacy CLI workflow and writes only to the ignored config path.
    """
    config = MarimoConfig(url, token).validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": config.url, "token": config.token}, indent=2) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


class MarimoClient:
    """Small authenticated client for the Marimo session/kernel API."""

    def __init__(self, config: MarimoConfig, *, timeout: float = DEFAULT_TIMEOUT_SECONDS, session: requests.Session | None = None):
        self.config = config.validate()
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.token}", "Content-Type": "application/json"}

    def session_ids(self) -> list[str]:
        try:
            response = self.session.get(
                f"{self.config.url}/api/sessions", headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise MarimoClientError(f"Unable to list Marimo sessions: {exc}") from exc
        if not isinstance(payload, dict) or not payload:
            raise MarimoClientError("No active Marimo sessions found")
        return [str(session_id) for session_id in payload]

    def execute_events(self, code: str) -> Iterator[tuple[str, dict]]:
        """Yield ``(event, payload)`` pairs from remote kernel execution."""
        if not code.strip():
            raise MarimoClientError("Cannot execute empty code")
        last_error: str | None = None
        for session_id in self.session_ids():
            headers = {**self.headers, "Marimo-Session-Id": session_id}
            try:
                response = self.session.post(
                    f"{self.config.url}/api/kernel/execute",
                    headers=headers,
                    json={"code": code},
                    stream=True,
                    timeout=self.timeout,
                )
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code} for session {session_id}"
                    response.close()
                    continue
                event_name = ""
                try:
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:"):
                            try:
                                payload = json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                continue
                            if isinstance(payload, dict):
                                yield event_name, payload
                finally:
                    response.close()
                return
            except requests.RequestException as exc:
                last_error = str(exc)
        raise MarimoClientError(last_error or "Failed to execute code on any active Marimo session")

    def execute(self, code: str) -> list[tuple[str, dict]]:
        return list(self.execute_events(code))
