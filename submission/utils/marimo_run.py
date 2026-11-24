#!/usr/bin/env python3
"""Compatibility CLI for executing code in a live Marimo kernel.

The reusable implementation lives in ``utils.marimo_client``.  Keep this file
as a thin command-line adapter so older session snippets continue to work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.marimo_client import (
    DEFAULT_CONFIG_PATH,
    MarimoClient,
    MarimoClientError,
    load_config,
    save_config,
)


def run_code_on_server(code: str, *, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    client = MarimoClient(load_config(config_path))
    for event, payload in client.execute_events(code):
        if event == "stdout":
            sys.stdout.write(str(payload.get("data", "")))
            sys.stdout.flush()
        elif event == "stderr":
            sys.stderr.write(str(payload.get("data", "")))
            sys.stderr.flush()
        elif event == "done":
            if payload.get("success") is False:
                error = payload.get("error", {})
                message = error.get("msg", "Unknown remote execution error") if isinstance(error, dict) else error
                raise MarimoClientError(f"Remote execution failed: {message}")
            output = payload.get("output", {})
            value = output.get("data", "") if isinstance(output, dict) else ""
            if value:
                print(f"\nResult: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Marimo remote helper utility")
    parser.add_argument("--set-config", nargs=2, metavar=("URL", "TOKEN"), help="Save ignored local Marimo config")
    parser.add_argument("-c", "--code", help="Run Python code on the remote kernel")
    parser.add_argument("-f", "--file", type=Path, help="Run a local Python file on the remote kernel")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Override local config path")
    args = parser.parse_args(argv)

    try:
        if args.set_config:
            save_config(args.set_config[0], args.set_config[1], args.config)
            print(f"Saved configuration to {args.config}")
            return 0
        if args.code is not None:
            run_code_on_server(args.code, config_path=args.config)
            return 0
        if args.file is not None:
            run_code_on_server(args.file.read_text(), config_path=args.config)
            return 0
        parser.print_help()
        return 0
    except (OSError, MarimoClientError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
