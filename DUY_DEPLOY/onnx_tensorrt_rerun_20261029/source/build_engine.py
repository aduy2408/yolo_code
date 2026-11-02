from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort

import evaluate


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--cache", type=Path, required=True)
args = parser.parse_args()

_, providers = evaluate.provider_config("tensorrt", args.cache)
print(f"available={ort.get_available_providers()}", flush=True)
print(f"requested={providers}", flush=True)
session = ort.InferenceSession(str(args.model), providers=providers)
session.disable_fallback()
print(f"session={session.get_providers()}", flush=True)
print("ENGINE_BUILD_OK", flush=True)
