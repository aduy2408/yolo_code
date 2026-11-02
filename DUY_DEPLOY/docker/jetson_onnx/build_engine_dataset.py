from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort

import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--dataset", choices=("levirship", "tinyperson"), required=True)
args = parser.parse_args()

provider_name, providers = evaluate.provider_config("tensorrt", args.cache)
session = ort.InferenceSession(str(args.model), providers=providers)
session.disable_fallback()
if session.get_providers()[0] != provider_name:
    raise RuntimeError(f"TensorRT was not first: {session.get_providers()}")
if args.dataset == "levirship":
    images, _ = evaluate.load_coco(args.data_root, "levirship", "val")
    probe = images[:1]
else:
    data = args.data_root / "tinyperson"
    rows = json.loads((data / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json").read_text())["images"]
    row = rows[0]
    rid = int(row["id"])
    probe = [{"id": rid, "width": int(row["width"]), "height": int(row["height"]), "path": data / "images/all" / (f"test_{rid:06d}.jpg")}]
evaluate.predictions(session, probe, 512 if args.dataset == "levirship" else 640, 1, 0.001, 0.5, 300)
print(f"ENGINE_BUILD_OK provider={session.get_providers()} cache={args.cache}", flush=True)
