from __future__ import annotations

import argparse
import json
from pathlib import Path


def files(root: Path, suffix: str) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in sorted(root.rglob(f"*{suffix}"))
        if path.is_file()
    }


def canonical(relative: str) -> str:
    return relative.rsplit("/", 1)[0] + "/best_precision_batch1.onnx"

def main() -> None:
    parser = argparse.ArgumentParser(description="Check paired FP32/FP16 ONNX and TensorRT artifacts.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--engines-root", type=Path)
    args = parser.parse_args()

    onnx = {precision: files(args.root / "onnx" / precision, ".onnx") for precision in ("fp32", "fp16")}
    if any(len(items) != args.expected_count for items in onnx.values()):
        raise SystemExit(f"ONNX_COUNT_MISMATCH {json.dumps({key: len(value) for key, value in onnx.items()}, sort_keys=True)}")
    if {canonical(relative) for relative in onnx["fp32"]} != {canonical(relative) for relative in onnx["fp16"]}:
        raise SystemExit("ONNX_IDENTITY_MISMATCH")
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = {
            f"{precision}/{job['job_id']}/best_{precision}_batch1.onnx"
            for job in manifest["jobs"]
            for precision in ("fp32", "fp16")
        }
        actual = {f"{precision}/{relative}" for precision, items in onnx.items() for relative in items}
        if actual != expected:
            raise SystemExit("ONNX_MANIFEST_IDENTITY_MISMATCH")
    if any(size <= 0 for items in onnx.values() for size in items.values()):
        raise SystemExit("EMPTY_ONNX_ARTIFACT")

    if args.engines_root is not None:
        engine_counts = {}
        for precision in ("fp32", "fp16"):
            paths = [path for path in (args.engines_root / precision).rglob("*") if path.is_file()]
            engine_counts[precision] = len(paths)
            if not paths:
                raise SystemExit(f"EMPTY_ENGINE_ROOT {precision}")
        if args.manifest is not None:
            job_ids = [str(job["job_id"]) for job in json.loads(args.manifest.read_text(encoding="utf-8"))["jobs"]]
            for precision in ("fp32", "fp16"):
                missing = [job_id for job_id in job_ids if not any((args.engines_root / precision / job_id).rglob("*"))]
                if missing:
                    raise SystemExit(f"ENGINE_IDENTITY_MISMATCH {precision} missing={missing}")
        print(json.dumps({"onnx": {key: len(value) for key, value in onnx.items()}, "engine_files": engine_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
