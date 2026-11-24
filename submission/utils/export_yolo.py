#!/usr/bin/env python3
"""Export a YOLO checkpoint to ONNX and/or TensorRT.

The checkpoint can be local, downloaded from a Hugging Face dataset repository,
or discovered from a Markdown report containing Hugging Face links.  Secrets are
read only from ``HF_TOKEN`` or the Hugging Face CLI cache.  Never put a token in
this file or on the command line.

Examples
--------
# Pick the newest report and its first checkpoint repository:
conda run -n ml2 python utils/export_yolo.py --latest-report --format onnx engine

# Use an explicit repository and checkpoint path:
HF_TOKEN=... conda run -n ml2 python utils/export_yolo.py \
  --repo-id duyle2408/varroa-yolo-baselines-mosaic-two-seed-runs \
  --hf-path runs/varroa/yolov8/seed_42/weights/best.pt \
  --format onnx engine --batch 1 8 --dynamic
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HF_RE = re.compile(
    r"https?://huggingface\.co/(?P<kind>datasets|models)/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?P<tail>/[^)\s>`]*)?",
    re.IGNORECASE,
)
CHECKPOINT_NAMES = ("weights/best.pt", "best.pt", "weights/last.pt", "last.pt")


@dataclass(frozen=True)
class HuggingFaceRef:
    repo_id: str
    repo_type: str
    path: str | None
    source: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Local .pt checkpoint.")
    source.add_argument("--repo-id", help="Hugging Face repo id, for example duyle2408/varroa-yolo-baselines.")
    source.add_argument("--report", type=Path, help="Markdown report to scan for Hugging Face links.")
    source.add_argument("--latest-report", action="store_true", help="Scan the newest Markdown report under docs/ and approach+results/.")
    parser.add_argument("--repo-pattern", help="When scanning a report, keep only repo URLs containing this text.")
    parser.add_argument("--repo-index", type=int, default=0, help="Repo link index after filtering a report, zero-based.")
    parser.add_argument("--hf-path", help="Exact file path inside the Hugging Face repo. Defaults to the best checkpoint found.")
    parser.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("exports"))
    parser.add_argument("--format", nargs="+", choices=("onnx", "engine"), default=("onnx", "engine"), help="Formats: onnx and/or engine (TensorRT).")
    parser.add_argument("--batch", nargs="+", type=int, default=(1,), help="One or more batch sizes. Each size is exported separately.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=(640,), help="Image size: one value or height width.")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0, cpu, or 0,1.")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic input shapes, including dynamic batch dimensions.")
    parser.add_argument("--half", action="store_true", help="Export FP16 where supported. Requires a CUDA-capable device for TensorRT.")
    parser.add_argument("--int8", action="store_true", help="Export INT8 where supported. Requires --data for TensorRT calibration.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), help="Explicit precision. Overrides the legacy --half/--int8 flags.")
    parser.add_argument("--data", type=Path, help="Dataset YAML used for INT8 calibration.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of calibration data used for INT8 export.")
    parser.add_argument("--workspace", type=float, help="TensorRT workspace size in GiB, passed to Ultralytics when supported.")
    parser.add_argument("--opset", type=int, help="ONNX opset override.")
    parser.add_argument("--overwrite", action="store_true", help="Allow an existing output directory to be reused.")
    return parser.parse_args()


def report_candidates() -> list[Path]:
    roots = [ROOT / "docs", ROOT / "approach+results"]
    return sorted(
        (path for root in roots if root.exists() for path in root.rglob("*.md")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def scan_report(path: Path, pattern: str | None) -> list[HuggingFaceRef]:
    text = path.read_text(encoding="utf-8", errors="replace")
    refs: list[HuggingFaceRef] = []
    seen: set[tuple[str, str, str | None]] = set()
    for match in HF_RE.finditer(text):
        repo_id = match.group("repo")
        if pattern and pattern.lower() not in repo_id.lower():
            continue
        tail = unquote(match.group("tail") or "")
        path_in_repo: str | None = None
        blob_match = re.search(r"/(?:blob|resolve)/[^/]+/(.+?)(?:[?#)\s`]|$)", tail)
        if blob_match:
            path_in_repo = blob_match.group(1).rstrip("/)")
        kind = match.group("kind").lower()
        repo_type = "dataset" if kind == "datasets" else "model"
        key = (repo_id, repo_type, path_in_repo)
        if key not in seen:
            seen.add(key)
            refs.append(HuggingFaceRef(repo_id, repo_type, path_in_repo, path))
    return refs


def resolve_ref(args: argparse.Namespace) -> HuggingFaceRef | None:
    if args.repo_id:
        return HuggingFaceRef(args.repo_id, args.repo_type, args.hf_path, None)
    report = args.report
    if args.latest_report:
        reports = report_candidates()
        report = next((candidate for candidate in reports if scan_report(candidate, args.repo_pattern)), None)
        if report is None:
            raise FileNotFoundError("No Markdown report with a Hugging Face link was found under docs/ or approach+results/.")
    if report:
        report = report.resolve()
        refs = scan_report(report, args.repo_pattern)
        if not refs:
            raise ValueError(f"No Hugging Face links found in {report}")
        if args.repo_index < 0 or args.repo_index >= len(refs):
            raise IndexError(f"--repo-index {args.repo_index} is outside 0..{len(refs) - 1}")
        ref = refs[args.repo_index]
        requested_path = args.hf_path or (ref.path if ref.path and ref.path.lower().endswith(".pt") else None)
        return HuggingFaceRef(ref.repo_id, ref.repo_type, requested_path, ref.source)
    return None


def choose_remote_path(files: Iterable[str], requested: str | None) -> str:
    files = sorted(set(files))
    if requested:
        if requested not in files:
            raise FileNotFoundError(f"Checkpoint {requested!r} is not present in the Hugging Face repository.")
        return requested
    candidates = [path for path in files if path.lower().endswith(".pt")]
    if not candidates:
        raise FileNotFoundError("No .pt checkpoint was found in the Hugging Face repository.")
    def rank(path: str) -> tuple[int, int, str]:
        normalized = path.lower()
        preference = next((index for index, name in enumerate(CHECKPOINT_NAMES) if normalized.endswith(name)), len(CHECKPOINT_NAMES))
        return preference, len(path), path
    return min(candidates, key=rank)


def download_checkpoint(ref: HuggingFaceRef, output_dir: Path) -> Path:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before downloading from Hugging Face.") from exc
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = [info.rfilename for info in api.list_repo_tree(ref.repo_id, repo_type=ref.repo_type, recursive=True) if getattr(info, "rfilename", "").lower().endswith(".pt")]
    checkpoint_path = choose_remote_path(files, ref.path)
    local = Path(hf_hub_download(repo_id=ref.repo_id, repo_type=ref.repo_type, filename=checkpoint_path, token=token, local_dir=str(output_dir / "hf_cache")))
    print(f"Resolved Hugging Face checkpoint: {ref.repo_id}/{checkpoint_path}")
    return local


def load_model(checkpoint: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is unavailable. Run this command in the project ML environment, e.g. conda run -n ml2 python ...") from exc
    return YOLO(str(checkpoint))


def quantize_onnx_int8(source: Path, target: Path) -> None:
    try:
        import onnx
        if not hasattr(onnx.TensorProto, "INT4"):
            onnx.TensorProto.INT4 = 22
        if not hasattr(onnx.TensorProto, "UINT4"):
            onnx.TensorProto.UINT4 = 21
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime and onnx to quantize ONNX INT8 outputs.") from exc
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )


def export_one(model, fmt: str, batch: int, args: argparse.Namespace, output_dir: Path) -> Path | str:
    output_dir.mkdir(parents=True, exist_ok=True)
    precision = args.precision
    if precision is None:
        precision = "int8" if args.int8 else "fp16" if args.half else "fp32"
    onnx_int8 = fmt == "onnx" and precision == "int8"
    export_precision = "fp32" if onnx_int8 else precision
    kwargs: dict[str, object] = {
        "format": fmt,
        "imgsz": args.imgsz[0] if len(args.imgsz) == 1 else tuple(args.imgsz),
        "batch": batch,
        "device": args.device,
        "dynamic": args.dynamic,
        "half": export_precision == "fp16",
        "int8": export_precision == "int8",
        "project": str(output_dir),
        "name": f"{fmt}_batch{batch}",
        "exist_ok": args.overwrite,
    }
    if args.data is not None:
        kwargs["data"] = str(args.data)
    if precision == "int8":
        kwargs["fraction"] = args.fraction
    if args.opset is not None:
        kwargs["opset"] = args.opset
    if args.workspace is not None:
        kwargs["workspace"] = args.workspace
    if fmt == "engine" and args.device == "cpu":
        raise ValueError("TensorRT export requires a CUDA device; do not use --device cpu.")
    print(f"Exporting format={fmt} precision={precision} batch={batch} dynamic={args.dynamic} device={args.device}")
    exported = Path(model.export(**kwargs))
    if not exported.is_file():
        raise FileNotFoundError(f"Ultralytics reported a missing export: {exported}")
    target = output_dir / f"{exported.stem}_{precision}_batch{batch}{exported.suffix}"
    if onnx_int8:
        quantize_onnx_int8(exported, target)
    elif exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
    return target


def main() -> None:
    args = parse_args()
    if any(batch < 1 for batch in args.batch):
        raise ValueError("All --batch values must be positive integers.")
    if len(args.imgsz) not in (1, 2) or any(size < 1 for size in args.imgsz):
        raise ValueError("--imgsz accepts one value or height width, both positive.")
    if args.precision and (args.half or args.int8):
        raise ValueError("Use either --precision or the legacy --half/--int8 flags, not both.")
    precision = args.precision or ("int8" if args.int8 else "fp16" if args.half else "fp32")
    if precision == "int8":
        if "engine" in args.format and args.data is None:
            raise ValueError("INT8 TensorRT export requires --data DATASET.yaml for calibration.")
        if args.data is not None:
            args.data = args.data.resolve()
            if not args.data.is_file():
                raise FileNotFoundError(args.data)
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and at most 1.")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref = resolve_ref(args)
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    elif ref:
        checkpoint = download_checkpoint(ref, args.output_dir)
    else:
        raise ValueError("Select a source with --checkpoint, --repo-id, --report, or --latest-report.")
    model = load_model(checkpoint)
    results: list[str] = []
    for batch in args.batch:
        for fmt in args.format:
            results.append(str(export_one(model, fmt, batch, args, args.output_dir)))
    print("Export complete:")
    for result in results:
        print(f"  {result}")


if __name__ == "__main__":
    main()
