#!/usr/bin/env python3
"""Train raw-cue architecture experiments #3, #4, #6a, #6b on LEVIR-Ship seed 42.

#3 DedicatedCueSlots        — direct injection, no learned mixing
#4 DetachedResidualFusion   — stop-grad residual + L1 recon aux supervision
#6a SplitChannelDetect       — channel split control (no aux supervision)
#6b GTChannelSpecialization  — channel split + D8 GT cue aux supervision

Upload protocol (mandatory per marimo-train workflow):
  - Fail fast before training if HF_TOKEN or hf_repo_id missing.
  - Upload + verify + write upload_complete.json after EACH variant/seed run.
  - Retry network errors at least 3 times.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misc.prepare_levir_ship import prepare


def local_ultralytics() -> None:
    local = ROOT / "models_related/ultralytics"
    if (local / "ultralytics/__init__.py").is_file() and str(local) not in sys.path:
        sys.path.insert(0, str(local))


CFG_DIR = ROOT / "models_related/models_config/yolov8/levir"

VARIANTS = {
    "yolov8n_p2_dedicated_cue_slots":  CFG_DIR / "yolov8n_p2_dedicated_cue_slots.yaml",    # #3
    "yolov8n_p2_detached_residual":    CFG_DIR / "yolov8n_p2_detached_residual.yaml",      # #4
    "yolov8n_p2_split_channel":        CFG_DIR / "yolov8n_p2_split_channel.yaml",          # #6a
    "yolov8n_p2_gt_channel_spec":      CFG_DIR / "yolov8n_p2_gt_channel_spec.yaml",        # #6b
}

SEEDS = [42]

# ── HF upload helpers ────────────────────────────────────────────────────────

def _hf_api(token: str):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def upload_run_to_hf(
    run_dir: Path,
    yaml_cfg: Path,
    hf_repo_id: str,
    hf_remote_prefix: str,
    hf_token: str,
    retries: int = 3,
) -> None:
    """Upload all mandatory artifacts for one run, verify, write upload_complete.json."""
    api = _hf_api(hf_token)

    required_locals = [
        run_dir / "weights/best.pt",
        run_dir / "weights/last.pt",
        run_dir / "results.csv",
        run_dir / "evaluation_metrics.json",
        run_dir / "args.yaml",
    ]
    extras = [yaml_cfg, Path(__file__)]

    to_upload = [(p, f"{hf_remote_prefix}/{p.name}") for p in required_locals if p.exists()]
    to_upload += [(p, f"{hf_remote_prefix}/{p.name}") for p in extras if p.exists()]

    for local_path, remote_path in to_upload:
        for attempt in range(1, retries + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=hf_repo_id,
                    repo_type="model",
                )
                print(f"  [HF] Uploaded: {remote_path}")
                break
            except Exception as e:
                if attempt == retries:
                    raise RuntimeError(f"Upload failed after {retries} retries: {local_path}") from e
                print(f"  [HF] Retry {attempt}/{retries}: {e}")
                time.sleep(5 * attempt)

    # Verify mandatory remote paths
    try:
        remote_files = set(api.list_repo_files(repo_id=hf_repo_id, repo_type="model"))
    except Exception as e:
        raise RuntimeError(f"Failed to list remote files for verification: {e}") from e

    mandatory = [
        f"{hf_remote_prefix}/best.pt",
        f"{hf_remote_prefix}/last.pt",
        f"{hf_remote_prefix}/results.csv",
        f"{hf_remote_prefix}/evaluation_metrics.json",
    ]
    missing = [p for p in mandatory if p not in remote_files]
    if missing:
        raise RuntimeError(f"HF verification failed — missing: {missing}")

    upload_complete = {
        "hf_repo_id": hf_repo_id,
        "hf_remote_prefix": hf_remote_prefix,
        "uploaded_files": [r for _, r in to_upload],
        "verified_paths": mandatory,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "upload_complete.json").write_text(json.dumps(upload_complete, indent=2))
    print(f"  [HF] Verification OK — upload_complete.json written.")


# ── Training helpers ─────────────────────────────────────────────────────────

def completed(run_dir: Path) -> bool:
    return all(
        (run_dir / f).is_file()
        for f in ("weights/best.pt", "weights/last.pt", "results.csv",
                  "evaluation_metrics.json", "upload_complete.json")
    )


def evaluate_run(model, data_yaml: str, run_dir: Path) -> dict:
    val_res = model.val(data=data_yaml, split="val", iou=0.5, save=False, verbose=False)
    test_res = model.val(data=data_yaml, split="test", iou=0.5, save=False, verbose=False)
    metrics = {
        "nms_iou": 0.5,
        "val": {
            "mp": float(val_res.results_dict.get("metrics/precision(B)", 0.0)),
            "mr": float(val_res.results_dict.get("metrics/recall(B)", 0.0)),
            "map50": float(val_res.results_dict.get("metrics/mAP50(B)", 0.0)),
            "map75": float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        },
        "test": {
            "mp": float(test_res.results_dict.get("metrics/precision(B)", 0.0)),
            "mr": float(test_res.results_dict.get("metrics/recall(B)", 0.0)),
            "map50": float(test_res.results_dict.get("metrics/mAP50(B)", 0.0)),
            "map75": float(test_res.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        },
    }
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def generate_summary(project_dir: Path, hf_repo_id: str, hf_token: str) -> dict:
    summary_data = {}
    md_lines = [
        "# Raw-Cue Architecture Experiments #3/#4/#6a/#6b Runs Summary",
        "",
        "| Variant | NMS IoU | VAL mAP50 | VAL mAP50-95 | TEST mAP50 | TEST mAP50-95 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for d in sorted(project_dir.iterdir()):
        if d.is_dir():
            mf = d / "evaluation_metrics.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                summary_data[d.name] = m
                val, test = m.get("val", {}), m.get("test", {})
                md_lines.append(
                    f"| {d.name} | 0.5 | {val.get('map50', 0):.4f} | {val.get('map75', 0):.4f}"
                    f" | **{test.get('map50', 0):.4f}** | {test.get('map75', 0):.4f} |"
                )
    (project_dir / "summary.json").write_text(json.dumps(summary_data, indent=2))
    (project_dir / "summary.md").write_text("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))

    # Upload summary to HF
    try:
        api = _hf_api(hf_token)
        for fname in ["summary.json", "summary.md"]:
            fpath = project_dir / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"runs/{fname}",
                    repo_id=hf_repo_id,
                    repo_type="model",
                )
    except Exception as e:
        print(f"[HF] Warning: summary upload failed: {e}")

    return summary_data


# ── Main runner ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # Fail fast: require HF credentials
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    hf_repo_id = args.hf_repo_id
    if not hf_token:
        raise SystemExit("FATAL: HF_TOKEN missing. Set --hf-token or HF_TOKEN env var.")
    if not hf_repo_id:
        raise SystemExit("FATAL: --hf-repo-id missing.")

    local_ultralytics()

    for seed in SEEDS:
        ds_dir = args.dataset_root / f"levir_ship_yolo_seed{seed}"
        data_root = args.data_root if (args.data_root / "All Images").is_dir() else Path("/marimo/LevirShipData")
        data_yaml = prepare(data_root, ds_dir, seed)

        for variant_name, yaml_cfg in VARIANTS.items():
            if args.variants and variant_name not in args.variants:
                continue

            run_name = f"{variant_name}_seed{seed}"
            run_dir = args.project / run_name
            hf_remote_prefix = f"runs/{variant_name}/seed_{seed}"

            if completed(run_dir):
                print(f"[SKIP] Already complete (incl. HF upload): {run_name}")
                continue

            from ultralytics import YOLO

            last = run_dir / "weights/last.pt"
            if last.is_file():
                print(f"[RESUME] {run_name}")
                model = YOLO(str(last))
                model.train(resume=True)
            else:
                print(f"[START] {run_name} — {yaml_cfg.name}")
                model = YOLO(str(yaml_cfg))
                model.train(
                    data=str(data_yaml),
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch_size,
                    device=args.device,
                    workers=args.workers,
                    patience=args.patience,
                    seed=seed,
                    deterministic=True,
                    project=str(args.project),
                    name=run_name,
                    exist_ok=True,
                )

            # Evaluate
            best_model = YOLO(str(run_dir / "weights/best.pt"))
            print(f"[EVAL] {run_name} nms_iou=0.5")
            metrics = evaluate_run(best_model, str(data_yaml), run_dir)
            print(json.dumps(metrics, indent=2))

            # Upload to HF — fail hard (don't silently skip to next variant)
            print(f"[HF] Uploading {run_name} → {hf_repo_id}/{hf_remote_prefix}")
            upload_run_to_hf(run_dir, yaml_cfg, hf_repo_id, hf_remote_prefix, hf_token)

    generate_summary(args.project, hf_repo_id, hf_token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",    type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project",      type=Path, default=ROOT / "runs/levir_raw_cue_archs")
    parser.add_argument("--epochs",       type=int,  default=100)
    parser.add_argument("--imgsz",        type=int,  default=512)
    parser.add_argument("--batch-size",   type=int,  default=8)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--workers",      type=int,  default=4)
    parser.add_argument("--patience",     type=int,  default=0)
    parser.add_argument("--hf-token",     default="", help="HuggingFace token (or set HF_TOKEN env)")
    parser.add_argument("--hf-repo-id",   default="duyle2408/levir-raw-cue-archs")
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help="Which variants to run. Default: all. E.g. --variants yolov8n_p2_split_channel"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
