#!/usr/bin/env python3
"""Train the requested baseline/W1 x context-augmentation ablation matrix."""
from __future__ import annotations
import argparse, json, os, random, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from misc.prepare_levir_ship import prepare
from utils.marimo_ops import require_training_context
ULTRA = ROOT / "models_related/ultralytics"
CFG = ROOT / "models_related/models_config/yolov8/levir"
RUNS = {
    "baseline_oacp": (CFG / "yolov8n_p2_levir_baseline.yaml", "oacp", {}),
    "baseline_cea": (CFG / "yolov8n_p2_levir_baseline.yaml", "cea", {}),
    "baseline_lea": (CFG / "yolov8n_p2_levir_baseline.yaml", "lea", {}),
    "w1_oacp": (CFG / "yolov8n_p2_levir_oaief_w1.yaml", "oacp", {}),
    "w1_cea": (CFG / "yolov8n_p2_levir_oaief_w1.yaml", "cea", {}),
    "w1_lea": (CFG / "yolov8n_p2_levir_oaief_w1.yaml", "lea", {}),
    # Legacy entries retain their original FTAL confound for provenance.
    "w1_api": (CFG / "yolov8n_p2_levir_oaief_w1_api.yaml", "none", {"api": True, "ftal": True, "legacy_api": True}),
    "w1_api_oacp": (CFG / "yolov8n_p2_levir_oaief_w1_api.yaml", "oacp", {"api": True, "ftal": True, "legacy_api": True}),
    "w1_api_cea": (CFG / "yolov8n_p2_levir_oaief_w1_api.yaml", "cea", {"api": True, "ftal": True, "legacy_api": True}),
    "w1_api_lea": (CFG / "yolov8n_p2_levir_oaief_w1_api.yaml", "lea", {"api": True, "ftal": True, "legacy_api": True}),
}
CORRECTED_API_RUNS = {
    "w1_control": (CFG / "yolov8n_p2_levir_oaief_w1.yaml", "none", {}),
    "w1_api_boxgrad": (CFG / "yolov8n_p2_levir_oaief_w1_api_boxgrad.yaml", "none", {"api": True}),
    "w1_api_boxgrad_ftal": (CFG / "yolov8n_p2_levir_oaief_w1_api_boxgrad.yaml", "none", {"api": True, "ftal": True}),
}
CORRECTED_FULL_RUNS = {
    "baseline_oacp": (CFG / "yolov8n_p2_levir_baseline.yaml", "oacp", {}),
    "baseline_cea": (CFG / "yolov8n_p2_levir_baseline.yaml", "cea", {}),
    "baseline_lea": (CFG / "yolov8n_p2_levir_baseline.yaml", "lea", {}),
    "w1_oacp": (CFG / "yolov8n_p2_levir_oaief_w1.yaml", "oacp", {}),
    "w1_cea": (CFG / "yolov8n_p2_levir_oaief_w1.yaml", "cea", {}),
    "w1_lea": (CFG / "yolov8n_p2_levir_oaief_w1.yaml", "lea", {}),
    "w1_api_oacp": (CFG / "yolov8n_p2_levir_oaief_w1_api_boxgrad.yaml", "oacp", {"api": True}),
    "w1_api_oacp_ftal": (CFG / "yolov8n_p2_levir_oaief_w1_api_boxgrad.yaml", "oacp", {"api": True, "ftal": True}),
    "w1_api_cea": (CFG / "yolov8n_p2_levir_oaief_w1_api_boxgrad.yaml", "cea", {"api": True}),
    "w1_api_lea": (CFG / "yolov8n_p2_levir_oaief_w1_api_boxgrad.yaml", "lea", {"api": True}),
    "w1_api": (CFG / "yolov8n_p2_levir_oaief_w1_api_boxgrad.yaml", "none", {"api": True}),
}
REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv")
COMPLETE = (*REQUIRED, "evaluation_metrics.json", "manifest.json")
FTAL = {"factorized_tal_target": True, "factorized_tal_mode": "legacy", "factorized_tal_tau": .75, "factorized_tal_kappa": 1.5, "factorized_tal_lambda": .5, "factorized_tal_s_max": 32., "factorized_tal_warmup_start": 5, "factorized_tal_warmup_end": 15, "factorized_tal_p2_only": True}

def _local(): sys.path.insert(0, str(ULTRA))
def _has(p, names): return all((p / n).is_file() for n in names)
def _upload_verified_for_repo(p, repo):
    try:
        return json.loads((p / "upload_complete.json").read_text()).get("repo_id") == repo
    except (FileNotFoundError, json.JSONDecodeError):
        return False
def _seed(s):
    import numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def _eval(run, data, a):
    _local(); from ultralytics import YOLO
    model = YOLO(run / "weights/best.pt"); metrics = {}
    for split in ("val", "test"):
        r = model.val(data=str(data), split=split, imgsz=a.imgsz, batch=a.batch_size, device=a.device, workers=a.workers, plots=False, iou=.5, project=str(run / "evaluation"), name=split, exist_ok=True)
        metrics.update({f"{split}/{k}": float(v) for k, v in (r.results_dict or {}).items()}); metrics[f"{split}/AP75"] = float(r.box.map75)
    (run / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
def _upload(run, name, repo):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"]); api.create_repo(repo_id=repo, repo_type="dataset", private=False, exist_ok=True)
    prefix = f"runs/{name}"; api.upload_folder(folder_path=str(run), path_in_repo=prefix, repo_id=repo, repo_type="dataset")
    remote = {x.rfilename for x in api.list_repo_tree(repo_id=repo, repo_type="dataset", path_in_repo=prefix, recursive=True) if hasattr(x, "rfilename")}
    missing = {f"{prefix}/{x}" for x in COMPLETE} - remote
    if missing: raise RuntimeError(f"Upload verification failed for {name}: {sorted(missing)}")
    (run / "upload_complete.json").write_text(json.dumps({"repo_id": repo, "remote_prefix": prefix}, indent=2) + "\n")
def run(a):
    require_training_context(hf_repo_id=a.hf_repo_id)
    split_seed = getattr(a, "split_seed", a.seed)
    data = prepare(a.data_root, a.dataset_root / "levir_ship_yolo", split_seed)
    _local(); from ultralytics import YOLO
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    catalog = CORRECTED_FULL_RUNS if a.corrected_full_matrix else CORRECTED_API_RUNS if a.corrected_api_ablation else RUNS
    selected = catalog if not a.only else {k: catalog[k] for k in a.only}
    for name, (config, aug, extra) in selected.items():
        run = a.project / name; run.mkdir(parents=True, exist_ok=True)
        if _has(run, COMPLETE) and _upload_verified_for_repo(run, a.hf_repo_id): continue
        if a.reuse_from and name in a.reuse_variants and not _has(run, COMPLETE):
            source = a.reuse_from / name
            if not (_has(source, COMPLETE) and (source / "upload_complete.json").is_file()):
                raise RuntimeError(f"Cannot reuse incomplete or unverified source run: {source}")
            shutil.copytree(source, run, dirs_exist_ok=True)
            reused_manifest = json.loads((run / "manifest.json").read_text())
            reused_manifest.update({
                "reused_from": str(source),
                "reused_source_commit_sha": reused_manifest.get("commit_sha"),
                "consumer_commit_sha": sha,
                "reused_reason": "Non-API baseline path is unchanged by the API loss-path fix.",
                "consumer_hf_repo_id": a.hf_repo_id,
            })
            (run / "manifest.json").write_text(json.dumps(reused_manifest, indent=2) + "\n")
        os.environ["YOLO_CONTEXT_AUG"] = aug; _seed(a.seed)
        from project_ultralytics.context_augment import augmentation_config
        if not (run / "manifest.json").is_file():
            (run / "manifest.json").write_text(json.dumps({"experiment": name, "config": str(config), "augmentation": aug, "augmentation_config": augmentation_config(), "api": extra.get("api", False), "api_target_mode": "boxgrad" if "boxgrad" in str(config) else ("foreground" if extra.get("api") else None), "ftal": extra.get("ftal", False), "legacy_api": extra.get("legacy_api", False), "commit_sha": sha, "seed": a.seed, "split_seed": split_seed, "split": ["val", "test"], "nms_iou": .5, "epochs": a.epochs, "patience": a.patience, "hf_repo_id": a.hf_repo_id}, indent=2) + "\n")
        if not _has(run, REQUIRED):
            model = YOLO(str(config)); model.load("yolov8n.pt", smart_transfer=True)
            kwargs = dict(data=str(data), epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch_size, device=a.device, workers=a.workers, amp=a.amp, seed=a.seed, deterministic=True, project=str(a.project), name=name, exist_ok=True)
            if extra.get("ftal"): kwargs.update(FTAL)
            model.train(**kwargs)
        if not _has(run, REQUIRED): raise FileNotFoundError(run)
        if not (run / "evaluation_metrics.json").is_file(): _eval(run, data, a)
        if not _has(run, COMPLETE): raise FileNotFoundError(run)
        _upload(run, name, a.hf_repo_id); print(f"COMPLETE {name}", flush=True)
def args():
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--dataset-root", type=Path, required=True); p.add_argument("--project", type=Path, required=True); p.add_argument("--hf-repo-id", required=True); p.add_argument("--only", nargs="*"); p.add_argument("--corrected-api-ablation", action="store_true", help="Run W1, detector-level API(boxgrad), and API+FTAL controls."); p.add_argument("--corrected-full-matrix", action="store_true", help="Run the 10-variant matrix with corrected detector-level W1+API."); p.add_argument("--reuse-from", type=Path, help="Reuse verified completed variants from an earlier matrix."); p.add_argument("--reuse-variants", nargs="*", default=[], help="Variant names eligible for --reuse-from."); p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=0); p.add_argument("--imgsz", type=int, default=512); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--device", default="0"); p.add_argument("--workers", type=int, default=4); p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True); return p.parse_args()
if __name__ == "__main__": run(args())
