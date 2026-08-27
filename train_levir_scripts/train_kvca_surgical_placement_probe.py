#!/usr/bin/env python3
"""Surgical KVCA placement probe on the canonical LEVIR GAP+FTAL checkpoint.

Each variant inserts one zero-initialized residual KVCA into the unchanged
canonical P2 FPN path. The canonical checkpoint is transferred into matching
layers, every non-probe layer is frozen through Ultralytics' layer freeze API,
and only the inserted KVCA is optimized.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = ROOT / "models_related/ultralytics"
CANONICAL_CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_channel_only.yaml"
CONFIGS = {
    "A": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_surgical_a_p3_context.yaml",
    "B": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_surgical_b_fusion_input.yaml",
    "C": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_surgical_c_final_p2.yaml",
}
KVCA_LAYERS = {"A": 16, "B": 18, "C": 19}
EXPECTED_KVCA = {"A": (64, 4), "B": (96, 8), "C": (32, 8)}
REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "evaluation_metrics.json", "experiment_manifest.json", "ranking_summary.json", "gradient_receptivity.json")


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_split(args: argparse.Namespace) -> Path:
    from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow
    data_yaml = workflow.prepare_fixed_split(argparse.Namespace(data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42))
    workflow.validate_split(data_yaml)
    return data_yaml


def model_for(placement: str, checkpoint: str):
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import KVCompressedAttention

    model = YOLO(CONFIGS[placement])
    source = YOLO(checkpoint).model
    source_state = source.state_dict()
    probe_index = KVCA_LAYERS[placement]
    mapped = {}
    for key, value in source_state.items():
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit():
            index = int(parts[1])
            if index >= probe_index:
                parts[1] = str(index + 1)
            key = ".".join(parts)
        if key in model.model.state_dict() and model.model.state_dict()[key].shape == value.shape:
            mapped[key] = value
    missing, unexpected = model.model.load_state_dict(mapped, strict=False)
    loaded_source_keys = len(mapped)
    if not loaded_source_keys:
        raise RuntimeError(f"{placement}: canonical checkpoint transfer loaded no tensors")
    if unexpected:
        raise RuntimeError(f"{placement}: unexpected transferred tensors: {unexpected[:5]}")
    model._surgical_transfer = {"loaded_tensors": loaded_source_keys, "missing_target_keys": list(missing)}
    # YOLO.train() rebuilds a DetectionModel when the wrapper has no checkpoint
    # object. Mark this manually-remapped wrapper as checkpoint-backed so the
    # trainer clones this exact target module instead of starting from YAML.
    model.ckpt = {"epoch": -1, "optimizer": None}
    model.ckpt_path = checkpoint
    layer = model.model.model[KVCA_LAYERS[placement]]
    if not isinstance(layer, KVCompressedAttention):
        raise TypeError(f"{placement}: expected KVCompressedAttention at layer {KVCA_LAYERS[placement]}, got {type(layer).__name__}")
    expected_channels, expected_sr = EXPECTED_KVCA[placement]
    if layer.c2 != expected_channels or layer.sr_ratio != expected_sr or layer.num_heads != 4:
        raise ValueError(f"{placement}: unexpected KVCA config c2={layer.c2}, heads={layer.num_heads}, sr={layer.sr_ratio}")
    head = model.model.model[-1]
    if head.f != [20] or head.stride.tolist() != [4.0]:
        raise ValueError(f"{placement}: expected GAP P2 Detect from [20], stride [4.0], got {head.f}, {head.stride.tolist()}")
    return model


def freeze_except_probe(model, placement: str) -> list[str]:
    probe_index = KVCA_LAYERS[placement]
    frozen = []
    for index, module in enumerate(model.model.model):
        if index == probe_index:
            continue
        for name, parameter in module.named_parameters(recurse=True):
            parameter.requires_grad = False
            frozen.append(f"model.{index}.{name}")
    trainable = [name for name, parameter in model.model.named_parameters() if parameter.requires_grad]
    expected_prefix = f"model.{probe_index}."
    if not trainable or any(not name.startswith(expected_prefix) for name in trainable):
        raise AssertionError(f"{placement}: trainable parameters escaped probe layer: {trainable[:8]}")
    return frozen


def install_gradient_probe(model, placement: str, run_dir: Path, limit: int = 500) -> None:
    """Record residual-output gradient norms for the first bounded train batches."""
    import torch

    rows: list[dict[str, float | int]] = []
    holder: dict[str, torch.Tensor] = {}
    handle = None

    def capture(_module, _inputs, output):
        if torch.is_tensor(output) and output.requires_grad:
            output.retain_grad()
            holder["output"] = output

    def on_train_start(trainer):
        # The trainer clones the target model during YOLO.train(). Attach to
        # that live module, not the pre-trainer wrapper module.
        nonlocal handle
        probe = trainer.model.model[KVCA_LAYERS[placement]]
        handle = probe.register_forward_hook(capture)

    def on_batch_end(trainer):
        if len(rows) >= limit:
            return
        output = holder.pop("output", None)
        if output is None or output.grad is None:
            return
        rows.append({"batch": len(rows), "g_total_l2": float(output.grad.detach().float().norm().cpu())})
        if len(rows) == limit:
            if handle is not None:
                handle.remove()
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "gradient_receptivity.json").write_text(json.dumps({"placement": placement, "batches": rows}, indent=2) + "\n")

    def on_train_end(trainer):
        if handle is not None:
            handle.remove()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "gradient_receptivity.json").write_text(json.dumps({"placement": placement, "batches": rows}, indent=2) + "\n")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_train_end", on_train_end)


def assert_identity_initialization(model, placement: str, imgsz: int, device: str) -> dict[str, float]:
    local_ultralytics()
    import torch
    probe = model.model.model[KVCA_LAYERS[placement]]
    probe.eval()
    x = torch.randn(1, probe.c2, imgsz // (8 if placement == "A" else 4), imgsz // (8 if placement == "A" else 4), device=device)
    with torch.inference_mode():
        y = probe(x)
    max_abs = float((y - x).abs().max().cpu())
    if max_abs > 1e-6:
        raise AssertionError(f"{placement}: zero-init KVCA is not identity, max_abs={max_abs}")
    return {"max_abs": max_abs}


def train_one(placement: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / placement / f"seed_{args.seed}"
    if all((run_dir / path).is_file() for path in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")):
        return run_dir
    seed_everything(args.seed)
    model = model_for(placement, args.canonical_checkpoint)
    freeze_except_probe(model, placement)
    install_gradient_probe(model, placement, run_dir, args.gradient_batches)
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
        device=args.device, workers=args.workers, patience=args.patience, seed=args.seed,
        deterministic=True, amp=args.amp, plots=False, project=str(args.project / placement),
        name=f"seed_{args.seed}", exist_ok=True, freeze=[i for i in range(len(model.model.model)) if i != KVCA_LAYERS[placement]],
        factorized_tal_target=True, factorized_tal_tau=0.75, factorized_tal_kappa=1.5,
        factorized_tal_lambda=0.5, factorized_tal_s_max=32.0, factorized_tal_warmup_start=5,
        factorized_tal_warmup_end=15, factorized_tal_p2_only=True,
    )
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float | str]:
    local_ultralytics()
    from ultralytics import YOLO
    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size, device=args.device, workers=args.workers, plots=False, iou=0.5, project=str(run_dir / "evaluation"), name=split, exist_ok=True)
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def write_ranking_summary(run_dir: Path, args: argparse.Namespace) -> None:
    local_ultralytics()
    from train_levir_scripts import analyze_p2_cbam_ranking as ranking
    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/val"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if args.ranking_limit:
        images = images[: args.ranking_limit]
    ranking.EXPECTED_LEVELS["surgical"] = 1
    device = args.device
    if isinstance(device, str) and device.isdigit():
        import torch
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    rows = ranking.inspect_model("surgical", run_dir / "weights/best.pt", images, argparse.Namespace(imgsz=args.imgsz, device=device, expected_seed=args.seed))
    ranking_summary = {
        "protocol": {"split": "val", "seed": args.seed, "nms_iou": 0.5,
                     "candidate_rule": "decoded P2 anchor center inside GT; overlapping anchors assigned to highest-IoU GT",
                     "prediction_stage": "decoded P2 boxes and sigmoid class scores before threshold and NMS"},
        "raw_p2": ranking.descriptive_summary(rows),
    }
    (run_dir / "ranking_summary.json").write_text(json.dumps(ranking_summary, indent=2, sort_keys=True) + "\n")


def write_manifest(placement: str, run_dir: Path, args: argparse.Namespace) -> None:
    shutil.copy2(CONFIGS[placement], run_dir / "config.yaml")
    manifest = {
        "placement": placement, "seed": args.seed, "split_seed": 42,
        "canonical_config": str(CANONICAL_CONFIG), "probe_config": str(CONFIGS[placement]),
        "canonical_checkpoint": args.canonical_checkpoint, "kvca_layer": KVCA_LAYERS[placement],
        "kvca_channels": EXPECTED_KVCA[placement][0], "kvca_sr_ratio": EXPECTED_KVCA[placement][1],
        "heads": 4, "zero_init_residual": True, "frozen_existing_parameters": True,
        "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz,
        "batch_size": args.batch_size, "nms_iou": 0.5, "ftal": {"tau": 0.75, "kappa": 1.5, "lambda": 0.5},
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--placements", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    p.add_argument("--project", type=Path, default=ROOT / "runs/levir_kvca_surgical_placement_probe")
    p.add_argument("--canonical-checkpoint", required=True)
    p.add_argument("--hf-repo-id", default=None, help="Optional Hugging Face repo for artifact upload")
    p.add_argument("--ranking-limit", type=int)
    p.add_argument("--gradient-batches", type=int, default=500)
    return p.parse_args(argv)


def main() -> None:
    from utils.marimo_ops import require_training_context

    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    data_yaml = prepare_split(args)
    if args.hf_repo_id:
        require_training_context(hf_repo_id=args.hf_repo_id)
    local_ultralytics()
    uploader = None
    if args.hf_repo_id:
        import train_all_levir_yolov8n_p2_gap_scale_temper as upload_base
        upload_base.REQUIRED = REQUIRED
        uploader = upload_base.Uploader(args.hf_repo_id)
    for placement in args.placements:
        run_dir = train_one(placement, data_yaml, args)
        evaluate(run_dir, data_yaml, args)
        write_ranking_summary(run_dir, args)
        write_manifest(placement, run_dir, args)
        missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
        if missing:
            raise RuntimeError(f"{placement}: missing required artifacts: {missing}")
        if uploader is not None:
            uploader.upload_run(placement, args.seed, run_dir)


if __name__ == "__main__":
    main()
