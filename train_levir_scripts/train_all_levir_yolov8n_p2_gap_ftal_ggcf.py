#!/usr/bin/env python3
"""Train/evaluate GAP+FTAL GGCF candidate-field experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG = ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_gap_ggcf.yaml"

workflow.EXPERIMENT = "levir_yolov8n_p2_gap_ftal_ggcf"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-gap-ftal-ggcf"
workflow.VARIANTS = {
    "G1_field_only": CONFIG,
    "G2_ggcf": CONFIG,
    "G3_ggcf_refined_assign": CONFIG,
    "G4_ggcf_standard_tal": CONFIG,
    "G5_field_only_refined_assign": CONFIG,
}

_base_model_for = workflow.model_for
_base_train_kwargs = workflow.train_kwargs
_base_uploader = workflow.Uploader


def _ftal(kwargs: dict[str, object], enabled: bool = True) -> None:
    kwargs.update(
        factorized_tal_target=enabled,
        factorized_tal_tau=0.75,
        factorized_tal_kappa=1.5,
        factorized_tal_lambda=0.5,
        factorized_tal_s_max=32.0,
        factorized_tal_warmup_start=5,
        factorized_tal_warmup_end=15,
        factorized_tal_p2_only=True,
    )


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    variant = getattr(args, "_variant", "")
    _ftal(kwargs, enabled=variant != "G4_ggcf_standard_tal")
    kwargs.update(
        ggcf_train_k=256,
        ggcf_hard_bg=128,
        ggcf_assign_refined=variant in {"G3_ggcf_refined_assign", "G4_ggcf_standard_tal", "G5_field_only_refined_assign"},
        ggcf_tal_diagnostics=variant == "G3_ggcf_refined_assign",
    )
    return kwargs


def model_for(variant: str, pretrained: str):
    model = _base_model_for(variant, pretrained)
    from ultralytics.nn.modules import ChannelAttention, Detect

    layers = model.model.model
    head = layers[-1]
    if not isinstance(layers[19], ChannelAttention) or not isinstance(head, Detect) or head.f != [19]:
        raise ValueError(f"{variant}: expected P2 -> ChannelAttention(avg) -> Detect([19])")
    if head.stride.tolist() != [4.0] or head.nl != 1 or not head.ggcf_refine:
        raise ValueError(f"{variant}: expected P2-only GGCF Detect stride [4], got {head.stride.tolist()}")
    head.ggcf_geometry = variant not in {"G1_field_only", "G5_field_only_refined_assign"}
    head.ggcf_encoder.geometry = head.ggcf_geometry
    return model


class Uploader(_base_uploader):
    REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv", "evaluation_metrics.json", "args.yaml")

    def upload_run(self, run_dir: Path, variant: str, seed: int) -> None:
        super().upload_run(run_dir, variant, seed)
        remote = f"runs/{variant}/seed_{seed}"
        files = set(self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset"))
        missing = [f"{remote}/{path}" for path in self.REQUIRED if f"{remote}/{path}" not in files]
        if missing:
            raise FileNotFoundError(f"HF upload verification failed: {missing}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(
            json.dumps({"repo_id": self.repo_id, "remote": remote, "verified": True}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.retry(lambda: self.api.upload_file(
            path_or_fileobj=marker,
            path_in_repo=f"{remote}/upload_complete.json",
            repo_id=self.repo_id,
            repo_type="dataset",
        ))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(workflow.VARIANTS), default=[
        "G1_field_only", "G2_ggcf", "G3_ggcf_refined_assign"
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{workflow.EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=workflow.HF_REPO)
    return parser.parse_args(argv)


def _test_ap(metrics: dict[str, float]) -> tuple[float, float]:
    return float(metrics.get("test/metrics/mAP50(B)", 0.0)), float(metrics.get("test/metrics/mAP50-95(B)", 0.0))


def main() -> None:
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.Uploader = Uploader
    workflow.parse_args = parse_args
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {}
        for variant in args.variants:
            args._variant = variant
            amp[variant] = workflow.smoke(variant, data_yaml, args)
    if args.smoke_only:
        return
    metrics_seen: dict[str, dict[str, float]] = {}
    for seed in args.seeds:
        for variant in args.variants:
            args._variant = variant
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            metrics_seen[variant] = workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)



if __name__ == "__main__":
    main()
