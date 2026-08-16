#!/usr/bin/env python3
"""Train/evaluate plain P2 + default TAL verifier tests on LEVIR dataset."""

from __future__ import annotations

import argparse
import json
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"

workflow.EXPERIMENT = "levir_yolov8n_p2_gap_ftal_verifiers"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-gap-ftal-verifiers-seed42"
workflow.VARIANTS = {
    "a1_box_fovea": CONFIG_ROOT / "yolov8n_p2_fpn_only_plain.yaml",
    "a3_semantic_structural": CONFIG_ROOT / "yolov8n_p2_fpn_only_plain.yaml",
    "a4_raw_adapted": CONFIG_ROOT / "yolov8n_p2_fpn_only_plain.yaml",
}

_BASE_TRAIN_KWARGS = workflow.train_kwargs
_BASE_MODEL_FOR = workflow.model_for
_BASE_SMOKE = workflow.smoke
_BASE_TRAIN = workflow.train


def create_scene_disjoint_split(data_root: Path, output: Path, seed: int) -> Path:
    """Prepare a reproducible scene-disjoint split targeting 2728/584/584 images."""
    image_dir, label_dir = data_root / "All Images", data_root / "All Annotations"
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    if image_stems != label_stems:
        raise ValueError("Image/label mismatch")
    
    import re
    from collections import defaultdict
    scene_re = re.compile(r"^(.*)_(-?\d+)_(-?\d+)$")
    scene_to_stems = defaultdict(list)
    for stem in sorted(image_stems):
        match = scene_re.match(stem)
        if not match:
            raise ValueError(f"Invalid stem: {stem}")
        scene = match.group(1)
        scene_to_stems[scene].append(stem)
        
    scenes = sorted(scene_to_stems.keys())
    import random
    random.Random(seed).shuffle(scenes)
    
    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    
    for scene in scenes:
        stems = scene_to_stems[scene]
        n = len(stems)
        if counts["train"] < 2728:
            splits["train"].extend(stems)
            counts["train"] += n
        elif counts["val"] < 584:
            splits["val"].extend(stems)
            counts["val"] += n
        else:
            splits["test"].extend(stems)
            counts["test"] += n
            
    print(f"Scene-disjoint split counts: {counts}")
    
    import shutil
    for generated in (output / "images", output / "labels"):
        if generated.exists():
            shutil.rmtree(generated)
            
    manifest = {"seed": seed, "splits": {}}
    for split, stems in splits.items():
        images_out = output / "images" / split
        labels_out = output / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            for source, destination in (
                (image_dir / f"{stem}.png", images_out / f"{stem}.png"),
                (label_dir / f"{stem}.txt", labels_out / f"{stem}.txt"),
            ):
                if not destination.exists():
                    try:
                        destination.symlink_to(source)
                    except OSError:
                        shutil.copy2(source, destination)
        manifest["splits"][split] = {"images": len(stems)}
        
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    data_yaml = output / "levir_ship.yaml"
    data_yaml.write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n",
        encoding="utf-8",
    )
    return data_yaml


def prepare_fixed_split(args: argparse.Namespace) -> Path:
    data_yaml = args.dataset_root / f"levir_ship_yolo_scene_seed{args.split_seed}" / "levir_ship.yaml"
    if not data_yaml.is_file():
        data_yaml = create_scene_disjoint_split(args.data_root, data_yaml.parent, args.split_seed)
    return data_yaml


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    # Use default TAL settings (remove FTAL)
    kwargs = _BASE_TRAIN_KWARGS(args, data_yaml, seed, amp)
    # Ensure FTAL flags are explicitly disabled
    kwargs.update(
        factorized_tal_target=False,
    )
    return kwargs


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    # Dynamically read and write a temporary config YAML file with verifier_mode injected
    plain_yaml_path = CONFIG_ROOT / "yolov8n_p2_fpn_only_plain.yaml"
    with open(plain_yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg_dict["verifier_mode"] = variant
    cfg_dict["verifier_alpha"] = 0.0
    cfg_dict["verifier_loss_gain"] = 0.5
    
    temp_yaml_path = CONFIG_ROOT / f"temp_{variant}.yaml"
    with open(temp_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f)
        
    try:
        from ultralytics import YOLO
        model = YOLO(str(temp_yaml_path))
        model.load(pretrained, smart_transfer=True)
    finally:
        if temp_yaml_path.is_file():
            temp_yaml_path.unlink()
            
    from ultralytics.nn.modules import Detect
    layers = model.model.model
    head = layers[-1]
    if not isinstance(head, Detect):
        raise ValueError(f"{variant}: expected Detect head at the end")
    if head.stride.tolist() != [4.0] or head.nl != 1:
        raise ValueError(f"{variant}: expected P2-only Detect stride [4], got {head.stride.tolist()}")
    return model


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    args.current_variant = variant
    return _BASE_TRAIN(variant, seed, data_yaml, amp, args)


def smoke(variant: str, data_yaml: Path, args: argparse.Namespace, amp: bool = True) -> bool:
    args.current_variant = variant
    return _BASE_SMOKE(variant, data_yaml, args, amp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(workflow.VARIANTS), default=list(workflow.VARIANTS))
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
    parser.add_argument("--verifier-alpha", type=float, default=0.0)
    parser.add_argument("--verifier-loss-gain", type=float, default=0.5)
    parser.set_defaults(current_variant="a1_box_fovea", runner=Path(__file__).resolve())
    return parser.parse_args()


def main() -> None:
    workflow.prepare_fixed_split = prepare_fixed_split
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.smoke = smoke
    workflow.train = train
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
