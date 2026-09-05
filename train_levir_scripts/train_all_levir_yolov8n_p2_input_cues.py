#!/usr/bin/env python3
"""Screen RGB control and ten fixed input cues on the LEVIR YOLOv8n-P2 model."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_WORKFLOW_SPEC = importlib.util.spec_from_file_location(
    "_input_cues_workflow", ROOT / "train_all_levir_yolov8n_p2_routing.py"
)
workflow = importlib.util.module_from_spec(_WORKFLOW_SPEC)
sys.modules[_WORKFLOW_SPEC.name] = workflow
_WORKFLOW_SPEC.loader.exec_module(workflow)

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
BASELINE = CONFIG_ROOT / "yolov8n_p2_levir_baseline.yaml"
TEMPLATE = CONFIG_ROOT / "yolov8n_p2_input_cue_template.yaml"
VARIANTS = (
    "rgb_control", "sobel_xy", "laplacian_split", "log", "haar", "lab_ab",
    "ycbcr_cbcr", "chromatic_edge", "local_zscore", "structure_coherence", "top_hat",
    "robust_ring_contrast", "lbp_stats", "multiscale_tophat", "local_rank", "phase_coherence",
)
GENERATED = CONFIG_ROOT / "generated_input_cues"


def resolved_configs() -> dict[str, Path]:
    """Generate one resolved YAML per variant from the plain P2 graph."""
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    GENERATED.mkdir(parents=True, exist_ok=True)
    paths = {"rgb_control": BASELINE}
    for cue_type in VARIANTS[1:]:
        config = copy.deepcopy(template)
        config["backbone"][0] = [-1, 1, "InputCueConv", [64, 3, 2, cue_type]]
        path = GENERATED / f"yolov8n_p2_input_cue_{cue_type}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths[cue_type] = path
    return paths


CONFIGS = resolved_configs()
workflow.EXPERIMENT = "levir_yolov8n_p2_input_cues"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-input-cues-seed42"
workflow.VARIANTS = CONFIGS


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import InputCueConv, copy_rgb_stem_weights

    rgb_model = YOLO(pretrained)
    model = YOLO(CONFIGS[variant])
    model.load(pretrained, smart_transfer=False)
    if variant != "rgb_control":
        stem = model.model.model[0]
        if not isinstance(stem, InputCueConv):
            raise TypeError(f"Expected InputCueConv stem for {variant}, got {type(stem).__name__}")
        copy_rgb_stem_weights(model.model, rgb_model.model)
        with __import__("torch").no_grad():
            stem.conv.weight[:, 3:].zero_()
    return model


def evaluate(run_dir: Path, data_yaml: Path, args):
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    workflow.local_ultralytics()
    from ultralytics import YOLO

    metrics = {}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/Precision(B)"] = float(result.box.mp)
        metrics[f"{split}/metrics/Recall(B)"] = float(result.box.mr)
        metrics[f"{split}/metrics/mAP50(B)"] = float(result.box.map50)
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
        metrics[f"{split}/metrics/mAP50-95(B)"] = float(result.box.map)
    metrics["nms_iou"] = 0.5
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


workflow.model_for = model_for
workflow.evaluate = evaluate

_WORKFLOW_PARSE_ARGS = workflow.parse_args


def parse_args(argv=None):
    args = _WORKFLOW_PARSE_ARGS(argv)
    if argv is None or "--seeds" not in argv:
        args.seeds = [42]
    args.variants = list(args.variants)
    args.runner = Path(__file__).resolve()
    return args


def main():
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
