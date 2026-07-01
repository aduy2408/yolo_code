#!/usr/bin/env python3
"""Run a tiny matched KVCA localization probe with Grad-CAM overlays."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = REPO_ROOT / "models_related" / "ultralytics"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from misc.prepare_dataset import prepare_dataset  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


@dataclass(frozen=True)
class ProbeModel:
    name: str
    cfg: Path
    cam_layer_index: int


MODELS = (
    ProbeModel("baseline", REPO_ROOT / "models_related/models_config/yolov8/yolov8_varroa_tiny_probe.yaml", 15),
    ProbeModel("kvca", REPO_ROOT / "models_related/models_config/yolov8/yolov8_varroa_tiny_kvca_probe.yaml", 16),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-12)


def center_error(pred: list[float], gt: list[float]) -> float:
    pcx, pcy = (pred[0] + pred[2]) * 0.5, (pred[1] + pred[3]) * 0.5
    gcx, gcy = (gt[0] + gt[2]) * 0.5, (gt[1] + gt[3]) * 0.5
    diag = max(((gt[2] - gt[0]) ** 2 + (gt[3] - gt[1]) ** 2) ** 0.5, 1e-12)
    return (((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5) / diag


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = [float(x) for x in parts]
        x1 = (cx - bw * 0.5) * width
        y1 = (cy - bh * 0.5) * height
        x2 = (cx + bw * 0.5) * width
        y2 = (cy + bh * 0.5) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def label_for_image(image_path: Path, data_yaml: Path, split: str) -> Path:
    root = data_yaml.parent
    return root / "labels" / split / image_path.with_suffix(".txt").name


def collect_images(data_yaml: Path, split: str, limit: int) -> list[Path]:
    root = data_yaml.parent
    images = sorted((root / "images" / split).glob("*.png"))
    positives: list[Path] = []
    for image_path in images:
        with Image.open(image_path) as img:
            label_path = label_for_image(image_path, data_yaml, split)
            if read_yolo_boxes(label_path, *img.size):
                positives.append(image_path)
    return positives[:limit]


def train_model(model_def: ProbeModel, data_yaml: Path, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    seed_everything(args.seed)
    model = YOLO(str(model_def.cfg))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=str(args.project / "train"),
        name=model_def.name,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=0,
        plots=False,
        pretrained=False,
        exist_ok=True,
        verbose=False,
    )
    return args.project / "train" / model_def.name / "weights" / "best.pt"


def eval_model(weights: Path, data_yaml: Path, model_name: str, args: argparse.Namespace) -> dict[str, float | str]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(args.project / "val"),
        name=model_name,
        plots=False,
        verbose=False,
    )
    return {
        "model": model_name,
        "weights": str(weights),
        "map50": float(getattr(metrics.box, "map50", 0.0)),
        "map50_95": float(getattr(metrics.box, "map", 0.0)),
        "precision": float(np.nanmean(getattr(metrics.box, "p", [0.0]))),
        "recall": float(np.nanmean(getattr(metrics.box, "r", [0.0]))),
    }


def prediction_localization(weights: Path, data_yaml: Path, model_name: str, args: argparse.Namespace) -> dict[str, float | str]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = collect_images(data_yaml, args.eval_split, args.cam_samples)
    rows = []
    for image_path in images:
        with Image.open(image_path) as img:
            width, height = img.size
        gt_boxes = read_yolo_boxes(label_for_image(image_path, data_yaml, args.eval_split), width, height)
        result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou, device=args.device, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.detach().cpu().tolist() if result.boxes is not None else []
        best_iou = 0.0
        best_center = ""
        for pred in pred_boxes:
            for gt in gt_boxes:
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_center = center_error(pred, gt)
        rows.append({"model": model_name, "image": str(image_path), "best_iou": best_iou, "center_error": best_center})

    valid_centers = [float(r["center_error"]) for r in rows if r["center_error"] != ""]
    return {
        "model": model_name,
        "mean_best_iou": float(np.mean([r["best_iou"] for r in rows])) if rows else 0.0,
        "mean_center_error": float(np.mean(valid_centers)) if valid_centers else 0.0,
    }


def preprocess_image(image_path: Path, imgsz: int, device: str) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((imgsz, imgsz))
    arr = np.asarray(resized).copy().transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
    return tensor.to(device).requires_grad_(True), image


def cam_stats(cam: np.ndarray, gt_boxes: list[list[float]]) -> tuple[float, float, float]:
    h, w = cam.shape
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in gt_boxes:
        mask[int(max(0, y1)) : int(min(h, y2)), int(max(0, x1)) : int(min(w, x2))] = True
    total = float(cam.sum())
    energy_inside = float(cam[mask].sum() / max(total, 1e-12))
    peak_y, peak_x = np.unravel_index(int(cam.argmax()), cam.shape)
    peak_inside = float(mask[peak_y, peak_x])
    cam_mask = cam >= max(float(cam.max()) * 0.5, 1e-12)
    inter = float(np.logical_and(cam_mask, mask).sum())
    union = float(np.logical_or(cam_mask, mask).sum())
    return energy_inside, peak_inside, inter / max(union, 1.0)


def save_overlay(image: Image.Image, cam: np.ndarray, gt_boxes: list[list[float]], out_path: Path) -> None:
    import matplotlib.cm as cm

    cam_img = Image.fromarray(np.uint8(cm.jet(cam)[..., :3] * 255)).resize(image.size)
    blended = Image.blend(image.convert("RGB"), cam_img, 0.45)
    draw = ImageDraw.Draw(blended)
    for box in gt_boxes:
        draw.rectangle(box, outline=(0, 255, 0), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(out_path)


def gradcam_model(weights: Path, model_def: ProbeModel, data_yaml: Path, args: argparse.Namespace) -> list[dict[str, float | str]]:
    from ultralytics import YOLO

    device = f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(weights))
    model = yolo.model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(True)
    target_layer = model.model[model_def.cam_layer_index]
    images = collect_images(data_yaml, args.eval_split, args.cam_samples)
    rows: list[dict[str, float | str]] = []

    for image_path in images:
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def fwd_hook(_module, _inputs, output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))

        handle = target_layer.register_forward_hook(fwd_hook)
        tensor, image = preprocess_image(image_path, args.imgsz, device)
        with torch.enable_grad():
            output = model(tensor)
            score = output[0][0, 4:, :].max() if isinstance(output, tuple) else output[0, 4:, :].max()
            model.zero_grad(set_to_none=True)
            score.backward()
        handle.remove()

        if not activations or not gradients:
            continue
        act = activations[-1].detach()
        grad = gradients[-1].detach()
        weights_cam = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights_cam * act).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-12)

        gt_boxes = read_yolo_boxes(label_for_image(image_path, data_yaml, args.eval_split), *image.size)
        energy, peak, cam_iou = cam_stats(cam, gt_boxes)
        out_path = args.project / "cam_overlays" / model_def.name / image_path.name
        save_overlay(image, cam, gt_boxes, out_path)
        rows.append(
            {
                "model": model_def.name,
                "image": str(image_path),
                "cam_energy_in_gt": energy,
                "cam_peak_in_gt": peak,
                "cam_iou_50": cam_iou,
                "overlay": str(out_path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, metric_rows: list[dict[str, object]], cam_rows: list[dict[str, object]]) -> None:
    by_model = {row["model"]: row for row in metric_rows}
    cam_summary = {}
    for name in ("baseline", "kvca"):
        rows = [r for r in cam_rows if r["model"] == name]
        cam_summary[name] = {
            "energy": float(np.mean([float(r["cam_energy_in_gt"]) for r in rows])) if rows else 0.0,
            "peak": float(np.mean([float(r["cam_peak_in_gt"]) for r in rows])) if rows else 0.0,
            "iou": float(np.mean([float(r["cam_iou_50"]) for r in rows])) if rows else 0.0,
        }
    verdict = "inconclusive"
    has_pair = "baseline" in by_model and "kvca" in by_model
    learned_signal = has_pair and max(float(by_model["baseline"]["map50"]), float(by_model["kvca"]["map50"])) > 0.0
    if (
        learned_signal
        and cam_summary["kvca"]["energy"] > cam_summary["baseline"]["energy"]
        and by_model["kvca"]["map50_95"] >= by_model["baseline"]["map50_95"]
    ):
        verdict = "KVCA helps localization in this quick probe"
    elif (
        learned_signal
        and cam_summary["kvca"]["energy"] < cam_summary["baseline"]["energy"]
        and by_model["kvca"]["map50_95"] <= by_model["baseline"]["map50_95"]
    ):
        verdict = "KVCA does not help localization in this quick probe"
    path.write_text(
        "\n".join(
            [
                "# KVCA Localization Probe",
                "",
                f"Verdict: {verdict}.",
                "",
                "## Metrics",
                "",
                "| model | mAP50 | mAP50-95 | precision | recall | mean best IoU | mean center error |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[
                    f"| {r['model']} | {float(r['map50']):.4f} | {float(r['map50_95']):.4f} | "
                    f"{float(r['precision']):.4f} | {float(r['recall']):.4f} | "
                    f"{float(r['mean_best_iou']):.4f} | {float(r['mean_center_error']):.4f} |"
                    for r in metric_rows
                ],
                "",
                "## CAM",
                "",
                "| model | energy in GT | peak in GT | CAM IoU@0.5 |",
                "| --- | ---: | ---: | ---: |",
                *[
                    f"| {name} | {vals['energy']:.4f} | {vals['peak']:.4f} | {vals['iou']:.4f} |"
                    for name, vals in cam_summary.items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data", help="Source dataset root containing train/val/test.")
    parser.add_argument("--project", type=Path, default=Path("yolo_related/runs/kvca_probe"))
    parser.add_argument("--dataset-dir", default="yolo_related/datasets/varroa_probe")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--cam-samples", type=int, default=24)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--skip-train", action="store_true", help="Use existing weights in project/train/*/weights/best.pt.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing weights and train only missing models.")
    parser.add_argument(
        "--models",
        default="baseline,kvca",
        help="Comma-separated model names to process. Use 'kvca' to retrain only the KVCA branch.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run limit=20, epochs=1, cam-samples=4.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.limit = 20
        args.epochs = 1
        args.cam_samples = 4
    args.project = args.project.resolve()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    seed_everything(args.seed)
    data_yaml = prepare_dataset(
        args.root,
        dataset_dir,
        limit=args.limit,
        gt_source="gt_one",
        only_positives=True,
        class_policy="map-3-to-1",
        seed=args.seed,
    )

    metric_rows: list[dict[str, object]] = []
    cam_rows: list[dict[str, object]] = []
    selected_models = {name.strip() for name in args.models.split(",") if name.strip()}
    for model_def in MODELS:
        if model_def.name not in selected_models:
            continue
        weights = args.project / "train" / model_def.name / "weights" / "best.pt"
        if args.reuse_existing and weights.is_file():
            pass
        elif not args.skip_train:
            weights = train_model(model_def, data_yaml, args)
        elif not weights.is_file():
            raise FileNotFoundError(f"--skip-train requested but weights are missing: {weights}")
        metrics = eval_model(weights, data_yaml, model_def.name, args)
        metrics.update(prediction_localization(weights, data_yaml, model_def.name, args))
        metric_rows.append(metrics)
        cam_rows.extend(gradcam_model(weights, model_def, data_yaml, args))

    write_csv(args.project / "metrics.csv", metric_rows)
    write_csv(args.project / "cam_localization.csv", cam_rows)
    write_report(args.project / "REPORT.md", metric_rows, cam_rows)
    print(f"Wrote {args.project / 'metrics.csv'}")
    print(f"Wrote {args.project / 'cam_localization.csv'}")
    print(f"Wrote {args.project / 'REPORT.md'}")


if __name__ == "__main__":
    main()
