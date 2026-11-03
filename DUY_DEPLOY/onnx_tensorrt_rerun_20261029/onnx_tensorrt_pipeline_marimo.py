# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "numpy<2",
#   "onnx",
#   "onnxruntime-gpu",
#   "pillow",
#   "torch",
#   "ultralytics",
# ]
# ///

import marimo

app = marimo.App(width="full")



@app.cell

def _():
    from datetime import datetime, timezone
    import importlib.util
    import json
    import os
    from pathlib import Path
    import sys
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    library_dirs = sorted(str(path) for path in (site_packages / "nvidia").glob("*/lib") if path.is_dir())
    library_dirs += [str(site_packages / "tensorrt_libs"), str(site_packages / "torch" / "lib")]
    existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(library_dirs + ([existing_library_path] if existing_library_path else []))
    import marimo as mo
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    return datetime, importlib, json, mo, onnx, os, ort, Path, sys, timezone, torch, YOLO


@app.cell

def _(Path, datetime, mo, os, ort, timezone, torch):
    repo_root = Path.cwd().resolve()
    if repo_root.name == "onnx_tensorrt_rerun_20261029":
        repo_root = repo_root.parent

    checkpoint = repo_root / "exports/baseline_onnx_matrix/levirship/no_mosaic/yolov8/seed_42/hf_cache/runs/levirship/yolov8/seed_42/weights/best.pt"
    data_root = Path(os.environ.get("DATA_ROOT", repo_root / "datasets"))
    datasets = ("levirship",)
    splits = ("test",)
    batch_size = 1
    confidence = 0.001
    nms_iou = 0.5
    max_det = 300
    run_id = datetime.now(timezone.utc).strftime("single_model_%Y%m%dT%H%M%SZ")
    run_root = repo_root / "DUY_DEPLOY/results/tensorrt/marimo_runs" / run_id
    onnx_path = run_root / "model_fp16_dynamic.onnx"
    trt_cache_root = run_root / "trt_engine_cache"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP16 export and TensorRT")
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"TensorRT provider unavailable: {ort.get_available_providers()}")
    run_root.mkdir(parents=True, exist_ok=True)

    mo.md(
        f"""
        **Checkpoint:** `{checkpoint}`  
        **Data root:** `{data_root}`  
        **Run root:** `{run_root}`  
        **CUDA:** `{torch.cuda.get_device_name(0)}`  
        **ONNX Runtime providers:** `{ort.get_available_providers()}`  
        **NMS IoU:** `{nms_iou}`
        """
    )
    return batch_size, confidence, checkpoint, data_root, datasets, max_det, nms_iou, onnx_path, repo_root, run_root, trt_cache_root


@app.cell

def _(Path, YOLO, checkpoint, onnx, onnx_path, torch):
    image_size = 512 if "levirship" in checkpoint.parts else 640
    model = YOLO(str(checkpoint))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=image_size,
            half=True,
            dynamic=True,
            batch=1,
            device=0,
            opset=17,
            simplify=True,
        )
    )
    exported_graph = onnx.load(str(exported_path), load_external_data=False)
    onnx.checker.check_model(exported_graph)
    onnx_path.write_bytes(exported_path.read_bytes())
    del exported_graph
    print(f"ONNX export/check passed: {onnx_path}")
    return image_size, model


@app.cell

def _(onnx_path, ort, trt_cache_root):
    import_options = ort.SessionOptions()
    import_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    trt_cache_root.mkdir(parents=True, exist_ok=True)
    provider_options = {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(trt_cache_root),
    }
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=import_options,
        providers=[("TensorrtExecutionProvider", provider_options)],
    )
    if session.get_providers()[0] != "TensorrtExecutionProvider":
        raise RuntimeError(f"TensorRT was not first: {session.get_providers()}")
    print("TensorRT session ready; engine builds lazily on first inference")
    print("Input:", session.get_inputs()[0].name, session.get_inputs()[0].shape, session.get_inputs()[0].type)
    return session


@app.cell

def _(importlib, repo_root, sys):
    evaluate_path = repo_root / "onnx_tensorrt_rerun_20261029/source/evaluate.py"
    spec = importlib.util.spec_from_file_location("marimo_pipeline_evaluate", evaluate_path)
    evaluate = importlib.util.module_from_spec(spec)
    sys.modules["marimo_pipeline_evaluate"] = evaluate
    assert spec.loader is not None
    spec.loader.exec_module(evaluate)
    return evaluate


@app.cell

def _(Path, batch_size, confidence, data_root, evaluate, json, max_det, nms_iou, repo_root, run_root, session, splits):
    import numpy as np
    from PIL import Image

    def load_local_dataset(dataset, split):
        if dataset == "levirship":
            dataset_root = data_root / "levir_ship_yolo_seed42"
            annotation_path = dataset_root / "annotations" / f"{split}.json"
            image_root = dataset_root / "degrade" / split
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        elif dataset == "tinyperson":
            dataset_root = data_root / "TinyPerson"
            annotation_path = dataset_root / "annotations" / f"tiny_set_{split}.json"
            image_root = dataset_root / split / "labeled_images"
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        elif dataset == "varroa":
            dataset_root = data_root / "varroa_yolo"
            image_root = dataset_root / "images" / split
            label_root = dataset_root / "labels" / split
            image_paths = sorted(path for path in image_root.iterdir() if path.suffix.lower() in evaluate.IMAGE_EXTENSIONS)
            images, ground_truth = [], {}
            for image_id, image_path in enumerate(image_paths):
                with Image.open(image_path) as source:
                    width, height = source.size
                images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height, "path": image_path})
                boxes = []
                label_path = label_root / f"{image_path.stem}.txt"
                if label_path.is_file():
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        fields = line.split()
                        if len(fields) >= 5:
                            _, cx, cy, box_width, box_height = map(float, fields[:5])
                            boxes.append(np.asarray([(cx - box_width / 2) * width, (cy - box_height / 2) * height, (cx + box_width / 2) * width, (cy + box_height / 2) * height], dtype=np.float32))
                ground_truth[image_id] = boxes
            return images, ground_truth
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        annotations = {}
        for row in payload["annotations"]:
            x, y, width, height = row["bbox"]
            annotations.setdefault(int(row["image_id"]), []).append(np.asarray([x, y, x + width, y + height], dtype=np.float32))
        images = []
        for row in payload["images"]:
            image_path = image_root / row["file_name"]
            if not image_path.is_file():
                image_path = image_root / Path(row["file_name"]).name
            if not image_path.is_file():
                continue
            images.append({**row, "path": image_path})
        return images, annotations

    def evaluate_dataset(dataset, split):
        images, ground_truth = load_local_dataset(dataset, split)
        predictions = evaluate.predictions(
            session,
            images,
            evaluate.image_size(dataset),
            batch_size,
            confidence,
            nms_iou,
            max_det,
        )
        metrics = evaluate.evaluate(images, ground_truth, predictions)
        row = {
            "dataset": dataset,
            "split": split,
            "images": len(images),
            "ground_truth_boxes": sum(map(len, ground_truth.values())),
            "provider": "TensorrtExecutionProvider",
            "batch_size": batch_size,
            "conf": confidence,
            "nms_iou": nms_iou,
            **metrics,
        }
        output = run_root / "metrics" / dataset / f"{split}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        return row

    return evaluate_dataset


@app.cell
def _(evaluate_dataset, splits):
    levirship_metrics = {split: evaluate_dataset("levirship", split) for split in splits}
    levirship_metrics
    return levirship_metrics


@app.cell
def _(evaluate_dataset, splits):
    tinyperson_metrics = {split: evaluate_dataset("tinyperson", split) for split in splits}
    tinyperson_metrics
    return tinyperson_metrics


@app.cell
def _(evaluate_dataset, splits):
    varroa_metrics = {split: evaluate_dataset("varroa", split) for split in splits}
    varroa_metrics
    return varroa_metrics


@app.cell

def _(checkpoint, datasets, json, levirship_metrics, nms_iou, onnx_path, run_root, tinyperson_metrics, trt_cache_root, varroa_metrics):
    manifest = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "run_root": str(run_root),
        "trt_engine_cache": str(trt_cache_root),
        "datasets": list(datasets),
        "provider": "TensorrtExecutionProvider",
        "nms_iou": nms_iou,
        "metrics": {
            "levirship": levirship_metrics,
            "tinyperson": tinyperson_metrics,
            "varroa": varroa_metrics,
        },
    }
    manifest_path = run_root / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return manifest


if __name__ == "__main__":
    app.run()
