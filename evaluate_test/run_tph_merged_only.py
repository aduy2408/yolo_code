import argparse
import json
import subprocess
from pathlib import Path

from train_scripts.train_all_tinyperson import evaluate_tiny_benchmark, nms_detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--patience", required=True)
    parser.add_argument("--workers", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--split-seed", required=True)
    parser.add_argument("--image-size", required=True)
    parser.add_argument("--model-yaml", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--nms-iou", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tph-root", type=Path, required=True)
    parser.add_argument("--tph-tinyperson", type=Path, required=True)
    args = parser.parse_args()

    test_dir = Path(args.data_root) / "tinyperson_test_corner_sw640_sh512"
    source_gt = Path("/marimo/TinyPerson/annotations/task/tiny_set_test_all.json")
    records = json.loads((test_dir / "corner_manifest.json").read_text())["images"]
    gt_data = json.loads(source_gt.read_text())
    original_id_by_name = {item["file_name"]: item["id"] for item in gt_data["images"]}
    record_by_prediction_id = {f"test_{item['image_id']}": item for item in records}

    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        "/tmp/uv-venv/bin/python",
        "val.py",
        "--data",
        str(Path(args.data_root) / "tinyperson_split_42_corner_sw640_sh512" / "tinyperson.yaml"),
        "--weights",
        str(args.tph_tinyperson),
        "--img",
        str(args.image_size),
        "--batch-size",
        "8",
        "--device",
        "0",
        "--task",
        "test",
        "--iou-thres",
        str(args.nms_iou),
        "--project",
        str(args.output / "tph_native"),
        "--name",
        "test",
        "--exist-ok",
        "--verbose",
        "--save-json",
    ]
    result = subprocess.run(command, cwd=args.tph_root, text=True, capture_output=True, check=False)
    (args.output / "tph_val_stdout.log").write_text(result.stdout)
    (args.output / "tph_val_stderr.log").write_text(result.stderr)
    prediction_files = sorted((args.output / "tph_native").rglob("*_predictions.json"))
    if not prediction_files:
        raise RuntimeError(f"TPH did not produce predictions JSON, exit={result.returncode}: {result.stderr[-2000:]}")

    by_original = {}
    for prediction in json.loads(prediction_files[-1].read_text()):
        record = record_by_prediction_id.get(str(prediction["image_id"]))
        if record is None:
            continue
        original_id = int(original_id_by_name[record["file_name"]])
        x, y, width, height = prediction["bbox"]
        corner = record["corner"]
        item = {"xyxy": [x + corner[0], y + corner[1], x + width + corner[0], y + height + corner[1]], "score": float(prediction["score"])}
        by_original.setdefault(original_id, []).append(item)

    predictions = []
    for original_id, items in by_original.items():
        for item in nms_detections(items, float(args.nms_iou)):
            x1, y1, x2, y2 = item["xyxy"]
            predictions.append({"image_id": original_id, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": item["score"]})

    prediction_path = args.output / "tph_test_merged_predictions.json"
    prediction_path.write_text(json.dumps(predictions))
    metrics = evaluate_tiny_benchmark(prediction_path, source_gt)
    summary = {"model": "tph", **metrics}
    (args.output / "merged_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
