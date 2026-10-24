# YOLO ONNX and TensorRT export

Use `utils/export_yolo.py` to export a local checkpoint or a checkpoint stored in a Hugging Face dataset/model repository.

The script supports:

- `--checkpoint path/to/best.pt` for a local checkpoint.
- `--repo-id USER/REPO --hf-path path/in/repo/weights/best.pt` for an explicit Hugging Face file.
- `--report docs/reports/baseline_main.md` to choose a repository link from a report.
- `--latest-report` to scan the newest Markdown report under `docs/` and `approach+results/`.
- `--format onnx engine` for ONNX and TensorRT.
- Multiple fixed batches with `--batch 1 4 8`, or dynamic shapes with `--dynamic`.
- `--precision fp32|fp16|int8` for explicit precision selection. The legacy `--half` and `--int8` flags remain supported.
- INT8 TensorRT calibration with `--data path/to/dataset.yaml` and optional `--fraction`.

Run in the project ML environment:

```bash
conda run -n ml2 python utils/export_yolo.py \
  --repo-id duyle2408/varroa-yolo-baselines-mosaic-two-seed-runs \
  --hf-path runs/varroa/yolov8/seed_42/weights/best.pt \
  --format onnx engine \
  --batch 1 8 \
  --dynamic \
  --output-dir exports/varroa_yolov8_seed42
```

For report discovery, filter links when a report contains several repositories:

```bash
HF_TOKEN="$HF_TOKEN" conda run -n ml2 python utils/export_yolo.py \
  --report docs/reports/baseline_main.md \
  --repo-pattern varroa \
  --repo-index 0 \
  --format onnx engine
```

`HF_TOKEN` is read from the environment and is never stored by the script. Do not paste tokens into source files, shell history, reports, or command output. The token shared in chat should be revoked and replaced before use.

For INT8 TensorRT, provide a dataset YAML so Ultralytics can calibrate the engine:

```bash
conda run -n ml2 python utils/export_yolo.py \
  --checkpoint exports/baseline_varroa_seed42/yolov8/runs/varroa/yolov8/seed_42/weights/best.pt \
  --format engine \
  --precision int8 \
  --batch 1 \
  --imgsz 640 \
  --device 0 \
  --data datasets/varroa_yolo/varroa.yaml \
  --fraction 0.25 \
  --output-dir exports/varroa_yolov8_int8
```

Use either `--precision` or the legacy `--half`/`--int8` flags, not both. INT8 is calibration-dependent, so compare validation and test metrics against the FP16 engine before adopting it.

Every export is also copied into `--output-dir` with a precision and batch suffix, so FP16 and INT8 outputs do not need to share the same `best.engine` path.

TensorRT export requires a working CUDA/TensorRT installation. `--half` requests FP16, `--int8` requests INT8 calibration, and `--workspace N` sets the TensorRT workspace size when supported by the installed Ultralytics version. Exporting multiple fixed batch sizes creates separate outputs. `--dynamic` creates a dynamic-shape export suitable for variable batch sizes, but the TensorRT engine still needs a compatible optimization profile in the installed exporter.

The script does not evaluate the exported model. Keep the original `.pt` checkpoint and compare ONNX/TensorRT inference separately after export.
