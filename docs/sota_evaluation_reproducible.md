# SOTA evaluation: reproducible run note

## Mục đích

File này là runbook evaluation-only cho bốn SOTA family:

- STW-YOLO
- TPH-YOLOv5
- FCOS-SET
- FCOS-SR-TOD

Không train lại. Không đổi checkpoint. Không suy luận model/config/dataset từ tên file.

Machine-readable manifest: `evaluate_test/sota_evaluation_manifest.json`.

## Contract chung

| Field | Value |
|---|---|
| Dataset | TinyPerson |
| Split seed | `42` |
| Training seeds | `42`, `43` |
| Image size | `640`; source windows `640x512` |
| NMS IoU | `0.50` |
| Native metric | corner-window detector val/test |
| Merged metric | map corner predictions to source-image IDs, NMS, official TinyBenchmark evaluator |
| Small metric | AP at IoU `0.50`, area `small`; not COCO AP-small averaged over IoUs |
| YOLO Python | `/tmp/uv-venv/bin/python` |
| MMDetection Python | `/marimo/mmdet-venv/bin/python` |
| Dataset root | `/marimo/TinyPerson` |
| HF upload | required for a newly executed evaluation; verify remote paths before reporting complete |

## Exact existing artifacts

### STW-YOLO

- seed42 checkpoint: `duyle2408/stw-yolo-runs:runs/tinyperson/seed_42/weights/best.pt`
- seed43 checkpoint: `duyle2408/stw-yolo-standard-seed43-backfill-runs:runs/tinyperson/seed_43/weights/best.pt`
- model config: `STW-YOLO/Lib/p2_rp5_yolo12s.yaml`
- seed42 merged artifact: `duyle2408/stw-tph-tinyperson-corrected-20261015-runs:tinyperson/seed42/corrected_gpu_r4/artifacts/final_results.json`
- legacy seed42 head needs the recorded `box_detail` compatibility fallback; weights are unchanged.

### TPH-YOLOv5

- checkpoint repo: `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`
- paths: `runs/tinyperson/seed_{42,43}/weights/best.pt`
- source commit: `bf07e94a1f3c189ea440651e96a621723279ee02`
- TPH commit: `052dfeb375e51756f17e9ca4f96b7e3e3a7cf3c4`
- seed42 merged artifact: `duyle2408/stw-tph-tinyperson-corrected-20261015-runs:tinyperson/seed42/corrected_gpu_r4/artifacts/final_results.json`

### FCOS-SET

- seed42 checkpoint/predictions: `duyle2408/set-fcos-stw-protocol-runs:tinyperson/mosaic/patience15/seed42/fcos_set/`
- seed42 merged detection list: `test_results/test/metrics_merged_detections.json`
- seed43 verified merged artifact: `duyle2408/fcos-tinyperson-eval-20261015-runs:tinyperson/seed43/fcos_set/`

### FCOS-SR-TOD

- seed42 run: `duyle2408/srtod-set-mosaic-matrix-runs:srtod/tinyperson/fcos/standard_mosaic/seed42_backfill_20261105/`
- seed43 verified merged artifact: `duyle2408/fcos-tinyperson-eval-20261015-runs:tinyperson/seed43/fcos_sr_tod/`
- seed42 `bucket_eval/bucket_metrics.json` is native bucket output; it must not be renamed to `test_merged/AP50-Small`.

## Commands

### STW/TPH native val/test

Use the existing explicit runner; it never trains or downloads:

```bash
/tmp/uv-venv/bin/python evaluate_test/evaluate_stw_tph_checkpoints.py \
  --backend stw \
  --checkpoint stw42=/path/to/stw42/best.pt \
  --checkpoint stw43=/path/to/stw43/best.pt \
  --data-yaml tinyperson=/path/to/tinyperson.yaml \
  --data-yaml levirship=/path/to/levirship.yaml \
  --data-yaml varroa=/path/to/varroa.yaml \
  --output /path/to/output \
  --device 0 --batch-size 8 --workers 8
```

For TPH, use `--backend tph --tph-root /path/to/tph-yolov5` and explicit TPH checkpoints.

### TPH merged TinyPerson

```bash
/tmp/uv-venv/bin/python evaluate_test/run_tph_merged_only.py \
  --epochs 100 --patience 0 --workers 8 --seed 43 --split-seed 42 \
  --image-size 640 \
  --model-yaml yolov5l-xs-tph \
  --data-root /marimo/yolo_code/datasets/tinyperson_eval \
  --hf-repo-id duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs \
  --nms-iou 0.5 \
  --output /path/to/tph_seed43_merged \
  --tph-root /marimo/yolo_code/vendor/tph-yolov5 \
  --tph-tinyperson /path/to/runs/tinyperson/seed_43/weights/best.pt
```

The data preparation must already have produced:

```text
/marimo/yolo_code/datasets/tinyperson_eval/tinyperson_test_corner_sw640_sh512/corner_manifest.json
/marimo/yolo_code/datasets/tinyperson_eval/tinyperson_split_42_corner_sw640_sh512/tinyperson.yaml
```

### MMDetection merged TinyPerson

```bash
/marimo/mmdet-venv/bin/python \
  /marimo/mmdet_code/evaluate_tinyperson_metrics.py \
  --res /path/to/predictions_coco.json \
  --corner-gt /marimo/TinyPerson/annotations/corner/task/tiny_set_test_sw640_sh512_all.json \
  --merged-gt /marimo/TinyPerson/annotations/task/tiny_set_test_all.json \
  --out /path/to/test_merged_metrics.json
```

This command is valid only when `predictions_coco.json` contains corner-window image IDs and corner-coordinate boxes. A native `bucket_metrics.json` scalar is not an input substitute.

## Reporting rule

- `AP50` and `mAP50-95` can be aggregated across seeds only when the split/protocol matches.
- `test_merged/AP50-Small` is a separate scalar from native `test_size/AP50-Small`.
- If a verified artifact contains only native buckets, report the native field with its namespace.
- If the merged scalar is absent, write `null` in the manifest; do not report zero and do not infer it.
- A completed training run is not automatically a completed merged evaluation run.

The current artifact/result map is recorded in `docs/reports/sota_results_reproducible_20261106.json`.
