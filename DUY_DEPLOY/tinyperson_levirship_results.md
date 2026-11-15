# tinyperson_levirship_results

Full deployment result tables for **TinyPerson** and **LEVIR-Ship** across OACP and Negative Copy-Paste. Includes every saved PT, TensorRT FP16, and TensorRT FP32 result with available seeds, plus mean ± sample standard deviation across seeds. TinyPerson metrics are final original-image merged metrics from TinyBenchmark; LEVIR-Ship metrics are native COCOeval metrics. All rows use `nms_iou: 0.5`.
The old YOLOv8 NC-R2 LEVIR seed-43 rows around 62% AP50 were superseded by a corrected retrain/backfill. The corrected TinyPerson PT, FP32, and FP16 rows are measured deployment results from the Colab export/evaluation run. Missing LEVIR-Ship FP32/FP16 rows for corrected checkpoints remain reconstructed estimates and are not newly measured TensorRT artifacts.
YOLOv11 NC-R4 LEVIR rows use corrected parity values from `docs/reports/copy_paste.md` section 12.2. PT rows are parity values; FP32/FP16 rows for those corrected LEVIR checkpoints are reconstructed estimates from the lost export results, not newly measured TensorRT artifacts.
## Verified TinyPerson Jetson `.pt` merge — mAP50-95

These rows were re-evaluated from the archived Jetson prediction artifacts in `DUY_DEPLOY/results/augmentation_runs_20261106.zip`. Corner-window predictions were merged in original-image coordinates with class-agnostic NMS at `IoU=0.5`; existing archived merged predictions were re-scored. COCOeval IoU range is `0.50:0.05:0.95`, `max_det=300`, and `nms_iou: 0.5`.

### Per-seed

|Variant|Model|Seed|AP50-all|AP50-Small|mAP50-95-all|mAP50-95-Small|
|---|---|---:|---:|---:|---:|---:|
|NC-R2|YOLOv8|43|46.39%|45.60%|16.64%|15.78%|
|NC-R2|YOLOv8|44|46.08%|45.20%|16.56%|15.48%|
|NC-R2|YOLO11|43|46.66%|45.40%|16.55%|15.48%|
|NC-R2|YOLO11|44|45.53%|44.46%|16.18%|15.13%|
|NC-R4|YOLOv8|43|46.05%|45.09%|16.46%|15.68%|
|NC-R4|YOLOv8|44|46.59%|45.64%|16.80%|15.92%|
|NC-R4|YOLO11|43|46.06%|45.05%|16.50%|15.56%|
|NC-R4|YOLO11|44|45.99%|44.98%|16.34%|15.37%|
|OACP-R2|YOLOv8|43|43.49%|42.86%|15.09%|14.59%|
|OACP-R2|YOLOv8|44|43.18%|42.28%|15.01%|14.36%|
|OACP-R2|YOLO11|43|43.25%|42.69%|15.22%|14.51%|
|OACP-R2|YOLO11|44|43.98%|42.88%|15.31%|14.58%|

### Mean ± sample std across seeds

|Variant|Model|n|AP50-all|AP50-Small|mAP50-95-all|mAP50-95-Small|
|---|---|---:|---:|---:|---:|---:|
|NC-R2|YOLOv8|2|46.23% ± 0.22 pp|45.40% ± 0.28 pp|16.60% ± 0.06 pp|15.63% ± 0.21 pp|
|NC-R2|YOLO11|2|46.09% ± 0.80 pp|44.93% ± 0.66 pp|16.36% ± 0.26 pp|15.30% ± 0.25 pp|
|NC-R4|YOLOv8|2|46.32% ± 0.38 pp|45.36% ± 0.39 pp|16.63% ± 0.24 pp|15.80% ± 0.17 pp|
|NC-R4|YOLO11|2|46.02% ± 0.05 pp|45.01% ± 0.05 pp|16.42% ± 0.11 pp|15.46% ± 0.13 pp|
|OACP-R2|YOLOv8|2|43.33% ± 0.22 pp|42.57% ± 0.41 pp|15.05% ± 0.05 pp|14.47% ± 0.16 pp|
|OACP-R2|YOLO11|2|43.61% ± 0.51 pp|42.79% ± 0.13 pp|15.27% ± 0.07 pp|14.55% ± 0.05 pp|


## OACP

## Per-seed — tinyperson

|Backend|Config|Model|Seed|AP50|AP75|mAP50-75|AP50-tiny|AP50-small|
|---|---|---|---|---|---|---|---|---|
|fp16|-|yolov11|s43|51.69%|8.49%|29.74%|48.89%|65.99%|
|fp16|-|yolov11|s44|52.40%|8.51%|30.09%|49.13%|66.74%|
|fp32|-|yolov11|s43|51.65%|8.44%|29.73%|48.83%|65.90%|
|fp32|-|yolov11|s44|52.46%|8.48%|30.10%|49.19%|67.10%|
|pt|-|yolov11|s43|51.52%|8.42%|29.71%|48.82%|65.68%|
|pt|-|yolov11|s44|52.12%|8.43%|29.91%|48.81%|66.76%|
|fp16|-|yolov8|s43|51.93%|8.11%|29.68%|49.28%|65.25%|
|fp16|-|yolov8|s44|51.72%|8.11%|29.44%|48.53%|65.66%|
|fp32|-|yolov8|s43|51.93%|8.10%|29.68%|49.27%|65.02%|
|fp32|-|yolov8|s44|51.79%|8.13%|29.48%|48.61%|65.67%|
|pt|-|yolov8|s43|51.55%|8.10%|29.56%|48.83%|64.88%|
|pt|-|yolov8|s44|51.53%|8.16%|29.39%|48.30%|65.63%|

## Mean ± std across seeds — tinyperson

|Dataset|Backend|Config|Model|n|AP50 mean±std|mAP mean±std|APsmall mean±std|AP75 mean±std|APtiny mean±std|
|---|---|---|---|---|---|---|---|---|
|tinyperson|fp16|-|yolov11|2|52.05% ± 0.50 pp|29.91% ± 0.24 pp|66.36% ± 0.53 pp|8.50% ± 0.01 pp|49.01% ± 0.17 pp|
|tinyperson|fp16|-|yolov8|2|51.83% ± 0.15 pp|29.56% ± 0.17 pp|65.45% ± 0.29 pp|8.11% ± 0.00 pp|48.90% ± 0.54 pp|
|tinyperson|fp32|-|yolov11|2|52.06% ± 0.58 pp|29.91% ± 0.27 pp|66.50% ± 0.85 pp|8.46% ± 0.03 pp|49.01% ± 0.26 pp|
|tinyperson|fp32|-|yolov8|2|51.86% ± 0.10 pp|29.58% ± 0.14 pp|65.35% ± 0.46 pp|8.12% ± 0.02 pp|48.94% ± 0.47 pp|
|tinyperson|pt|-|yolov11|2|51.82% ± 0.43 pp|29.81% ± 0.14 pp|66.22% ± 0.76 pp|8.42% ± 0.01 pp|48.81% ± 0.01 pp|
|tinyperson|pt|-|yolov8|2|51.54% ± 0.02 pp|29.47% ± 0.12 pp|65.26% ± 0.53 pp|8.13% ± 0.04 pp|48.56% ± 0.38 pp|

## Per-seed — levirship

|Backend|Config|Model|Seed|AP50-all|mAP50-95-all|AP50-Small|mAP50-95-Small|
|---|---|---|---|---|---|---|---|
|fp16|-|yolov11|s43|73.24%|27.58%|73.34%|27.65%|
|fp16|-|yolov11|s44|71.88%|26.33%|71.64%|26.24%|
|fp32|-|yolov11|s43|73.24%|27.59%|73.35%|27.64%|
|fp32|-|yolov11|s44|71.89%|26.34%|71.68%|26.26%|
|pt|-|yolov11|s43|73.35%|27.61%|73.43%|27.67%|
|pt|-|yolov11|s44|71.89%|26.34%|71.71%|26.28%|
|fp16|-|yolov8|s43|76.10%|27.48%|76.14%|27.44%|
|fp16|-|yolov8|s44|74.16%|28.17%|74.38%|28.31%|
|fp32|-|yolov8|s43|76.21%|27.47%|76.21%|27.43%|
|fp32|-|yolov8|s44|74.18%|28.19%|74.40%|28.37%|
|pt|-|yolov8|s43|76.08%|27.47%|76.11%|27.43%|
|pt|-|yolov8|s44|74.18%|28.18%|74.39%|28.37%|

## Mean ± std across seeds — levirship

|Dataset|Backend|Config|Model|n|AP50 mean±std|mAP mean±std|APsmall mean±std|mAPsmall mean±std|
|---|---|---|---|---|---|---|---|---|
|levirship|fp16|-|yolov11|2|72.56% ± 0.96 pp|26.96% ± 0.89 pp|72.49% ± 1.20 pp|26.94% ± 1.00 pp|
|levirship|fp16|-|yolov8|2|75.13% ± 1.38 pp|27.82% ± 0.49 pp|75.26% ± 1.24 pp|27.88% ± 0.61 pp|
|levirship|fp32|-|yolov11|2|72.57% ± 0.96 pp|26.97% ± 0.88 pp|72.52% ± 1.18 pp|26.95% ± 0.98 pp|
|levirship|fp32|-|yolov8|2|75.19% ± 1.44 pp|27.83% ± 0.51 pp|75.30% ± 1.28 pp|27.90% ± 0.66 pp|
|levirship|pt|-|yolov11|2|72.62% ± 1.03 pp|26.98% ± 0.90 pp|72.57% ± 1.21 pp|26.97% ± 0.98 pp|
|levirship|pt|-|yolov8|2|75.13% ± 1.34 pp|27.83% ± 0.50 pp|75.25% ± 1.22 pp|27.90% ± 0.66 pp|

## Negative Copy-Paste

## Per-seed — tinyperson

|Backend|Config|Model|Seed|AP50|AP75|mAP50-75|AP50-tiny|AP50-small|
|---|---|---|---|---|---|---|---|---|
|fp16|r2|yolov11|s43|55.12%|9.63%|32.39%|51.12%|69.21%|
|fp16|r2|yolov11|s44|53.87%|9.41%|31.63%|50.73%|68.14%|
|fp32|r2|yolov11|s43|55.15%|9.65%|32.39%|51.15%|69.33%|
|fp32|r2|yolov11|s44|53.90%|9.44%|31.66%|50.77%|68.17%|
|pt|r2|yolov11|s43|55.25%|9.58%|32.50%|51.29%|69.31%|
|pt|r2|yolov11|s44|53.88%|9.49%|31.69%|50.82%|67.85%|
|fp16|r4|yolov11|s43|54.34%|9.94%|32.28%|50.85%|68.50%|
|fp16|r4|yolov11|s44|54.70%|9.41%|32.16%|51.52%|68.76%|
|fp32|r4|yolov11|s43|54.34%|9.93%|32.28%|50.80%|68.52%|
|fp32|r4|yolov11|s44|54.67%|9.41%|32.15%|51.49%|68.71%|
|pt|r4|yolov11|s43|54.31%|9.84%|32.20%|51.00%|68.63%|
|pt|r4|yolov11|s44|54.81%|9.42%|32.12%|51.63%|68.71%|
|fp16|r2|yolov8|s43|54.84%|10.11%|32.61%|51.70%|69.11%|
|fp16|r2|yolov8|s44|54.29%|9.54%|31.83%|51.30%|67.81%|
|fp32|r2|yolov8|s43|54.85%|10.09%|32.60%|51.71%|69.18%|
|fp32|r2|yolov8|s44|54.29%|9.54%|31.84%|51.32%|67.86%|
|pt|r2|yolov8|s43|54.89%|10.12%|32.56%|51.65%|69.18%|
|pt|r2|yolov8|s44|54.23%|9.60%|31.89%|51.13%|67.98%|
|fp16|r4|yolov8|s43|54.63%|9.81%|32.21%|51.64%|68.13%|
|fp16|r4|yolov8|s44|55.40%|10.02%|32.93%|52.18%|69.17%|
|fp32|r4|yolov8|s43|54.61%|9.76%|32.21%|51.63%|68.06%|
|fp32|r4|yolov8|s44|55.39%|10.06%|32.94%|52.24%|69.21%|
|pt|r4|yolov8|s43|54.44%|9.74%|32.14%|51.10%|68.89%|
|pt|r4|yolov8|s44|55.27%|10.09%|32.83%|51.92%|68.81%|

## Mean ± std across seeds — tinyperson

|Dataset|Backend|Config|Model|n|AP50 mean±std|mAP mean±std|APsmall mean±std|AP75 mean±std|APtiny mean±std|
|---|---|---|---|---|---|---|---|---|---|
|tinyperson|fp16|r2|yolov11|2|54.49% ± 0.89 pp|32.01% ± 0.54 pp|68.68% ± 0.76 pp|9.52% ± 0.16 pp|50.92% ± 0.28 pp|
|tinyperson|fp16|r2|yolov8|2|54.56% ± 0.39 pp|32.22% ± 0.55 pp|68.46% ± 0.91 pp|9.83% ± 0.40 pp|51.50% ± 0.28 pp|
|tinyperson|fp16|r4|yolov11|2|54.52% ± 0.25 pp|32.22% ± 0.09 pp|68.63% ± 0.18 pp|9.68% ± 0.38 pp|51.19% ± 0.47 pp|
|tinyperson|fp16|r4|yolov8|2|55.02% ± 0.55 pp|32.57% ± 0.50 pp|68.65% ± 0.74 pp|9.92% ± 0.15 pp|51.91% ± 0.38 pp|
|tinyperson|fp32|r2|yolov11|2|54.53% ± 0.89 pp|32.02% ± 0.51 pp|68.75% ± 0.82 pp|9.55% ± 0.15 pp|50.96% ± 0.27 pp|
|tinyperson|fp32|r2|yolov8|2|54.57% ± 0.40 pp|32.22% ± 0.54 pp|68.52% ± 0.93 pp|9.82% ± 0.39 pp|51.51% ± 0.28 pp|
|tinyperson|fp32|r4|yolov11|2|54.51% ± 0.23 pp|32.22% ± 0.09 pp|68.62% ± 0.13 pp|9.67% ± 0.37 pp|51.15% ± 0.49 pp|
|tinyperson|fp32|r4|yolov8|2|55.00% ± 0.55 pp|32.57% ± 0.52 pp|68.63% ± 0.81 pp|9.91% ± 0.21 pp|51.94% ± 0.43 pp|
|tinyperson|pt|r2|yolov11|2|54.57% ± 0.97 pp|32.09% ± 0.57 pp|68.58% ± 1.03 pp|9.54% ± 0.07 pp|51.05% ± 0.33 pp|
|tinyperson|pt|r2|yolov8|2|54.56% ± 0.47 pp|32.23% ± 0.47 pp|68.58% ± 0.85 pp|9.86% ± 0.36 pp|51.39% ± 0.37 pp|
|tinyperson|pt|r4|yolov11|2|54.56% ± 0.35 pp|32.16% ± 0.06 pp|68.67% ± 0.05 pp|9.63% ± 0.30 pp|51.32% ± 0.45 pp|
|tinyperson|pt|r4|yolov8|2|54.86% ± 0.59 pp|32.49% ± 0.49 pp|68.85% ± 0.06 pp|9.92% ± 0.25 pp|51.51% ± 0.58 pp|

## Per-seed — levirship

|Backend|Config|Model|Seed|AP50-all|mAP50-95-all|AP50-Small|mAP50-95-Small|
|---|---|---|---|---|---|---|---|
|fp16|r2|yolov11|s43|76.30%|28.42%|76.53%|28.55%|
|fp16|r2|yolov11|s44|75.91%|28.27%|76.69%|28.49%|
|fp32|r2|yolov11|s43|76.48%|28.52%|76.73%|28.66%|
|fp32|r2|yolov11|s44|75.92%|28.30%|76.70%|28.53%|
|pt|r2|yolov11|s43|76.37%|28.56%|76.63%|28.72%|
|pt|r2|yolov11|s44|76.08%|28.28%|76.89%|28.52%|
|fp16|r4|yolov11|s43|79.04%|29.31%|79.41%|—|
|fp16|r4|yolov11|s44|79.28%|30.11%|78.22%|—|
|fp32|r4|yolov11|s43|79.18%|29.46%|79.07%|—|
|fp32|r4|yolov11|s44|79.41%|30.27%|78.91%|—|
|pt|r4|yolov11|s43|79.12%|29.38%|79.28%|—|
|pt|r4|yolov11|s44|79.33%|30.32%|78.57%|—|
|fp16|r2|yolov8|s43|80.54%|30.02%|83.18%|—|
|fp16|r2|yolov8|s44|80.91%|30.48%|85.21%|—|
|fp32|r2|yolov8|s43|80.69%|30.14%|83.53%|—|
|fp32|r2|yolov8|s44|80.86%|30.27%|84.88%|—|
|pt|r2|yolov8|s43|80.63%|30.10%|83.40%|—|
|pt|r2|yolov8|s44|80.80%|30.35%|85.03%|—|
|fp16|r4|yolov8|s43|78.48%|29.93%|78.91%|30.08%|
|fp16|r4|yolov8|s44|78.86%|30.40%|79.39%|30.68%|
|fp32|r4|yolov8|s43|78.47%|29.89%|78.89%|30.04%|
|fp32|r4|yolov8|s44|78.72%|30.33%|79.28%|30.61%|
|pt|r4|yolov8|s43|78.47%|29.89%|78.90%|30.02%|
|pt|r4|yolov8|s44|78.81%|30.41%|79.38%|30.67%|

## Mean ± std across seeds — levirship

|Dataset|Backend|Config|Model|n|AP50 mean±std|mAP mean±std|APsmall mean±std|mAPsmall mean±std|
|---|---|---|---|---|---|---|---|---|
|levirship|fp16|r2|yolov11|2|76.11% ± 0.28 pp|28.35% ± 0.10 pp|76.61% ± 0.11 pp|28.52% ± 0.04 pp|
|levirship|fp16|r2|yolov8|2|80.72% ± 0.26 pp|30.25% ± 0.33 pp|84.19% ± 1.44 pp|—|
|levirship|fp16|r4|yolov11|2|79.16% ± 0.17 pp|29.71% ± 0.57 pp|78.81% ± 0.84 pp|—|
|levirship|fp16|r4|yolov8|2|78.67% ± 0.26 pp|30.16% ± 0.33 pp|79.15% ± 0.34 pp|30.38% ± 0.42 pp|
|levirship|fp32|r2|yolov11|2|76.20% ± 0.39 pp|28.41% ± 0.16 pp|76.71% ± 0.02 pp|28.60% ± 0.09 pp|
|levirship|fp32|r2|yolov8|2|80.78% ± 0.12 pp|30.20% ± 0.09 pp|84.20% ± 0.95 pp|—|
|levirship|fp32|r4|yolov11|2|79.30% ± 0.16 pp|29.87% ± 0.57 pp|78.99% ± 0.11 pp|—|
|levirship|fp32|r4|yolov8|2|78.59% ± 0.17 pp|30.11% ± 0.31 pp|79.09% ± 0.28 pp|30.32% ± 0.40 pp|
|levirship|pt|r2|yolov11|2|76.23% ± 0.21 pp|28.42% ± 0.20 pp|76.76% ± 0.19 pp|28.62% ± 0.14 pp|
|levirship|pt|r2|yolov8|2|80.71% ± 0.12 pp|30.23% ± 0.18 pp|84.22% ± 1.15 pp|—|
|levirship|pt|r4|yolov11|2|79.23% ± 0.15 pp|29.85% ± 0.66 pp|78.93% ± 0.50 pp|—|
|levirship|pt|r4|yolov8|2|78.64% ± 0.24 pp|30.15% ± 0.37 pp|79.14% ± 0.34 pp|30.35% ± 0.46 pp|

## Compact baseline comparison

These are the recovered OACP-R2 runtime measurements. The same baseline
runtime is attached to the NC-R2/NC-R4 rows because augmentation changes the
checkpoint weights and accuracy, not the exported model graph or deployment
backend. Each row reports `AP50 / AP50-Small / mAP50-95` separately from
`FPS end-to-end / latency / inference FPS`. Latency is median full-pipeline
latency in milliseconds; inference FPS is `1000 / model-only median latency`.
`nms_iou: 0.5`.

### TinyPerson

| Model | Backend | AP50 | AP50-Small | mAP50-95 | FPS end-to-end | Latency (ms) | Inference FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8 | PT FP32 | 43.33% | 42.57% | 15.05% | 26.69 | 37.47 | 43.44 |
| YOLOv8 | PT FP16 | 43.33% | 42.57% | 15.05% | 32.75 | 30.56 | 50.25 |
| YOLOv8 | TensorRT FP32 | 43.44% | 42.74% | 15.03% | 29.02 | 34.47 | 51.28 |
| YOLOv8 | TensorRT FP16 | 43.40% | 42.70% | 15.02% | 44.45 | 22.58 | 95.06 |
| YOLO11 | PT FP32 | 43.61% | 42.79% | 15.27% | 27.82 | 35.95 | 36.13 |
| YOLO11 | PT FP16 | 43.61% | 42.79% | 15.28% | 32.10 | 31.17 | 44.56 |
| YOLO11 | TensorRT FP32 | 43.71% | 42.87% | 15.29% | 28.39 | 35.24 | 48.17 |
| YOLO11 | TensorRT FP16 | 43.63% | 42.84% | 15.28% | 43.56 | 23.02 | 85.69 |
| NC-R2 | YOLOv8 | PT FP32 | 54.56% | 68.58% | 32.23% | 26.69 | 37.47 | 43.44 |
| NC-R2 | YOLOv8 | PT FP16 | 54.56% | 68.58% | 32.23% | 32.75 | 30.56 | 50.25 |
| NC-R2 | YOLOv8 | TensorRT FP32 | — | — | — | 29.02 | 34.47 | 51.28 |
| NC-R2 | YOLOv8 | TensorRT FP16 | — | — | — | 44.45 | 22.58 | 95.06 |
| NC-R2 | YOLO11 | PT FP32 | 54.57% | 68.58% | 32.09% | 27.82 | 35.95 | 36.13 |
| NC-R2 | YOLO11 | PT FP16 | 54.49% | 68.68% | 32.01% | 32.10 | 31.17 | 44.56 |
| NC-R2 | YOLO11 | TensorRT FP32 | 54.53% | 68.75% | 32.02% | 28.39 | 35.24 | 48.17 |
| NC-R2 | YOLO11 | TensorRT FP16 | 54.49% | 68.68% | 32.01% | 43.56 | 23.02 | 85.69 |
| NC-R4 | YOLOv8 | PT FP32 | 54.86% | 68.85% | 32.49% | 26.69 | 37.47 | 43.44 |
| NC-R4 | YOLOv8 | PT FP16 | 55.02% | 68.65% | 32.57% | 32.75 | 30.56 | 50.25 |
| NC-R4 | YOLOv8 | TensorRT FP32 | 55.00% | 68.63% | 32.57% | 29.02 | 34.47 | 51.28 |
| NC-R4 | YOLOv8 | TensorRT FP16 | 55.02% | 68.65% | 32.57% | 44.45 | 22.58 | 95.06 |
| NC-R4 | YOLO11 | PT FP32 | 54.56% | 68.67% | 32.16% | 27.82 | 35.95 | 36.13 |
| NC-R4 | YOLO11 | PT FP16 | 54.52% | 68.63% | 32.22% | 32.10 | 31.17 | 44.56 |
| NC-R4 | YOLO11 | TensorRT FP32 | 54.51% | 68.62% | 32.22% | 28.39 | 35.24 | 48.17 |
| NC-R4 | YOLO11 | TensorRT FP16 | 54.52% | 68.63% | 32.22% | 43.56 | 23.02 | 85.69 |

### LEVIR-Ship

| Model | Backend | AP50 | AP50-Small | mAP50-95 | FPS end-to-end | Latency (ms) | Inference FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8 | PT FP32 | 75.13% | 75.25% | 27.83% | 23.88 | 41.90 | 41.50 |
| YOLOv8 | PT FP16 | 75.13% | 75.26% | 27.82% | 29.56 | 33.84 | 59.38 |
| YOLOv8 | TensorRT FP32 | 75.19% | 75.30% | 27.83% | 26.78 | 37.38 | 76.05 |
| YOLOv8 | TensorRT FP16 | 75.13% | 75.26% | 27.82% | 39.75 | 25.15 | 136.61 |
| YOLO11 | PT FP32 | 72.62% | 72.57% | 26.98% | 25.12 | 39.83 | 44.13 |
| YOLO11 | PT FP16 | 72.56% | 72.49% | 26.96% | 25.60 | 39.09 | 45.77 |
| YOLO11 | TensorRT FP32 | 72.57% | 72.52% | 26.97% | 26.07 | 38.36 | 70.13 |
| YOLO11 | TensorRT FP16 | 72.56% | 72.49% | 26.96% | 38.43 | 26.05 | 119.47 |
| NC-R2 | YOLOv8 | PT FP32 | 80.71% | 84.22% | 30.23% | 23.88 | 41.90 | 41.50 |
| NC-R2 | YOLOv8 | PT FP16 | 80.72%* | 84.19%* | 30.25%* | 29.56 | 33.84 | 59.38 |
| NC-R2 | YOLOv8 | TensorRT FP32 | 80.78%* | 84.20%* | 30.20%* | 26.78 | 37.38 | 76.05 |
| NC-R2 | YOLOv8 | TensorRT FP16 | 80.72%* | 84.19%* | 30.25%* | 39.75 | 25.15 | 136.61 |
| NC-R2 | YOLO11 | Server PT | 80.56 ± 1.76% | 80.18 ± 0.38% | 29.65 ± 0.20% | — | — | — |
| NC-R2 | YOLO11 | Jetson PT | 76.23% | 76.76% | 28.42% | 25.12 | 39.83 | 44.13 |
| NC-R2 | YOLO11 | PT FP16 | 76.11% | 76.61% | 28.35% | 25.60 | 39.09 | 45.77 |
| NC-R2 | YOLO11 | TensorRT FP32 | 76.20% | 76.71% | 28.41% | 26.07 | 38.36 | 70.13 |
| NC-R2 | YOLO11 | TensorRT FP16 | 76.11% | 76.61% | 28.35% | 38.43 | 26.05 | 119.47 |
| NC-R4 | YOLOv8 | PT FP32 | 78.64% | 79.14% | 30.15% | 23.88 | 41.90 | 41.50 |
| NC-R4 | YOLOv8 | PT FP16 | 78.67% | 79.15% | 30.16% | 29.56 | 33.84 | 59.38 |
| NC-R4 | YOLOv8 | TensorRT FP32 | 78.59% | 79.09% | 30.11% | 26.78 | 37.38 | 76.05 |
| NC-R4 | YOLOv8 | TensorRT FP16 | 78.67% | 79.15% | 30.16% | 39.75 | 25.15 | 136.61 |
| NC-R4 | YOLO11 | PT FP32 | 79.23% | 78.93% | 29.85% | 25.12 | 39.83 | 44.13 |
| NC-R4 | YOLO11 | PT FP16 | 79.16%* | 78.81%* | 29.71%* | 25.60 | 39.09 | 45.77 |
| NC-R4 | YOLO11 | TensorRT FP32 | 79.30%* | 78.99%* | 29.87%* | 26.07 | 38.36 | 70.13 |
| NC-R4 | YOLO11 | TensorRT FP16 | 79.16%* | 78.81%* | 29.71%* | 38.43 | 26.05 | 119.47 |
Cột runtime của các dòng NC dùng lại benchmark baseline cùng model/backend; accuracy vẫn là số liệu riêng của Copy-Paste.
`*` Reconstructed accuracy estimate from the lost export result; runtime is
inherited from the matching baseline benchmark.

Runtime values on NC-R2/NC-R4 are inherited from the matching baseline
model/backend benchmark; NC accuracy values remain variant-specific. No
separate NC runtime CSV survives.

## Corrected TensorRT OACP parity rerun

This section is the authoritative deployment result for the corrected OACP
parity rerun. Source:
`duyle2408/oacp-r2-parity-rerun-seed43-44/deployment_trt_results_20261108/`.
It supersedes the older OACP baseline rows above for these TensorRT
evaluations. All rows use `nms_iou: 0.5`.

TinyPerson was evaluated from merged original-image predictions with the
custom TinyBenchmark COCO evaluator (`IoU=0.50:0.05:0.95`). LEVIR-Ship was
evaluated with the Ultralytics evaluator; AP50-small uses the `20²–32²`
bucket. Values are percentages.

### Per-seed test metrics

| Dataset | Model | Precision | Seed | AP50 | AP50-small | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| TinyPerson | YOLOv8 | FP32 | 43 | 54.89% | 67.83% | 19.96% |
| TinyPerson | YOLOv8 | FP32 | 44 | 54.29% | 68.44% | 19.83% |
| TinyPerson | YOLOv8 | FP16 | 43 | 54.77% | 67.76% | 19.88% |
| TinyPerson | YOLOv8 | FP16 | 44 | 54.31% | 68.27% | 19.75% |
| TinyPerson | YOLO11 | FP32 | 43 | 54.51% | 67.44% | 19.93% |
| TinyPerson | YOLO11 | FP32 | 44 | 54.63% | 68.55% | 19.89% |
| TinyPerson | YOLO11 | FP16 | 43 | 54.41% | 67.39% | 19.85% |
| TinyPerson | YOLO11 | FP16 | 44 | 54.53% | 68.69% | 19.81% |
| LEVIR-Ship | YOLOv8 | FP32 | 43 | 79.80% | 87.91% | 29.76% |
| LEVIR-Ship | YOLOv8 | FP32 | 44 | 78.33% | 86.08% | 29.79% |
| LEVIR-Ship | YOLOv8 | FP16 | 43 | 79.15% | 87.10% | 29.51% |
| LEVIR-Ship | YOLOv8 | FP16 | 44 | 78.21% | 85.92% | 29.46% |
| LEVIR-Ship | YOLO11 | FP32 | 43 | 77.66% | 86.10% | 28.51% |
| LEVIR-Ship | YOLO11 | FP32 | 44 | 77.27% | 86.16% | 28.01% |
| LEVIR-Ship | YOLO11 | FP16 | 43 | 77.20% | 86.17% | 28.37% |
| LEVIR-Ship | YOLO11 | FP16 | 44 | 77.00% | 85.91% | 28.15% |

### Mean ± sample std across seeds

| Dataset | Model | Precision | AP50 | AP50-small | mAP50-95 |
|---|---|---|---:|---:|---:|
| TinyPerson | YOLOv8 | FP32 | 54.59% ± 0.43 pp | 68.13% ± 0.43 pp | 19.89% ± 0.09 pp |
| TinyPerson | YOLOv8 | FP16 | 54.54% ± 0.32 pp | 68.01% ± 0.36 pp | 19.81% ± 0.09 pp |
| TinyPerson | YOLO11 | FP32 | 54.57% ± 0.09 pp | 67.99% ± 0.79 pp | 19.91% ± 0.03 pp |
| TinyPerson | YOLO11 | FP16 | 54.47% ± 0.08 pp | 68.04% ± 0.91 pp | 19.83% ± 0.03 pp |
| LEVIR-Ship | YOLOv8 | FP32 | 79.06% ± 1.04 pp | 87.00% ± 1.29 pp | 29.77% ± 0.02 pp |
| LEVIR-Ship | YOLOv8 | FP16 | 78.68% ± 0.66 pp | 86.51% ± 0.83 pp | 29.48% ± 0.04 pp |
| LEVIR-Ship | YOLO11 | FP32 | 77.47% ± 0.28 pp | 86.13% ± 0.04 pp | 28.26% ± 0.35 pp |
| LEVIR-Ship | YOLO11 | FP16 | 77.10% ± 0.14 pp | 86.04% ± 0.18 pp | 28.26% ± 0.16 pp |
