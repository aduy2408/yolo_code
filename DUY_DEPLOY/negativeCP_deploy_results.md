# negativeCP_deploy_results

Deployment test results for `nc`. `PT` is the native `.pt` result; `FP32` and `FP16` are TensorRT engine results. `nms_iou: 0.5` for every row. TinyPerson uses corrected TinyBenchmark (`AP50`, `AP75`, `mAP50-75`); LEVIR-Ship and Varroa use native COCOeval (`AP50-all`, `mAP50-95-all`). Mean/std is across available seeds only; `n` is the number of seeds.
YOLOv11 NC-R4 and YOLOv8 NC-R2 on LEVIR-Ship use corrected parity PT values from `docs/reports/copy_paste.md` section 12.2. The TinyPerson PT, FP32, and FP16 rows are measured deployment results from the Colab export/evaluation run. Missing LEVIR-Ship FP32/FP16 rows remain reconstructed estimates and are not newly measured TensorRT artifacts.

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

## TinyPerson canonical merged mAP50-95

The TinyPerson evaluator's complete protocol is IoU `0.50:0.05:0.95`. The values below are the verified canonical merged metrics by model/config/seed; they are not backend-specific FP32/FP16 measurements. The deployment rows above retain their legacy `mAP50-75` field because the lost backend artifacts do not contain a verified per-backend `mAP50-95`.

| Model | Config | Seed | Canonical merged mAP50-95 |
|---|---|---:|---:|
| YOLO11 | r2 | s43 | 19.95% |
| YOLO11 | r2 | s44 | 19.52% |
| YOLO11 | r4 | s43 | 19.88% |
| YOLO11 | r4 | s44 | 19.77% |
| YOLOv8 | r2 | s43 | 20.04% |
| YOLOv8 | r2 | s44 | 19.72% |
| YOLOv8 | r4 | s43 | 19.81% |
| YOLOv8 | r4 | s44 | 20.32% |

Mean ± sample std:

| Model | Config | Canonical merged mAP50-95 |
|---|---|---:|
| YOLO11 | r2 | 19.74% ± 0.30 pp |
| YOLO11 | r4 | 19.82% ± 0.08 pp |
| YOLOv8 | r2 | 19.88% ± 0.23 pp |
| YOLOv8 | r4 | 20.06% ± 0.36 pp |

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
