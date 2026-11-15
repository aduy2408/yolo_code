# oacp_deploy_results

Deployment test results for `oacp`. `PT` is the native `.pt` result; `FP32` and `FP16` are TensorRT engine results. `nms_iou: 0.5` for every row. TinyPerson uses corrected TinyBenchmark (`AP50`, `AP75`, `mAP50-75`); LEVIR-Ship and Varroa use native COCOeval (`AP50-all`, `mAP50-95-all`). Mean/std is across available seeds only; `n` is the number of seeds.

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
|---|---|---|---|---|---|---|---|---|---|
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

## Per-seed — varroa

|Backend|Config|Model|Seed|AP50-all|mAP50-95-all|AP50-Small|mAP50-95-Small|
|---|---|---|---|---|---|---|---|
|fp16|r2|yolov11|s43|89.58%|33.11%|87.81%|32.02%|
|fp16|r2|yolov11|s44|92.75%|33.86%|91.83%|32.73%|
|fp32|r2|yolov11|s43|89.58%|33.16%|88.39%|32.33%|
|fp32|r2|yolov11|s44|92.55%|33.92%|91.86%|32.73%|
|pt|r2|yolov11|s43|89.50%|33.05%|88.66%|32.14%|
|pt|r2|yolov11|s44|92.57%|34.20%|91.62%|32.91%|
|fp16|r2|yolov8|s43|90.11%|33.90%|88.40%|32.40%|
|fp16|r2|yolov8|s44|91.62%|32.67%|89.07%|31.11%|
|fp32|r2|yolov8|s43|90.07%|33.92%|88.39%|32.39%|
|fp32|r2|yolov8|s44|91.62%|32.70%|89.07%|31.08%|
|pt|r2|yolov8|s43|90.01%|33.77%|88.10%|32.23%|
|pt|r2|yolov8|s44|91.56%|32.95%|89.02%|31.28%|

## Mean ± std across seeds — varroa

|Dataset|Backend|Config|Model|n|AP50 mean±std|mAP mean±std|APsmall mean±std|mAPsmall mean±std|
|---|---|---|---|---|---|---|---|---|
|varroa|fp16|r2|yolov11|2|91.17% ± 2.24 pp|33.49% ± 0.53 pp|89.82% ± 2.84 pp|32.38% ± 0.50 pp|
|varroa|fp16|r2|yolov8|2|90.87% ± 1.07 pp|33.29% ± 0.87 pp|88.73% ± 0.48 pp|31.75% ± 0.91 pp|
|varroa|fp32|r2|yolov11|2|91.07% ± 2.10 pp|33.54% ± 0.53 pp|90.13% ± 2.45 pp|32.53% ± 0.28 pp|
|varroa|fp32|r2|yolov8|2|90.84% ± 1.09 pp|33.31% ± 0.86 pp|88.73% ± 0.48 pp|31.73% ± 0.93 pp|
|varroa|pt|r2|yolov11|2|91.03% ± 2.17 pp|33.62% ± 0.81 pp|90.14% ± 2.09 pp|32.53% ± 0.55 pp|
|varroa|pt|r2|yolov8|2|90.78% ± 1.09 pp|33.36% ± 0.58 pp|88.56% ± 0.65 pp|31.76% ± 0.68 pp|


## Varroa OACP research table — seeds 42/43

This table combines the earlier Table V.5 values (seed 42) with a fresh
Colab evaluation of the seed-43 checkpoints from
`duyle2408/varroa-yolov8-yolo11-oacp-r2-seed43-44-runs`. Mean and standard
deviation are across the two training seeds; standard deviation is sample
std. `nms_iou: 0.5`.

| Detector | Split | AP50 mean ± std (%) | mAP50:95 mean ± std (%) | AP50-Small mean ± std (%) |
|---|---|---:|---:|---:|
| YOLOv8 + OACP | Validation | 91.82 ± 0.11 | 33.52 ± 0.16 | 87.02 ± 4.08 |
| YOLOv8 + OACP | Test | 90.52 ± 0.55 | 33.90 ± 0.16 | 83.40 ± 4.19 |
| YOLO11 + OACP | Validation | 91.14 ± 1.50 | 33.73 ± 0.22 | 84.04 ± 0.26 |
| YOLO11 + OACP | Test | 91.12 ± 1.04 | 33.54 ± 0.52 | 83.99 ± 2.54 |

### Source values

| Detector | Seed | Val AP50 | Val mAP50:95 | Val AP50-Small | Test AP50 | Test mAP50:95 | Test AP50-Small |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8 + OACP | 42 | 91.74 | 33.41 | 89.90 | 90.13 | 34.01 | 86.36 |
| YOLOv8 + OACP | 43 | 91.90 | 33.64 | 84.13 | 90.91 | 33.79 | 80.44 |
| YOLO11 + OACP | 42 | 90.08 | 33.89 | 83.86 | 91.86 | 33.91 | 85.78 |
| YOLO11 + OACP | 43 | 92.21 | 33.58 | 84.22 | 90.39 | 33.18 | 82.19 |

The seed-43 AP50-Small values came from the native bucket evaluator; the
earlier seed-42 values came from the original Table V.5 evaluation. Treat
AP50-Small comparisons as protocol-sensitive.


## Varroa TensorRT AP50-Small audit

The first no-Docker TensorRT AP50-Small table is withheld from report use.
The native OACP reference table above uses the TinyBenchmark size-bucket
protocol, while the deployment evaluator used a different COCOeval path.
Although AP50-all is close, AP50-Small is not yet protocol-parity verified.
Use the audited OACP reference values until the TensorRT predictions are
rescored with the same size-bucket evaluator.

