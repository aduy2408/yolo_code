# YOLO Augmentation Main Results

Generated from `duyle2408/model-augmentation-seed42-size-metrics-runs` after verifying 32/32 `evaluation_metrics.json` and `size_metrics_complete.json` markers. All values are split-labeled. AP values are reported as fractions, not percentages.

## Required metrics

| # | Dataset | Model | Method | Variant | Mosaic | val/AP50 | val/AP50-Small | val/mAP50-95 | test/AP50 | test/AP50-Small | test/mAP50-95 | Source prefix |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | levir | yolov11 | copy_paste | cp2_single2 | no_mosaic | 0.8216 | 0.8068 | 0.3216 | 0.7807 | 0.8287 | 0.3026 | `yolov11/levir/copy_paste/cp2_single2/no_mosaic/seed_42` |
| 2 | levir | yolov11 | copy_paste | negative_canvas_r2 | no_mosaic | 0.8425 | 0.8476 | 0.3248 | 0.7984 | 0.8442 | 0.2989 | `yolov11/levir/copy_paste/negative_canvas_r2/no_mosaic/seed_42` |
| 3 | levir | yolov11 | copy_paste | negative_canvas_r4 | no_mosaic | 0.8191 | 0.8141 | 0.3222 | 0.7954 | 0.8479 | 0.2944 | `yolov11/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_42` |
| 4 | tinyperson | yolov11 | copy_paste | cp2_single2 | mosaic | 0.5293 | 0.5416 | 0.1968 | 0.5204 | 0.5281 | 0.1886 | `yolov11/tinyperson/copy_paste/cp2_single2/mosaic/seed_42` |
| 5 | tinyperson | yolov11 | copy_paste | negative_canvas_r2 | mosaic | 0.5450 | 0.5020 | 0.2013 | 0.5215 | 0.5229 | 0.1886 | `yolov11/tinyperson/copy_paste/negative_canvas_r2/mosaic/seed_42` |
| 6 | tinyperson | yolov11 | copy_paste | negative_canvas_r4 | mosaic | 0.5472 | 0.5058 | 0.2015 | 0.5179 | 0.5202 | 0.1871 | `yolov11/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_42` |
| 7 | varroa | yolov11 | copy_paste | cp2_single2 | mosaic | 0.8821 | 0.7796 | 0.3254 | 0.8619 | 0.7705 | 0.3084 | `yolov11/varroa/copy_paste/cp2_single2/mosaic/seed_42` |
| 8 | levir | yolov8 | copy_paste | cp2_single2 | no_mosaic | 0.6993 | 0.6873 | 0.2614 | 0.6838 | 0.6578 | 0.2380 | `yolov8/levir/copy_paste/cp2_single2/no_mosaic/seed_42` |
| 9 | levir | yolov8 | copy_paste | negative_canvas_r2 | no_mosaic | 0.8181 | 0.8614 | 0.3227 | 0.7899 | 0.8308 | 0.2961 | `yolov8/levir/copy_paste/negative_canvas_r2/no_mosaic/seed_42` |
| 10 | levir | yolov8 | copy_paste | negative_canvas_r4 | no_mosaic | 0.8012 | 0.7761 | 0.3130 | 0.7652 | 0.8015 | 0.2902 | `yolov8/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_42` |
| 11 | varroa | yolov8 | copy_paste | cp2_single2 | mosaic | 0.8577 | 0.7379 | 0.3225 | 0.8368 | 0.7509 | 0.3006 | `yolov8/varroa/copy_paste/cp2_single2/mosaic/seed_42` |
| 12 | levir | yolov9 | copy_paste | cp2_single2 | no_mosaic | 0.8128 | 0.8310 | 0.3229 | 0.7904 | 0.8279 | 0.2968 | `yolov9/levir/copy_paste/cp2_single2/no_mosaic/seed_42` |
| 13 | levir | yolov9 | copy_paste | negative_canvas_r2 | no_mosaic | 0.8196 | 0.8436 | 0.3251 | 0.7986 | 0.8160 | 0.3095 | `yolov9/levir/copy_paste/negative_canvas_r2/no_mosaic/seed_42` |
| 14 | levir | yolov9 | copy_paste | negative_canvas_r4 | no_mosaic | 0.8152 | 0.8476 | 0.3243 | 0.7863 | 0.8300 | 0.3036 | `yolov9/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_42` |
| 15 | tinyperson | yolov8 | copy_paste | cp2_single2 | mosaic | 0.5098 | 0.5197 | 0.1847 | 0.5031 | 0.5329 | 0.1834 | `yolov8/tinyperson/copy_paste/cp2_single2/mosaic/seed_42` |
| 16 | tinyperson | yolov8 | copy_paste | negative_canvas_r2 | mosaic | 0.5084 | 0.4721 | 0.1876 | 0.5105 | 0.5192 | 0.1837 | `yolov8/tinyperson/copy_paste/negative_canvas_r2/mosaic/seed_42` |
| 17 | tinyperson | yolov8 | copy_paste | negative_canvas_r4 | mosaic | 0.5282 | 0.4663 | 0.1919 | 0.5128 | 0.5311 | 0.1843 | `yolov8/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_42` |
| 18 | levir | yolov9 | copy_paste | negative_canvas_r2 | no_mosaic | 0.8196 | 0.8436 | 0.3251 | 0.7986 | 0.8160 | 0.3095 | `yolov9/levir/copy_paste/negative_canvas_r2/no_mosaic/seed_42` |
| 19 | levir | yolov9 | copy_paste | negative_canvas_r4 | no_mosaic | 0.8152 | 0.8476 | 0.3243 | 0.7863 | 0.8300 | 0.3036 | `yolov9/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_42` |
| 20 | tinyperson | yolov9 | copy_paste | cp2_single2 | mosaic | 0.5061 | 0.4934 | 0.1822 | 0.5006 | 0.5300 | 0.1814 | `yolov9/tinyperson/copy_paste/cp2_single2/mosaic/seed_42` |
| 21 | tinyperson | yolov9 | copy_paste | negative_canvas_r2 | mosaic | 0.4984 | 0.4670 | 0.1827 | 0.4806 | 0.5029 | 0.1711 | `yolov9/tinyperson/copy_paste/negative_canvas_r2/mosaic/seed_42` |
| 22 | tinyperson | yolov9 | copy_paste | negative_canvas_r4 | mosaic | 0.4951 | 0.5139 | 0.1816 | 0.4892 | 0.5220 | 0.1751 | `yolov9/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_42` |
| 23 | tinyperson | yolov8 | oacp | r2 | mosaic | 0.5178 | 0.4648 | 0.1872 | 0.5091 | 0.5334 | 0.1834 | `yolov8/tinyperson/oacp/r2/mosaic/seed_42` |
| 24 | tinyperson | yolov9 | oacp | r2 | mosaic | 0.5212 | 0.4641 | 0.1879 | 0.5067 | 0.5195 | 0.1812 | `yolov9/tinyperson/oacp/r2/mosaic/seed_42` |
| 25 | varroa | yolov9 | oacp | r2 | mosaic | 0.9140 | 0.8501 | 0.3392 | 0.9063 | 0.8111 | 0.3375 | `yolov9/varroa/oacp/r2/mosaic/seed_42` |
| 26 | levir | yolov11 | oacp | r2 | no_mosaic | 0.8304 | 0.8267 | 0.3240 | 0.8180 | 0.8377 | 0.3126 | `yolov11/levir/oacp/r2/no_mosaic/seed_42` |
| 27 | tinyperson | yolov11 | oacp | r2 | mosaic | 0.5530 | 0.4920 | 0.2008 | 0.5242 | 0.5367 | 0.1919 | `yolov11/tinyperson/oacp/r2/mosaic/seed_42` |
| 28 | varroa | yolov11 | oacp | r2 | mosaic | 0.9008 | 0.8386 | 0.3389 | 0.9186 | 0.8578 | 0.3391 | `yolov11/varroa/oacp/r2/mosaic/seed_42` |
| 29 | levir | yolov8 | oacp | r2 | no_mosaic | 0.8003 | 0.8136 | 0.3152 | 0.7944 | 0.7911 | 0.3024 | `yolov8/levir/oacp/r2/no_mosaic/seed_42` |
| 30 | tinyperson | yolov8 | oacp | r2 | mosaic | 0.5540 | 0.5367 | 0.2021 | 0.5243 | 0.5253 | 0.1885 | `yolov8/tinyperson/oacp/r2/mosaic/seed_42` |
| 31 | varroa | yolov8 | oacp | r2 | mosaic | 0.9174 | 0.8990 | 0.3341 | 0.9013 | 0.8636 | 0.3401 | `yolov8/varroa/oacp/r2/mosaic/seed_42` |
| 32 | levir | yolov9 | oacp | r2 | no_mosaic | 0.8076 | 0.8017 | 0.3247 | 0.7914 | 0.8238 | 0.2944 | `yolov9/levir/oacp/r2/no_mosaic/seed_42` |

## Metric protocols

1. `levir/yolov11/copy_paste/cp2_single2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
2. `levir/yolov11/copy_paste/negative_canvas_r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
3. `levir/yolov11/copy_paste/negative_canvas_r4`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
4. `tinyperson/yolov11/copy_paste/cp2_single2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
5. `tinyperson/yolov11/copy_paste/negative_canvas_r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
6. `tinyperson/yolov11/copy_paste/negative_canvas_r4`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
7. `varroa/yolov11/copy_paste/cp2_single2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: Varroa standard held-out test split.
8. `levir/yolov8/copy_paste/cp2_single2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
9. `levir/yolov8/copy_paste/negative_canvas_r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
10. `levir/yolov8/copy_paste/negative_canvas_r4`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
11. `varroa/yolov8/copy_paste/cp2_single2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: Varroa standard held-out test split.
12. `levir/yolov9/copy_paste/cp2_single2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
13. `levir/yolov9/copy_paste/negative_canvas_r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
14. `levir/yolov9/copy_paste/negative_canvas_r4`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
15. `tinyperson/yolov8/copy_paste/cp2_single2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
16. `tinyperson/yolov8/copy_paste/negative_canvas_r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
17. `tinyperson/yolov8/copy_paste/negative_canvas_r4`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
18. `levir/yolov9/copy_paste/negative_canvas_r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
19. `levir/yolov9/copy_paste/negative_canvas_r4`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
20. `tinyperson/yolov9/copy_paste/cp2_single2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
21. `tinyperson/yolov9/copy_paste/negative_canvas_r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
22. `tinyperson/yolov9/copy_paste/negative_canvas_r4`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
23. `tinyperson/yolov8/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
24. `tinyperson/yolov9/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
25. `varroa/yolov9/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: Varroa standard held-out test split.
26. `levir/yolov11/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
27. `tinyperson/yolov11/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
28. `varroa/yolov11/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: Varroa standard held-out test split.
29. `levir/yolov8/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.
30. `tinyperson/yolov8/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: TinyPerson official corner-window merged evaluator.
31. `varroa/yolov8/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: Varroa standard held-out test split.
32. `levir/yolov9/oacp/r2`: val size protocol: TinyBenchmark area buckets on native YOLO val images; IoU=0.50:0.05:0.75; maxDets=200; test protocol: LEVIR-Ship standard held-out test split.

## Provenance

- Output repository: `duyle2408/model-augmentation-seed42-size-metrics-runs`.
- Split seed: `42`; training seed: `42`.
- Completion markers verified: `32/32` for both `evaluation_metrics.json` and `size_metrics_complete.json`.
- `val/AP50-Small` and `test/AP50-Small` come from the size-bucket evaluator. `N/A` means the corresponding bucket was unavailable in the artifact.
