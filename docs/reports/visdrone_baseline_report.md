# VisDrone2019-DET YOLO baseline results

**Compiled:** 2026-09-22 (ICT)
**Source:** [duyle2408/visdrone-yolo-baselines-runs](https://huggingface.co/datasets/duyle2408/visdrone-yolo-baselines-runs)
**Coverage:** 30/30 upload markers and 30/30 validated `evaluation_metrics.json` artifacts

## Executive summary

- Matrix: YOLOv5n, YOLOv8n, YOLOv9t, YOLOv10n, and YOLO11n.
- Each model has seeds 42, 43, and 44 under `mosaic` and `no_mosaic` policies.
- Training contract: 100 epochs, patience 0, batch 8, workers 8, CUDA, optimizer `auto` with the runner recording MuSGD behavior, and NMS IoU 0.5.
- Dataset: official VisDrone2019-DET train/val/test-dev split, converted from `/marimo/VisDrone2019`.
- All rows report explicit validation and test metrics. No validation value is relabeled as test.

## Metric and test protocol

The report uses the split-qualified fields from each uploaded `evaluation_metrics.json`:
`val/AP50`, `val/mAP50-95`, `test/AP50`, and `test/mAP50-95`.
The test split is the native VisDrone2019-DET test-dev protocol. The same artifacts also contain the size-bucket protocol with `test_size/AP50-Tiny1`, `Tiny2`, `Tiny3`, `Small`, and `Medium`.

## Per-run results

| Model | Augmentation | Seed | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | test_size/AP50-Tiny1 | test_size/AP50-Tiny2 | test_size/AP50-Tiny3 | test_size/AP50-Small | test_size/AP50-Medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11n | mosaic | 42 | 0.3358 | 0.1838 | 0.2812 | 0.1535 | 0.0099 | 0.0966 | 0.2827 | 0.5229 | 0.7422 |
| yolo11n | mosaic | 43 | 0.3349 | 0.1858 | 0.2800 | 0.1521 | 0.0169 | 0.0920 | 0.2852 | 0.5220 | 0.7417 |
| yolo11n | mosaic | 44 | 0.3343 | 0.1847 | 0.2768 | 0.1501 | 0.0099 | 0.0968 | 0.2852 | 0.5143 | 0.7410 |
| yolo11n | no_mosaic | 42 | 0.3223 | 0.1780 | 0.2682 | 0.1449 | 0.0099 | 0.0891 | 0.2674 | 0.5084 | 0.7397 |
| yolo11n | no_mosaic | 43 | 0.3240 | 0.1794 | 0.2680 | 0.1440 | 0.0099 | 0.0817 | 0.2682 | 0.5019 | 0.7312 |
| yolo11n | no_mosaic | 44 | 0.3253 | 0.1791 | 0.2698 | 0.1454 | 0.0099 | 0.0889 | 0.2637 | 0.4997 | 0.7304 |
| yolov10n | mosaic | 42 | 0.3255 | 0.1824 | 0.2695 | 0.1470 | 0.0099 | 0.0822 | 0.2573 | 0.4828 | 0.7109 |
| yolov10n | mosaic | 43 | 0.3213 | 0.1817 | 0.2697 | 0.1476 | 0.0170 | 0.0859 | 0.2658 | 0.4906 | 0.7112 |
| yolov10n | mosaic | 44 | 0.3220 | 0.1796 | 0.2665 | 0.1460 | 0.0099 | 0.0823 | 0.2580 | 0.4906 | 0.7095 |
| yolov10n | no_mosaic | 42 | 0.3102 | 0.1756 | 0.2591 | 0.1409 | 0.0099 | 0.0725 | 0.2401 | 0.4712 | 0.7000 |
| yolov10n | no_mosaic | 43 | 0.3136 | 0.1754 | 0.2547 | 0.1386 | 0.0099 | 0.0748 | 0.2453 | 0.4691 | 0.6919 |
| yolov10n | no_mosaic | 44 | 0.3082 | 0.1731 | 0.2551 | 0.1388 | 0.0099 | 0.0731 | 0.2385 | 0.4687 | 0.7012 |
| yolov5n | mosaic | 42 | 0.3226 | 0.1783 | 0.2709 | 0.1473 | 0.0099 | 0.0846 | 0.2607 | 0.4949 | 0.7320 |
| yolov5n | mosaic | 43 | 0.3238 | 0.1787 | 0.2719 | 0.1473 | 0.0173 | 0.0866 | 0.2660 | 0.5028 | 0.7325 |
| yolov5n | mosaic | 44 | 0.3291 | 0.1810 | 0.2719 | 0.1476 | 0.0099 | 0.0784 | 0.2640 | 0.5017 | 0.7313 |
| yolov5n | no_mosaic | 42 | 0.3127 | 0.1719 | 0.2575 | 0.1393 | 0.0099 | 0.0740 | 0.2472 | 0.4816 | 0.7201 |
| yolov5n | no_mosaic | 43 | 0.3158 | 0.1723 | 0.2627 | 0.1416 | 0.0099 | 0.0735 | 0.2481 | 0.4898 | 0.7207 |
| yolov5n | no_mosaic | 44 | 0.3094 | 0.1695 | 0.2585 | 0.1393 | 0.0099 | 0.0806 | 0.2537 | 0.4909 | 0.7205 |
| yolov8n | mosaic | 42 | 0.3392 | 0.1873 | 0.2766 | 0.1513 | 0.0099 | 0.0963 | 0.2756 | 0.5214 | 0.7413 |
| yolov8n | mosaic | 43 | 0.3370 | 0.1852 | 0.2788 | 0.1520 | 0.0169 | 0.0921 | 0.2782 | 0.5209 | 0.7417 |
| yolov8n | mosaic | 44 | 0.3370 | 0.1856 | 0.2774 | 0.1509 | 0.0099 | 0.0951 | 0.2818 | 0.5217 | 0.7414 |
| yolov8n | no_mosaic | 42 | 0.3286 | 0.1826 | 0.2686 | 0.1456 | 0.0099 | 0.0867 | 0.2652 | 0.5105 | 0.7310 |
| yolov8n | no_mosaic | 43 | 0.3242 | 0.1786 | 0.2701 | 0.1460 | 0.0099 | 0.0796 | 0.2667 | 0.5103 | 0.7325 |
| yolov8n | no_mosaic | 44 | 0.3273 | 0.1816 | 0.2681 | 0.1447 | 0.0099 | 0.0885 | 0.2750 | 0.5113 | 0.7315 |
| yolov9t | mosaic | 42 | 0.3379 | 0.1873 | 0.2871 | 0.1575 | 0.0099 | 0.0801 | 0.2747 | 0.5157 | 0.7432 |
| yolov9t | mosaic | 43 | 0.3329 | 0.1854 | 0.2872 | 0.1559 | 0.0099 | 0.0713 | 0.2699 | 0.5147 | 0.7424 |
| yolov9t | mosaic | 44 | 0.3394 | 0.1866 | 0.2903 | 0.1581 | 0.0099 | 0.0784 | 0.2751 | 0.5142 | 0.7431 |
| yolov9t | no_mosaic | 42 | 0.3292 | 0.1833 | 0.2758 | 0.1495 | 0.0099 | 0.0703 | 0.2553 | 0.5034 | 0.7422 |
| yolov9t | no_mosaic | 43 | 0.3322 | 0.1845 | 0.2766 | 0.1509 | 0.0099 | 0.0703 | 0.2621 | 0.5034 | 0.7419 |
| yolov9t | no_mosaic | 44 | 0.3323 | 0.1840 | 0.2743 | 0.1503 | 0.0099 | 0.0724 | 0.2464 | 0.4923 | 0.7328 |

## Mean ± standard deviation by model and augmentation

| Model | Augmentation | n | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| yolo11n | mosaic | 3 | 0.3350 ± 0.0008 | 0.1848 ± 0.0010 | 0.2794 ± 0.0023 | 0.1519 ± 0.0017 |
| yolo11n | no_mosaic | 3 | 0.3239 ± 0.0015 | 0.1788 ± 0.0008 | 0.2687 ± 0.0010 | 0.1448 ± 0.0007 |
| yolov10n | mosaic | 3 | 0.3229 ± 0.0022 | 0.1812 ± 0.0014 | 0.2686 ± 0.0018 | 0.1469 ± 0.0008 |
| yolov10n | no_mosaic | 3 | 0.3107 ± 0.0027 | 0.1747 ± 0.0014 | 0.2563 ± 0.0024 | 0.1394 ± 0.0013 |
| yolov5n | mosaic | 3 | 0.3252 ± 0.0035 | 0.1794 ± 0.0015 | 0.2716 ± 0.0006 | 0.1474 ± 0.0002 |
| yolov5n | no_mosaic | 3 | 0.3126 ± 0.0032 | 0.1713 ± 0.0015 | 0.2596 ± 0.0027 | 0.1401 ± 0.0014 |
| yolov8n | mosaic | 3 | 0.3378 ± 0.0013 | 0.1860 ± 0.0011 | 0.2776 ± 0.0012 | 0.1514 ± 0.0005 |
| yolov8n | no_mosaic | 3 | 0.3267 ± 0.0023 | 0.1809 ± 0.0020 | 0.2689 ± 0.0010 | 0.1454 ± 0.0007 |
| yolov9t | mosaic | 3 | 0.3367 ± 0.0034 | 0.1865 ± 0.0009 | 0.2882 ± 0.0018 | 0.1572 ± 0.0011 |
| yolov9t | no_mosaic | 3 | 0.3313 ± 0.0018 | 0.1839 ± 0.0006 | 0.2756 ± 0.0011 | 0.1502 ± 0.0007 |

## Seed-aggregated test ranking

Ranking uses the mean `test/mAP50-95` across seeds 42-44. Mosaic and no-Mosaic are kept separate.

| Rank | Model | Augmentation | Mean test/mAP50-95 | SD |
|---:|---|---|---:|---:|
| 1 | yolov9t | mosaic | 0.1572 | 0.0011 |
| 2 | yolo11n | mosaic | 0.1519 | 0.0017 |
| 3 | yolov8n | mosaic | 0.1514 | 0.0005 |
| 4 | yolov9t | no_mosaic | 0.1502 | 0.0007 |
| 5 | yolov5n | mosaic | 0.1474 | 0.0002 |
| 6 | yolov10n | mosaic | 0.1469 | 0.0008 |
| 7 | yolov8n | no_mosaic | 0.1454 | 0.0007 |
| 8 | yolo11n | no_mosaic | 0.1448 | 0.0007 |
| 9 | yolov5n | no_mosaic | 0.1401 | 0.0014 |
| 10 | yolov10n | no_mosaic | 0.1394 | 0.0013 |

## Artifact and provenance audit

- Remote completion evidence: every prefix has `upload_complete.json` and `evaluation_metrics.json` in the task-specific HF dataset repository.
- Metric audit: all 30 metric files contain finite values for the four required split-qualified metrics and all five required size-bucket AP50 fields.
- Local recovery audit: the seven runs generated by the final recovery checkout had `weights/best.pt`, `weights/last.pt`, `results.csv`, `evaluation_metrics.json`, `experiment_manifest.json`, and `upload_complete.json`.
- The final wrapper log recorded `COMPLETE` for the last missing run and `SKIP_VERIFIED` for already uploaded prefixes.
- Caveat: the aggregate `utils.marimo_ops complete_verified` wrapper gate reported no recorded clean process exit (`not_running_unverified`) even though all 30 remote markers and metrics were independently verified. Treat the HF artifact set as complete, while retaining this orchestration caveat in provenance.

## Reproducibility configuration

| Field | Value |
|---|---|
| Commit | `0a9c9d22b5a2d03729ad48a81bc0b0dd773a5573` |
| Dataset root | `/marimo/VisDrone2019` |
| Dataset YAML | `datasets/visdrone_baselines/visdrone.yaml` |
| Official split | VisDrone2019-DET train / val / test-dev |
| Models | `yolov5n`, `yolov8n`, `yolov9t`, `yolov10n`, `yolo11n` |
| Training seeds | 42, 43, 44 |
| Augmentations | `mosaic`, `no_mosaic` |
| Epochs / patience | 100 / 0 |
| Batch / workers | 8 / 8 |
| Optimizer | `auto` (MuSGD behavior recorded by runner) |
| NMS IoU | 0.5 |
| HF repository | `duyle2408/visdrone-yolo-baselines-runs` |

## Artifact links

- [HF dataset repository](https://huggingface.co/datasets/duyle2408/visdrone-yolo-baselines-runs)
- Per-run artifacts follow `runs/<model>/<augmentation>/seed_<seed>/`.

## Interpretation limits

- The three seeds provide a small stability estimate, not a definitive uncertainty interval.
- Comparisons should use the test metrics and preserve the matched detector/data/training contract.
- The test-size fields are a secondary diagnostic protocol and should not replace the aggregate test metrics.
