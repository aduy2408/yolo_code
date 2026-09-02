# Recovered baseline validation/test results

Checked on 2026-09-02 against the uploaded Hugging Face Dataset repositories. Values below are copied from `evaluation_metrics.json`, aggregate CSV files, and the repository reports. No metric has been inferred from plots.

## Hugging Face repositories

- LEVIR-Ship YOLO baselines: <https://huggingface.co/datasets/duyle2408/levir-ship-yolo-baselines>
- Varroa baselines, missing part 1: <https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-missing-part1-missing>
- Varroa baselines, missing part 2: <https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-missing-part2-missing>
- Varroa baselines, missing part 3: <https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-missing-part3-missing>
- Varroa baselines, full part 1: <https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-part1-full>
- Varroa baselines, full part 2: <https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-part2-full>
- Varroa DBSS/HIT ablation results, separate experiment: <https://huggingface.co/datasets/duyle2408/varroa-yolo-dbss-hit-3seed>
- TinyPerson YOLO baselines: <https://huggingface.co/datasets/duyle2408/tinyperson-yolo-baselines>

## Important protocol warning

The LEVIR-Ship HF baseline repository contains several evaluation generations. The verified aggregate report uses explicit NMS IoU 0.5 and reports validation and test metrics together. The numbers in the current thesis table, such as 27.11 for YOLOv8 and 30.78 for YOLOv10, were not found verbatim in the checked repository artifacts. They should not be silently replaced by the values below until the original run/protocol is identified.

## LEVIR-Ship: recovered YOLO baseline rows

Values are mean ± sample standard deviation over seeds 42, 43, and 44. The first block is the repository's standard stored evaluation; the second block is the explicit NMS IoU 0.50 re-evaluation.

| Model | Val P | Val R | Val mAP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test AP75 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0.7999 ± 0.0174 | 0.6999 ± 0.0297 | 0.7519 ± 0.0218 | — | 0.2858 ± 0.0148 | 0.7818 ± 0.0193 | 0.6684 ± 0.0294 | 0.7176 ± 0.0313 | 0.1032 ± 0.0022 | 0.2637 ± 0.0128 |
| YOLOv10n | 0.7609 ± 0.0417 | 0.6742 ± 0.0064 | 0.7366 ± 0.0345 | — | 0.2981 ± 0.0220 | 0.7434 ± 0.0504 | 0.6615 ± 0.0221 | 0.7294 ± 0.0303 | 0.1366 ± 0.0019 | 0.2892 ± 0.0148 |

| Model | Val P | Val R | Val mAP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test AP75 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n, NMS 0.50 | 0.8328 ± 0.0098 | 0.7680 ± 0.0269 | 0.8144 ± 0.0167 | 0.1443 ± 0.0235 | 0.3202 ± 0.0093 | 0.8173 ± 0.0345 | 0.7730 ± 0.0308 | 0.8146 ± 0.0350 | 0.1288 ± 0.0249 | 0.3151 ± 0.0280 |
| YOLOv10n, NMS 0.50 | 0.6918 ± 0.0143 | 0.6715 ± 0.0104 | 0.7243 ± 0.0136 | 0.1647 ± 0.0239 | 0.2997 ± 0.0052 | 0.6894 ± 0.0421 | 0.6561 ± 0.0343 | 0.7173 ± 0.0491 | 0.1561 ± 0.0148 | 0.2908 ± 0.0215 |

Source report: [`docs/reports/report_yolo.md`](report_yolo.md).

## Varroa: recovered baseline provenance and metrics

The Varroa baseline table in the thesis is assembled from the five repositories listed below, not from the separate DBSS/HIT ablation repository:

- `varroa-yolo-baselines-missing-part1-missing`
- `varroa-yolo-baselines-missing-part2-missing`
- `varroa-yolo-baselines-missing-part3-missing`
- `varroa-yolo-baselines-part1-full`
- `varroa-yolo-baselines-part2-full`

The aggregate validation values were recovered from the per-run `results.csv` files in the corresponding full repositories by selecting the epoch with the highest validation `mAP50-95`. The relevant validation/test aggregates are:

| Model | Seeds | Val mAP50 | Val mAP50-95 | Val Precision | Val Recall | Test mAP50 | Test mAP50-95 | Test Precision | Test Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8-n | 42,43,44 | 0.9107 ± 0.0049 | 0.3522 ± 0.0035 | 0.9084 ± 0.0077 | 0.8858 ± 0.0058 | 0.9002 ± 0.0081 | 0.3262 ± 0.0051 | 0.8998 ± 0.0046 | 0.8965 ± 0.0128 |
| YOLOv10-n | 42,43,44 | 0.8798 ± 0.0129 | 0.3395 ± 0.0074 | 0.8452 ± 0.0133 | 0.8318 ± 0.0096 | 0.8800 ± 0.0089 | 0.3333 ± 0.0049 | 0.8439 ± 0.0147 | 0.8328 ± 0.0156 |

The five repositories contain the source training logs and test summaries. Validation was therefore reconstructed from the uploaded per-run `results.csv` files using the best-validation-epoch rule, rather than from an invented or unrelated summary file.

### Separate Varroa DBSS/HIT experiment

The following table is not the source of the thesis baseline rows. It is retained only as a separate ablation reference from `varroa-yolo-dbss-hit-3seed`; the current snapshot contains seeds 42 and 43 despite the repository name containing `3seed`.

| Model | Mechanism | Seeds | Val P | Val R | Val mAP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test AP75 | Test mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | DBSS | 42,43 | 0.9036 ± 0.0141 | 0.8862 ± 0.0084 | 0.9146 ± 0.0080 | 0.1563 ± 0.0141 | 0.3586 ± 0.0100 | 0.9069 ± 0.0151 | 0.8759 ± 0.0126 | 0.9036 ± 0.0080 | 0.1519 ± 0.0040 | 0.3515 ± 0.0032 |
| YOLOv10n | DBSS | 42,43 | 0.8187 ± 0.0391 | 0.8346 ± 0.0097 | 0.8808 ± 0.0049 | 0.1378 ± 0.0084 | 0.3409 ± 0.0030 | 0.8304 ± 0.0351 | 0.8200 ± 0.0292 | 0.8680 ± 0.0232 | 0.1398 ± 0.0015 | 0.3318 ± 0.0062 |
| YOLOv8n | HIT | 42,43 | 0.8978 ± 0.0094 | 0.8985 ± 0.0036 | 0.9150 ± 0.0038 | 0.1543 ± 0.0122 | 0.3557 ± 0.0140 | 0.9049 ± 0.0235 | 0.8947 ± 0.0323 | 0.9128 ± 0.0112 | 0.1574 ± 0.0068 | 0.3518 ± 0.0024 |
| YOLOv10n | HIT | 42,43 | 0.8587 ± 0.0161 | 0.8433 ± 0.0048 | 0.9003 ± 0.0040 | 0.1444 ± 0.0192 | 0.3473 ± 0.0028 | 0.8537 ± 0.0102 | 0.8482 ± 0.0255 | 0.8931 ± 0.0107 | 0.1446 ± 0.0128 | 0.3347 ± 0.0084 |

Source report: [`docs/reports/report_varroa_dbss.md`](report_varroa_dbss.md).

## TinyPerson: recovered standard validation/test metrics

The TinyPerson repository contains 15 runs for YOLOv5n, YOLOv8n, YOLOv9t, and YOLOv10n across seeds 42, 43, and 44. YOLO11n has only seeds 42 and 43. Values below are mean ± sample standard deviation. These are the standard detector metrics, not the merged corner-window TinyBenchmark metrics.

| Model | Seeds | Val P | Val R | Val mAP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test AP75 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 42,43,44 | 0.6250 ± 0.0606 | 0.5005 ± 0.0426 | 0.5120 ± 0.0506 | 0.0746 ± 0.0082 | 0.1754 ± 0.0185 | 0.5930 ± 0.0287 | 0.4918 ± 0.0383 | 0.4945 ± 0.0411 | 0.0811 ± 0.0144 | 0.1760 ± 0.0198 |
| YOLOv8n | 42,43,44 | 0.6236 ± 0.0522 | 0.5130 ± 0.0357 | 0.5194 ± 0.0365 | 0.0802 ± 0.0021 | 0.1795 ± 0.0153 | 0.5945 ± 0.0269 | 0.4898 ± 0.0211 | 0.4990 ± 0.0265 | 0.0829 ± 0.0122 | 0.1782 ± 0.0129 |
| YOLOv9t | 42,43,44 | 0.6378 ± 0.0630 | 0.5035 ± 0.0329 | 0.5191 ± 0.0570 | 0.0825 ± 0.0198 | 0.1780 ± 0.0278 | 0.6067 ± 0.0153 | 0.4997 ± 0.0193 | 0.5097 ± 0.0225 | 0.0847 ± 0.0082 | 0.1827 ± 0.0109 |
| YOLOv10n | 42,43,44 | 0.5888 ± 0.0521 | 0.5008 ± 0.0063 | 0.5072 ± 0.0177 | 0.0830 ± 0.0070 | 0.1814 ± 0.0097 | 0.5623 ± 0.0063 | 0.4724 ± 0.0046 | 0.4814 ± 0.0022 | 0.0923 ± 0.0021 | 0.1803 ± 0.0021 |
| YOLO11n | 42,43 | 0.5943 ± 0.0093 | 0.5218 ± 0.0037 | 0.5262 ± 0.0090 | 0.0896 ± 0.0092 | 0.1861 ± 0.0114 | 0.6053 ± 0.0012 | 0.5058 ± 0.0108 | 0.5182 ± 0.0042 | 0.0888 ± 0.0019 | 0.1865 ± 0.0023 |

### TinyPerson merged-corner test metrics available in the repository

The merged evaluation is incomplete in the current snapshot. It is available for YOLOv8 seed 43, YOLOv9 seed 43, and YOLO11 seed 42 only. Therefore these values must not be presented as three-seed aggregates:

| Run | AP50 | AP75 | mAP50-75 | AP-Tiny1 | AP-Tiny2 | AP-Tiny3 | AP-Small | AP-Medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n, seed 43 | 0.5369 | 0.0945 | 0.3176 | 0.1595 | 0.3084 | 0.3684 | 0.4390 | 0.4352 |
| YOLOv9t, seed 43 | 0.5055 | 0.0838 | 0.2926 | 0.1229 | 0.2689 | 0.3500 | — | — |
| YOLO11n, seed 42 | 0.5450 | 0.0933 | 0.3183 | — | — | — | — | — |

The uploaded run manifests specify 100 epochs, patience 0, image size 640, batch size 8, NMS IoU 0.5, and the official TinyPerson corner-window plus source-image-grouped 90/10 split.

## Recommendation for the thesis

1. Keep the current thesis tables unchanged until the original source of the LEVIR values 27.11 and 30.78 is located.
2. Add separate validation and test columns using one consistent protocol. Do not mix the stored evaluation block with the explicit NMS 0.50 re-evaluation block.
3. For TinyPerson, use the standard validation/test block above for the baseline detector table. Put merged-corner AP-Tiny/AP-Small/AP-Medium values in a separate table and label the available runs explicitly.
4. Do not claim complete merged TinyBenchmark results for all 15 TinyPerson runs. The HF snapshot currently lacks merged metrics for most runs.
