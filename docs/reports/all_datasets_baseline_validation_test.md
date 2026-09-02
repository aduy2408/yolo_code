# Complete baseline validation and test results across all three datasets

Checked against local uploaded artifacts and public Hugging Face repositories on 2026-09-02. Values are mean ± sample standard deviation across the listed seeds unless stated otherwise. No value is inferred from plots.

## Scope and protocol

This report deliberately includes all baseline model families available in the recovered artifacts, not only YOLOv8 and YOLOv10. Validation metrics are taken from the best validation `mAP50-95` epoch where per-run `results.csv` is available. Test metrics are taken from the corresponding uploaded test summary or evaluation artifact. AP75 is reported only where the artifact actually contains it.

---

## 1. LEVIR-Ship

### Source repository

- [duyle2408/levir-ship-yolo-baselines](https://huggingface.co/datasets/duyle2408/levir-ship-yolo-baselines)

The LEVIR baseline repository contains YOLOv5nu, YOLOv8n, YOLOv9t, YOLOv10n, and YOLO11n runs for seeds 42, 43, and 44. The table below uses the standard stored evaluation aggregate from [`docs/reports/report_yolo.md`](report_yolo.md). A separate explicit-NMS-0.50 re-evaluation also exists in that report and must not be mixed with this block.

| Model | Seeds | Val P | Val R | Val mAP50 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5nu | 42,43,44 | 0.7184 ± 0.1030 | 0.6393 ± 0.0574 | 0.6741 ± 0.0865 | 0.2577 ± 0.0388 | 0.7317 ± 0.0538 | 0.6014 ± 0.0780 | 0.6571 ± 0.0822 | 0.2425 ± 0.0281 |
| YOLOv8n | 42,43,44 | 0.7999 ± 0.0174 | 0.6999 ± 0.0297 | 0.7519 ± 0.0218 | 0.2858 ± 0.0148 | 0.7818 ± 0.0193 | 0.6684 ± 0.0294 | 0.7176 ± 0.0313 | 0.2637 ± 0.0128 |
| YOLOv9t | 42,43,44 | 0.7810 ± 0.0401 | 0.7019 ± 0.0143 | 0.7441 ± 0.0485 | 0.2859 ± 0.0224 | 0.7836 ± 0.0314 | 0.6869 ± 0.0344 | 0.7377 ± 0.0261 | 0.2773 ± 0.0105 |
| YOLOv10n | 42,43,44 | 0.7609 ± 0.0417 | 0.6742 ± 0.0064 | 0.7366 ± 0.0345 | 0.2981 ± 0.0220 | 0.7434 ± 0.0504 | 0.6615 ± 0.0221 | 0.7294 ± 0.0303 | 0.2892 ± 0.0148 |
| YOLO11n | 42,43,44 | 0.7813 ± 0.0341 | 0.6677 ± 0.0467 | 0.7238 ± 0.0621 | 0.2852 ± 0.0382 | 0.7417 ± 0.0425 | 0.6661 ± 0.0322 | 0.7070 ± 0.0494 | 0.2656 ± 0.0254 |

The corresponding explicit NMS IoU 0.50 block, including AP75, is retained in [`docs/reports/report_yolo.md`](report_yolo.md). The current thesis values `27.11` and `30.78` were not found verbatim in the checked HF artifacts, so their original protocol remains unresolved.

---

## 2. Varroa

### Correct baseline repositories

The Varroa baseline results come from these five repositories:

- [missing part 1](https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-missing-part1-missing)
- [missing part 2](https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-missing-part2-missing)
- [missing part 3](https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-missing-part3-missing)
- [full part 1](https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-part1-full)
- [full part 2](https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-part2-full)

The `missing-part*` repositories complement the full repositories. The separate [DBSS/HIT repository](https://huggingface.co/datasets/duyle2408/varroa-yolo-dbss-hit-3seed) is an ablation source, not the baseline source.

### All full-repository baseline models

Validation was calculated from the full repositories' per-run `results.csv` files by selecting the best validation `mAP50-95` epoch. Test aggregates are from the five-repository baseline summary in [`misc/baseline_reuslts/BASELINE_RESULTS_SUMMARY.md`](../../misc/baseline_reuslts/BASELINE_RESULTS_SUMMARY.md).

| Model | Seeds | Val mAP50 | Val mAP50-95 | Val P | Val R | Test mAP50 | Test mAP50-95 | Test P | Test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 42,43,44 | 0.9124 ± 0.0061 | 0.3518 ± 0.0027 | 0.9212 ± 0.0083 | 0.8861 ± 0.0284 | 0.8961 ± 0.0133 | 0.3227 ± 0.0063 | 0.9032 ± 0.0075 | 0.8966 ± 0.0082 |
| YOLOv5s | 42,43,44 | 0.9097 ± 0.0034 | 0.3486 ± 0.0017 | 0.9038 ± 0.0023 | 0.8850 ± 0.0068 | 0.8956 ± 0.0085 | 0.3277 ± 0.0013 | 0.8958 ± 0.0142 | 0.8959 ± 0.0031 |
| YOLOv5l | 42,43,44 | 0.9097 ± 0.0046 | 0.3457 ± 0.0030 | 0.9083 ± 0.0103 | 0.8597 ± 0.0254 | 0.9111 ± 0.0098 | 0.3301 ± 0.0022 | 0.9085 ± 0.0074 | 0.8904 ± 0.0144 |
| YOLOv8n | 42,43,44 | 0.9107 ± 0.0049 | 0.3522 ± 0.0035 | 0.9084 ± 0.0077 | 0.8858 ± 0.0058 | 0.9002 ± 0.0081 | 0.3262 ± 0.0051 | 0.8998 ± 0.0046 | 0.8965 ± 0.0128 |
| YOLOv8s | 42,43,44 | 0.9107 ± 0.0076 | 0.3468 ± 0.0013 | 0.9024 ± 0.0186 | 0.8839 ± 0.0121 | 0.8972 ± 0.0069 | 0.3295 ± 0.0013 | 0.8947 ± 0.0094 | 0.8806 ± 0.0081 |
| YOLOv8l | 42,43,44 | 0.9081 ± 0.0167 | 0.3477 ± 0.0007 | 0.8980 ± 0.0050 | 0.8791 ± 0.0087 | 0.8992 ± 0.0077 | 0.3287 ± 0.0022 | 0.8944 ± 0.0134 | 0.8843 ± 0.0126 |
| YOLOv9t | 42,43,44 | 0.9105 ± 0.0030 | 0.3543 ± 0.0039 | 0.8984 ± 0.0127 | 0.8813 ± 0.0262 | 0.9056 ± 0.0172 | 0.3293 ± 0.0051 | 0.9036 ± 0.0070 | 0.8951 ± 0.0071 |
| YOLOv9s | 42,43,44 | 0.8981 ± 0.0090 | 0.3487 ± 0.0041 | 0.8979 ± 0.0021 | 0.8692 ± 0.0122 | 0.9011 ± 0.0065 | 0.3327 ± 0.0065 | 0.8983 ± 0.0124 | 0.8979 ± 0.0062 |
| YOLOv9c | 42,43,44 | 0.9146 ± 0.0053 | 0.3493 ± 0.0020 | 0.9115 ± 0.0105 | 0.8934 ± 0.0064 | 0.9011 ± 0.0089 | 0.3298 ± 0.0050 | 0.8998 ± 0.0101 | 0.9020 ± 0.0058 |
| YOLOv10n | 42,43,44 | 0.8798 ± 0.0129 | 0.3395 ± 0.0074 | 0.8452 ± 0.0133 | 0.8318 ± 0.0096 | 0.8800 ± 0.0089 | 0.3333 ± 0.0049 | 0.8439 ± 0.0147 | 0.8328 ± 0.0156 |
| YOLOv10s | 42,43,44 | 0.8844 ± 0.0065 | 0.3336 ± 0.0070 | 0.8712 ± 0.0113 | 0.8567 ± 0.0032 | 0.8708 ± 0.0069 | 0.3247 ± 0.0057 | 0.8570 ± 0.0146 | 0.8256 ± 0.0073 |
| YOLOv10l | 42,43,44 | 0.8733 ± 0.0179 | 0.3405 ± 0.0046 | 0.8543 ± 0.0300 | 0.8263 ± 0.0215 | 0.8762 ± 0.0101 | 0.3338 ± 0.0096 | 0.8475 ± 0.0176 | 0.8270 ± 0.0010 |
| YOLO11n | 42,43,44 | 0.9097 ± 0.0085 | 0.3520 ± 0.0052 | 0.9102 ± 0.0294 | 0.8730 ± 0.0108 | 0.9009 ± 0.0149 | 0.3260 ± 0.0098 | 0.9108 ± 0.0149 | 0.8986 ± 0.0094 |
| YOLO11s | 42,43,44 | 0.9078 ± 0.0113 | 0.3458 ± 0.0032 | 0.8930 ± 0.0114 | 0.8799 ± 0.0185 | 0.9067 ± 0.0124 | 0.3326 ± 0.0050 | 0.9053 ± 0.0086 | 0.9048 ± 0.0123 |
| YOLO11l | 42,43,44 | 0.9158 ± 0.0021 | 0.3509 ± 0.0015 | 0.9167 ± 0.0048 | 0.8847 ± 0.0117 | 0.8959 ± 0.0060 | 0.3340 ± 0.0011 | 0.8985 ± 0.0037 | 0.8986 ± 0.0068 |

The missing-part repositories also contain larger m/x/e variants, but those are marked as incomplete in the aggregate inventory. They should be added as separate rows only when the corresponding test summary and validation run coverage are both confirmed.

---

## 3. TinyPerson

### Source repository

- [duyle2408/tinyperson-yolo-baselines](https://huggingface.co/datasets/duyle2408/tinyperson-yolo-baselines)

The repository contains 15 standard detector runs: YOLOv5n, YOLOv8n, YOLOv9t, and YOLOv10n for seeds 42, 43, and 44, plus YOLO11n for seeds 42 and 43. The values below are standard validation/test metrics from each run's `evaluation_metrics.json`, aggregated across available seeds.

| Model | Seeds | Val P | Val R | Val mAP50 | Val AP75 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test AP75 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 42,43,44 | 0.6250 ± 0.0606 | 0.5005 ± 0.0426 | 0.5120 ± 0.0506 | 0.0746 ± 0.0082 | 0.1754 ± 0.0185 | 0.5930 ± 0.0287 | 0.4918 ± 0.0383 | 0.4945 ± 0.0411 | 0.0811 ± 0.0144 | 0.1760 ± 0.0198 |
| YOLOv8n | 42,43,44 | 0.6236 ± 0.0522 | 0.5130 ± 0.0357 | 0.5194 ± 0.0365 | 0.0802 ± 0.0021 | 0.1795 ± 0.0153 | 0.5945 ± 0.0269 | 0.4898 ± 0.0211 | 0.4990 ± 0.0265 | 0.0829 ± 0.0122 | 0.1782 ± 0.0129 |
| YOLOv9t | 42,43,44 | 0.6378 ± 0.0630 | 0.5035 ± 0.0329 | 0.5191 ± 0.0570 | 0.0825 ± 0.0198 | 0.1780 ± 0.0278 | 0.6067 ± 0.0153 | 0.4997 ± 0.0193 | 0.5097 ± 0.0225 | 0.0847 ± 0.0082 | 0.1827 ± 0.0109 |
| YOLOv10n | 42,43,44 | 0.5888 ± 0.0521 | 0.5008 ± 0.0063 | 0.5072 ± 0.0177 | 0.0830 ± 0.0070 | 0.1814 ± 0.0097 | 0.5623 ± 0.0063 | 0.4724 ± 0.0046 | 0.4814 ± 0.0022 | 0.0923 ± 0.0021 | 0.1803 ± 0.0021 |
| YOLO11n | 42,43 | 0.5943 ± 0.0093 | 0.5218 ± 0.0037 | 0.5262 ± 0.0090 | 0.0896 ± 0.0092 | 0.1861 ± 0.0114 | 0.6053 ± 0.0012 | 0.5058 ± 0.0108 | 0.5182 ± 0.0042 | 0.0888 ± 0.0019 | 0.1865 ± 0.0023 |

TinyPerson also contains merged corner-window test metrics, but only some runs currently expose `test_merged/*` fields. Those merged metrics must be reported separately from the standard detector metrics and must not be presented as complete three-seed aggregates.

---

## Thesis usage notes

- Use one consistent metric block per dataset. Do not mix standard stored evaluation with explicit-NMS re-evaluation.
- Report `mean ± sample standard deviation` only for models with multiple available seeds.
- Keep AP75 as `—` when the source artifact does not contain it.
- For Varroa, cite the five baseline repositories above. Cite DBSS/HIT only in an ablation section.
- For TinyPerson, distinguish standard detector test metrics from merged-corner TinyBenchmark metrics.
- The comprehensive source-linked report is intended as the data source for the Results and Discussion chapter. The thesis `.tex` source itself was not present in this workspace, so direct chapter compilation was not performed.
