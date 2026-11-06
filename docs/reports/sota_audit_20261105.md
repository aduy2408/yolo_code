# SOTA Audit: FCOS-SET and FCOS-SR-TOD Seed Coverage

**Audit date:** 2026-11-06  
**Scope:** Historical FCOS-SET and FCOS-SR-TOD seed coverage across LEVIR-Ship, Varroa, and TinyPerson.  
**NMS IoU:** `0.50` where recorded.

## Finding

Seed42 **does exist** in historical Hugging Face artifacts. The previous report wording meant that no **protocol-matched seed42 validation artifact** was found for the current FCOS standard-Mosaic 640×640 LEVIR/TinyPerson rows. It did not mean that seed42 checkpoints were absent.

## FCOS-SET results

FCOS-SET has two usable seeds for LEVIR-Ship and TinyPerson. The screenshot row is seed43, not a two-seed aggregate.

### LEVIR-Ship

| Training seed | Val AP50 | Val mAP50-95 | Val AP50-small | Test AP50 | Test mAP50-95 | Test AP50-small |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7324 | 0.2598 | 0.7339 | 0.7740 | 0.2785 | 0.7714 |
| 43 | 0.7656 | 0.2762 | 0.7683 | 0.7358 | 0.2708 | 0.7330 |

Seed42:

```text
duyle2408/set-fcos-stw-protocol-runs:
  levirship/no_mosaic/patience15/seed42/fcos_set/final_results.json
```

Seed43:

```text
duyle2408/set_fcos_runs:
  sota_fcos_set/levir_ship/seed43/seed43/fcos_set/
```

### TinyPerson

| Training seed | Val AP50 | Val mAP50-95 | Val AP50-small | Test AP50 | Test mAP50-95 | Test AP50-small |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.4148 | 0.1494 | 0.5516 | 0.4259 | 0.1519 | 0.5722 |
| 43 | 0.4084 | 0.1487 | 0.5366 | 0.4247 | 0.1523 | 0.5916 |

Seed42:

```text
duyle2408/set-fcos-stw-protocol-runs:
  tinyperson/mosaic/patience15/seed42/fcos_set/final_results.json
```

Seed43:

```text
duyle2408/fcos-tinyperson-eval-20261015-runs:
  tinyperson/seed43/fcos_set/final_results.json
  checkpoint: sota_fcos_set/tinyperson/seed43/fcos_set/best_coco_bbox_mAP_epoch_37.pth
```

### FCOS-SET full size-bucket metrics

The `AP50-small` values above are the size-bucket AP at IoU 0.50, not COCO `AP-small` averaged over IoU 0.50:0.95. The historical FCOS-SET artifacts also contain the complete TinyBenchmark bucket fields:

| Dataset | Seed | Split | AP50-Tiny1 | AP50-Tiny2 | AP50-Tiny3 | AP50-Small | AP50-Medium |
|---|---:|---|---:|---:|---:|---:|---:|
| LEVIR-Ship | 42 | Val | -- | -- | -- | 0.7339 | 0.2194 |
| LEVIR-Ship | 42 | Test | -- | -- | -- | 0.7714 | 0.2881 |
| LEVIR-Ship | 43 | Val | -- | -- | -- | 0.7683 | 0.2123 |
| LEVIR-Ship | 43 | Test | -- | -- | -- | 0.7330 | 0.3282 |
| TinyPerson | 42 | Val | 0.1292 | 0.4140 | 0.4580 | 0.5516 | 0.2925 |
| TinyPerson | 42 | Test | 0.1253 | 0.3410 | 0.4632 | 0.5722 | 0.2653 |
| TinyPerson | 43 | Val | 0.1445 | 0.4129 | 0.4406 | 0.5366 | 0.2920 |
| TinyPerson | 43 | Test | 0.1096 | 0.2986 | 0.4708 | 0.5916 | 0.2477 |

HF source files:

```text
duyle2408/set-fcos-stw-protocol-runs:
  levirship/no_mosaic/patience15/seed42/fcos_set/final_results.json
  tinyperson/mosaic/patience15/seed42/fcos_set/final_results.json

duyle2408/set_fcos_runs:
  sota_fcos_set/levir_ship/seed43/seed43/fcos_set/final_results.json
  sota_fcos_set/tinyperson/seed43/fcos_set/final_results.json
```

The screenshot’s FCOS-SET test row is seed43: LEVIR-Ship `0.7358 / 0.2708 / 0.7330` and TinyPerson `0.4247 / 0.1523 / 0.5916`; the final value in each triplet is the AP50-Small bucket metric.


The validation two-seed summaries are:

```text
LEVIR-Ship: AP50 0.7490 ± 0.0235; mAP50-95 0.2680 ± 0.0116; AP50-small 0.7511 ± 0.0243
TinyPerson: AP50 0.4116 ± 0.0045; mAP50-95 0.1491 ± 0.0005; AP50-small 0.5441 ± 0.0106
```

Varroa has historical seed42 test evidence and a selected seed43 row, but seed42 does not expose complete split-qualified validation fields for a matched two-seed aggregate.

## Complete two-seed SOTA matrix: LEVIR-Ship and TinyPerson

This is the complete seed inventory for the four requested SOTA families. It intentionally keeps missing fields as `--`; a metric from another protocol is not substituted.

| Model | Dataset | Seed | Val AP50 | Val mAP50-95 | Val AP50-small | Test AP50 | Test mAP50-95 | Test AP50-small | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| STW-YOLO | LEVIR-Ship | 42 | 0.8321 | 0.3325 | 0.8340 | 0.7934 | 0.2861 | 0.8408 | Complete native backfill |
| STW-YOLO | LEVIR-Ship | 43 | 0.8295 | 0.3385 | -- | 0.8205 | 0.3208 | -- | Core split metrics; bucket fields not recorded in this audit |
| STW-YOLO | TinyPerson | 42 | 0.6190 | 0.2313 | 0.5982 | 0.5684 | 0.2149 | 0.5419 | Complete native backfill |
| STW-YOLO | TinyPerson | 43 | 0.6241 | 0.2394 | -- | 0.5792 | 0.2204 | -- | Core split metrics; bucket fields not recorded in this audit |
| TPH-YOLOv5 | LEVIR-Ship | 42 | 0.7140 | 0.2590 | 0.7862 | 0.6740 | 0.2370 | 0.7180 | Core metrics; AP50-small source-qualified for seed42 |
| TPH-YOLOv5 | LEVIR-Ship | 43 | 0.7550 | 0.2880 | -- | 0.7430 | 0.2690 | -- | Core split metrics; native AP50-small absent |
| TPH-YOLOv5 | TinyPerson | 42 | 0.5420 | 0.1920 | -- | 0.5500 | 0.2010 | -- | Core split metrics; native AP50-small absent |
| TPH-YOLOv5 | TinyPerson | 43 | 0.4920 | 0.1690 | -- | 0.4890 | 0.1750 | -- | Core split metrics; native AP50-small absent |
| FCOS-SET | LEVIR-Ship | 42 | 0.7324 | 0.2598 | 0.7339 | 0.7740 | 0.2785 | 0.7714 | Complete |
| FCOS-SET | LEVIR-Ship | 43 | 0.7656 | 0.2762 | 0.7683 | 0.7358 | 0.2708 | 0.7330 | Complete |
| FCOS-SET | TinyPerson | 42 | 0.4148 | 0.1494 | 0.5516 | 0.4259 | 0.1519 | 0.5722 | Complete |
| FCOS-SET | TinyPerson | 43 | 0.4084 | 0.1487 | 0.5366 | 0.4247 | 0.1523 | 0.5916 | Complete |
| FCOS-SR-TOD | LEVIR-Ship | 42 | 0.5630 | 0.1800 | 0.8226 | 0.4350 | 0.1180 | 0.5622 | Matched seed42 bucket backfill |
| FCOS-SR-TOD | LEVIR-Ship | 43 | 0.7330 | 0.2670 | 0.7375 | 0.6520 | 0.2070 | 0.6466 | Complete |
| FCOS-SR-TOD | TinyPerson | 42 | 0.4800 | 0.1590 | 0.6696 | 0.3710 | 0.1280 | 0.6289 | Matched seed42 bucket backfill |
| FCOS-SR-TOD | TinyPerson | 43 | 0.4180 | 0.1500 | 0.3712 | 0.3730 | 0.1290 | 0.3466 | Complete |

**Status under the requested report contract:** all four selected SOTA models have the required val/test AP50 and mAP50-95 on LEVIR-Ship and TinyPerson. TinyPerson uses merged AP50-small where requested; the verified TPH merged result is seed42 (`val=0.6507`, `test=0.6369`). The native per-seed AP50-small `--` cells for TPH are not blockers and must not be described as missing training seeds or missing core results. The STW seed43 bucket fields are likewise outside the selected seed42 backfill scope.

## Final two-seed mean ± sample standard deviation

Means and standard deviations below use training seeds 42 and 43 and the same metric/protocol within each row. `--` means the requested metric is not available for both seeds; no cross-protocol substitution is made.

| Model | Dataset | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Val AP50-small | Test AP50-small |
|---|---|---:|---:|---:|---:|---:|---:|
| STW-YOLO | LEVIR-Ship | 0.8308 ± 0.0018 | 0.3355 ± 0.0042 | 0.8070 ± 0.0192 | 0.3034 ± 0.0245 | -- | -- |
| STW-YOLO | TinyPerson | 0.6216 ± 0.0036 | 0.2354 ± 0.0057 | 0.5738 ± 0.0076 | 0.2177 ± 0.0039 | -- | -- |
| TPH-YOLOv5 | LEVIR-Ship | 0.7345 ± 0.0290 | 0.2735 ± 0.0205 | 0.7085 ± 0.0488 | 0.2530 ± 0.0226 | -- | -- |
| TPH-YOLOv5 | TinyPerson | 0.5170 ± 0.0354 | 0.1805 ± 0.0163 | 0.5195 ± 0.0431 | 0.1880 ± 0.0184 | -- | -- |
| FCOS-SET | LEVIR-Ship | 0.7490 ± 0.0235 | 0.2680 ± 0.0116 | 0.7549 ± 0.0270 | 0.2747 ± 0.0054 | 0.7511 ± 0.0243 | 0.7522 ± 0.0272 |
| FCOS-SET | TinyPerson | 0.4116 ± 0.0045 | 0.1491 ± 0.0005 | 0.4253 ± 0.0008 | 0.1521 ± 0.0003 | 0.5441 ± 0.0106 | 0.5819 ± 0.0137 |
| FCOS-SR-TOD | LEVIR-Ship | 0.6480 ± 0.1202 | 0.2235 ± 0.0615 | 0.5435 ± 0.1534 | 0.1625 ± 0.0629 | 0.7801 ± 0.0602 | 0.6044 ± 0.0597 |
| FCOS-SR-TOD | TinyPerson | 0.4490 ± 0.0438 | 0.1545 ± 0.0064 | 0.3720 ± 0.0014 | 0.1285 ± 0.0007 | 0.5204 ± 0.2116 | 0.4878 ± 0.1998 |


## Corrected complete merged TinyPerson table

The table below supersedes the older native-size table for the requested merged fields. It uses source-image merged test metrics, NMS IoU `0.50`, seeds `42/43`, and sample standard deviation.

| Model | AP50 | mAP50-95 | Merged AP50-small |
|---|---:|---:|---:|
| STW-YOLO | 0.5924 ± 0.0046 | 0.2243 ± 0.0003 | 0.7050 ± 0.0065 |
| TPH-YOLOv5 | 0.5247 ± 0.0321 | 0.1945 ± 0.0136 | 0.6045 ± 0.0459 |
| FCOS-SET | 0.4253 ± 0.0009 | 0.1521 ± 0.0003 | 0.5961 ± 0.0064 |
| FCOS-SR-TOD | 0.4583 ± 0.0047 | 0.1632 ± 0.0015 | 0.5949 ± 0.0007 |

The reproducible evaluator is `evaluate_test/evaluate_sota_models.py`; per-seed source values are in `sota_results_reproducible_20261106.json`.

## Cross-model YOLO seed coverage

The earlier compact summary incorrectly implied that STW and TPH had only seed42. Verified seed43 artifacts also exist; no retraining is required for these rows.

| Model | Dataset | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Status |
|---|---|---:|---:|---:|---:|---:|---|
| TPH-YOLOv5 | LEVIR-Ship | 42 | 0.7140 | 0.2590 | 0.6740 | 0.2370 | Complete |
| TPH-YOLOv5 | LEVIR-Ship | 43 | 0.7550 | 0.2880 | 0.7430 | 0.2690 | Complete |
| TPH-YOLOv5 | TinyPerson | 42 | 0.5420 | 0.1920 | 0.5500 | 0.2010 | Complete |
| TPH-YOLOv5 | TinyPerson | 43 | 0.4920 | 0.1690 | 0.4890 | 0.1750 | Complete |
| STW-YOLO | LEVIR-Ship | 42 | 0.8321 | 0.3325 | 0.7934 | 0.2861 | Complete split-qualified backfill |
| STW-YOLO | LEVIR-Ship | 43 | 0.8295 | 0.3385 | 0.8205 | 0.3208 | Complete split-qualified backfill |
| STW-YOLO | TinyPerson | 42 | 0.6190 | 0.2313 | 0.5684 | 0.2149 | Complete split-qualified backfill |
| STW-YOLO | TinyPerson | 43 | 0.6241 | 0.2394 | 0.5792 | 0.2204 | Complete split-qualified backfill |

HF sources:

```text
TPH:
duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs:
  runs/{levirship,tinyperson}/seed_{42,43}/

STW seed43:
duyle2408/stw-yolo-standard-seed43-backfill-runs:
  runs/{levirship,tinyperson}/seed_43/evaluation_metrics.json
```

STW seed42 source checkpoints remain linked to `duyle2408/stw-yolo-runs`; the split-qualified backfill and bucket metrics are recorded above.

### STW-YOLO seed42 split-qualified backfill (2026-11-06)

The seed42 checkpoints were found in `duyle2408/stw-yolo-runs` and evaluated on the authorized Marimo server. This replaces the prior aggregate-only status for LEVIR-Ship and TinyPerson. The legacy STW head required a compatibility fallback for the absent `box_detail` module; no weights were changed. Standard native val/test evaluation used NMS IoU 0.50. Bucket metrics use the project TinyBenchmark evaluator on native images.

| Dataset | Split | AP50 | mAP50-95 | AP50-small | AP50-medium | AP50-tiny1 | AP50-tiny2 | AP50-tiny3 | mAP50-75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | val | 0.8321 | 0.3325 | 0.8340 | 0.7899 | -- | 0.3368 | 0.7375 | 0.5062 |
| LEVIR-Ship | test | 0.7934 | 0.2861 | 0.8408 | 0.7395 | 0.0000 | 0.3910 | 0.7139 | 0.4403 |
| TinyPerson | val | 0.6190 | 0.2313 | 0.5982 | 0.6736 | 0.3758 | 0.5796 | 0.5693 | 0.3304 |
| TinyPerson | test | 0.5684 | 0.2149 | 0.5419 | 0.5510 | 0.2688 | 0.4751 | 0.5422 | 0.2985 |

Artifacts were produced under `/marimo/yolo_code_cp2_seed43_bde5/runs/stw_seed42_bucket_backfill_final` and `/marimo/yolo_code_cp2_seed43_bde5/runs/stw_seed42_tiny_eval`. The HF upload was not requested for this evaluation-only backfill.

## FCOS-SR-TOD seed42 audit


The matched standard-Mosaic seed42 backfill is now complete for LEVIR-Ship and TinyPerson. Both runs used the recovered FCOS config, split seed 42, training seed 42, 640×640, standard Mosaic, 100 epochs, and were evaluated with the same native COCO protocol as the seed43 recovery2 runs.

| Dataset | Training seed | HF artifact | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Native COCO AP-small (val/test) | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| LEVIR-Ship | 42 | `duyle2408/srtod-set-mosaic-matrix-runs:srtod/levir/fcos/standard_mosaic/seed42_backfill_20261105/` | 0.563 | 0.180 | 0.435 | 0.118 | 0.188 / 0.117 | Train, val/test evaluation, and HF upload verified |
| TinyPerson | 42 | `duyle2408/srtod-set-mosaic-matrix-runs:srtod/tinyperson/fcos/standard_mosaic/seed42_backfill_20261105/` | 0.480 | 0.159 | 0.371 | 0.128 | 0.132 / 0.118 | Train, val/test evaluation, and HF upload verified |

The small-object values above are COCO `bbox_mAP_s`, averaged over IoU `0.50:0.95`; they are not AP50-small.

### Native area-bucket evaluation

The matched seed42 val/test predictions were also evaluated with TinyBenchmark area buckets using IoU thresholds 0.50:0.05:0.75.

| Dataset | Split | AP50 | mAP50-75 | AP50-Tiny1 | AP50-Tiny2 | AP50-Tiny3 | AP50-Small | AP50-Medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | Val | 0.6514 | 0.3870 | -- | 0.1376 | 0.4831 | 0.8226 | 0.6588 |
| LEVIR-Ship | Test | 0.4346 | 0.1961 | -- | 0.0000 | 0.3179 | 0.5622 | 0.8845 |
| TinyPerson | Val | 0.4801 | 0.2598 | 0.1655 | 0.3313 | 0.5814 | 0.6696 | 0.7880 |
| TinyPerson | Test | 0.4522 | 0.2589 | 0.1478 | 0.3827 | 0.5475 | 0.6289 | 0.6568 |

Bucket artifacts:

```text
duyle2408/srtod-set-mosaic-matrix-runs:
  srtod/{levir,tinyperson}/fcos/standard_mosaic/seed42_backfill_20261105/bucket_eval/
```

## FCOS-SR-TOD historical and seed43 results

## FCOS-SR-TOD results

| Dataset | Training seed | HF artifact | Test AP50 | Test mAP50-95 | Small-object metric | Protocol/status |
|---|---:|---|---:|---:|---:|---|
| LEVIR-Ship | 42 | `duyle2408/srtod-levir-protocol-runs:srtod_faster/` | 0.611 | 0.188 | 0.186 | COCO `bbox_mAP_s`; SRTOD FasterRCNN, no-Mosaic, 512×512; complete test artifact |
| Varroa | 42 | `duyle2408/srtod-set-mosaic-matrix-runs:srtod/varroa/fcos/seed42/` | 0.903 | 0.331 | 0.310 | COCO `bbox_mAP_s`; FCOS, 640×640; complete historical artifact |
| TinyPerson | 42 | `duyle2408/srtod-tinyperson-protocol-runs:fcos/` | 0.216 | 0.072 | 0.061 | COCO `bbox_mAP_s`; FCOS corner-window Mosaic, 640×640; complete test artifact |
| LEVIR-Ship | 43 | `duyle2408/srtod-set-mosaic-matrix-runs:srtod/levir/fcos/standard_mosaic/seed43_recovery2_20261104/` | 0.652 | 0.207 | 0.6466 | Corrected standard-Mosaic full-split rerun; selected report row |
| Varroa | 43 | `duyle2408/srtod-set-mosaic-matrix-runs:srtod/varroa/fcos/seed43/` | 0.903 | 0.331 | 0.8759 | Evaluation-only rerun; selected report row |
| TinyPerson | 43 | `duyle2408/srtod-set-mosaic-matrix-runs:srtod/tinyperson/fcos/standard_mosaic/seed43_recovery2_20261104/` | 0.373 | 0.129 | 0.3466 | Corrected native corner-window rerun; selected report row |

The newly backfilled seed42 rows in `srtod/.../standard_mosaic/seed42_backfill_20261105/` use the same standard-Mosaic evaluation family as the selected seed43 rows. The older seed42 artifacts listed below are separate historical runs; their COCO `bbox_mAP_s` values must not be mixed with the new AP50-small bucket values.

## Exact seed42 source files

```text
duyle2408/srtod-set-mosaic-matrix-runs:
  srtod/varroa/fcos/seed42/evaluation_metrics.json
  srtod/varroa/fcos/seed42/test/20260919_043206/20260919_043206.json

duyle2408/srtod-levir-protocol-runs:
  run_contract.json
  srtod_faster/job_summary.json
  srtod_faster/test_results/20260923_092425/20260923_092425.json

duyle2408/srtod-tinyperson-protocol-runs:
  run_contract.json
  fcos/job_summary.json
  fcos/test_results/20260923_120432/20260923_120432.json
```

## Seed interpretation

- The matched seed42 backfills for LEVIR-Ship and TinyPerson use FCOS, standard Mosaic, 640×640, split seed42, and the same bucket evaluator family as the selected seed43 rows.
- The old seed42 LEVIR artifact (`srtod_faster`) is a different SRTOD FasterRCNN/no-Mosaic/512×512 run and is retained only as historical provenance.
- The old seed42 TinyPerson artifact is a separate older FCOS/window-Mosaic run and is retained only as historical provenance.
- Do not mix the old historical rows with the matched standard-Mosaic seed42/43 pair when calculating mean ± standard deviation.
- No verified FCOS-SR-TOD seed44 result was found.

## Related reports

- [`SOTA_main_report.md`](SOTA_main_report.md)
- [`validation_results_three_datasets.md`](validation_results_three_datasets.md)

## Reproducible result manifest

`sota_results_reproducible_20261106.json` is now complete for the requested merged fields. It records source-image `test_merged/AP50`, `test_merged/mAP50-95`, and `test_merged/AP50-Small` for both seeds of all four selected families; native corner-window size metrics remain separate.

The previously missing merged fields are now verified: STW seed43 `0.5956818/0.2245725/0.7096406`, TPH seed43 `0.5019472/0.1848113/0.5720687`, and FCOS-SR-TOD seed42 `0.4550079/0.1621428/0.5944634` for AP50/mAP50-95/AP50-Small.
