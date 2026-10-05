# MMDetection Baseline Runs: Varroa, LEVIR-Ship, TinyPerson

**Updated:** 2026-10-05  
**Base report:** `yolo_related/docs/reports/complete_main_report.md`  
**Scope:** YOLO-matched MMDetection baseline runs only. SOTA and architecture variants are excluded from the main matrix.  
**Metric policy:** values are copied from remote Hugging Face artifacts without inference. `--` means the requested field is absent from the source artifact and should be evaluated or backfilled on the server.

## 1. Executive status

| Dataset | HF repository | Verified jobs | Protocols | Seeds | Metric status |
|---|---|---:|---|---|---|
| Varroa | `duyle2408/varroa_mmdet_yolo_protocol_runs` | 24 | `no_mosaic`, `mosaic` | 42, 43 | Test metrics present. Validation metrics are absent from the uploaded aggregate artifact. |
| LEVIR-Ship | `duyle2408/levirship_mmdet_yolo_protocol_seed42_runs` + `duyle2408/levirship_mmdet_yolo_protocol_seed43_runs` | 24 planned baseline prefixes | `no_mosaic`, `mosaic` | 42, 43 | Both seeds are present. Seed42 no-mosaic RetinaNet lacks `final_results.json`; seed43 rows are complete. |
| TinyPerson | `duyle2408/tinyperson_mmdet_yolo_protocol_seed42_runs` + `duyle2408/tinyperson_mmdet_yolo_protocol_seed43_runs` | 24 planned baseline prefixes | `no_mosaic`, `mosaic` | 42, 43 | Both seeds are present. Seed43 no-mosaic RetinaNet lacks `final_results.json`; other rows have validation/test metrics. |

The remote configs inspected for the YOLO-matched matrix record **MuSGD, 100 epochs, early-stop patience 15, batch size 8, split seed 42, and NMS IoU 0.5**. Varroa records image size 640x640. LEVIR-Ship and TinyPerson retain their dataset-specific baseline resize/window protocols.

## 2. Complete validation/test metric table

Columns are AP50, AP75, mAP50-95, AP50-Small, and TinyPerson AP50-Tiny1/2/3, split by validation and test. TinyPerson Tiny1/2/3 are dataset-specific metrics. `--` is not a zero.

| Dataset | Protocol | Seed | Model | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Val T1 | Val T2 | Val T3 | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Test T1 | Test T2 | Test T3 | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LEVIR-Ship | no_mosaic | 42 | fcos | 0.7227 | 0.0945 | 0.2487 | 0.7222 | -- | -- | -- | 0.7602 | 0.0988 | 0.2872 | 0.7583 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | faster_rcnn | 0.7420 | 0.1318 | 0.2846 | 0.7459 | -- | -- | -- | 0.7344 | 0.1297 | 0.2749 | 0.7331 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | cascade_rcnn | 0.7265 | 0.0957 | 0.2568 | 0.7316 | -- | -- | -- | 0.6556 | 0.0959 | 0.2310 | 0.6512 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | rtmdet | 0.5875 | 0.1378 | 0.2435 | 0.5889 | -- | -- | -- | 0.5804 | 0.0802 | 0.2162 | 0.5857 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | atss | 0.7370 | 0.1029 | 0.2611 | 0.7372 | -- | -- | -- | 0.7714 | 0.1002 | 0.2795 | 0.7679 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | retinanet | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | missing `final_results.json`; re-eval/finalize |
| LEVIR-Ship | mosaic | 42 | fcos | 0.7306 | 0.1109 | 0.2726 | 0.7336 | -- | -- | -- | 0.6991 | 0.0837 | 0.2455 | 0.6947 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | faster_rcnn | 0.7262 | 0.1275 | 0.2776 | 0.7299 | -- | -- | -- | 0.7073 | 0.1100 | 0.2560 | 0.7056 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | cascade_rcnn | 0.7596 | 0.1151 | 0.2850 | 0.7614 | -- | -- | -- | 0.6479 | 0.0968 | 0.2347 | 0.6450 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | rtmdet | 0.7186 | 0.1205 | 0.2805 | 0.7272 | -- | -- | -- | 0.6341 | 0.0897 | 0.2313 | 0.6402 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | atss | 0.7486 | 0.0954 | 0.2647 | 0.7476 | -- | -- | -- | 0.7556 | 0.1212 | 0.2866 | 0.7563 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | retinanet | 0.6990 | 0.1022 | 0.2511 | 0.6994 | -- | -- | -- | 0.7378 | 0.1215 | 0.2829 | 0.7391 | -- | -- | -- | complete |
| TinyPerson | no_mosaic | 42 | fcos | 0.4154 | 0.0608 | 0.1411 | 0.5449 | 0.1215 | 0.3972 | 0.4781 | 0.4305 | 0.0672 | 0.1484 | 0.5666 | 0.1079 | 0.3299 | 0.4979 | complete |
| TinyPerson | no_mosaic | 42 | faster_rcnn | 0.3092 | 0.0557 | 0.1105 | 0.5954 | 0.0000 | 0.1604 | 0.4499 | 0.3688 | 0.0682 | 0.1361 | 0.6278 | 0.0000 | 0.1326 | 0.4649 | complete |
| TinyPerson | no_mosaic | 42 | cascade_rcnn | 0.5106 | 0.0923 | 0.1903 | 0.6034 | 0.2788 | 0.5493 | 0.5235 | 0.5139 | 0.0909 | 0.1896 | 0.6025 | 0.2639 | 0.4869 | 0.5501 | complete |
| TinyPerson | no_mosaic | 42 | rtmdet | 0.5061 | 0.0873 | 0.1818 | 0.6124 | 0.2627 | 0.5265 | 0.5154 | 0.5056 | 0.0953 | 0.1844 | 0.6118 | 0.2103 | 0.4712 | 0.5607 | complete |
| TinyPerson | no_mosaic | 42 | atss | 0.4392 | 0.0706 | 0.1525 | 0.5617 | 0.1674 | 0.4519 | 0.4698 | 0.4413 | 0.0718 | 0.1553 | 0.5768 | 0.1351 | 0.3593 | 0.5008 | complete |
| TinyPerson | no_mosaic | 42 | retinanet | 0.4790 | 0.0549 | 0.1514 | 0.5530 | 0.2278 | 0.5630 | 0.4833 | 0.4668 | 0.0463 | 0.1497 | 0.5383 | 0.2172 | 0.4686 | 0.5203 | complete |
| TinyPerson | mosaic | 42 | fcos | 0.4060 | 0.0557 | 0.1394 | 0.5862 | 0.1267 | 0.3710 | 0.4143 | 0.4415 | 0.0693 | 0.1523 | 0.5981 | 0.1252 | 0.3395 | 0.5013 | complete |
| TinyPerson | mosaic | 42 | faster_rcnn | 0.2904 | 0.0515 | 0.1084 | 0.5822 | 0.0000 | 0.1467 | 0.3820 | 0.3526 | 0.0681 | 0.1329 | 0.6185 | 0.0000 | 0.1229 | 0.4326 | complete |
| TinyPerson | mosaic | 42 | cascade_rcnn | 0.5272 | 0.0979 | 0.1981 | 0.5981 | 0.2737 | 0.5932 | 0.5218 | 0.4810 | 0.0944 | 0.1820 | 0.5865 | 0.1991 | 0.4270 | 0.5238 | complete |
| TinyPerson | mosaic | 42 | rtmdet | 0.5054 | 0.0861 | 0.1837 | 0.5813 | 0.2520 | 0.5368 | 0.5091 | 0.5191 | 0.0980 | 0.1902 | 0.6348 | 0.2179 | 0.4693 | 0.5640 | complete |
| TinyPerson | mosaic | 42 | atss | 0.4298 | 0.0660 | 0.1500 | 0.5995 | 0.1571 | 0.4366 | 0.4949 | 0.4469 | 0.0707 | 0.1550 | 0.5984 | 0.1356 | 0.3639 | 0.5135 | complete |
| TinyPerson | mosaic | 42 | retinanet | 0.4515 | 0.0532 | 0.1484 | 0.5020 | 0.2273 | 0.5134 | 0.4677 | 0.4344 | 0.0478 | 0.1405 | 0.4980 | 0.2059 | 0.4146 | 0.4925 | complete |
| Varroa | no_mosaic | 42 | fcos | -- | -- | -- | -- | -- | -- | -- | 0.8987 | 0.0997 | 0.3257 | 0.8875 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | faster_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8929 | 0.0919 | 0.3241 | 0.8750 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | cascade_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8802 | 0.1006 | 0.3128 | 0.8534 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | rtmdet | -- | -- | -- | -- | -- | -- | -- | 0.9047 | 0.1088 | 0.3387 | 0.8930 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | atss | -- | -- | -- | -- | -- | -- | -- | 0.8892 | 0.1021 | 0.3169 | 0.8547 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | retinanet | -- | -- | -- | -- | -- | -- | -- | 0.8819 | 0.1028 | 0.3268 | 0.8727 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | fcos | -- | -- | -- | -- | -- | -- | -- | 0.9133 | 0.0983 | 0.3257 | 0.8985 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | faster_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8854 | 0.0910 | 0.3108 | 0.8676 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | cascade_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8774 | 0.0879 | 0.3085 | 0.8408 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | rtmdet | -- | -- | -- | -- | -- | -- | -- | 0.9191 | 0.1193 | 0.3427 | 0.8986 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | atss | -- | -- | -- | -- | -- | -- | -- | 0.8793 | 0.0859 | 0.3086 | 0.8588 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | retinanet | -- | -- | -- | -- | -- | -- | -- | 0.8923 | 0.1036 | 0.3303 | 0.8673 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | fcos | -- | -- | -- | -- | -- | -- | -- | 0.8939 | 0.1089 | 0.3207 | 0.8671 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | faster_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8886 | 0.1013 | 0.3216 | 0.8764 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | cascade_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8929 | 0.1084 | 0.3157 | 0.8709 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | rtmdet | -- | -- | -- | -- | -- | -- | -- | 0.9050 | 0.1257 | 0.3447 | 0.8795 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | atss | -- | -- | -- | -- | -- | -- | -- | 0.8778 | 0.0932 | 0.3218 | 0.8434 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | retinanet | -- | -- | -- | -- | -- | -- | -- | 0.8938 | 0.1105 | 0.3316 | 0.8630 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | fcos | -- | -- | -- | -- | -- | -- | -- | 0.9007 | 0.1162 | 0.3247 | 0.8623 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | faster_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8857 | 0.1036 | 0.3198 | 0.8741 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | cascade_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8913 | 0.1134 | 0.3267 | 0.8743 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | rtmdet | -- | -- | -- | -- | -- | -- | -- | 0.9134 | 0.1166 | 0.3289 | 0.8991 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | atss | -- | -- | -- | -- | -- | -- | -- | 0.8927 | 0.0837 | 0.3132 | 0.8615 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | retinanet | -- | -- | -- | -- | -- | -- | -- | 0.8954 | 0.0930 | 0.3352 | 0.8757 | -- | -- | -- | test complete; validation missing |


### 2.1 Seed 43 supplemental table

The original table above contains seed42 for LEVIR-Ship and TinyPerson. The following rows are the additional seed43 runs found in the separate HF repositories. Seed44 was not treated as absent: the corresponding repository request returned unauthorized, so its status remains **unknown** until access is granted.

| Dataset | Protocol | Seed | Model | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Val T1 | Val T2 | Val T3 | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Test T1 | Test T2 | Test T3 | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LEVIR-Ship | no_mosaic | 43 | fcos | 0.7051 | 0.0890 | 0.2485 | 0.7042 | -- | -- | -- | 0.7084 | 0.0642 | 0.2315 | 0.7046 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 43 | faster_rcnn | 0.7548 | 0.1156 | 0.2867 | 0.7558 | -- | -- | -- | 0.7107 | 0.0816 | 0.2524 | 0.7069 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 43 | cascade_rcnn | 0.7465 | 0.1197 | 0.2798 | 0.7515 | -- | -- | -- | 0.6522 | 0.0920 | 0.2422 | 0.6437 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 43 | rtmdet | 0.7062 | 0.1472 | 0.2884 | 0.7094 | -- | -- | -- | 0.6712 | 0.1178 | 0.2658 | 0.6790 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 43 | atss | 0.7265 | 0.1012 | 0.2596 | 0.7289 | -- | -- | -- | 0.7514 | 0.0956 | 0.2812 | 0.7478 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 43 | retinanet | 0.7184 | 0.0840 | 0.2575 | 0.7200 | -- | -- | -- | 0.7076 | 0.0936 | 0.2579 | 0.7093 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 43 | fcos | 0.7372 | 0.0906 | 0.2578 | 0.7430 | -- | -- | -- | 0.7217 | 0.0856 | 0.2496 | 0.7179 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 43 | faster_rcnn | 0.7569 | 0.1203 | 0.2836 | 0.7591 | -- | -- | -- | 0.7020 | 0.0912 | 0.2542 | 0.6980 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 43 | cascade_rcnn | 0.7355 | 0.1176 | 0.2701 | 0.7393 | -- | -- | -- | 0.7128 | 0.1223 | 0.2684 | 0.7051 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 43 | rtmdet | 0.7362 | 0.1329 | 0.2849 | 0.7384 | -- | -- | -- | 0.7558 | 0.1061 | 0.2896 | 0.7581 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 43 | atss | 0.7609 | 0.0868 | 0.2616 | 0.7575 | -- | -- | -- | 0.7472 | 0.1127 | 0.2818 | 0.7417 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 43 | retinanet | 0.7241 | 0.0983 | 0.2583 | 0.7276 | -- | -- | -- | 0.6598 | 0.1046 | 0.2509 | 0.6595 | -- | -- | -- | complete |
| TinyPerson | no_mosaic | 43 | fcos | 0.4269 | 0.0614 | 0.1485 | 0.5662 | 0.1281 | 0.4049 | 0.4853 | 0.4276 | 0.0650 | 0.1469 | 0.5653 | 0.1238 | 0.3284 | 0.4901 | complete |
| TinyPerson | no_mosaic | 43 | faster_rcnn | 0.2895 | 0.0604 | 0.1103 | 0.5605 | 0.0000 | 0.1107 | 0.4132 | 0.3488 | 0.0640 | 0.1285 | 0.6231 | 0.0000 | 0.1001 | 0.4324 | complete |
| TinyPerson | no_mosaic | 43 | cascade_rcnn | 0.5207 | 0.1047 | 0.1946 | 0.5384 | 0.2856 | 0.5646 | 0.5315 | 0.5195 | 0.1015 | 0.1938 | 0.6092 | 0.2392 | 0.5011 | 0.5724 | complete |
| TinyPerson | no_mosaic | 43 | rtmdet | 0.4971 | 0.0878 | 0.1788 | 0.5741 | 0.2416 | 0.5280 | 0.5271 | 0.5109 | 0.0973 | 0.1889 | 0.6168 | 0.2178 | 0.4742 | 0.5684 | complete |
| TinyPerson | no_mosaic | 43 | atss | 0.4331 | 0.0732 | 0.1542 | 0.5988 | 0.1143 | 0.4229 | 0.4937 | 0.4347 | 0.0666 | 0.1514 | 0.5957 | 0.1072 | 0.3416 | 0.5001 | complete |
| TinyPerson | no_mosaic | 43 | retinanet | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | missing `final_results.json`; re-eval/finalize |
| TinyPerson | mosaic | 43 | fcos | 0.4048 | 0.0562 | 0.1354 | 0.5682 | 0.1315 | 0.3869 | 0.4454 | 0.4325 | 0.0623 | 0.1488 | 0.5839 | 0.1189 | 0.3129 | 0.4913 | complete |
| TinyPerson | mosaic | 43 | faster_rcnn | 0.3049 | 0.0532 | 0.1120 | 0.5902 | 0.0000 | 0.1935 | 0.4051 | 0.3722 | 0.0739 | 0.1404 | 0.6238 | 0.0000 | 0.1308 | 0.4652 | complete |
| TinyPerson | mosaic | 43 | cascade_rcnn | 0.5172 | 0.0991 | 0.1912 | 0.5416 | 0.3033 | 0.5817 | 0.5299 | 0.5185 | 0.0986 | 0.1928 | 0.5998 | 0.2653 | 0.5065 | 0.5673 | complete |
| TinyPerson | mosaic | 43 | rtmdet | 0.4910 | 0.0859 | 0.1809 | 0.5625 | 0.2485 | 0.5025 | 0.4985 | 0.5204 | 0.0963 | 0.1907 | 0.6376 | 0.2214 | 0.4707 | 0.5774 | complete |
| TinyPerson | mosaic | 43 | atss | 0.4237 | 0.0612 | 0.1480 | 0.5196 | 0.1623 | 0.4226 | 0.4483 | 0.4348 | 0.0697 | 0.1516 | 0.5555 | 0.1289 | 0.3533 | 0.4833 | complete |
| TinyPerson | mosaic | 43 | retinanet | 0.4496 | 0.0509 | 0.1444 | 0.5091 | 0.2101 | 0.5130 | 0.4548 | 0.4485 | 0.0508 | 0.1437 | 0.5249 | 0.1921 | 0.4349 | 0.5087 | complete |

## 3. Missing metrics and server re-evaluation backlog

### 3.1 Varroa validation metrics

The Varroa remote repository contains complete training/test artifacts and a consolidated `evaluation/20260924/varroa_metrics.json`, but that aggregate artifact is test-only. All 24 Varroa baseline rows therefore have `--` for validation metrics. Run validation-only evaluation on the uploaded best checkpoints and upload one split-qualified metrics artifact per existing prefix.

| Protocols | Seeds | Models | Missing fields |
|---|---|---|---|
| `no_mosaic`, `mosaic` | 42, 43 | fcos, faster_rcnn, cascade_rcnn, rtmdet, atss, retinanet | Val AP50, AP75, mAP50-95, AP50-Small |

### 3.2 LEVIR-Ship and TinyPerson missing rows

- LEVIR-Ship seed42 no-mosaic RetinaNet: checkpoint/config/test artifacts exist, but no split-qualified `final_results.json` was found.
- TinyPerson seed43 no-mosaic RetinaNet: no split-qualified `final_results.json` was found in the seed43 repository.
- Seed44 for both datasets is **unknown**, not confirmed missing. The repository requests returned unauthorized with the current HF credential.

### 3.3 Other fields

- LEVIR-Ship: all other core validation/test fields are present.
- TinyPerson: all core validation/test fields are present, including Tiny1/Tiny2/Tiny3.
- Large-object AP is not included in the main table because the source schema exposes `ap_small` and `ap_medium`, while this report focuses on AP50, AP75, mAP50-95, and AP50-Small. The raw source artifacts remain available for additional fields.

## 4. FCOS-SET special baseline runs

These are separate FCOS-SET runs, not part of the six-model baseline matrix. They are included because they were recorded under the STW protocol repository.

Repository: `duyle2408/set-fcos-stw-protocol-runs`

| Dataset | Protocol | Seed | Model | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LEVIR-Ship | no_mosaic | 42 | fcos_set | 0.7324 | 0.1057 | 0.2598 | 0.7339 | 0.7740 | 0.0926 | 0.2785 | 0.7714 | complete |
| TinyPerson | mosaic | 42 | fcos_set | 0.4148 | 0.0704 | 0.1494 | 0.5516 | 0.4259 | 0.0742 | 0.1519 | 0.5722 | complete |

## 5. Provenance and acceptance evidence

- Varroa manifests: `experiment_manifest.json`, `job_summary.json`, `patched_config.py`, checkpoints, and test results exist under each remote prefix.
- LEVIR-Ship and TinyPerson: `final_results.json`, `patched_config.py`, validation/test metrics, checkpoints, and prediction artifacts exist under the baseline prefixes, except the noted LEVIR no-mosaic RetinaNet `final_results.json` gap.
- Baseline matrix model registry: `fcos`, `faster_rcnn`, `cascade_rcnn`, `rtmdet`, `atss`, `retinanet`.
- Remote repositories were queried through authenticated Hugging Face Hub API on 2026-10-05. The token value is intentionally not recorded in this report.
- Local `mmdetection/work_dirs` is not the completion source of truth. The remote HF artifacts are the authoritative run evidence.

## 6. Recommended server work

1. Re-evaluate the 24 Varroa best checkpoints on validation with split seed 42 and the original dataset-specific protocol.
2. Evaluate the existing LEVIR-Ship seed42 no-mosaic RetinaNet checkpoint and the TinyPerson seed43 no-mosaic RetinaNet row, then upload split-qualified `final_results.json` artifacts.
3. Request or restore access to the seed44 repositories before deciding whether those seeds were run.
4. Upload validation metrics under the existing Varroa prefixes without overwriting the original test metrics.
5. Preserve checkpoint, source commit, model, protocol, seed, NMS IoU, and image-size metadata.
6. After upload, replace only the relevant `--` fields in this report with values from verified remote artifacts.

## 7. Excluded repositories

Repositories containing DynamicVis, PH-DETR, DGFE/API, SRTOD, DINO/DETR, or other architecture and augmentation variants are excluded from the baseline tables unless explicitly listed above. They should be reported separately as SOTA or variant runs to avoid mixing control and treatment experiments.
