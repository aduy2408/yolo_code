# VisDrone2019-DET baseline results at 1536 pixels

**Updated:** 2026-10-01 (ICT)
**Dataset:** official VisDrone2019-DET train / val / test-dev split
**Common settings:** image size `1536×1536`, batch `8`, workers `8`, NMS IoU `0.5`, Mosaic-style training augmentation.

This report contains separate YOLO and MMDetection tables. The corrected YOLO table now uses the TPH repositories identified from the prior run: `duyle2408/visdrone-yolov9-yolov10-yolo11-tph-runs` and `duyle2408/visdrone-yolov5-yolov8-tph-runs`. Their manifests explicitly record `imgsz: 1536`. Values are seed means. YOLOv5n has two available seeds, 43 and 44. The other YOLO models have three seeds, 42, 43, and 44.

## 1. YOLO baseline results, `imgsz=1536`

These are the verified TPH YOLO Mosaic runs from the two Hugging Face repositories above. The uploaded artifacts expose validation aggregate metrics, test aggregate metrics, and test AP50 size buckets. `Tiny1`, `Tiny2`, `Tiny3`, `Small`, and `Medium` are the existing size-bucket fields from `evaluation_metrics.json`.

| Model | Augmentation | n | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | test/AP50-Tiny1 | test/AP50-Tiny2 | test/AP50-Tiny3 | test/AP50-Small | test/AP50-Medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | Mosaic | 2 | 0.5204 | 0.3136 | 0.4247 | 0.2444 | 0.0935 | 0.2945 | 0.5118 | 0.7176 | 0.8351 |
| YOLOv8n | Mosaic | 3 | 0.5306 | 0.3210 | 0.4300 | 0.2481 | 0.1022 | 0.3036 | 0.5227 | 0.7285 | 0.8438 |
| YOLOv9t | Mosaic | 3 | 0.5461 | 0.3338 | 0.4467 | 0.2592 | 0.1014 | 0.3077 | 0.5266 | 0.7329 | 0.8514 |
| YOLOv10n | Mosaic | 3 | 0.5011 | 0.3100 | 0.4206 | 0.2457 | 0.0826 | 0.2733 | 0.4905 | 0.6938 | 0.8228 |
| YOLO11n | Mosaic | 3 | 0.5237 | 0.3173 | 0.4348 | 0.2502 | 0.0978 | 0.3108 | 0.5253 | 0.7232 | 0.8442 |

## 2. MMDetection baseline results

MMDetection models use the shared CachedMosaic, RandomAffine, HSV, horizontal-flip, resize, and padding pipeline recorded in the run manifests. The remote MMDetection artifact reports the test aggregate AP fields and the recorded `AP50` size buckets shown above. It does not provide validation size buckets in the consolidated source report, so no validation size value is inferred.

| Model | n | val/AP50 | val/AP50-95 | test/AP | test/AP50 | test/AP-small | test/AP50-small | test/AP50-medium | test/AP50-large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cascade R-CNN R50-FPN | 2 | 0.418 | 0.259 | 0.1981 | 0.3348 | 0.1104 | 0.2210 | 0.5083 | 0.3889 |
| Faster R-CNN R50-FPN | 2 | 0.404 | 0.231 | 0.1809 | 0.3282 | 0.0946 | 0.2072 | 0.5034 | 0.3521 |
| RetinaNet R50-FPN | 2 | 0.378 | 0.227 | 0.1771 | 0.3102 | 0.0839 | 0.1775 | 0.4946 | 0.3581 |
| FCOS R50-FPN-GN | 2 | 0.283 | 0.155 | 0.1329 | 0.2525 | 0.0671 | 0.1536 | 0.3881 | 0.2769 |
| RTMDet-S* | 2 | 0.417 | 0.238 | 0.1243 | 0.2371 | 0.0836 | 0.1820 | 0.2954 | 0.1392 |

`*` RTMDet test area metrics use the fixed-shape recovery evaluation because the original uploaded 1536 test config produced a CSPNeXt-PAN feature-map size mismatch on the evaluation server. Its original standard-protocol test metrics remain authoritative: seed42 `test/mAP50-95 = 0.194`, seed43 `test/mAP50-95 = 0.189`. The recovery row must not be mixed with the original-protocol RTMDet ranking.

## 3. DETR-R18 and RT-DETR-R18 VisDrone matrix runs

These additional runs come from `duyle2408/detr_r18_rtdetr_r18_matrix_runs`, under `detr_matrix/visdrone/`. Their manifests record official VisDrone train/val/test-dev, image size `1536×1536`, batch `8`, workers `8`, MuSGD `lr=0.01`, patience `15`, split seed `42`, and training seeds `42` and `43`. The table below reports the test metrics recorded in the uploaded MMDetection JSON logs. A separate validation-metric artifact was not present for these VisDrone matrix prefixes, so validation values are not inferred.

| Model | Seed | val/AP50 | val/AP50-95 | test/AP | test/AP50 | test/AP75 | test/AP-small | test/AP-medium | test/AP-large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DETR-R18 | 42 | not uploaded | not uploaded | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| DETR-R18 | 43 | not uploaded | not uploaded | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| RT-DETR-R18 | 42 | not uploaded | not uploaded | 0.000004 | 0.000022 | 0.000000 | 0.000000 | 0.000007 | 0.000022 |
| RT-DETR-R18 | 43 | not uploaded | not uploaded | 0.000075 | 0.000256 | 0.000000 | 0.000000 | 0.000102 | 0.000010 |

**Interpretation caution:** these are the metrics recorded by the uploaded test JSON logs, not a new evaluation. The DETR logs report all-zero test metrics, and the RT-DETR logs report near-zero metrics. They should be kept as a separate matrix result and not silently merged into the main YOLO/MMDetection ranking until the uploaded predictions and an independent COCO re-evaluation are verified.

## Protocol and provenance

- DETR/RT-DETR matrix source: `duyle2408/detr_r18_rtdetr_r18_matrix_runs`
- DETR matrix prefixes: `detr_matrix/visdrone/detr_r18/seed42` and `seed43`
- RT-DETR matrix prefixes: `detr_matrix/visdrone/rtdetr_r18/seed42` and `seed43`
- DETR config: `configs/detr/detr_r18_8xb2-500e_coco.py`
- RT-DETR config/repository: `rtdetr_r18vd_8xe-72e_coco.py` from `flytocc/rtdetr-mmdet`, commit `66365c1553ffd121ca4ee2be9d091735faf3c182`
- DETR/RT-DETR matrix source commit: `e069bcee9adba52fda5bf19f71ea0087b6b021e5`
- YOLOv9/v10/11 TPH source: `duyle2408/visdrone-yolov9-yolov10-yolo11-tph-runs`
- YOLO manifests verified with `imgsz: 1536`, batch `8`, workers `8`, Mosaic, and NMS IoU `0.5`.
- The earlier repository `duyle2408/visdrone-yolo-baselines-runs` contains separate `imgsz: 640` runs and is not used in the corrected YOLO 1536 table.
- MMDetection source: `duyle2408/visdrone2019-mmdet-baselines-1536-seeds`
- MMDetection executable: `/marimo/mmdet-venv/bin/python`
- MMDetection remote area-metric artifacts: `/marimo/visdrone_full_eval_1536/`
- YOLO artifacts contain `val/AP50`, `val/mAP50-95`, `test/AP50`, `test/mAP50-95`, and test AP50 size buckets. They do not expose the same full COCO area-field set as the MMDetection re-evaluation.

## Reproducibility

- YOLO training matrix in the corrected table: YOLOv5n, YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n, Mosaic, image size 1536.
- DETR-R18 and RT-DETR-R18 matrix: image size `1536×1536`, batch `8`, workers `8`, 100 epochs maximum, patience `15`, AMP off, MuSGD `lr=0.01`, split seed `42`, training seeds `42` and `43`.
- MMDetection training matrix: Faster R-CNN, Cascade R-CNN, RetinaNet, RTMDet-S, FCOS, seeds 42-43.
- The YOLOv5n TPH repository currently contains seeds 43 and 44 only, so its aggregate uses n=2. Other listed YOLO models use seeds 42-44.
- The uploaded VisDrone matrix test logs do not include a separate validation metrics artifact; the report leaves those cells unavailable rather than inferring them.
- Required report artifact: `yolo_related/docs/reports/visdrone_baseline_report.md`.
