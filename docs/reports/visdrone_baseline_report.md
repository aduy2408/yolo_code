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

## Protocol and provenance

- YOLOv5/v8 TPH source: `duyle2408/visdrone-yolov5-yolov8-tph-runs`
- YOLOv9/v10/11 TPH source: `duyle2408/visdrone-yolov9-yolov10-yolo11-tph-runs`
- YOLO manifests verified with `imgsz: 1536`, batch `8`, workers `8`, Mosaic, and NMS IoU `0.5`.
- The earlier repository `duyle2408/visdrone-yolo-baselines-runs` contains separate `imgsz: 640` runs and is not used in the corrected YOLO 1536 table.
- MMDetection source: `duyle2408/visdrone2019-mmdet-baselines-1536-seeds`
- MMDetection executable: `/marimo/mmdet-venv/bin/python`
- MMDetection remote area-metric artifacts: `/marimo/visdrone_full_eval_1536/`
- YOLO artifacts contain `val/AP50`, `val/mAP50-95`, `test/AP50`, `test/mAP50-95`, and test AP50 size buckets. They do not expose the same full COCO area-field set as the MMDetection re-evaluation.

## Reproducibility

- YOLO training matrix in the corrected table: YOLOv5n, YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n, Mosaic, image size 1536.
- MMDetection training matrix: Faster R-CNN, Cascade R-CNN, RetinaNet, RTMDet-S, FCOS, seeds 42-43.
- The YOLOv5n TPH repository currently contains seeds 43 and 44 only, so its aggregate uses n=2. Other listed YOLO models use seeds 42-44.
- Required report artifact: `yolo_related/docs/reports/visdrone_baseline_report.md`.
