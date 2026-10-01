# VisDrone2019-DET baseline results, provenance audit

**Updated:** 2026-10-01 (ICT)
**Dataset:** official VisDrone2019-DET train / val / test-dev split
**MMDetection settings:** image size `1536×1536`, batch `8`, workers `8`, NMS IoU `0.5`.

**Important provenance correction:** the YOLO artifacts previously used in this report are **not 1536-pixel runs**. Their Hugging Face manifests explicitly record `imgsz: 640`. Therefore the YOLO numbers below must not be compared as 1536-pixel results against MMDetection. The MMDetection rows remain 1536-pixel results.

## 1. YOLO baseline results currently available, confirmed as 640 pixels

The table below is retained for provenance, but it is **not** a 1536-pixel table. The source manifests for `duyle2408/visdrone-yolo-baselines-runs` record `imgsz: 640` for the runs, including YOLOv9t Mosaic seed42 and YOLO11n Mosaic seed42. Values are seed means. YOLO rows aggregate seeds 42, 43, and 44.

The candidate repository `duyle2408/visdrone2019-mmdet-yoloaug-1536-runs` currently exposes no run artifacts beyond `.gitattributes`, so verified YOLO 1536 metrics are not available from the checked Hugging Face sources.

| Model | Augmentation | n | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | test/AP50-Tiny1 | test/AP50-Tiny2 | test/AP50-Tiny3 | test/AP50-Small | test/AP50-Medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Status** | **YOLO table below is 640-pixel provenance only** | — | — | — | — | — | — | — | — | — | — |
| YOLOv5n | Mosaic | 3 | 0.3252 | 0.1793 | 0.2716 | 0.1474 | 0.0124 | 0.0832 | 0.2636 | 0.4998 | 0.7319 |
| YOLOv5n | No Mosaic | 3 | 0.3126 | 0.1712 | 0.2596 | 0.1401 | 0.0099 | 0.0760 | 0.2497 | 0.4874 | 0.7204 |
| YOLOv8n | Mosaic | 3 | 0.3377 | 0.1860 | 0.2776 | 0.1514 | 0.0122 | 0.0945 | 0.2785 | 0.5213 | 0.7415 |
| YOLOv8n | No Mosaic | 3 | 0.3267 | 0.1809 | 0.2689 | 0.1454 | 0.0099 | 0.0849 | 0.2690 | 0.5107 | 0.7317 |
| YOLOv9t | Mosaic | 3 | 0.3367 | 0.1864 | 0.2882 | 0.1572 | 0.0099 | 0.0766 | 0.2732 | 0.5149 | 0.7429 |
| YOLOv9t | No Mosaic | 3 | 0.3312 | 0.1839 | 0.2756 | 0.1502 | 0.0099 | 0.0710 | 0.2546 | 0.4997 | 0.7390 |
| YOLOv10n | Mosaic | 3 | 0.3229 | 0.1812 | 0.2686 | 0.1469 | 0.0123 | 0.0835 | 0.2604 | 0.4880 | 0.7105 |
| YOLOv10n | No Mosaic | 3 | 0.3107 | 0.1747 | 0.2563 | 0.1394 | 0.0099 | 0.0735 | 0.2413 | 0.4697 | 0.6977 |
| YOLO11n | Mosaic | 3 | 0.3350 | 0.1848 | 0.2793 | 0.1519 | 0.0122 | 0.0951 | 0.2844 | 0.5197 | 0.7416 |
| YOLO11n | No Mosaic | 3 | 0.3239 | 0.1788 | 0.2687 | 0.1448 | 0.0099 | 0.0866 | 0.2664 | 0.5033 | 0.7338 |

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

- YOLO source actually used by the table: `duyle2408/visdrone-yolo-baselines-runs`, whose manifests record `imgsz: 640`.
- Candidate YOLO 1536 source checked: `duyle2408/visdrone2019-mmdet-yoloaug-1536-runs`, currently empty apart from `.gitattributes`.
- YOLO artifacts contain `val/AP50`, `val/mAP50-95`, `test/AP50`, `test/mAP50-95`, and test AP50 size buckets. They do not expose the same full COCO area-field set as the MMDetection re-evaluation.
- **Do not use the YOLO table for a 1536-vs-1536 comparison until actual YOLO 1536 checkpoints and manifests are found or rerun.**

## Reproducibility

- YOLO training matrix: YOLOv5n, YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n, Mosaic and no-Mosaic, seeds 42-44.
- MMDetection training matrix: Faster R-CNN, Cascade R-CNN, RetinaNet, RTMDet-S, FCOS, seeds 42-43.
- Required report artifact: `yolo_related/docs/reports/visdrone_baseline_report.md`.
