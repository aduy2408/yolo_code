# Varroa Dataset And Model Notes

Date: 2026-06-18

## Dataset Location

- Source dataset in this workspace: `/mnt/data/varroa/yolo_related/data`
- Split layout:
  - `data/train/videos`, `data/train/labels`
  - `data/val/videos`, `data/val/labels`
  - `data/test/videos`, `data/test/labels`
- YOLO-converted dataset used in Marimo:
  - `/marimo/data/datasets/varroa_yolo/varroa.yaml`

Example YOLO dataset YAML:

```yaml
path: /marimo/data/datasets/varroa_yolo
train: images/train
val: images/val
test: images/test
nc: 1
names: [varroa]
```

## Dataset Statistics

- Images: 13,507
- Objects after converter-style clamping: 4,626
- Image size: all images are `160 x 280`
- Most images have no object.
- Max objects per image: 3

Positive image rate:

| Split | Positive / Total | Rate |
| --- | ---: | ---: |
| train | 2,553 / 8,223 | 31.0% |
| val | 451 / 1,876 | 24.0% |
| test | 942 / 3,408 | 27.6% |

Native bbox size:

- Median box: about `32 x 31 px`
- 25-75% range: about `27-37 px` wide and `27-37 px` high
- COCO-style size distribution:
  - small: 53.2%
  - medium: 46.8%
  - large: 0%

At `imgsz=640` with letterbox scaling:

- Median min side: about `65 px`
- Median object coverage:
  - P2 stride 4: `16.3` cells
  - P3 stride 8: `8.15` cells
  - P4 stride 16: `4.08` cells
- Only about `0.1%` of objects have min side `<32 px` at 640.

At smaller training sizes:

- `imgsz=320`: median min side about `32.6 px`; P3 sees about 4 cells.
- `imgsz=160`: median min side about `16.3 px`; P3 sees about 2 cells, P2 sees about 4 cells.

## Raw Class 3 Tightness Check

- In the raw `gt_one.csv` files, the second column contains labels/states `0`, `1`, and `3`.
- `gt_filtered.csv` keeps only positive class `1`.
- The YOLO-converted dataset uses `nc: 1`, so raw class `3` is not represented as a separate YOLO class. It is collapsed or omitted depending on the converter path.

Raw `gt_one.csv` counts across `data/{train,val,test}`:

| Raw label/state | Count |
| --- | ---: |
| `0` | 9,562 |
| `1` | 3,083 |
| `3` | 864 |

Bbox tightness was measured with a dark-pixel proxy inside each annotated box, not with segmentation masks:

| Raw label/state | Mean box size | Mean proxy margin |
| --- | ---: | ---: |
| `1` | `33.1 x 31.9 px` | `0.50 px` |
| `3` | `32.3 x 31.5 px` | `0.36 px` |

By this heuristic, raw class `3` boxes are at least as tight as class `1` boxes, and likely slightly tighter. Re-run the read-only check after any dataset converter changes, verify counts against `data/{train,val,test}/gt_one.csv`, and spot-check several class `3` images visually before treating the proxy metric as final evidence.

## Data Quality Notes

- The first line in many original label files does not match the number of parsed boxes.
- This is not fatal for current training because `prepare_dataset.py` ignores the first line and reads boxes from later lines.
- One train box is invalid or outside bounds and gets dropped by converter-style clamping.
- Bigger risk: limited diversity. The data comes from only 13 videos, so frames are heavily correlated and validation/test difficulty can depend strongly on video split.

## Model Decision Check

Current decision: use a YOLOv8 model with P2 detection and WIoU for localization.

This is reasonable for the dataset, with one caveat:

- P2 is useful for localization precision because the native objects are small and bbox quality matters.
- At `imgsz=640`, the objects are not extremely small anymore. P3 already sees the median object at about 8 cells, and P4 sees about 4 cells.
- Therefore, P2 should help with tighter boxes, but it is not expected to be the only factor controlling recall.
- WIoU plus higher `box` and `dfl` gains is a good fit for improving localization quality.
- Heavy attention on high-resolution P2 is not the first thing to optimize because it can add significant compute while the dataset is diversity-limited.

Recommended ablation order:

1. `P2-P5 + WIoU`: keep the 4-head model as the accuracy-oriented baseline.
2. `P2-P4 + WIoU`: drop the P5 detection branch to reduce inference/training cost; likely safe because there are no large objects.
3. `P3-P5 + WIoU`: compare against no-P2 to measure whether P2 is actually improving box quality at `imgsz=640`.
4. Try `imgsz=512` or `320` only if speed is more important than max localization accuracy. At smaller image sizes, P2 becomes more important.

Practical recommendation:

- Keep `models_related/models_config/yolov8_varroa_p2_wiou.yaml` as the main 4-head baseline.
- Add a separate `P2-P4` YAML for speed ablation rather than modifying the 4-head baseline.
- Do not remove P4 initially; `P2-P3` alone may lose context and increase false positives.
