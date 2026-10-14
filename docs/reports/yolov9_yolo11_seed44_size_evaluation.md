# YOLOv9/YOLO11 seed44 checkpoint evaluation report

**Generated:** 2026-10-14T03:08:52+00:00
**HF output repository:** `duyle2408/yolov9-yolo11-seed44-size-eval-runs`
**HF source repository:** `duyle2408/yolov9-yolo11-cp-nc-seed44-runs`
**Coverage:** 20/20 evaluation leaves with `size_metrics_complete.json`
**Commit:** `783b7abcf37ebd88cefa21b9bf4d3b484fd6fe15`
**Seeds:** training seed 44, split seed 42
**NMS IoU:** 0.50
**Image size:** 640
**Batch/workers:** 8 / 8

## 1. Completion and protocols

- `evaluation_metrics.json`: **20/20**
- `size_metrics_complete.json`: **20/20**
- Core metrics required on every row: `val/AP50`, `val/mAP50-95`, `test/AP50`, `test/mAP50-95`.
- LEVIR/Varroa test protocol: standard held-out test split.
- TinyPerson test protocol: official corner-window merged evaluator, reported separately as `test_merged/*`.
- Native size protocol: IoU thresholds 0.50:0.05:0.75, maxDets=200.

## 2. Core metrics and headline bucket metrics

| Model | Dataset | Variant | Mosaic | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | val AP75 | val AP50-Small | test AP75 | test AP50-Small | merged AP50-Small |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov11 | levir | cp1_single1 | no_mosaic | 0.831106 | 0.328351 | 0.812895 | 0.305535 | 0.161185 | 0.849258 | 0.122272 | 0.853481 | -- |
| yolov11 | levir | cp2_single2 | no_mosaic | 0.821948 | 0.328284 | 0.806159 | 0.306006 | 0.163629 | 0.835423 | 0.140718 | 0.844701 | -- |
| yolov11 | levir | negative_canvas_r2 | no_mosaic | 0.824498 | 0.326964 | 0.817999 | 0.297791 | 0.154239 | 0.839309 | 0.138407 | 0.863412 | -- |
| yolov11 | levir | negative_canvas_r4 | no_mosaic | 0.826592 | 0.331348 | 0.793301 | 0.303076 | 0.143869 | 0.829160 | 0.138473 | 0.806672 | -- |
| yolov11 | tinyperson | cp1_single1 | mosaic | 0.549654 | 0.201250 | 0.522657 | 0.188030 | 0.097780 | 0.544692 | 0.089867 | 0.537896 | 0.697463 |
| yolov11 | tinyperson | cp2_single2 | mosaic | 0.541952 | 0.201215 | 0.515849 | 0.188188 | 0.105409 | 0.509129 | 0.090278 | 0.539168 | 0.688307 |
| yolov11 | tinyperson | negative_canvas_r2 | mosaic | 0.538402 | 0.197360 | 0.513245 | 0.184560 | 0.099537 | 0.511639 | 0.087433 | 0.520366 | 0.678929 |
| yolov11 | tinyperson | negative_canvas_r4 | mosaic | 0.539917 | 0.199006 | 0.520172 | 0.186942 | 0.106216 | 0.475210 | 0.086422 | 0.525918 | 0.685296 |
| yolov11 | varroa | cp1_single1 | mosaic | 0.882046 | 0.327633 | 0.856823 | 0.312937 | 0.100032 | 0.800733 | 0.090939 | 0.774276 | -- |
| yolov11 | varroa | cp2_single2 | mosaic | 0.880828 | 0.326859 | 0.858780 | 0.320274 | 0.105185 | 0.775007 | 0.096206 | 0.782445 | -- |
| yolov9 | levir | cp1_single1 | no_mosaic | 0.835017 | 0.324726 | 0.811530 | 0.304400 | 0.140337 | 0.854773 | 0.145233 | 0.854333 | -- |
| yolov9 | levir | cp2_single2 | no_mosaic | 0.805009 | 0.333446 | 0.796946 | 0.308439 | 0.158256 | 0.811671 | 0.145211 | 0.821976 | -- |
| yolov9 | levir | negative_canvas_r2 | no_mosaic | 0.820313 | 0.323856 | 0.804971 | 0.305774 | 0.171862 | 0.838582 | 0.144926 | 0.823360 | -- |
| yolov9 | levir | negative_canvas_r4 | no_mosaic | 0.805172 | 0.313601 | 0.784613 | 0.294080 | 0.150070 | 0.828916 | 0.130780 | 0.823127 | -- |
| yolov9 | tinyperson | cp1_single1 | mosaic | 0.458256 | 0.164463 | 0.478797 | 0.170250 | 0.084957 | 0.460689 | 0.080454 | 0.511270 | 0.647098 |
| yolov9 | tinyperson | cp2_single2 | mosaic | 0.467739 | 0.167467 | 0.478731 | 0.169195 | 0.084546 | 0.464871 | 0.083708 | 0.522089 | 0.652089 |
| yolov9 | tinyperson | negative_canvas_r2 | mosaic | 0.467195 | 0.168368 | 0.475247 | 0.169444 | 0.093779 | 0.443261 | 0.078555 | 0.501528 | 0.640092 |
| yolov9 | tinyperson | negative_canvas_r4 | mosaic | 0.460574 | 0.166524 | 0.471195 | 0.166707 | 0.085012 | 0.454521 | 0.076592 | 0.503795 | 0.628518 |
| yolov9 | varroa | cp1_single1 | mosaic | 0.885457 | 0.322798 | 0.874110 | 0.322858 | 0.096269 | 0.770625 | 0.103062 | 0.758396 | -- |
| yolov9 | varroa | cp2_single2 | mosaic | 0.889610 | 0.328582 | 0.864908 | 0.313520 | 0.102776 | 0.791979 | 0.096016 | 0.776832 | -- |

## 3. Full per-run bucket metrics

Each subsection preserves split-qualified names. `-1.000000` means the evaluator had no eligible ground-truth objects for that bucket, not a zero score.

### 1. yolov11 / levir / cp1_single1 / no_mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/levir/copy_paste/cp1_single1/no_mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/levir/copy_paste/cp1_single1/no_mosaic/seed_44/evaluation_metrics.json`
- Test protocol: LEVIR-Ship standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.831106 |
| `val/mAP50-95` | 0.328351 |
| `test/AP50` | 0.812895 |
| `test/mAP50-95` | 0.305535 |
| `val_size/AP75` | 0.161185 |
| `val_size/AP50-Tiny1` | -1.000000 |
| `val_size/AP50-Tiny2` | 0.148515 |
| `val_size/AP50-Tiny3` | 0.693608 |
| `val_size/AP50-Small` | 0.849258 |
| `val_size/AP50-Medium` | 0.801980 |
| `test_size/AP75` | 0.122272 |
| `test_size/AP50-Tiny1` | 0.000000 |
| `test_size/AP50-Tiny2` | 0.262376 |
| `test_size/AP50-Tiny3` | 0.710674 |
| `test_size/AP50-Small` | 0.853481 |
| `test_size/AP50-Medium` | 0.797383 |


### 2. yolov11 / levir / cp2_single2 / no_mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/levir/copy_paste/cp2_single2/no_mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/levir/copy_paste/cp2_single2/no_mosaic/seed_44/evaluation_metrics.json`
- Test protocol: LEVIR-Ship standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.821948 |
| `val/mAP50-95` | 0.328284 |
| `test/AP50` | 0.806159 |
| `test/mAP50-95` | 0.306006 |
| `val_size/AP75` | 0.163629 |
| `val_size/AP50-Tiny1` | -1.000000 |
| `val_size/AP50-Tiny2` | 0.252475 |
| `val_size/AP50-Tiny3` | 0.709944 |
| `val_size/AP50-Small` | 0.835423 |
| `val_size/AP50-Medium` | 0.900990 |
| `test_size/AP75` | 0.140718 |
| `test_size/AP50-Tiny1` | 0.000000 |
| `test_size/AP50-Tiny2` | 0.196040 |
| `test_size/AP50-Tiny3` | 0.750175 |
| `test_size/AP50-Small` | 0.844701 |
| `test_size/AP50-Medium` | 0.710258 |


### 3. yolov11 / levir / negative_canvas_r2 / no_mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/levir/copy_paste/negative_canvas_r2/no_mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/levir/copy_paste/negative_canvas_r2/no_mosaic/seed_44/evaluation_metrics.json`
- Test protocol: LEVIR-Ship standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.824498 |
| `val/mAP50-95` | 0.326964 |
| `test/AP50` | 0.817999 |
| `test/mAP50-95` | 0.297791 |
| `val_size/AP75` | 0.154239 |
| `val_size/AP50-Tiny1` | -1.000000 |
| `val_size/AP50-Tiny2` | 0.140028 |
| `val_size/AP50-Tiny3` | 0.704801 |
| `val_size/AP50-Small` | 0.839309 |
| `val_size/AP50-Medium` | 0.801980 |
| `test_size/AP75` | 0.138407 |
| `test_size/AP50-Tiny1` | 0.000000 |
| `test_size/AP50-Tiny2` | 0.240099 |
| `test_size/AP50-Tiny3` | 0.722021 |
| `test_size/AP50-Small` | 0.863412 |
| `test_size/AP50-Medium` | 0.698401 |


### 4. yolov11 / levir / negative_canvas_r4 / no_mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_44/evaluation_metrics.json`
- Test protocol: LEVIR-Ship standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.826592 |
| `val/mAP50-95` | 0.331348 |
| `test/AP50` | 0.793301 |
| `test/mAP50-95` | 0.303076 |
| `val_size/AP75` | 0.143869 |
| `val_size/AP50-Tiny1` | -1.000000 |
| `val_size/AP50-Tiny2` | 0.175068 |
| `val_size/AP50-Tiny3` | 0.693322 |
| `val_size/AP50-Small` | 0.829160 |
| `val_size/AP50-Medium` | 0.801980 |
| `test_size/AP75` | 0.138473 |
| `test_size/AP50-Tiny1` | 0.000000 |
| `test_size/AP50-Tiny2` | 0.242536 |
| `test_size/AP50-Tiny3` | 0.708428 |
| `test_size/AP50-Small` | 0.806672 |
| `test_size/AP50-Medium` | 0.767822 |


### 5. yolov11 / tinyperson / cp1_single1 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/tinyperson/copy_paste/cp1_single1/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/tinyperson/copy_paste/cp1_single1/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: TinyPerson official corner-window merged evaluator
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.549654 |
| `val/mAP50-95` | 0.201250 |
| `test/AP50` | 0.522657 |
| `test/mAP50-95` | 0.188030 |
| `val_size/AP75` | 0.097780 |
| `val_size/AP50-Tiny1` | 0.253668 |
| `val_size/AP50-Tiny2` | 0.489360 |
| `val_size/AP50-Tiny3` | 0.530012 |
| `val_size/AP50-Small` | 0.544692 |
| `val_size/AP50-Medium` | 0.619773 |
| `test_size/AP75` | 0.089867 |
| `test_size/AP50-Tiny1` | 0.214785 |
| `test_size/AP50-Tiny2` | 0.413740 |
| `test_size/AP50-Tiny3` | 0.488866 |
| `test_size/AP50-Small` | 0.537896 |
| `test_size/AP50-Medium` | 0.558044 |
| `test_merged/AP75` | 0.095099 |
| `test_merged/AP50-Tiny1` | 0.333089 |
| `test_merged/AP50-Tiny2` | 0.540900 |
| `test_merged/AP50-Tiny3` | 0.633400 |
| `test_merged/AP50-Small` | 0.697463 |
| `test_merged/AP50-Medium` | 0.671061 |


### 6. yolov11 / tinyperson / cp2_single2 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/tinyperson/copy_paste/cp2_single2/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/tinyperson/copy_paste/cp2_single2/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: TinyPerson official corner-window merged evaluator
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.541952 |
| `val/mAP50-95` | 0.201215 |
| `test/AP50` | 0.515849 |
| `test/mAP50-95` | 0.188188 |
| `val_size/AP75` | 0.105409 |
| `val_size/AP50-Tiny1` | 0.267211 |
| `val_size/AP50-Tiny2` | 0.484874 |
| `val_size/AP50-Tiny3` | 0.544673 |
| `val_size/AP50-Small` | 0.509129 |
| `val_size/AP50-Medium` | 0.619541 |
| `test_size/AP75` | 0.090278 |
| `test_size/AP50-Tiny1` | 0.202083 |
| `test_size/AP50-Tiny2` | 0.407032 |
| `test_size/AP50-Tiny3` | 0.490211 |
| `test_size/AP50-Small` | 0.539168 |
| `test_size/AP50-Medium` | 0.558715 |
| `test_merged/AP75` | 0.098398 |
| `test_merged/AP50-Tiny1` | 0.313806 |
| `test_merged/AP50-Tiny2` | 0.548555 |
| `test_merged/AP50-Tiny3` | 0.626289 |
| `test_merged/AP50-Small` | 0.688307 |
| `test_merged/AP50-Medium` | 0.666801 |


### 7. yolov11 / tinyperson / negative_canvas_r2 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/tinyperson/copy_paste/negative_canvas_r2/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/tinyperson/copy_paste/negative_canvas_r2/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: TinyPerson official corner-window merged evaluator
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.538402 |
| `val/mAP50-95` | 0.197360 |
| `test/AP50` | 0.513245 |
| `test/mAP50-95` | 0.184560 |
| `val_size/AP75` | 0.099537 |
| `val_size/AP50-Tiny1` | 0.275892 |
| `val_size/AP50-Tiny2` | 0.479089 |
| `val_size/AP50-Tiny3` | 0.532947 |
| `val_size/AP50-Small` | 0.511639 |
| `val_size/AP50-Medium` | 0.624775 |
| `test_size/AP75` | 0.087433 |
| `test_size/AP50-Tiny1` | 0.209067 |
| `test_size/AP50-Tiny2` | 0.425166 |
| `test_size/AP50-Tiny3` | 0.494226 |
| `test_size/AP50-Small` | 0.520366 |
| `test_size/AP50-Medium` | 0.546054 |
| `test_merged/AP75` | 0.094155 |
| `test_merged/AP50-Tiny1` | 0.326327 |
| `test_merged/AP50-Tiny2` | 0.561713 |
| `test_merged/AP50-Tiny3` | 0.629440 |
| `test_merged/AP50-Small` | 0.678929 |
| `test_merged/AP50-Medium` | 0.662349 |


### 8. yolov11 / tinyperson / negative_canvas_r4 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: TinyPerson official corner-window merged evaluator
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.539917 |
| `val/mAP50-95` | 0.199006 |
| `test/AP50` | 0.520172 |
| `test/mAP50-95` | 0.186942 |
| `val_size/AP75` | 0.106216 |
| `val_size/AP50-Tiny1` | 0.256392 |
| `val_size/AP50-Tiny2` | 0.502152 |
| `val_size/AP50-Tiny3` | 0.484234 |
| `val_size/AP50-Small` | 0.475210 |
| `val_size/AP50-Medium` | 0.628356 |
| `test_size/AP75` | 0.086422 |
| `test_size/AP50-Tiny1` | 0.206919 |
| `test_size/AP50-Tiny2` | 0.427737 |
| `test_size/AP50-Tiny3` | 0.491371 |
| `test_size/AP50-Small` | 0.525918 |
| `test_size/AP50-Medium` | 0.544677 |
| `test_merged/AP75` | 0.093219 |
| `test_merged/AP50-Tiny1` | 0.334415 |
| `test_merged/AP50-Tiny2` | 0.559906 |
| `test_merged/AP50-Tiny3` | 0.634922 |
| `test_merged/AP50-Small` | 0.685296 |
| `test_merged/AP50-Medium` | 0.670117 |


### 9. yolov11 / varroa / cp1_single1 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/varroa/copy_paste/cp1_single1/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/varroa/copy_paste/cp1_single1/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: Varroa standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.882046 |
| `val/mAP50-95` | 0.327633 |
| `test/AP50` | 0.856823 |
| `test/mAP50-95` | 0.312937 |
| `val_size/AP75` | 0.100032 |
| `val_size/AP50-Tiny1` | 0.000000 |
| `val_size/AP50-Tiny2` | -1.000000 |
| `val_size/AP50-Tiny3` | 0.287129 |
| `val_size/AP50-Small` | 0.800733 |
| `val_size/AP50-Medium` | 0.863863 |
| `test_size/AP75` | 0.090939 |
| `test_size/AP50-Tiny1` | -1.000000 |
| `test_size/AP50-Tiny2` | -1.000000 |
| `test_size/AP50-Tiny3` | 0.099010 |
| `test_size/AP50-Small` | 0.774276 |
| `test_size/AP50-Medium` | 0.849929 |


### 10. yolov11 / varroa / cp2_single2 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov11/varroa/copy_paste/cp2_single2/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov11/varroa/copy_paste/cp2_single2/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: Varroa standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.880828 |
| `val/mAP50-95` | 0.326859 |
| `test/AP50` | 0.858780 |
| `test/mAP50-95` | 0.320274 |
| `val_size/AP75` | 0.105185 |
| `val_size/AP50-Tiny1` | 0.000000 |
| `val_size/AP50-Tiny2` | -1.000000 |
| `val_size/AP50-Tiny3` | 0.148515 |
| `val_size/AP50-Small` | 0.775007 |
| `val_size/AP50-Medium` | 0.868126 |
| `test_size/AP75` | 0.096206 |
| `test_size/AP50-Tiny1` | -1.000000 |
| `test_size/AP50-Tiny2` | -1.000000 |
| `test_size/AP50-Tiny3` | 0.099010 |
| `test_size/AP50-Small` | 0.782445 |
| `test_size/AP50-Medium` | 0.831445 |


### 11. yolov9 / levir / cp1_single1 / no_mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/levir/copy_paste/cp1_single1/no_mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/levir/copy_paste/cp1_single1/no_mosaic/seed_44/evaluation_metrics.json`
- Test protocol: LEVIR-Ship standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.835017 |
| `val/mAP50-95` | 0.324726 |
| `test/AP50` | 0.811530 |
| `test/mAP50-95` | 0.304400 |
| `val_size/AP75` | 0.140337 |
| `val_size/AP50-Tiny1` | -1.000000 |
| `val_size/AP50-Tiny2` | 0.262662 |
| `val_size/AP50-Tiny3` | 0.726289 |
| `val_size/AP50-Small` | 0.854773 |
| `val_size/AP50-Medium` | 0.801980 |
| `test_size/AP75` | 0.145233 |
| `test_size/AP50-Tiny1` | 0.000000 |
| `test_size/AP50-Tiny2` | 0.260777 |
| `test_size/AP50-Tiny3` | 0.752412 |
| `test_size/AP50-Small` | 0.854333 |
| `test_size/AP50-Medium` | 0.790759 |


### 12. yolov9 / levir / cp2_single2 / no_mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/levir/copy_paste/cp2_single2/no_mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/levir/copy_paste/cp2_single2/no_mosaic/seed_44/evaluation_metrics.json`
- Test protocol: LEVIR-Ship standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.805009 |
| `val/mAP50-95` | 0.333446 |
| `test/AP50` | 0.796946 |
| `test/mAP50-95` | 0.308439 |
| `val_size/AP75` | 0.158256 |
| `val_size/AP50-Tiny1` | -1.000000 |
| `val_size/AP50-Tiny2` | 0.357756 |
| `val_size/AP50-Tiny3` | 0.720960 |
| `val_size/AP50-Small` | 0.811671 |
| `val_size/AP50-Medium` | 0.891089 |
| `test_size/AP75` | 0.145211 |
| `test_size/AP50-Tiny1` | 0.000000 |
| `test_size/AP50-Tiny2` | 0.109991 |
| `test_size/AP50-Tiny3` | 0.724474 |
| `test_size/AP50-Small` | 0.821976 |
| `test_size/AP50-Medium` | 0.737624 |


### 13. yolov9 / levir / negative_canvas_r2 / no_mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/levir/copy_paste/negative_canvas_r2/no_mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/levir/copy_paste/negative_canvas_r2/no_mosaic/seed_44/evaluation_metrics.json`
- Test protocol: LEVIR-Ship standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.820313 |
| `val/mAP50-95` | 0.323856 |
| `test/AP50` | 0.804971 |
| `test/mAP50-95` | 0.305774 |
| `val_size/AP75` | 0.171862 |
| `val_size/AP50-Tiny1` | -1.000000 |
| `val_size/AP50-Tiny2` | 0.356597 |
| `val_size/AP50-Tiny3` | 0.741211 |
| `val_size/AP50-Small` | 0.838582 |
| `val_size/AP50-Medium` | 0.801980 |
| `test_size/AP75` | 0.144926 |
| `test_size/AP50-Tiny1` | 0.000000 |
| `test_size/AP50-Tiny2` | 0.372161 |
| `test_size/AP50-Tiny3` | 0.753242 |
| `test_size/AP50-Small` | 0.823360 |
| `test_size/AP50-Medium` | 0.745316 |


### 14. yolov9 / levir / negative_canvas_r4 / no_mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_44/evaluation_metrics.json`
- Test protocol: LEVIR-Ship standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.805172 |
| `val/mAP50-95` | 0.313601 |
| `test/AP50` | 0.784613 |
| `test/mAP50-95` | 0.294080 |
| `val_size/AP75` | 0.150070 |
| `val_size/AP50-Tiny1` | -1.000000 |
| `val_size/AP50-Tiny2` | 0.320874 |
| `val_size/AP50-Tiny3` | 0.715590 |
| `val_size/AP50-Small` | 0.828916 |
| `val_size/AP50-Medium` | 0.801980 |
| `test_size/AP75` | 0.130780 |
| `test_size/AP50-Tiny1` | 0.000000 |
| `test_size/AP50-Tiny2` | 0.258652 |
| `test_size/AP50-Tiny3` | 0.725956 |
| `test_size/AP50-Small` | 0.823127 |
| `test_size/AP50-Medium` | 0.762376 |


### 15. yolov9 / tinyperson / cp1_single1 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/tinyperson/copy_paste/cp1_single1/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/tinyperson/copy_paste/cp1_single1/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: TinyPerson official corner-window merged evaluator
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.458256 |
| `val/mAP50-95` | 0.164463 |
| `test/AP50` | 0.478797 |
| `test/mAP50-95` | 0.170250 |
| `val_size/AP75` | 0.084957 |
| `val_size/AP50-Tiny1` | 0.186018 |
| `val_size/AP50-Tiny2` | 0.380846 |
| `val_size/AP50-Tiny3` | 0.410155 |
| `val_size/AP50-Small` | 0.460689 |
| `val_size/AP50-Medium` | 0.570850 |
| `test_size/AP75` | 0.080454 |
| `test_size/AP50-Tiny1` | 0.153311 |
| `test_size/AP50-Tiny2` | 0.374378 |
| `test_size/AP50-Tiny3` | 0.457759 |
| `test_size/AP50-Small` | 0.511270 |
| `test_size/AP50-Medium` | 0.536456 |
| `test_merged/AP75` | 0.083745 |
| `test_merged/AP50-Tiny1` | 0.245558 |
| `test_merged/AP50-Tiny2` | 0.510095 |
| `test_merged/AP50-Tiny3` | 0.592616 |
| `test_merged/AP50-Small` | 0.647098 |
| `test_merged/AP50-Medium` | 0.631507 |


### 16. yolov9 / tinyperson / cp2_single2 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/tinyperson/copy_paste/cp2_single2/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/tinyperson/copy_paste/cp2_single2/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: TinyPerson official corner-window merged evaluator
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.467739 |
| `val/mAP50-95` | 0.167467 |
| `test/AP50` | 0.478731 |
| `test/mAP50-95` | 0.169195 |
| `val_size/AP75` | 0.084546 |
| `val_size/AP50-Tiny1` | 0.203322 |
| `val_size/AP50-Tiny2` | 0.381058 |
| `val_size/AP50-Tiny3` | 0.446494 |
| `val_size/AP50-Small` | 0.464871 |
| `val_size/AP50-Medium` | 0.587573 |
| `test_size/AP75` | 0.083708 |
| `test_size/AP50-Tiny1` | 0.165519 |
| `test_size/AP50-Tiny2` | 0.370344 |
| `test_size/AP50-Tiny3` | 0.467299 |
| `test_size/AP50-Small` | 0.522089 |
| `test_size/AP50-Medium` | 0.536465 |
| `test_merged/AP75` | 0.089689 |
| `test_merged/AP50-Tiny1` | 0.252608 |
| `test_merged/AP50-Tiny2` | 0.499558 |
| `test_merged/AP50-Tiny3` | 0.600527 |
| `test_merged/AP50-Small` | 0.652089 |
| `test_merged/AP50-Medium` | 0.646757 |


### 17. yolov9 / tinyperson / negative_canvas_r2 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/tinyperson/copy_paste/negative_canvas_r2/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/tinyperson/copy_paste/negative_canvas_r2/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: TinyPerson official corner-window merged evaluator
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.467195 |
| `val/mAP50-95` | 0.168368 |
| `test/AP50` | 0.475247 |
| `test/mAP50-95` | 0.169444 |
| `val_size/AP75` | 0.093779 |
| `val_size/AP50-Tiny1` | 0.169689 |
| `val_size/AP50-Tiny2` | 0.397411 |
| `val_size/AP50-Tiny3` | 0.472825 |
| `val_size/AP50-Small` | 0.443261 |
| `val_size/AP50-Medium` | 0.566207 |
| `test_size/AP75` | 0.078555 |
| `test_size/AP50-Tiny1` | 0.159000 |
| `test_size/AP50-Tiny2` | 0.365885 |
| `test_size/AP50-Tiny3` | 0.468740 |
| `test_size/AP50-Small` | 0.501528 |
| `test_size/AP50-Medium` | 0.543964 |
| `test_merged/AP75` | 0.084746 |
| `test_merged/AP50-Tiny1` | 0.243961 |
| `test_merged/AP50-Tiny2` | 0.494718 |
| `test_merged/AP50-Tiny3` | 0.601162 |
| `test_merged/AP50-Small` | 0.640092 |
| `test_merged/AP50-Medium` | 0.637382 |


### 18. yolov9 / tinyperson / negative_canvas_r4 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: TinyPerson official corner-window merged evaluator
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.460574 |
| `val/mAP50-95` | 0.166524 |
| `test/AP50` | 0.471195 |
| `test/mAP50-95` | 0.166707 |
| `val_size/AP75` | 0.085012 |
| `val_size/AP50-Tiny1` | 0.175090 |
| `val_size/AP50-Tiny2` | 0.381831 |
| `val_size/AP50-Tiny3` | 0.443586 |
| `val_size/AP50-Small` | 0.454521 |
| `val_size/AP50-Medium` | 0.578446 |
| `test_size/AP75` | 0.076592 |
| `test_size/AP50-Tiny1` | 0.146708 |
| `test_size/AP50-Tiny2` | 0.361659 |
| `test_size/AP50-Tiny3` | 0.462010 |
| `test_size/AP50-Small` | 0.503795 |
| `test_size/AP50-Medium` | 0.530027 |
| `test_merged/AP75` | 0.082479 |
| `test_merged/AP50-Tiny1` | 0.244150 |
| `test_merged/AP50-Tiny2` | 0.486697 |
| `test_merged/AP50-Tiny3` | 0.575447 |
| `test_merged/AP50-Small` | 0.628518 |
| `test_merged/AP50-Medium` | 0.621981 |


### 19. yolov9 / varroa / cp1_single1 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/varroa/copy_paste/cp1_single1/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/varroa/copy_paste/cp1_single1/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: Varroa standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.885457 |
| `val/mAP50-95` | 0.322798 |
| `test/AP50` | 0.874110 |
| `test/mAP50-95` | 0.322858 |
| `val_size/AP75` | 0.096269 |
| `val_size/AP50-Tiny1` | 0.000000 |
| `val_size/AP50-Tiny2` | -1.000000 |
| `val_size/AP50-Tiny3` | 0.148515 |
| `val_size/AP50-Small` | 0.770625 |
| `val_size/AP50-Medium` | 0.846283 |
| `test_size/AP75` | 0.103062 |
| `test_size/AP50-Tiny1` | -1.000000 |
| `test_size/AP50-Tiny2` | -1.000000 |
| `test_size/AP50-Tiny3` | 0.099010 |
| `test_size/AP50-Small` | 0.758396 |
| `test_size/AP50-Medium` | 0.834008 |


### 20. yolov9 / varroa / cp2_single2 / mosaic / seed_44

- Source: `duyle2408/yolov9-yolo11-cp-nc-seed44-runs::yolov9-yolo11-cp-nc-seed44-runs/yolov9/varroa/copy_paste/cp2_single2/mosaic/seed_44`
- Evaluated artifact: `duyle2408/yolov9-yolo11-seed44-size-eval-runs/yolov9-yolo11-cp-nc-seed44-runs/yolov9/varroa/copy_paste/cp2_single2/mosaic/seed_44/evaluation_metrics.json`
- Test protocol: Varroa standard held-out test split
- Size metric family: --

| Metric | Value |
|---|---:|
| `val/AP50` | 0.889610 |
| `val/mAP50-95` | 0.328582 |
| `test/AP50` | 0.864908 |
| `test/mAP50-95` | 0.313520 |
| `val_size/AP75` | 0.102776 |
| `val_size/AP50-Tiny1` | 0.000000 |
| `val_size/AP50-Tiny2` | -1.000000 |
| `val_size/AP50-Tiny3` | 0.148515 |
| `val_size/AP50-Small` | 0.791979 |
| `val_size/AP50-Medium` | 0.871853 |
| `test_size/AP75` | 0.096016 |
| `test_size/AP50-Tiny1` | -1.000000 |
| `test_size/AP50-Tiny2` | -1.000000 |
| `test_size/AP50-Tiny3` | 0.188119 |
| `test_size/AP50-Small` | 0.776832 |
| `test_size/AP50-Medium` | 0.849569 |


## 4. Notes

- `val/*` and `test/*` are kept explicitly split-qualified.
- TinyPerson `test_merged/*` metrics are not substituted for native `test/*` metrics.
- Size-bucket metrics are read from the uploaded evaluation artifacts, not inferred from plots or another run.
- The HF completion marker for each leaf is `size_metrics_complete.json`; all 20 leaves are present in the output repository.

