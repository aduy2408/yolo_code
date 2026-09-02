# LEVIR-Ship Dataset Statistics

## Scope and provenance

This report describes the fixed LEVIR-Ship YOLO split used by the repository:

| Split | Images | GT objects | Empty images | Empty ratio | GT/image |
|---|---:|---:|---:|---:|---:|
| train | 2,320 | 1,862 | 1,157 | 49.87% | 0.8026 |
| val | 788 | 661 | 392 | 49.75% | 0.8388 |
| test | 788 | 696 | 374 | 47.46% | 0.8832 |
| **total** | **3,896** | **3,219** | **1,923** | **49.36%** | **0.8262** |

Sources:

- Dataset images and YOLO labels: `datasets/levir_ship_yolo_seed42/`.
- Existing split and object geometry summaries: `diagnostics/levir_dbss_generalization/split_summary.csv` and `objects.csv`.
- Existing per-image background measurements: `diagnostics/levir_dbss_generalization/background_statistics.csv`.

All image-level statistics below use 512x512 images. Bounding-box values are in image pixels, not normalized YOLO coordinates.

## Object-size distribution

The size bins use bbox area:

| Group | Definition |
|---|---|
| tiny | area `<100 px²` |
| small | `100 <= area <= 400 px²` |
| medium | `400 < area <= 1024 px²` |
| large | area `>1024 px²` |

The boundary convention assigns exactly `400 px²` to `small` and exactly `1024 px²` to `medium`.

| Split | tiny | small | medium | large | Total |
|---|---:|---:|---:|---:|---:|
| train | 31 (1.66%) | 962 (51.66%) | 823 (44.20%) | 46 (2.47%) | 1,862 |
| val | 8 (1.21%) | 358 (54.16%) | 285 (43.12%) | 10 (1.51%) | 661 |
| test | 14 (2.01%) | 389 (55.89%) | 276 (39.66%) | 17 (2.44%) | 696 |
| **total** | **53 (1.65%)** | **1,709 (53.09%)** | **1,384 (42.99%)** | **73 (2.27%)** | **3,219** |

### Bounding-box statistics

Each cell is `mean +/- std [min, max]`. Aspect ratio is `width / height`.

| Split | Group | N | Width (px) | Height (px) | Area (px²) | sqrt(area) (px) | Aspect ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | tiny | 31 | 8.31 +/- 1.65 [6.12, 10.92] | 9.47 +/- 1.65 [6.37, 13.00] | 78.37 +/- 14.77 [50.07, 97.35] | 8.81 +/- 0.86 [7.08, 9.87] | 0.91 +/- 0.21 [0.54, 1.47] |
| train | small | 962 | 15.96 +/- 3.88 [7.43, 30.00] | 16.18 +/- 3.35 [7.00, 30.00] | 259.04 +/- 81.30 [100.13, 400.00] | 15.88 +/- 2.60 [10.01, 20.00] | 1.03 +/- 0.33 [0.33, 3.57] |
| train | medium | 823 | 25.62 +/- 5.18 [13.00, 43.00] | 23.86 +/- 4.88 [13.00, 41.00] | 605.57 +/- 158.68 [400.52, 1024.00] | 24.41 +/- 3.12 [20.01, 32.00] | 1.13 +/- 0.38 [0.38, 2.80] |
| train | large | 46 | 37.37 +/- 5.56 [26.00, 51.00] | 32.57 +/- 4.60 [23.00, 45.00] | 1202.43 +/- 169.04 [1025.00, 1800.00] | 34.60 +/- 2.33 [32.02, 42.43] | 1.18 +/- 0.30 [0.65, 2.00] |
| val | tiny | 8 | 9.49 +/- 1.92 [7.43, 13.00] | 9.43 +/- 1.24 [7.00, 11.14] | 87.50 +/- 9.80 [66.99, 99.00] | 9.34 +/- 0.54 [8.18, 9.95] | 1.05 +/- 0.37 [0.70, 1.86] |
| val | small | 358 | 16.12 +/- 4.03 [6.99, 27.00] | 16.25 +/- 3.49 [7.00, 27.00] | 262.73 +/- 84.72 [100.00, 400.00] | 15.98 +/- 2.69 [10.00, 20.00] | 1.04 +/- 0.34 [0.43, 3.00] |
| val | medium | 285 | 25.50 +/- 5.14 [15.00, 42.00] | 23.88 +/- 4.75 [14.00, 41.00] | 602.60 +/- 151.87 [405.00, 1023.00] | 24.36 +/- 2.99 [20.12, 31.98] | 1.12 +/- 0.36 [0.48, 2.57] |
| val | large | 10 | 39.52 +/- 7.13 [32.00, 54.00] | 30.59 +/- 4.58 [20.00, 38.00] | 1195.22 +/- 241.99 [1056.00, 1862.00] | 34.42 +/- 3.22 [32.50, 43.15] | 1.35 +/- 0.48 [0.97, 2.70] |
| test | tiny | 14 | 6.57 +/- 1.23 [4.37, 8.74] | 8.51 +/- 2.00 [5.84, 11.26] | 56.70 +/- 19.12 [25.50, 92.71] | 7.42 +/- 1.28 [5.05, 9.63] | 0.81 +/- 0.21 [0.50, 1.20] |
| test | small | 389 | 16.07 +/- 3.71 [8.00, 27.00] | 16.22 +/- 3.58 [8.00, 30.00] | 261.23 +/- 81.00 [101.29, 400.00] | 15.96 +/- 2.58 [10.06, 20.00] | 1.03 +/- 0.32 [0.36, 2.40] |
| test | medium | 276 | 25.70 +/- 5.70 [14.00, 48.00] | 24.46 +/- 4.89 [13.00, 43.00] | 619.49 +/- 150.83 [403.00, 1023.00] | 24.71 +/- 2.96 [20.07, 31.98] | 1.11 +/- 0.40 [0.40, 3.00] |
| test | large | 17 | 37.24 +/- 6.73 [27.00, 49.00] | 36.65 +/- 8.88 [22.00, 63.00] | 1349.65 +/- 391.51 [1034.00, 2520.00] | 36.42 +/- 4.82 [32.16, 50.20] | 1.09 +/- 0.42 [0.63, 2.14] |

## Image and background statistics

Each cell is `mean +/- std [min, max]` across images in that split. `gray_mean` and `gray_std` are grayscale intensity statistics on `[0, 255]`. `gradient_mean` and `gradient_std` are image-gradient statistics in pixel-intensity units. `entropy32` is entropy from a 32-bin grayscale histogram. `black_ratio` is the fraction of pixels classified as black border/background by the existing diagnostic.

| Split | Black ratio | Gray mean | Gray std | Gradient mean | Gradient std | Entropy32 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.0243 +/- 0.1190 [0.0000, 0.9998] | 74.98 +/- 37.58 [0.01, 255.00] | 13.54 +/- 17.63 [0.11, 101.17] | 1.1009 +/- 1.0364 [0.0017, 13.8835] | 2.0271 +/- 1.8067 [0.0680, 24.7391] | 1.3843 +/- 1.1154 [0.0000, 4.6931] |
| val | 0.0203 +/- 0.1122 [0.0000, 0.9333] | 76.86 +/- 37.41 [4.11, 248.09] | 14.16 +/- 18.31 [0.19, 87.69] | 1.1830 +/- 1.0645 [0.0227, 6.6769] | 2.1438 +/- 1.8930 [0.1738, 15.9330] | 1.4408 +/- 1.1553 [0.0000, 4.6559] |
| test | 0.0279 +/- 0.1308 [0.0000, 1.0000] | 75.19 +/- 34.75 [0.00, 214.29] | 14.03 +/- 17.96 [0.19, 105.47] | 1.1283 +/- 1.0875 [0.0004, 8.9044] | 2.0779 +/- 1.8739 [0.2151, 13.3036] | 1.4008 +/- 1.1282 [0.0000, 4.6043] |

## Local GT contrast and frequency

The following measurements are computed once for every GT object and grouped using the geometry in `objects.csv`, so the sample count exactly matches the bbox table.

- **Gray contrast:** `abs(mean(object) - mean(ring)) / (mean(ring) + 1e-6)`.
- **Center-ring contrast:** same formula, using the central 50% of the bbox as the object region.
- **Ring:** pixels outside the bbox in a crop centered on the GT and expanded to approximately 2x the bbox dimensions.
- **High-frequency ratio:** grayscale bbox crop resized to `32x32`; `sum(|FFT|, radius > 8) / sum(|FFT|)`, after subtracting the crop mean.

Values below are `mean +/- std [min, max]`; contrast values are dimensionless and frequency ratio is in `[0, 1]` in the normal case.

| Split | Group | N | Gray contrast | Center-ring contrast | High-frequency ratio |
|---|---|---:|---:|---:|---:|
| train | tiny | 31 | 0.0362 +/- 0.0196 [0.0016, 0.0915] | 0.0826 +/- 0.0456 [0.0031, 0.1667] | 0.1596 +/- 0.0527 [0.0804, 0.2682] |
| train | small | 962 | 0.0433 +/- 0.0464 [0.0001, 0.5269] | 0.1231 +/- 0.1277 [0.0004, 1.4186] | 0.1699 +/- 0.0580 [0.0579, 0.3797] |
| train | medium | 823 | 0.0626 +/- 0.0857 [0.0000, 0.9179] | 0.1806 +/- 0.2365 [0.0001, 2.5129] | 0.2252 +/- 0.0743 [0.0671, 0.4821] |
| train | large | 46 | 0.0488 +/- 0.0561 [0.0019, 0.3175] | 0.1456 +/- 0.1430 [0.0008, 0.6093] | 0.2819 +/- 0.0663 [0.1385, 0.4563] |
| val | tiny | 8 | 0.0794 +/- 0.1174 [0.0007, 0.3858] | 0.1595 +/- 0.2389 [0.0227, 0.7880] | 0.1580 +/- 0.0387 [0.0928, 0.2174] |
| val | small | 358 | 0.0494 +/- 0.0523 [0.0000, 0.3628] | 0.1399 +/- 0.1362 [0.0017, 0.8201] | 0.1616 +/- 0.0590 [0.0534, 0.3537] |
| val | medium | 285 | 0.0596 +/- 0.0929 [0.0001, 1.0068] | 0.1706 +/- 0.2612 [0.0007, 3.0693] | 0.2223 +/- 0.0762 [0.0870, 0.4757] |
| val | large | 10 | 0.0400 +/- 0.0372 [0.0055, 0.1419] | 0.1304 +/- 0.1055 [0.0272, 0.4140] | 0.2807 +/- 0.0716 [0.1866, 0.4273] |
| test | tiny | 14 | 0.0508 +/- 0.0270 [0.0041, 0.1003] | 0.1253 +/- 0.0589 [0.0640, 0.2615] | 0.1604 +/- 0.0369 [0.0971, 0.2283] |
| test | small | 389 | 0.0498 +/- 0.0511 [0.0008, 0.4889] | 0.1450 +/- 0.1476 [0.0027, 1.4851] | 0.1617 +/- 0.0565 [0.0543, 0.3799] |
| test | medium | 276 | 0.0564 +/- 0.0754 [0.0002, 0.5566] | 0.1593 +/- 0.1973 [0.0004, 1.4689] | 0.2302 +/- 0.0806 [0.0764, 0.5199] |
| test | large | 17 | 0.0631 +/- 0.0522 [0.0015, 0.1999] | 0.1806 +/- 0.1420 [0.0017, 0.5586] | 0.2802 +/- 0.0791 [0.1109, 0.4200] |

## Interpretation and limitations

- `97.73%` of all GT objects are in the `tiny + small + medium` bins; only `2.27%` are `large` under the area definition above.
- Most objects are small: `53.09%` are in the `small` bin and `42.99%` are in the `medium` bin.
- The mean background grayscale, gradient, and entropy values are close across splits. The large per-image ranges show that the dataset still contains heterogeneous scenes; similar means do not imply identical distributions.
- High-frequency ratio increases with object size in these raw-image crops. This is a crop statistic, not a measurement of P2 feature preservation and not evidence for a model mechanism.
- Local contrast values can exceed `1` because the denominator is the ring mean and the metric is not clipped. They should be used for relative subgroup comparison, not as calibrated probabilities.
- The `tiny` and `large` groups have small sample counts, especially in validation and test. Their means are therefore more sensitive to individual objects.
- This report does not use predictions, checkpoints, NMS, AP, or P2 feature maps. It describes dataset GT and raw image statistics only.
