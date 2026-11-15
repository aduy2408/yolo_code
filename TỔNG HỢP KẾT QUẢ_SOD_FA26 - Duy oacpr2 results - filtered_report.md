# Filtered runtime metrics — OACP R2

## Decision

This report keeps only completed Jetson-comparable runtime rows:

- Keep `PyTorch .pt` and `TensorRT` only.
- Keep FP32 and FP16.
- Treat seed 43/44 as repeated measurements of the same runtime model/config, not separate model variants; the report-level row is the aggregate across both runs.
- Report runtime variation as mean ± sample std only as a stability check; do not interpret it as accuracy seed variance.
- Use median full-pipeline latency as the primary latency metric.
- Use `FPS from median = 1000 / median latency` as the primary FPS metric.
- Exclude `ONNX Runtime CPU`: it is a CPU baseline, not comparable to the Jetson edge backend rows.
- Exclude export/build time and mode wall time from inference latency.

Input rows: 72. Retained rows: 48. Excluded rows: 24.

## Report aggregation rule

The main runtime comparison intentionally omits the augmentation case (`OACP`, `NC-R2`, `NC-R4`). For runtime, these are the same YOLOv8/YOLO11 architecture family; the report row should be keyed by `dataset × model × backend × precision`, with mean ± std over repeated seed measurements. When NC runtime files are added, they should be pooled with the corresponding OACP rows rather than split into separate FPS/latency rows.

## Summary across retained runs

|Dataset|Model|Backend|Precision|n|Median latency ms (mean ± std)|FPS from median (mean ± std)|Mean latency ms|P95 latency ms|Model-only median ms|
|---|---|---|---|---:|---:|---:|---:|---:|---:|
|levir_ship|YOLO11|PyTorch .pt|FP16|2|39.09 ± 1.42|25.60 ± 0.93|40.47|50.10|21.85|
|levir_ship|YOLO11|PyTorch .pt|FP32|2|39.83 ± 1.42|25.12 ± 0.90|41.07|50.08|22.66|
|levir_ship|YOLO11|TensorRT|FP16|2|26.05 ± 1.26|38.43 ± 1.86|27.23|36.04|8.37|
|levir_ship|YOLO11|TensorRT|FP32|2|38.36 ± 0.74|26.07 ± 0.50|40.24|57.42|14.26|
|levir_ship|YOLOv8|PyTorch .pt|FP16|2|33.84 ± 0.68|29.56 ± 0.59|35.39|47.62|16.84|
|levir_ship|YOLOv8|PyTorch .pt|FP32|2|41.90 ± 1.18|23.88 ± 0.67|43.19|55.53|24.10|
|levir_ship|YOLOv8|TensorRT|FP16|2|25.15 ± 0.07|39.75 ± 0.11|26.54|39.95|7.32|
|levir_ship|YOLOv8|TensorRT|FP32|2|37.38 ± 1.50|26.78 ± 1.08|38.84|54.67|13.15|
|tinyperson|YOLO11|PyTorch .pt|FP16|2|31.17 ± 0.91|32.10 ± 0.94|60.25|264.56|22.44|
|tinyperson|YOLO11|PyTorch .pt|FP32|2|35.95 ± 0.85|27.82 ± 0.66|61.00|221.28|27.68|
|tinyperson|YOLO11|TensorRT|FP16|2|23.02 ± 1.72|43.56 ± 3.26|47.85|197.95|11.67|
|tinyperson|YOLO11|TensorRT|FP32|2|35.24 ± 0.97|28.39 ± 0.78|62.88|222.08|20.76|
|tinyperson|YOLOv8|PyTorch .pt|FP16|2|30.56 ± 1.27|32.75 ± 1.36|55.29|232.00|19.90|
|tinyperson|YOLOv8|PyTorch .pt|FP32|2|37.47 ± 0.11|26.69 ± 0.08|58.31|194.70|23.02|
|tinyperson|YOLOv8|TensorRT|FP16|2|22.58 ± 1.96|44.45 ± 3.86|44.25|162.19|10.52|
|tinyperson|YOLOv8|TensorRT|FP32|2|34.47 ± 0.49|29.02 ± 0.41|60.40|191.46|19.50|
|varroa|YOLO11|PyTorch .pt|FP16|2|33.47 ± 0.50|29.88 ± 0.45|34.39|40.47|21.98|
|varroa|YOLO11|PyTorch .pt|FP32|2|34.04 ± 0.89|29.39 ± 0.77|34.89|41.12|21.91|
|varroa|YOLO11|TensorRT|FP16|2|28.83 ± 0.29|34.69 ± 0.35|29.81|36.16|11.67|
|varroa|YOLO11|TensorRT|FP32|2|44.60 ± 0.86|22.42 ± 0.43|46.15|55.04|20.81|
|varroa|YOLOv8|PyTorch .pt|FP16|2|28.63 ± 0.30|34.93 ± 0.36|30.21|39.60|16.55|
|varroa|YOLOv8|PyTorch .pt|FP32|2|34.91 ± 0.35|28.64 ± 0.28|36.40|45.83|22.50|
|varroa|YOLOv8|TensorRT|FP16|2|27.75 ± 0.09|36.03 ± 0.12|28.69|36.20|10.54|
|varroa|YOLOv8|TensorRT|FP32|2|43.18 ± 0.43|23.16 ± 0.23|44.24|55.18|19.55|

## Per-seed retained rows

|Dataset|Model|Backend|Precision|Seed|Median latency ms|FPS from median|Mean latency ms|P95 latency ms|Model-only median ms|
|---|---|---|---|---:|---:|---:|---:|---:|---:|
|levir_ship|YOLO11|PyTorch .pt|FP16|43|40.098|24.939|41.153|48.740|21.845|
|levir_ship|YOLO11|PyTorch .pt|FP16|44|38.087|26.256|39.782|51.450|21.851|
|levir_ship|YOLO11|PyTorch .pt|FP32|43|40.841|24.485|41.702|48.698|22.659|
|levir_ship|YOLO11|PyTorch .pt|FP32|44|38.828|25.755|40.438|51.471|22.659|
|levir_ship|YOLO11|TensorRT|FP16|43|26.944|37.114|27.765|34.425|8.380|
|levir_ship|YOLO11|TensorRT|FP16|44|25.164|39.739|26.695|37.656|8.367|
|levir_ship|YOLO11|TensorRT|FP32|43|38.883|25.718|39.759|51.266|14.246|
|levir_ship|YOLO11|TensorRT|FP32|44|37.834|26.431|40.721|63.566|14.266|
|levir_ship|YOLOv8|PyTorch .pt|FP16|43|34.318|29.139|35.782|46.968|16.824|
|levir_ship|YOLOv8|PyTorch .pt|FP16|44|33.361|29.975|34.991|48.281|16.861|
|levir_ship|YOLOv8|PyTorch .pt|FP32|43|42.731|23.402|43.609|55.453|24.040|
|levir_ship|YOLOv8|PyTorch .pt|FP32|44|41.064|24.352|42.770|55.597|24.161|
|levir_ship|YOLOv8|TensorRT|FP16|43|25.104|39.834|26.255|36.701|7.309|
|levir_ship|YOLOv8|TensorRT|FP16|44|25.204|39.676|26.821|43.203|7.337|
|levir_ship|YOLOv8|TensorRT|FP32|43|38.436|26.017|39.142|51.658|13.146|
|levir_ship|YOLOv8|TensorRT|FP32|44|36.314|27.538|38.536|57.684|13.150|
|tinyperson|YOLO11|PyTorch .pt|FP16|43|31.814|31.433|61.974|266.889|22.461|
|tinyperson|YOLO11|PyTorch .pt|FP16|44|30.522|32.763|58.526|262.234|22.418|
|tinyperson|YOLO11|PyTorch .pt|FP32|43|36.555|27.356|63.730|237.705|27.657|
|tinyperson|YOLO11|PyTorch .pt|FP32|44|35.346|28.292|58.276|204.862|27.708|
|tinyperson|YOLO11|TensorRT|FP16|43|24.237|41.260|49.721|192.709|11.673|
|tinyperson|YOLO11|TensorRT|FP16|44|21.803|45.866|45.974|203.193|11.665|
|tinyperson|YOLO11|TensorRT|FP32|43|35.921|27.839|63.181|208.593|20.723|
|tinyperson|YOLO11|TensorRT|FP32|44|34.551|28.943|62.581|235.557|20.789|
|tinyperson|YOLOv8|PyTorch .pt|FP16|43|31.453|31.793|56.228|248.336|19.933|
|tinyperson|YOLOv8|PyTorch .pt|FP16|44|29.663|33.712|54.347|215.657|19.874|
|tinyperson|YOLOv8|PyTorch .pt|FP32|43|37.545|26.635|58.888|195.513|22.931|
|tinyperson|YOLOv8|PyTorch .pt|FP32|44|37.386|26.748|57.736|193.895|23.105|
|tinyperson|YOLOv8|TensorRT|FP16|43|23.967|41.724|45.344|168.714|10.535|
|tinyperson|YOLOv8|TensorRT|FP16|44|21.196|47.179|43.153|155.667|10.513|
|tinyperson|YOLOv8|TensorRT|FP32|43|34.811|28.727|62.046|203.533|19.504|
|tinyperson|YOLOv8|TensorRT|FP32|44|34.125|29.304|58.750|179.380|19.489|
|varroa|YOLO11|PyTorch .pt|FP16|43|33.116|30.197|33.553|38.013|22.054|
|varroa|YOLO11|PyTorch .pt|FP16|44|33.821|29.567|35.226|42.924|21.898|
|varroa|YOLO11|PyTorch .pt|FP32|43|33.409|29.932|33.853|38.351|21.887|
|varroa|YOLO11|PyTorch .pt|FP32|44|34.667|28.846|35.920|43.891|21.927|
|varroa|YOLO11|TensorRT|FP16|43|28.622|34.938|29.000|33.009|11.661|
|varroa|YOLO11|TensorRT|FP16|44|29.031|34.446|30.625|39.309|11.678|
|varroa|YOLO11|TensorRT|FP32|43|43.993|22.731|44.779|50.607|20.798|
|varroa|YOLO11|TensorRT|FP32|44|45.211|22.118|47.519|59.482|20.812|
|varroa|YOLOv8|PyTorch .pt|FP16|43|28.424|35.182|28.803|32.640|16.583|
|varroa|YOLOv8|PyTorch .pt|FP16|44|28.844|34.670|31.618|46.559|16.519|
|varroa|YOLOv8|PyTorch .pt|FP32|43|34.670|28.843|35.075|39.072|22.523|
|varroa|YOLOv8|PyTorch .pt|FP32|44|35.159|28.442|37.725|52.586|22.479|
|varroa|YOLOv8|TensorRT|FP16|43|27.689|36.115|28.129|33.085|10.538|
|varroa|YOLOv8|TensorRT|FP16|44|27.821|35.944|29.248|39.324|10.550|
|varroa|YOLOv8|TensorRT|FP32|43|42.879|23.322|43.383|51.518|19.516|
|varroa|YOLOv8|TensorRT|FP32|44|43.483|22.997|45.098|58.848|19.591|

## Scope note

This source CSV contains OACP-R2 rows only. It does not contain NC-R2 or NC-R4 runtime rows; those must be added from their corresponding benchmark artifacts before claiming a complete OACP/NC runtime matrix.
