# Kiến trúc cụ thể YOLOv9, YOLOv10, YOLO11 và đối chiếu YOLOv8

> Báo cáo được neo vào các YAML và implementation đã kiểm tra trong commit upstream
> `vendor/ultralytics_upstream` (`e6754ce4c`). `project_ultralytics/architectures.py`
> chứa graph khai báo độc lập dependency để đọc, kiểm thử và tái sử dụng trong repo.
>
> Lưu ý tên: tài liệu Ultralytics gọi phiên bản thứ tư là **YOLO11**, còn nhiều
> log/repository gọi là `yolo11` hoặc `YOLOv11`.

## 1. Graph chuẩn cho detection

Các bản dưới đây đều nhận ảnh RGB, giảm mẫu qua các mức stride 2, 4, 8, 16, 32,
và phát hiện trên ba feature map `P3/8`, `P4/16`, `P5/32`. Channel trong graph là
channel trước compound scaling. Cấu hình `n`/`t` chỉ thay đổi depth, width và giới
hạn channel, không đổi topology chính.

### YOLOv8

Nguồn: `vendor/ultralytics_upstream/ultralytics/cfg/models/v8/yolov8.yaml`.

```text
Backbone:
Conv(64,3,2) -> Conv(128,3,2) -> C2f(128)x3
-> Conv(256,3,2) -> C2f(256)x6
-> Conv(512,3,2) -> C2f(512)x6
-> Conv(1024,3,2) -> C2f(1024)x3 -> SPPF(1024,5)

Neck/head: upsample x2 + concat(P4) + C2f(512)x3
-> upsample x2 + concat(P3) + C2f(256)x3 = P3
-> Conv(256,3,2) + concat(P4 head) + C2f(512)x3 = P4
-> Conv(512,3,2) + concat(P5 backbone) + C2f(1024)x3 = P5
-> Detect(P3,P4,P5)
```

### YOLOv9

Nguồn: `vendor/ultralytics_upstream/ultralytics/cfg/models/v9/yolov9t.yaml` và
`vendor/ultralytics_upstream/ultralytics/nn/modules/block.py`.

```text
Backbone (GELAN):
Conv(16,3,2) -> Conv(32,3,2) -> ELAN1(32,32,16)
-> AConv(64) -> RepNCSPELAN4(64,64,32,3)
-> AConv(96) -> RepNCSPELAN4(96,96,48,3)
-> AConv(128) -> RepNCSPELAN4(128,128,64,3)
-> SPPELAN(128,64)

Neck/head:
upsample + concat(P4) + RepNCSPELAN4(96,96,48,3)
-> upsample + concat(P3) + RepNCSPELAN4(64,64,32,3) = P3
-> AConv(48) + concat(P4 head) + RepNCSPELAN4(96,96,48,3) = P4
-> AConv(64) + concat(P5 backbone) + RepNCSPELAN4(128,128,64,3) = P5
-> Detect(P3,P4,P5)
```

**Module mechanics.** `RepNCSPELAN4` là CSP-ELAN: 1x1 projection, tách nhánh,
hai nhánh `RepCSP` + Conv 3x3, concat rồi 1x1 fusion. `ELAN1` là biến thể nhẹ
hơn dùng các Conv 3x3. `AConv` average-pool trước rồi Conv stride 2. `SPPELAN`
là nhiều max-pool stride 1 nối tiếp rồi concat. PGI (Programmable Gradient
Information) là cơ chế huấn luyện truyền thông tin gradient/progressive information
qua auxiliary/progammable gradient paths, không phải một feature-map output mới
trong inference graph chuẩn YAML.

### YOLOv10

Nguồn: `vendor/ultralytics_upstream/ultralytics/cfg/models/v10/yolov10n.yaml`,
`vendor/ultralytics_upstream/ultralytics/nn/modules/block.py`, và
`vendor/ultralytics_upstream/ultralytics/nn/modules/head.py`.

```text
Backbone:
Conv(64,3,2) -> Conv(128,3,2) -> C2f(128,True)x3
-> Conv(256,3,2) -> C2f(256,True)x6
-> SCDown(512,3,2) -> C2f(512,True)x6
-> SCDown(1024,3,2) -> C2f(1024,True)x3
-> SPPF(1024,5) -> PSA(1024)

Neck/head:
upsample + concat(P4) + C2f(512)x3
-> upsample + concat(P3) + C2f(256)x3 = P3
-> Conv(256,3,2) + concat(P4 head) + C2f(512)x3 = P4
-> SCDown(512,3,2) + concat(P5 backbone) + C2fCIB(1024,True,True)x3 = P5
-> v10Detect(P3,P4,P5)
```

**Module mechanics.** `SCDown` tách channel projection và spatial downsampling
bằng `Conv(c1,c2,1,1)` rồi depthwise `Conv(c2,c2,k,s,g=c2,act=False)` theo bản
implementation. `PSA` là partial self-attention, chỉ đưa một phần channel vào
attention để giảm chi phí. `C2fCIB` là C2f dùng các CIB block ở nhánh bên trong.

Khác biệt quyết định nằm ở `v10Detect`: head có nhánh one-to-many và one-to-one.
Huấn luyện dùng dual assignment để giữ nhiều positive và học nhánh one-to-one;
suy luận dùng consistent one-to-one predictions, cho phép NMS-free/end-to-end
inference. Vì vậy không nên mô tả YOLOv10 chỉ là YOLOv8 thay Conv bằng SCDown.

### YOLO11 / YOLOv11

Nguồn: `vendor/ultralytics_upstream/ultralytics/cfg/models/11/yolo11.yaml`.

```text
Backbone:
Conv(64,3,2) -> Conv(128,3,2) -> C3k2(256,False,0.25)x2
-> Conv(256,3,2) -> C3k2(512,False,0.25)x2
-> Conv(512,3,2) -> C3k2(512,True)x2
-> Conv(1024,3,2) -> C3k2(1024,True)x2
-> SPPF(1024,5) -> C2PSA(1024)x2

Neck/head:
upsample + concat(P4) + C3k2(512,False)x2
-> upsample + concat(P3) + C3k2(256,False)x2 = P3
-> Conv(256,3,2) + concat(P4 head) + C3k2(512,False)x2 = P4
-> Conv(512,3,2) + concat(P5 backbone) + C3k2(1024,True)x2 = P5
-> Detect(P3,P4,P5)
```

**Module mechanics.** `C3k2` là block CSP hai-conv nhẹ hơn, có thể chọn inner
`C3k` bottleneck. `C2PSA` bọc các `PSABlock` trong một partial-channel CSP path.
YOLO11 tiếp tục dùng `Detect` anchor-free thông thường, không có `v10Detect`.

## 2. Bảng so sánh kiến trúc

| Thành phần | YOLOv8 | YOLOv9 | YOLOv10 | YOLO11 |
|---|---|---|---|---|
| Backbone block chính | C2f | GELAN + RepNCSPELAN4 | C2f + C2fCIB | C3k2 |
| Downsampling nổi bật | Conv stride 2 | AConv | SCDown ở P4/P5 và neck P5 | Conv stride 2 |
| Context/attention sâu | SPPF | SPPELAN | SPPF + PSA | SPPF + C2PSA |
| Neck fusion | PAN/FPN, C2f | PAN/FPN, RepNCSPELAN4 | PAN/FPN, C2f/C2fCIB | PAN/FPN, C3k2 |
| Detection head | Detect | Detect | v10Detect, one-to-many + one-to-one | Detect |
| NMS-free native inference | Không | Không | Có mục tiêu thiết kế | Không phải điểm thiết kế chính |
| Gradient/training innovation | TAL/DFL pipeline | PGI | dual assignment + consistent dual predictions | training/inference Detect pipeline kế thừa |
| Outputs chuẩn | P3/P4/P5 | P3/P4/P5 | P3/P4/P5 | P3/P4/P5 |

## 3. Kết luận thực dụng khi chọn model

- **YOLOv8:** baseline dễ hiểu, parser/runtime ổn định, phù hợp làm control.
- **YOLOv9:** thay đổi backbone mạnh nhất về kiểu graph. Chọn khi cần GELAN,
  re-parameterizable CSP-ELAN và lợi ích huấn luyện PGI; PGI cần được phân biệt
  với YAML inference graph.
- **YOLOv10:** thay đổi cả downsampling, attention và đặc biệt detection head.
  Chọn khi latency/NMS-free end-to-end là yêu cầu chính.
- **YOLO11:** tiến hóa gọn trên backbone/attention (`C3k2`, `C2PSA`) nhưng vẫn
  giữ Detect head quen thuộc. Đây thường là migration ít phá vỡ pipeline hơn
  YOLOv10 nếu không cần NMS-free.

Không so sánh số parameter/FLOPs giữa các YAML chưa scale cùng `n/s/m/l/x`, cùng
input size, cùng `nc` và cùng implementation commit. Các graph channel ở trên là
channel canonical trước scaling.

## 4. Cách dùng code trong repo

```python
from project_ultralytics.architectures import get_architecture, graph_lines

for name in ("yolov8", "yolov9", "yolov10", "yolo11"):
    print("\n".join(graph_lines(get_architecture(name))))
```

Code khai báo: `project_ultralytics/architectures.py`.
Canonical runtime YAML: `vendor/ultralytics_upstream/ultralytics/cfg/models/`.
Legacy compatibility YAML: `models_related/ultralytics/ultralytics/cfg/models/`.
Không sửa trực tiếp vendor hoặc legacy fork.

## 5. Tài liệu tham khảo

- [YOLOv9 paper](https://arxiv.org/abs/2402.13616)
- [YOLOv10 paper](https://arxiv.org/abs/2405.14458)
- [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics model YAMLs trong repo](../..//vendor/ultralytics_upstream/ultralytics/cfg/models/)
