# Kiến trúc cụ thể YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11

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

### YOLOv5

Nguồn: `vendor/ultralytics_upstream/ultralytics/cfg/models/v5/yolov5.yaml`.

```text
Backbone:
Conv(64,6,2,2) -> Conv(128,3,2) -> C3(128)x3
-> Conv(256,3,2) -> C3(256)x6
-> Conv(512,3,2) -> C3(512)x9
-> Conv(1024,3,2) -> C3(1024)x3 -> SPPF(1024,5)

Neck/head:
Conv1x1(512) -> upsample + concat(P4) + C3(512)x3
-> Conv1x1(256) -> upsample + concat(P3) + C3(256)x3 = P3
-> Conv(256,3,2) + concat(P4 head) + C3(512)x3 = P4
-> Conv(512,3,2) + concat(P5 backbone) + C3(1024)x3 = P5
-> Detect(P3,P4,P5, anchors)
```

YOLOv5 dùng `C3` kiểu CSP và detection head anchor-based với anchor groups được
khai báo trực tiếp trong YAML. Đây là khác biệt lớn so với `Detect` anchor-free
của YOLOv8, YOLOv9 và YOLO11, cũng như `v10Detect` của YOLOv10. Các YAML
YOLOv5 lịch sử có thể dùng `Focus`, nhưng canonical YAML hiện tại trong repo dùng
stem `Conv(6, stride=2)`.

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

## 2. Giải thích từng block

Phần này mô tả theo notation `B×C×H×W`, trong đó `B` là batch, `C` là số
channel và `H,W` là kích thước spatial. Các block dùng chung được giải thích
trước, sau đó là những block đặc trưng của từng phiên bản.

### 2.1 Block nền tảng dùng chung

#### `Conv`

`Conv(c1,c2,k,s)` trong Ultralytics là chuỗi:

```text
Conv2d(c1 -> c2, kernel=k, stride=s, padding=autopad(k))
-> BatchNorm2d(c2) -> activation (mặc định SiLU)
```

`s=2` giảm `H,W` một nửa và đồng thời tăng/đổi channel. Đây là phép downsample
chính ở YOLOv8 và YOLO11, cũng xuất hiện ở stem và một nhánh neck của YOLOv9/v10.

#### `Bottleneck`

```text
x -> Conv(c1,c_hidden,k1,1) -> Conv(c_hidden,c2,k2,1,g) -> y
output = y + x  nếu shortcut=True và c1=c2
        = y      ngược lại
```

`c_hidden = floor(e*c2)`. Shortcut giữ đường gradient ngắn và tái sử dụng đặc
trưng; grouped convolution `g>1` giảm số phép tính. `C2f`, `C3k2`, `CIB` và
`RepCSP` đều xây dựng từ biến thể bottleneck này.

#### `C2f` (YOLOv8, YOLOv10)

Đây là CSP bottleneck dạng nhiều nhánh:

```text
u = Conv1x1(x)                   # 2*c_hidden channels
[a,b] = split(u, 2)
y = [a, b]
for i in 1..n:
    y.append(Bottleneck(y[-1]))
output = Conv1x1(concat(y))      # (n+2)*c_hidden -> c2
```

Khác `C2`, `C2f` giữ lại output của **từng** bottleneck để concat, không chỉ giữ
nhánh bypass cuối. Vì thế feature ở nhiều độ sâu được đưa tới fusion, trong khi
chỉ một phần channel đi qua chuỗi bottleneck. `shortcut` trong YAML quyết định
bottleneck bên trong có residual hay không.

#### `C3` (YOLOv5)

`C3` là CSP block ba convolution:

```text
a = Sequential(Bottleneck_1,...,Bottleneck_n)(Conv1x1(x))
b = Conv1x1(x)
output = Conv1x1(concat(a,b))
```

Một nhánh đi qua chuỗi bottleneck, nhánh còn lại bypass trực tiếp. Khác `C2f`,
`C3` chỉ concat hai nhánh cuối thay vì giữ output của từng bottleneck. Vì vậy
YOLOv5 có graph đơn giản hơn nhưng ít multi-level feature reuse hơn C2f. Tham số
`shortcut=False` ở neck tắt residual trong các bottleneck của nhánh xử lý.

#### `SPPF`

SPPF là Spatial Pyramid Pooling nhanh, giữ nguyên kích thước spatial:

```text
x -> Conv1x1 -> z
z1 = MaxPool(z, k=5, s=1)
z2 = MaxPool(z1, k=5, s=1)
z3 = MaxPool(z2, k=5, s=1)
output = Conv1x1(concat(z, z1, z2, z3))
```

Ba lần pooling 5×5 tuần tự tạo receptive field tương đương các mức pooling lớn
hơn với chi phí thấp hơn chạy nhiều kernel độc lập. Block này nằm tại P5 trước
neck trong YOLOv8, YOLOv10 và YOLO11.

#### `Upsample`, `Concat` và neck PAN/FPN

`Upsample(scale=2, nearest)` tăng gấp đôi `H,W` mà không học tham số. `Concat`
nối tensor theo chiều channel, nên hai tensor phải có cùng `B,H,W`. Nhánh
top-down lấy feature P5 giàu ngữ nghĩa, upsample rồi concat với P4/P3 của
backbone. Nhánh bottom-up sau đó downsample lại để trộn thông tin localization
ở P3 với ngữ nghĩa sâu hơn ở P4/P5. Đây là lý do cả năm graph đều tạo ba output
P3/P4/P5.

#### `Detect`

Head `Detect` nhận `[P3,P4,P5]`. Mỗi mức có các nhánh convolution tách cho box
regression và classification. Regression dùng DFL với `reg_max` bin cho mỗi
tọa độ, sau đó giải mã phân phối thành `(x,y,w,h)` theo grid và stride. Tổng số
điểm dự đoán là tổng số pixel của ba feature map. Head hiện đại này là
anchor-free: không tạo các anchor box cố định cho từng cell.

### 2.2 Block riêng của YOLOv9

#### `ELAN1`

`ELAN1` là một CSP-ELAN nhẹ:

```text
u = Conv1x1(x) -> split(u, 2) = [a,b]
c = Conv3x3(a)
d = Conv3x3(c)
output = Conv1x1(concat(a,b,c,d))
```

Các nhánh có độ sâu khác nhau được concat để duy trì nhiều đường truyền gradient
và nhiều receptive field. Nó được dùng ngay sau stem P2/4 trong YOLOv9t.

#### `RepBottleneck`, `RepCSP` và `RepNCSPELAN4`

`RepBottleneck` thay bottleneck thường bằng `RepConv`, thường có nhiều nhánh
training (3×3, 1×1 và identity nếu hợp lệ) rồi có thể fuse thành convolution
đơn khi deploy. `RepCSP` đưa chuỗi RepBottleneck qua một nhánh CSP.

`RepNCSPELAN4` ghép các ý tưởng đó:

```text
u = Conv1x1(x) -> split(u, 2) = [a,b]
c = Conv1x1/RepCSP(a) -> Conv3x3(c)
d = RepCSP(c) -> Conv3x3(d)
output = Conv1x1(concat(a,b,c,d))
```

Trong implementation, hai nhánh `cv2` và `cv3` lần lượt xử lý phần feature
trước đó rồi concat cùng hai phần split ban đầu. `c3` là kích thước trung gian,
`c4` là kích thước của các RepCSP branch và `n` là số RepCSP lặp. Đây là block
chính của cả backbone lẫn neck YOLOv9.

#### `AConv`

`AConv` thực hiện average pooling kernel 2, stride 1 trước một Conv 3×3 stride
2. Pooling làm trơn và giảm aliasing trước downsample; convolution sau đó học
phép chiếu channel. Nó thay các Conv stride-2 ở P3/P4/P5 của graph YOLOv9.

#### `SPPELAN`

```text
z = Conv1x1(x)
[z, MaxPool(z), MaxPool²(z), MaxPool³(z)]
-> concat theo channel -> Conv1x1
```

Khác `SPPF`, implementation dùng một chuỗi max-pool trên feature đã chiếu bởi
Conv1x1 và tổ chức nó trong kiểu ELAN. Output giữ nguyên kích thước P5 nhưng có
ngữ cảnh local ở nhiều receptive field.

#### `PGI`

PGI không phải layer inference nằm giữa P3/P4/P5 trong YAML detection tối giản.
Đây là cơ chế training dùng programmable/auxiliary gradient paths để truyền
thông tin gradient hữu ích về các tầng nông và giảm information bottleneck. Vì
vậy khi vẽ graph deploy chỉ nên vẽ GELAN + Detect; khi mô tả training phải ghi
thêm PGI và auxiliary branch tương ứng của implementation/paper.

### 2.3 Block riêng của YOLOv10

#### `SCDown`

Implementation chính xác là:

```text
x -> Conv1x1(c1 -> c2) -> depthwise Conv(k, stride=2, groups=c2, no activation)
```

Pointwise convolution đảm nhiệm channel projection trước; depthwise convolution
đảm nhiệm spatial downsample sau. So với một Conv thường `c1 -> c2, k=3, s=2`,
SCDown giảm coupling giữa hai nhiệm vụ và số tham số/Multiply-Adds ở phần spatial.

#### `CIB` và `C2fCIB`

`CIB` (inverted bottleneck kiểu lightweight) dùng projection channel, depthwise
spatial convolution và projection ngược. Ý tưởng là phần spatial dùng
depthwise convolution rẻ, còn pointwise convolution đảm nhiệm trộn channel.

`C2fCIB` kế thừa khung `C2f`: một phần feature bypass, phần còn lại đi qua chuỗi
`CIB`, sau đó concat và fuse bằng 1×1. Trong YAML YOLOv10, block này được đặt ở
P5 cuối neck để tăng chất lượng feature sâu mà không thay toàn bộ C2f.

#### `PSABlock` và `PSA`

`PSABlock` gồm self-attention và feed-forward network với residual:

```text
y = x + Attention(x)
output = y + FFN(y)
```

Attention tạo Q/K/V từ feature và dùng nhiều head; FFN là hai projection 1×1.
`PSA` bao ngoài block bằng partial channel split:

```text
[a,b] = Conv1x1(x).split(2)
b = b + Attention(b)
b = b + FFN(b)
output = Conv1x1(concat(a,b))
```

Chỉ nhánh `b` đi qua attention, nên chi phí thấp hơn self-attention toàn bộ
channel. YOLOv10 đặt PSA sau SPPF tại P5.

#### `v10Detect`

`v10Detect` kế thừa `Detect` nhưng giữ hai bộ head classification:

```text
feature -> one-to-many head  -> nhiều positive khi training
feature -> one-to-one head   -> một prediction nhất quán khi inference
```

Trong training, dual assignment tối ưu hai nhánh theo mục tiêu khác nhau. Khi
deploy/fuse, one-to-many branch bị bỏ và one-to-one branch cho phép chọn top
prediction trực tiếp, loại bỏ nhu cầu NMS trong pipeline end-to-end. Đây là thay
đổi ở detection head và assignment, không chỉ là thay một block backbone.

### 2.4 Block riêng của YOLO11

#### `C3k` và `C3k2`

`C3k` là biến thể C3 dùng bottleneck với kernel cấu hình được, thường là hai
convolution 3×3 trong inner block. `C3k2` là khung kiểu C2f/CSP nhẹ hơn:

```text
u = Conv1x1(x) -> split(u, 2) = [a,b]
for mỗi block:
    b = Bottleneck(b)              # hoặc C3k(b) nếu c3k=True
output = Conv1x1(concat(a,b_1,...,b_n))
```

Tham số `e=0.25` ở các tầng P2/P3 giảm hidden channels mạnh. Ở P4/P5,
`shortcut=True` tăng tái sử dụng residual. Nhờ vậy YOLO11 thay C2f bằng block
gọn hơn nhưng vẫn giữ kiểu CSP nhiều đường gradient.

#### `C2PSA`

`C2PSA` là C2/CSP wrapper cho nhiều `PSABlock`:

```text
[a,b] = Conv1x1(x).split(2)
b = Sequential(PSABlock_1, ..., PSABlock_n)(b)
output = Conv1x1(concat(a,b))
```

Chỉ một phần channel chịu self-attention, phần còn lại bypass để giữ thông tin
local và giảm memory. YAML YOLO11 dùng `C2PSA(1024)` sau SPPF ở P5. Khác YOLOv10,
YOLO11 dùng `C2PSA` nhưng detection head cuối vẫn là `Detect` thông thường.

### 2.5 Đọc một dòng YAML như thế nào

Một dòng có dạng `[from, repeats, module, args]`:

```yaml
[-1, 6, C2f, [256, True]]
```

nghĩa là lấy output layer trước đó (`from=-1`), tạo `C2f` lặp 6 lần, với output
channel 256 và shortcut bật. Dòng:

```yaml
[[-1, 4], 1, Concat, [1]]
```

ghép output hiện tại với output layer 4 theo chiều channel (`dim=1`). Cuối cùng:

```yaml
[[16, 19, 22], 1, Detect, [nc]]
```

đưa ba feature map P3/P4/P5 vào detection head. Cách đọc này giúp phân biệt
**topology thực thi** trong YAML với các cơ chế training như PGI hoặc dual
assignment không hiện thành layer riêng.

## 3. Bảng so sánh kiến trúc

| Thành phần | YOLOv5 | YOLOv8 | YOLOv9 | YOLOv10 | YOLO11 |
|---|---|---|---|---|---|
| Backbone block chính | C3 | C2f | GELAN + RepNCSPELAN4 | C2f + C2fCIB | C3k2 |
| Downsampling nổi bật | Conv stride 2 | Conv stride 2 | AConv | SCDown ở P4/P5 và neck P5 | Conv stride 2 |
| Context/attention sâu | SPPF | SPPF | SPPELAN | SPPF + PSA | SPPF + C2PSA |
| Neck fusion | PAN/FPN, C3 | PAN/FPN, C2f | PAN/FPN, RepNCSPELAN4 | PAN/FPN, C2f/C2fCIB | PAN/FPN, C3k2 |
| Detection head | Detect, anchor-based | Detect, anchor-free | Detect, anchor-free | v10Detect, one-to-many + one-to-one | Detect, anchor-free |
| NMS-free native inference | Không | Không | Không | Có mục tiêu thiết kế | Không phải điểm thiết kế chính |
| Gradient/training innovation | Anchor matching + loss pipeline | TAL/DFL pipeline | PGI | dual assignment + consistent dual predictions | training/inference Detect pipeline kế thừa |
| Outputs chuẩn | P3/P4/P5 | P3/P4/P5 | P3/P4/P5 | P3/P4/P5 | P3/P4/P5 |

## 4. Kết luận thực dụng khi chọn model

- **YOLOv5:** baseline lịch sử rõ ràng, C3/CSP đơn giản và anchor-based. Phù hợp
  khi cần tái hiện pipeline cũ hoặc so sánh ảnh hưởng của anchor-free head.
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

## 5. Cách dùng code trong repo

```python
from project_ultralytics.architectures import get_architecture, graph_lines

for name in ("yolov8", "yolov9", "yolov10", "yolo11"):
    print("\n".join(graph_lines(get_architecture(name))))
```

Code khai báo: `project_ultralytics/architectures.py`.
Canonical runtime YAML: `vendor/ultralytics_upstream/ultralytics/cfg/models/`.
Legacy compatibility YAML: `models_related/ultralytics/ultralytics/cfg/models/`.
Không sửa trực tiếp vendor hoặc legacy fork.

## 6. Tài liệu tham khảo

- [YOLOv5 Ultralytics model docs](https://docs.ultralytics.com/models/yolov5/)
- [YOLOv9 paper](https://arxiv.org/abs/2402.13616)
- [YOLOv10 paper](https://arxiv.org/abs/2405.14458)
- [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics model YAMLs trong repo](../..//vendor/ultralytics_upstream/ultralytics/cfg/models/)
