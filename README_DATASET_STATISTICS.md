# Thống kê dữ liệu VisDrone2019 và TinyPerson

Tài liệu này ghi lại thống kê dữ liệu dùng trong repository cho hai bộ dữ liệu **VisDrone2019-DET** và **TinyPerson**. Các con số hình học được tính trực tiếp từ toàn bộ annotation cục bộ. Background, entropy và contrast của **TinyPerson** đã được tính exhaustive trên toàn bộ 746 ảnh train và 786 ảnh test bằng server Marimo; các chỉ số ảnh của VisDrone trong README hiện vẫn là mẫu 60 ảnh mỗi split vì server Marimo chưa có mount VisDrone2019.

> **Phạm vi:** đây là thống kê ground-truth, không phải kết quả dự đoán, AP hay mAP. Không được dùng các số contrast/background bên dưới như metric đánh giá model.

## 1. Quy ước và nguồn dữ liệu

| Dataset | Split | Annotation | Ảnh được đếm |
|---|---|---|---:|
| VisDrone2019-DET | train | `datasets/VisDrone2019/VisDrone2019-DET-train/annotations/*.txt` | 6,471 |
| VisDrone2019-DET | val | `datasets/VisDrone2019/VisDrone2019-DET-val/annotations/*.txt` | 548 |
| VisDrone2019-DET | test-dev | `datasets/VisDrone2019/VisDrone2019-DET-test-dev/annotations/*.txt` | 1,610 |
| TinyPerson | train | `datasets/TinyPerson/annotations/tiny_set_train.json` | 746 |
| TinyPerson | test | `datasets/TinyPerson/annotations/tiny_set_test.json` | 786 |

### Cách lọc annotation

- **VisDrone:** annotation có `category_id` từ `1` đến `10` được xem là object hợp lệ. `category_id=0` và `category_id=11` là vùng/nhãn bỏ qua.
- **TinyPerson:** object hợp lệ là annotation có `ignore=false` và `uncertain=false`. Các annotation có một trong hai cờ này được giữ trong `objects_all` nhưng không đưa vào thống kê object hợp lệ.
- Diện tích bbox là `width × height`, tính theo pixel của ảnh gốc.
- Các bin kích thước dùng chung:

| Bin | Điều kiện diện tích |
|---|---:|
| `tiny` | `< 256 px²` |
| `small` | `256–<1,024 px²` |
| `medium` | `1,024–<9,216 px²` |
| `large` | `≥ 9,216 px²` |

## 2. Tổng quan số ảnh và object

| Dataset / split | Ảnh | Object trong annotation | Object hợp lệ | Bỏ qua | Ảnh không có object hợp lệ | Object/ảnh: mean ± std | Median | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VisDrone train | 6,471 | 353,550 | 343,205 | 10,345 | 0 | 53.04 ± 43.84 | 42 | 902 |
| VisDrone val | 548 | 40,169 | 38,759 | 1,410 | 0 | 70.73 ± 45.95 | 65 | 317 |
| VisDrone test-dev | 1,610 | 77,547 | 75,102 | 2,445 | 0 | 46.65 ± 44.04 | 36 | 461 |
| TinyPerson train | 746 | 25,288 | 18,433 | 6,855 | 58 | 24.74 ± 37.72 | 7 | 177 |
| TinyPerson test | 786 | 18,508 | 13,687 | 4,821 | 20 | 17.46 ± 27.85 | 6 | 173 |

### Mật độ object và khoảng cách

Khoảng cách được đo bằng **khoảng cách Euclid giữa tâm bbox và object gần nhất trong cùng ảnh**, trên tọa độ pixel ảnh gốc. Không tính khoảng cách giữa các ảnh khác nhau.

| Dataset / split | Nearest-center mean ± std (px) | P05 | Median | P95 | Nhận xét |
|---|---:|---:|---:|---:|---|
| VisDrone train | 43.53 ± 55.86 | 4.24 | 24.86 | 144.26 | Cụm object dày, nhiều cặp rất gần nhau |
| VisDrone val | 30.53 ± 38.74 | 3.54 | 17.87 | 100.16 | Split val dày hơn train theo khoảng cách tâm |
| VisDrone test-dev | 44.74 ± 56.77 | 4.03 | 25.62 | 146.99 | Vẫn có nhiều cảnh mật độ cao nhưng phân tán lớn |
| TinyPerson train | 42.63 ± 78.75 | 5.18 | 20.60 | 142.91 | Phần lớn là nhóm nhỏ, có cảnh rất dày |
| TinyPerson test | 80.45 ± 192.71 | 5.17 | 25.33 | 317.17 | Có nhiều ảnh thưa hơn và đuôi phân phối dài |

> Khoảng cách tâm không phải khoảng cách mép bbox. Hai bbox có thể chồng lấn nhưng vẫn có khoảng cách tâm dương. Các giá trị bằng `0` có thể xuất hiện do annotation rất nhỏ hoặc tâm làm tròn về cùng tọa độ.

## 3. Phân phối kích thước object

### VisDrone2019-DET

| Split | Tiny | Small | Medium | Large | Tiny + Small |
|---|---:|---:|---:|---:|---:|
| Train | 89,209 (25.99%) | 118,316 (34.47%) | 116,696 (34.00%) | 18,984 (5.53%) | **60.47%** |
| Val | 11,950 (30.83%) | 14,625 (37.73%) | 11,116 (28.68%) | 1,068 (2.76%) | **68.56%** |
| Test-dev | 27,341 (36.41%) | 23,475 (31.26%) | 21,876 (29.13%) | 2,410 (3.21%) | **67.66%** |

Kích thước bbox hợp lệ, ghi theo `mean width × mean height`, diện tích median và căn bậc hai diện tích trung bình:

| Split | Width mean | Height mean | Area mean ± std (px²) | Area median | √area mean |
|---|---:|---:|---:|---:|---:|
| Train | 38.59 px | 37.40 px | 2,448 ± 6,598 | 680 | 36.44 px |
| Val | 31.06 px | 31.35 px | 1,512 ± 4,385 | 520 | 29.85 px |
| Test-dev | 30.66 px | 32.38 px | 1,687 ± 4,844 | 480 | 30.08 px |

| Split | P05 width × height | Median width × height | P95 width × height | Median aspect ratio `w/h` |
|---|---:|---:|---:|---:|
| Train | 6 × 9 px | 25 × 27 px | 115 × 101 px | 0.91 |
| Val | 6 × 8 px | 20 × 25 px | 94 × 74 px | 0.83 |
| Test-dev | 4 × 7 px | 20 × 23 px | 94 × 88 px | 0.80 |

### TinyPerson

| Split | Tiny | Small | Medium | Large | Tiny + Small |
|---|---:|---:|---:|---:|---:|
| Train | 11,362 (61.64%) | 4,487 (24.34%) | 2,354 (12.77%) | 230 (1.25%) | **85.98%** |
| Test | 7,267 (53.09%) | 4,096 (29.93%) | 2,155 (15.74%) | 169 (1.23%) | **83.02%** |

| Split | Width mean | Height mean | Area mean ± std (px²) | Area median | √area mean |
|---|---:|---:|---:|---:|---:|
| Train | 16.00 px | 25.11 px | 786 ± 2,785 | 168 | 19.38 px |
| Test | 18.69 px | 25.87 px | 841 ± 2,602 | 230 | 21.12 px |

| Split | P05 width × height | Median width × height | P95 width × height | Median aspect ratio `w/h` |
|---|---:|---:|---:|---:|
| Train | 3.78 × 5.91 px | 10.30 × 16.47 px | 46.69 × 74.81 px | 0.60 |
| Test | 3.87 × 6.56 px | 12.08 × 18.56 px | 54.96 × 68.99 px | 0.63 |

**Kết luận về kích thước:** TinyPerson khó hơn về scale object. Gần 86% object train và 83% object test nằm trong `tiny + small`, trong khi VisDrone nằm khoảng 60–69%. TinyPerson cũng có bbox cao và hẹp hơn, thể hiện rõ ở aspect ratio median khoảng `0.60–0.63`.

## 4. Phân phối class

### VisDrone2019-DET

| ID | Class | Train | Val | Test-dev |
|---:|---|---:|---:|---:|
| 1 | pedestrian | 79,337 | 8,844 | 21,006 |
| 2 | people | 27,059 | 5,125 | 6,376 |
| 3 | bicycle | 10,480 | 1,287 | 1,302 |
| 4 | car | 144,867 | 14,064 | 28,074 |
| 5 | van | 24,956 | 1,975 | 5,771 |
| 6 | truck | 12,875 | 750 | 2,659 |
| 7 | tricycle | 4,812 | 1,045 | 530 |
| 8 | awning-tricycle | 3,246 | 532 | 599 |
| 9 | bus | 5,926 | 251 | 2,940 |
| 10 | motor | 29,647 | 4,886 | 5,845 |

Class `car` chiếm lớn nhất ở cả ba split, khoảng 36–42% object hợp lệ. Các class `awning-tricycle`, `tricycle`, `bus` có ít mẫu hơn đáng kể, vì vậy nên báo cáo metric theo class khi so sánh model.

### TinyPerson

| Category ID | Class | Train | Test |
|---:|---|---:|---:|
| 1 | `sea_person` | 6,318 | 4,939 |
| 2 | `earth_person` | 12,115 | 8,748 |

TinyPerson là bài toán hai class theo annotation COCO. `earth_person` chiếm khoảng 65.7% train và 63.9% test object hợp lệ.

## 5. Contrast cục bộ và background

### Định nghĩa

- **Gray mean/std:** trung bình và độ lệch chuẩn grayscale của toàn ảnh. Ảnh được thu nhỏ tối đa còn 512 px ở cạnh dài trước khi tính.
- **Ring background:** vùng bao quanh bbox, xấp xỉ mở rộng thêm 50% chiều rộng và chiều cao bbox theo mỗi phía, sau đó loại phần bbox.
- **Object-ring contrast:**

```text
abs(mean(object) - mean(ring)) / (abs(mean(ring)) + 1e-6)
```

Giá trị càng thấp nghĩa là object khó tách khỏi vùng xung quanh theo grayscale đơn giản. Giá trị lớn hơn `1` có thể xảy ra và không phải xác suất.

Kết quả dưới đây phân biệt rõ:

- **VisDrone:** thống kê background/contrast dùng mẫu cố định `seed=42`, `60 ảnh mỗi split`, vì server Marimo được cung cấp không có dataset VisDrone2019.
- **TinyPerson:** thống kê background/contrast/entropy chạy exhaustive trên toàn bộ ảnh có annotation bằng server Marimo. Ảnh được resize tối đa cạnh dài `1536 px` trước khi tính pixel statistics, nhưng không bỏ ảnh nào: train `746/746`, test `786/786`. Geometry và nearest-center luôn dùng toàn bộ annotation.

| Dataset / split | Gray mean | Gray std | Entropy32 | Ring gray mean | Object-ring contrast mean | Contrast median | Contrast P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VisDrone train, mẫu 60 | 99.82 ± 35.85 | 45.34 ± 10.53 | chưa tính | 111.71 ± 36.78 | 0.160 ± 0.129 | 0.136 | 0.406 |
| VisDrone val, mẫu 60 | 111.98 ± 22.65 | 47.82 ± 10.76 | chưa tính | 110.30 ± 31.11 | 0.151 ± 0.115 | 0.127 | 0.369 |
| VisDrone test-dev, mẫu 60 | 82.33 ± 35.60 | 43.17 ± 9.80 | chưa tính | 95.67 ± 37.30 | 0.187 ± 0.158 | 0.153 | 0.483 |
| TinyPerson train, **toàn bộ 746 ảnh** | 119.96 ± 28.31 | 48.71 ± 13.43 | 4.152 ± 0.445 | 134.09 ± 42.24 | 0.181 ± 0.142 | 0.156 | 0.422 |
| TinyPerson test, **toàn bộ 786 ảnh** | 125.34 ± 20.92 | 47.22 ± 13.08 | 4.070 ± 0.477 | 139.30 ± 37.29 | 0.167 ± 0.119 | 0.147 | 0.388 |

### Diễn giải background

- VisDrone có độ biến thiên background lớn giữa các scene, nhưng các chỉ số image-level hiện chỉ là mẫu 60 ảnh do dataset chưa có trên server Marimo.
- TinyPerson đã được đo trên toàn bộ ảnh. Train có ring gray mean khoảng `134.09`, test khoảng `139.30`; contrast trung bình lần lượt `0.181` và `0.167`.
- Không nên kết luận TinyPerson luôn sáng hơn hoặc luôn khó hơn chỉ từ mean. Cần giữ nguyên preprocessing, độ phân giải và protocol khi so sánh.
- Contrast grayscale không phản ánh texture, màu, bóng, vật cản, blur, JPEG artifact hoặc độ tương phản trên feature map của model.

## 6. Kích thước ảnh và biến thiên scene

Kích thước ảnh không đồng nhất, đặc biệt ở TinyPerson. Một số kích thước xuất hiện trong mẫu 60 ảnh:

| Dataset / split | Các kích thước nổi bật |
|---|---|
| VisDrone train | `1400×1050`, `1360×765`, `2000×1500`, `1400×788`, `1920×1080` |
| VisDrone val | `1360×765`, `960×540`, `1920×1080` |
| VisDrone test-dev | `1400×788`, `1400×1050`, `1360×765`, `1916×1078` |
| TinyPerson train | `1920×1080`, `1280×720`, `1920×1072`, cùng một số ảnh panorama/độ phân giải cao |
| TinyPerson test | `3840×2160`, `1920×1080`, `1280×720`, cùng các kích thước không đồng nhất khác |

Điều này ảnh hưởng trực tiếp đến object pixel size, density theo diện tích và hiệu quả resize. Khi báo cáo thí nghiệm, cần ghi rõ `imgsz`, letterbox/crop strategy và có dùng corner crop hay không.

## 7. Tóm tắt thực hành cho thiết kế detector

1. **Giữ feature map độ phân giải cao:** cả hai dataset đều có nhiều object nhỏ; TinyPerson có tỷ lệ `tiny + small` cao hơn rõ rệt.
2. **Không chỉ tối ưu object trung bình:** median object nhỏ hơn mean rất nhiều do phân phối có đuôi dài.
3. **Cần đánh giá theo density:** VisDrone có nhiều ảnh 40–70 object; TinyPerson có nhiều ảnh ít object hơn nhưng vẫn có cảnh rất dày.
4. **Cần kiểm tra background/contrast:** TinyPerson có contrast grayscale thấp hơn trong mẫu, nên các module context hoặc feature enhancement cần được đánh giá bằng ablation, không suy diễn từ một metric ảnh đơn lẻ.
5. **Báo cáo đúng split:** các số trên là train/val/test-dev của dataset, không thay thế cho `val/AP50`, `val/mAP50-95`, `test/AP50`, hoặc `test/mAP50-95` của model.

## 8. Tái lập thống kê

Các con số được sinh từ các lượt phân tích cục bộ và remote sau:

- Annotation geometry: toàn bộ annotation của 5 split.
- TinyPerson image/background/contrast: server Marimo `sb-9e3c1ab80319a9e9.sb.molab.run`, file kết quả `/marimo/tinyperson_full_stats.json`, toàn bộ ảnh, resize tối đa `1536 px`.
- VisDrone image/background/contrast: mẫu `60 ảnh/split`, `seed=42`, vì remote server không có thư mục/mount VisDrone2019. Cần mount hoặc upload dataset trước khi chạy exhaustive VisDrone.

JSON remote là artifact trung gian, không commit vào repository.
