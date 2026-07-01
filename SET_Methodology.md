# Phương pháp luận: Spectral Enhancement for Tiny Object Detection (SET)

Kiến trúc mạng nơ-ron sâu thường gặp khó khăn trong việc phát hiện đối tượng nhỏ (tiny objects) do đặc trưng của chúng bị suy giảm sau nhiều bước down-sampling và dễ bị chìm lấp bởi nhiễu tần số cao từ hậu cảnh. Để giải quyết vấn đề này, bài báo đề xuất phương pháp **Spectral Enhancement for Tiny object detection (SET)**, hoạt động dựa trên một kiến trúc không đồng nhất (heterogeneous architecture) với hai mô-đun cốt lõi: **Hierarchical Background Smoothing (HBS)** và **Adversarial Perturbation Injection (API)**.

Dưới đây là chi tiết về cơ sở toán học và cơ chế hoạt động của từng mô-đun.

---

## 1. Hierarchical Background Smoothing (HBS)

Mục tiêu của HBS là triệt tiêu các thông tin nhiễu tần số cao (high-frequency noise) ở vùng hậu cảnh (background) trong khi vẫn bảo toàn các chi tiết của tiền cảnh (foreground). Điều này giúp làm nổi bật các đặc trưng tần số của đối tượng nhỏ.

### 1.1. Phân tách đặc trưng (Feature Decoupling)
Cho một ảnh đầu vào $X \in \mathbb{R}^{3 \times W \times H}$, một mặt nạ nhị phân (binary mask) $M$ được tạo ra dựa trên ground truth bounding box $B$:

$$M_{i,j} = \mathbf{1}[(i,j) \in B]$$

Trong mạng Feature Pyramid Network (FPN) gồm $N$ lớp, đặc trưng tại lớp thứ $i$ được ký hiệu là $P_i$. HBS thực hiện phân tách và xử lý đặc trưng như sau:

$$P_i^E = P_i^{fg} + \phi_i(P_i^{bg}, r) = P_i \otimes M + \phi_i(P_i \otimes 
eg M, r)$$

Trong đó:
* $P_i^E$, $P_i^{fg}$, và $P_i^{bg}$ lần lượt là đặc trưng đã được tăng cường, đặc trưng tiền cảnh và đặc trưng hậu cảnh.
* $
eg M$ là phần bù của mặt nạ nhị phân.
* $\otimes$ là phép nhân Hadamard (Hadamard product) có hỗ trợ broadcasting.

### 1.2. Thao tác làm mịn thích ứng (Adaptive Smoothing Operation)
Thay vì sử dụng các bộ lọc tĩnh tĩnh, HBS áp dụng một hàm làm mịn thích ứng $\phi_i(\cdot, r)$ với tỷ lệ giảm kênh (channel reduction rate) $r$:

$$\phi_i(P_i^{bg}, r) = \sigma(w_i^e \otimes \sigma(w_i^r \otimes P_i^{bg})) + P_i^{bg}$$

Trong đó:
* $w_i^r \in \mathbb{R}^{C_i \times C_{i/r} \times K_i \times K_i}$ là trọng số của lớp tích chập giảm kênh.
* $w_i^e \in \mathbb{R}^{C_{i/r} \times C_i \times K_i \times K_i}$ là trọng số của lớp tích chập khôi phục (tăng) kênh.
* $\sigma$ là hàm kích hoạt phi tuyến ReLU.

Quá trình giảm chiều (reduction) giúp nén thông tin, và khi mở rộng (expansion) trở lại, các thông tin tần số cao rất khó được tái tạo, dẫn đến một không gian hậu cảnh mượt mà hơn. 

Kích thước kernel $K_i$ được xác định linh hoạt dựa trên stride $S_i$ của lớp FPN tương ứng:

$$K_i = g(S_i) = \left( \lfloor \frac{\log_2(S_i)}{2} \rfloor \times 2 \right) + 1$$

---

## 2. Adversarial Perturbation Injection (API)

Do đối tượng nhỏ có tín hiệu tần số cao rất yếu và khó phân biệt, API được thiết kế để tiêm các nhiễu đối kháng (adversarial perturbations) ở cấp độ đặc trưng (feature-level). Quá trình này giúp tăng cường độ nổi bật (saliency) của các vùng chứa thông tin quan trọng và củng cố tính biểu diễn đặc trưng (feature representation) thông qua quá trình huấn luyện đối kháng (adversarial training).

### 2.1. Tối ưu hóa Min-Max
Mục tiêu của API cho mỗi lớp FPN được định nghĩa bằng bài toán tối ưu hóa:

$$\min_{\theta_i} \left( \max_{||\epsilon_{i,cls}|| \le \rho} \mathcal{L}_{cls}(P_i + \epsilon_{i,cls}) + \gamma ||P_i||_2^2 \right)$$

Trong đó:
* $\mathcal{L}_{cls}$ là hàm mất mát phân loại (classification loss).
* $\rho$ giới hạn kích thước của nhiễu.
* $\gamma$ là siêu tham số kiểm soát chuẩn hóa $L_2$.

Nghiệm đóng (closed-form solution) cho nhiễu đối kháng dưới chuẩn $L_2$ được tính bằng đạo hàm của hàm loss:

$$\epsilon_{i,cls}^* = \rho \frac{\nabla_{P_i} \mathcal{L}_{cls}(P_i)}{||\nabla_{P_i} \mathcal{L}_{cls}(P_i)||_2}$$

Việc tiêm nhiễu này làm tăng gradient tại các vùng không gian quan trọng, buộc mô hình phải tập trung học các đặc trưng có tính phân biệt cao hơn.

### 2.2. Tổng hợp nhiễu đa nhánh
Trong một kiến trúc detector (như FCOS có các nhánh classification, regression, và center-ness), nhiễu đối kháng tổng hợp $\epsilon_i^{adv}$ được tính bằng kỳ vọng trên $M$ nhánh:

$$\epsilon_i^{adv} = \sum_{m=1}^M \lambda_m \cdot \epsilon_{i,m}^*$$

Sự kết hợp này đảm bảo tính cân bằng về ngữ nghĩa phát hiện đối tượng giữa các nhánh dự đoán.

---

## 3. Tối ưu hóa phụ trợ (Auxiliary Optimization)

Hệ thống sử dụng một hàm mất mát tổng hợp (total loss function) kết hợp giữa hàm mất mát phát hiện đối tượng gốc $\mathcal{L}_{detection}$ và hàm mất mát phụ trợ (auxiliary loss) tính trên các đặc trưng đã được tinh chỉnh bởi HBS và API:

$$\mathcal{L} = \mathcal{L}_{detection} + \lambda \cdot \sum_{i=1}^N \mathcal{L}_i^{aux}(P_i^E + \epsilon_i^{adv})$$

Trong đó, $\lambda$ là tham số cân bằng tỷ trọng của nhánh phụ trợ. Việc cập nhật tham số mô hình thông qua cả hai luồng loss giúp tạo ra các biểu diễn đặc trưng mạnh mẽ và nhạy cảm hơn đối với các vật thể có kích thước cực nhỏ.
