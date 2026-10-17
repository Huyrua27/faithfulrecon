# Geometric Failure, Labels and Intervention Protocol — Technical Report

**Tác giả:** Trường Thịnh  
**Phân công:** Research Track 2 — Geometric Failure & Intervention Analysis  
**Đề tài:** FaithfulRecon: An Intervention-Based Protocol for Evaluating Spatial Reliability Signals in 3D Gaussian Splatting  
**Mã nguồn triển khai:** `frecon/benchmark/`  
**Dữ liệu xuất bản:** `runs/d1/synthetic/v8_az360/seed_{0,1,2}/geometric_labels.npz`

```text
Gaussian Model (mu_i) → Reference Surface (S) → Continuous Error (d_i) → Failure Label (f_i)
Under-observed Condition (m_i)  ≠  Geometric Failure (f_i)
```

---

## 1. Định nghĩa bài toán (Problem Definition)

3D Gaussian Splatting (3DGS) mô hình hóa cảnh 3D bằng tập hợp các hạt Gaussian dị hướng và tối ưu hóa thông qua hàm mất mát trắc quang 2D (photometric loss). Trong điều kiện thu nhận góc nhìn thưa (sparse-view), bài toán tối ưu trở nên thiếu ràng buộc nghiêm trọng (ill-posed inverse problem). 

Nhiều cụm Gaussian trôi nổi tự do ngoài không gian (floaters) xuất hiện nhưng vẫn tái tạo hoàn hảo trên các ảnh huấn luyện 2D. Các phương pháp hiện tại thường dựa vào các tín hiệu nội tại per-primitive (như gradient, opacity, visibility) để quyết định mật độ hóa hoặc tỉa bớt hạt, nhưng chưa từng có một giao thức độc lập nào đánh giá xem các tín hiệu này có thực sự nhận diện đúng các hạt liên quan đến lỗi hình học hay không.

Nhiệm vụ của Track 2 là:
1. Xây dựng định nghĩa hình học chuẩn xác và không thiên vị (ground-truth geometric failure labels).
2. Tách biệt nguyên nhân gây lỗi (under-observed condition $m_i$) và kết quả lỗi hình học ($f_i$).
3. Thiết lập giao thức can thiệp (intervention protocol) nhằm kiểm chứng phản ứng nhân quả của mô hình.

---

## 2. Điều kiện thiếu quan sát (Under-observed Condition — $m_i$)

### 2.1. Bản chất lý thuyết
Điều kiện thiếu quan sát $m_i$ là nhãn điều kiện (condition label), **hoàn toàn độc lập với mô hình Gaussian đã huấn luyện và không phụ thuộc vào bất kỳ tín hiệu (signal) nào**. $m_i$ chỉ phụ thuộc duy nhất vào cấu hình camera huấn luyện và hình học bề mặt vật thể chuẩn $\mathcal{S}$.

Chuỗi quan hệ nhân quả được giả định như sau:
$$	ext{Góc nhìn thưa (Sparse views)} \longrightarrow 	ext{Vùng thiếu quan sát } (m_i) \longrightarrow 	ext{Mơ hồ tối ưu (Ambiguity)} \longrightarrow 	ext{Lỗi hình học } (f_i)$$

### 2.2. Công thức toán học
Một điểm tham chiếu $P \in \mathcal{S}$ bị coi là thiếu quan sát nếu thỏa mãn ít nhất một trong hai tiêu chí:
1. Số lượng camera huấn luyện nhìn thấy điểm $P$ nhỏ hơn ngưỡng $k_{	ext{views}}$:
   $$V(P) = \sum_{v=1}^{V} \mathbb{I}(P 	ext{ visible in view } v) < k_{	ext{views}}$$
2. Góc tam giác đạc cực đại (maximum pairwise triangulation angle) giữa các tia nhìn thấy điểm $P$ nhỏ hơn ngưỡng $	heta$:
   $$	heta_{\max}(P) = \max_{a, b \in 	ext{Seen}(P)} rccos\left(rac{r_a \cdot r_b}{\|r_a\| \|r_b\|}ight) < 	heta$$

Với mỗi hạt Gaussian thứ $i$ có tâm $\mu_i$, ta tìm điểm tham chiếu gần nhất trên bề mặt chuẩn:
$$P_i^* = rg\min_{P \in \mathcal{S}} \|\mu_i - P\|_2$$
Nhãn điều kiện được gán: $m_i = \mathbb{I}(P_i^* 	ext{ is under-observed})$.

### 2.3. Xử lý che khuất và hướng bề mặt (Occlusion & Normal Facing)
Một điểm $P$ có vector pháp tuyến bề mặt $ec{n}_P$ chỉ được coi là nhìn thấy từ camera có tâm $C_v$ nếu:
- Điểm nằm trong trường nhìn (frustum): $u \in [0, W]$, $v \in [0, H]$ và khoảng cách chiều sâu $z_{	ext{cam}} > 0$.
- Bề mặt quay về phía camera (backface culling):
  $$\cos\phi = ec{n}_P \cdot rac{C_v - P}{\|C_v - P\|} > 0.05$$
Điều này loại bỏ triệt để hiện tượng nhìn xuyên qua vật thể đối với các camera đặt ở phía đối diện.

---

## 3. Định nghĩa lỗi hình học (Geometric Failure Outcome — $f_i$)

### 3.1. Bản chất lý thuyết
Nhãn lỗi hình học $f_i$ là nhãn kết quả (outcome label), đánh giá xem hạt Gaussian $i$ sau quá trình huấn luyện có thực sự là một cấu trúc lỗi (floater) hay không.

### 3.2. Công thức toán học
Hạt Gaussian thứ $i$ bị coi là lỗi hình học ($f_i = 1$) khi và chỉ khi thỏa mãn đồng thời 3 điều kiện:
$$f_i = \mathbb{I}(d_i > 	au) \land \mathbb{I}(	ext{contrib}_i > \epsilon) \land \mathbb{I}(	ext{opacity}_i > 	ext{min\_opacity})$$

Trong đó:
- $d_i$: Khoảng cách Euclidean từ tâm Gaussian $\mu_i$ tới bề mặt chuẩn $\mathcal{S}$. Ngưỡng mặc định $	au = 0.03$ world units.
- $	ext{contrib}_i$: Tổng mức độ đóng góp tạo ảnh tích lũy của hạt trên toàn bộ các góc nhìn kiểm thử (held-out views):
  $$	ext{contrib}_i = \sum_{c \in 	ext{TestCams}} 	ext{render}(c, 	ext{return\_contrib=True})_i$$
  Ngưỡng tối thiểu $\epsilon = 0.01$. Điều kiện này loại bỏ các hạt nằm xa nhưng hoàn toàn tàng hình (không ảnh hưởng tới hình ảnh).
- $	ext{opacity}_i > 0.05$: Loại bỏ các hạt gần như trong suốt.

---

## 4. Hình học bề mặt tham chiếu (Reference Geometry)

* **Nguồn dữ liệu:** Cảnh Synthetic gồm các vật thể hình học giải tích (hình cầu bán kính $R=0.38$, hình hộp chữ nhật, 3 thanh que mỏng `thin rods` bán kính $r=0.02$).
* **Hệ tọa độ:** Đồng nhất với hệ tọa độ thế giới (world coordinate frame) của mô hình Gaussian (OpenCV convention: $x$ sang phải, $y$ hướng xuống, $z$ hướng tới trước). Không gian cảnh chuẩn hóa nằm trong $[-1, 1]^3$.
* **Mật độ điểm:** Bề mặt được lấy mẫu dầy đặc với $M = 60,000$ điểm tham chiếu kèm pháp tuyến $ec{n}$, đảm bảo khoảng cách giữa các điểm lân cận $\ll 	au = 0.03$.
* **Kiểm tra độ bao phủ (Frustum Coverage Check):** Hàm `check_alignment()` kiểm tra hình chiếu của $M$ điểm tham chiếu lên tập camera huấn luyện, ghi nhận tỷ lệ bao phủ trung bình đạt trên $30\% - 45\%$ số điểm trên mỗi góc nhìn, xác nhận dữ liệu hình học nằm trọn vẹn trong không gian quan sát của hệ thống camera.

---

## 5. Sai số hình học liên tục (Continuous Geometric Error — $d_i$)

Khoảng cách continuous $d_i$ đo độ lệch thực tế của từng hạt:
$$d_i = \min_{j=1..M} \|\mu_i - P_j\|_2$$

* **Triển khai tối ưu bộ nhớ:** Để tránh tràn VRAM GPU khi nhân ma trận kích thước lớn, hàm `point_to_surface_distance` sử dụng kỹ thuật chia khối 2 chiều (double-chunking) với kích thước block Gaussian $512$ và block tham chiếu $8192$ kết hợp `torch.cdist(..., p=2)`.
* **Phân bố thực nghiệm:** Trên cả 3 seed, khoảng cách $d_i$ dao động liên tục từ $0.0000$ đến $1.1460$ world units. Phân bố có dạng hàm mũ dốc đứng: đa số hạt tập trung sát bề mặt ($d_i < 0.01$), trong khi đuôi phân bố kéo dài đại diện cho các floaters bay lơ lửng ngoài không gian.

---

## 6. Phân tích nhãn nhị phân và Tính ổn định qua Seed

Bảng thống kê tỷ lệ nhãn thực nghiệm được tính toán độc lập qua 3 seed với cấu hình chuẩn ($	au = 0.03$, $k_{	ext{views}} = 2$, $	heta = 10^\circ$):

| Seed | Tổng số Gaussian ($N$) | Tỷ lệ Under-observed ($m_i$) | Tỷ lệ Geometric Failure ($f_i$) | Khoảng cách $d_i$ (min / max) |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 27,733 | 1,160 hạt (**4.18%**) | 2,329 hạt (**8.40%**) | 0.0000 / 1.1460 |
| **Seed 1** | 27,984 | 1,401 hạt (**5.01%**) | 2,531 hạt (**9.04%**) | 0.0000 / 1.0713 |
| **Seed 2** | 28,691 | 1,396 hạt (**4.87%**) | 2,467 hạt (**8.60%**) | 0.0000 / 1.0626 |
| **Trung bình** | **28,136 ± 496** | **4.69% ± 0.44%** | **8.68% ± 0.33%** | — |

* **Tính ổn định (Seed Consistency):** Độ lệch chuẩn giữa các seed cực nhỏ ($< 0.5\%$), chứng minh quy trình sinh nhãn có độ tin cậy và độ lặp lại thống kê rất cao.
* **Mối tương quan $m_i 
eq f_i$:** Tỷ lệ $m_i pprox 4.7\%$ trong khi $f_i pprox 8.7\%$. Điều này chứng minh thực tế: không phải mọi vùng thiếu quan sát đều sinh ra floaters, và floaters có thể xuất hiện do sự lan truyền lỗi tối ưu ra ngoài vùng thiếu góc nhìn.

---

## 7. Giả thuyết can thiệp (Intervention Hypothesis)

Nếu một tín hiệu nội tại $S$ (ví dụ GradMag hay Opacity) thực sự phản ánh độ kém tin cậy hình học, thì việc can thiệp loại bỏ Top-K% hạt có score cao nhất phải dẫn đến:
1. **Phản ứng có thể đo lường (Measurable):** Cường độ phản ứng phải vượt qua ngưỡng nhiễu nội tại của mô hình ($	ext{SNR}_{	ext{in}} \ge 2.0$).
2. **Tính cục bộ (Locality):** Phản ứng thay đổi phải tập trung tại vùng không gian chứa các hạt bị xóa, không làm xáo trộn toàn bộ cảnh.
3. **Đúng chiều kỳ vọng (Correct Direction):** Việc dọn dẹp các hạt lỗi phải giúp giảm sai số trên góc nhìn mới và cải thiện độ chuẩn xác hình học so với việc xóa ngẫu nhiên (Matched Random Control).

---

## 8. Tiêu chí tính cục bộ (Locality Criterion)

* **Trong không gian ảnh (Image-space Locality Ratio — $	ext{LR}_{	ext{img}}$):**
  $$	ext{LR}_{	ext{img}} = rac{d_{	ext{in}}}{d_{	ext{in}} + d_{	ext{out}}}$$
  Trong đó $d_{	ext{in}}$ và $d_{	ext{out}}$ là sai khác ảnh trung bình per-pixel bên trong và bên ngoài mặt nạ chiếu của các hạt bị xóa.
* **Trong không gian 3D (Parametric Displacement — $	ext{Disp}$):**
  Đo khoảng cách dịch chuyển vị trí trung bình $\|\Delta \mu\|$ của các hạt nằm trong quả cầu lân cận bán kính $r = 0.05$ xung quanh vị trí các hạt bị xóa so với phần còn lại của mô hình.

---

## 9. Tiêu chí chiều phản ứng (Direction Criterion)

Chiều phản ứng đo lường mức độ cải thiện của mô hình sau khi tối ưu lại:
$$	ext{Dir} = -(	ext{Error}_{	ext{post-intervention}} - 	ext{Error}_{	ext{baseline}})$$
Giá trị $	ext{Dir} > 0$ đồng nghĩa với việc sai số giảm (mô hình tốt lên).
So sánh với đối chứng ngẫu nhiên:
$$	ext{Gain\_Dir} = 	ext{Dir}(	ext{Signal}) - 	ext{Dir}(	ext{Random})$$
Một tín hiệu được xác nhận hỗ trợ hình học khi $	ext{Gain\_Dir} > 0$ một cách nhất quán qua các seed.

---

## 10. Tiêu chí ngưỡng nhiễu (Noise-floor Criterion)

Quá trình tối ưu lại (re-optimization) bằng SGD vốn có tính ngẫu nhiên nội tại. Ngưỡng nhiễu được đo lường thông qua phép thử đối chứng không can thiệp (Null run):
- `null_a` và `null_b`: Tối ưu lại độc lập 1000 bước từ cùng một checkpoint ban đầu.
- Sai lệch giữa `null_a` và `null_b` thiết lập nên ngưỡng nhiễu nền (Noise floor $\sigma_{	ext{noise}}$).
- Tỷ số tín hiệu trên nhiễu:
  $$	ext{SNR}_{	ext{in}} = rac{|\Delta_{	ext{signal}}|}{\sigma_{	ext{noise}}}$$
Chỉ khi $	ext{SNR} \ge 2.0$, phản ứng mới được coi là có ý nghĩa thực nghiệm.

---

## 11. Các hạn chế đã biết (Known Limitations)

1. **Khoảng cách điểm rời rạc:** Khoảng cách $d_i$ được tính tới điểm tham chiếu gần nhất trong số $60,000$ điểm được lấy mẫu, không phải khoảng cách giải tích tới bề mặt lưới liên tục.
2. **Xử lý che khuất:** Điều kiện tầm nhìn $m_i$ sử dụng kiểm tra frustum kết hợp hướng pháp tuyến (normal-facing check), chưa tích hợp ray-tracing depth buffer per-pixel hoàn chỉnh.
3. **Cộng dồn đóng góp trắc quang:** Giá trị `heldout_contrib` hiện được tính bằng tổng đóng góp alpha qua toàn bộ 16 test views (`sum_over_test_views`), áp dụng ngưỡng cố định $\epsilon = 0.01$.

---

## 12. Đề xuất cải tiến cho các giai đoạn tiếp theo (Refinements)

1. **Mặt nạ chiếu thích nghi (Adaptive Projection Masking):** Giảm diện tích mặt nạ 2D bằng cách ngưỡng hóa theo độ mờ đục thực tế của hạt thay vì gán giá trị giả định 0.99, giúp tránh hiện tượng bão hòa $	ext{LR}_{	ext{img}} pprox 0.90$.
2. **Tách biệt chỉ số dọn dẹp lỗi và bảo tồn bề mặt:** Bóc tách Chamfer Distance thành $	ext{Floater Clearance}$ và $	ext{Surface Damage}$ để đánh giá công bằng các tín hiệu nhạy cảm với cạnh mỏng như GradMag.
3. **Đối chứng phân tầng theo Opacity (Stratified Random Control):** Lấy mẫu ngẫu nhiên có cùng phân bố opacity với nhóm Top-K để loại bỏ yếu tố gây nhiễu từ độ mờ đục.

---

## 13. Hình ảnh trực quan thực nghiệm (Figures)

Toàn bộ các biểu đồ và hình vẽ 3D được lưu trữ độc lập theo từng seed tại `results/benchmark/figures/seed_{0,1,2}/`:

1. **`ref_vs_gaussians.png`:** Hình ảnh đối chiếu không gian giữa bề mặt chuẩn (màu xám) và vị trí các tâm Gaussian (màu đỏ).
2. **`under_observed_map.png`:** Bản đồ không gian 3D phân bố nhãn $m_i = 1$ (các điểm màu đỏ tập trung ở phần đáy quả cầu và đáy hộp, nơi thiếu góc nhìn từ hệ thống camera ngẩng).
3. **`geometric_failure_map.png`:** Bản đồ các hạt lỗi hình học $f_i = 1$ (các floaters trôi nổi ngoài bề mặt vật thể).
4. **`error_distribution.png`:** Biểu đồ histogram phân bố khoảng cách sai lệch $d_i$, so sánh trực quan giữa nhóm lỗi hình học ($f_i=1$) và nhóm không lỗi ($f_i=0$).

---

## 14. Tài liệu tham khảo (References)

1. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics, 42(4).
2. FaithfulRecon Team. (2026). *FaithfulRecon: An Intervention-Based Protocol for Evaluating Spatial Reliability Signals in 3D Gaussian Splatting*. Research Overview & Specification Document.
