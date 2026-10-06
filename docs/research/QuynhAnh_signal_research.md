# Reliability signals in 3DGS — research notes

**Owner:** Quỳnh Anh · **Branch:** `feature/quynhanh-signals` · **Status:** TODO

> Câu hỏi cần trả lời: với từng reliability signal, signal thực sự đang đo cái gì, được tính như
> thế nào trong 3DGS, và có cơ sở nào để kỳ vọng nó liên quan đến geometric reliability?

Code: `frecon/signals/` · Export: `scripts/export_signals.py` · Test: `tests/test_signals.py`
Quy ước: score cao = **kém** tin cậy hơn; output `(N,)` float32, theo thứ tự checkpoint (xem `docs/data_format.md`).

Mỗi signal khoảng 1–2 trang, chỉ những gì cần cho thí nghiệm.

---

## Signal 1 — Opacity (`frecon/signals/opacity.py`)

### 1. Signal definition
TODO

### 2. Mathematical formulation
TODO — hiện tại: `score_i = 1 − sigmoid(opacity_logit_i)`.

### 3. Implementation location in 3DGS
TODO — tham số nào, được kích hoạt ở đâu (`frecon/gaussians.py`), bị ảnh hưởng bởi opacity reset / prune thế nào (`frecon/train.py`).

### 4. Input / output
TODO

### 5. Expected interpretation
TODO

### 6. Potential confounders
TODO — gợi ý: Gaussian gần trong suốt không đóng góp vào ảnh → xoá không đổi gì (Phase 0: SNR_in ≈ 1.1).

### 7. Limitations
TODO

### 8. Relation to FaithfulRecon
TODO — dùng trong D1/D2 như thế nào.

### 9. References
TODO

---

## Signal 2 — Gradient magnitude (`frecon/signals/gradmag.py`)

### 1. Signal definition
TODO
### 2. Mathematical formulation
TODO — hiện tại: trung bình qua các view nhìn thấy của ‖∂L_c/∂μ2D_i‖ (NDC).
### 3. Implementation location in 3DGS
TODO — so với bộ tích luỹ densification trong `Trainer.train_step`.
### 4. Input / output
TODO
### 5. Expected interpretation
TODO
### 6. Potential confounders
TODO — kích thước trên màn hình, opacity, số view.
### 7. Limitations
TODO
### 8. Relation to FaithfulRecon
TODO — giải thích kết quả Phase 0: response đo được (SNR_in 4.7) nhưng hình học cục bộ xấu hơn random.
### 9. References
TODO

---

## Signal 3 — Visibility (`frecon/signals/visibility.py`)

### 1. Signal definition
TODO
### 2. Mathematical formulation
TODO — hiện tại: −(số view train mà Gaussian đóng góp > 1e-3) + tie-break ngẫu nhiên.
### 3. Implementation location in 3DGS
TODO
### 4. Input / output
TODO
### 5. Expected interpretation
TODO
### 6. Potential confounders
TODO — Gaussian không bao giờ nhìn thấy; tie; gần với định nghĩa `m_i` (nguy cơ đánh giá vòng tròn).
### 7. Limitations
TODO
### 8. Relation to FaithfulRecon
TODO
### 9. References
TODO

---

## Signal 4 — Cross-view gradient inconsistency (`frecon/signals/gradient_inconsistency.py`)

### 1. Signal definition
TODO
### 2. Mathematical formulation
TODO — `v_i = Tr(Cov_c[g_ic]) / (‖E_c[g_ic]‖² + δ)`, `g_ic = ∂L_c/∂μ_i`.
### 3. Implementation location in 3DGS
TODO
### 4. Input / output
TODO
### 5. Expected interpretation
TODO — đây là giả thuyết, không phải giả định.
### 6. Potential confounders
TODO
### 7. Limitations
TODO
### 8. Relation to FaithfulRecon
TODO — thành phần `v_i` của `s_geom`.
### 9. References
TODO

---

## Phân tích Phase 0 (mục 1.3 của phân công)

Dùng `docs/templates/experiment_report.md`. Số liệu: `results/phase0/report.md`.
TODO: signal nào response mạnh/yếu, khác random không, locality, ổn định qua seed, trường hợp đo được nhưng ngược chiều.

## Đề xuất refinement (mục 1.4)

Mỗi đề xuất: Problem → Why it matters → Proposed change → What experiment would validate it.
Không tự đổi protocol chính khi chưa trao đổi.

TODO
