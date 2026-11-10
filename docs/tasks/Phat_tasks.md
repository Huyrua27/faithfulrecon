# FaithfulRecon — Phân công: Lê Trần Tấn Phát

## Vai trò

**Research Track 4 — Second-order signals & D3 (held-out predictive validity)**

Phát cùng Huy làm phần lõi nghiên cứu. Hai việc chính dưới đây **không ai trong 3 bạn còn lại
làm**, nhưng bắt buộc phải có để trả lời H1 (signal bậc hai so với baseline) và RQ3/RQ4
(predictive validity, và nó có đi kèm intervention response hay không).

| | Huy | Phát |
|---|---|---|
| Nghiên cứu | D2 (intervention protocol), Phase 0.5 | Signal bậc hai, `s_geom`, D3 |
| Hạ tầng | lõi `frecon/`, renderer, GPU / gsplat, scene thật | — |
| Điều phối | review PR, data contract, ghép nối end-to-end | review PR của Quỳnh Anh (phần signal) |
| Viết | Intro, Method (D2), báo cáo GVHD | Related work (FisherRF, PUP, uncertainty), Method (signal bậc hai, D3) |

Branch: `feature/phat-second-order`, `feature/phat-d3`.

---

## 1. Signal bậc hai (hạn: 25/10 — mốc W4 "signal zoo")

### 1.1. Ước lượng khối Gauss–Newton / Fisher theo từng Gaussian

Cả ba signal dưới đây cần cùng một đại lượng: với mỗi Gaussian *i* và nhóm tham số θ_i
(vị trí μ_i, hoặc μ_i + scale), khối

```text
F_i = Σ_views Σ_pixels J_{p,i}^T J_{p,i}        (J_{p,i} = ∂I_p / ∂θ_i)
```

Không tính Jacobian từng pixel (quá đắt). Dùng **ước lượng ngẫu nhiên kiểu Hutchinson**:
với vector ngẫu nhiên v (Rademacher, ±1 cho mỗi pixel và kênh màu),

```text
g = J^T v = ∂<v, I> / ∂θ            (một lần backward)
E[g_i g_i^T] = Σ_p J_{p,i}^T J_{p,i} = F_i      vì E[v v^T] = I
```

→ mỗi view lấy S mẫu v, cộng dồn `g_i g_i^T` (3×3 hoặc 6×6 cho mỗi Gaussian).
Viết một hàm dùng chung `fisher_blocks(ctx, params=("means",), n_samples=S)` trong
`frecon/signals/second_order.py`, có cache trong `ctx.cache` giống `view_statistics`.

**Bắt buộc kiểm chứng:** trên cảnh rất nhỏ (vài chục Gaussian, ảnh 16×16), tính Jacobian chính xác
bằng `torch.autograd.functional.jacobian` và so với ước lượng → sai số tương đối giảm theo 1/√S.

### 1.2. Các signal

| Signal | Công thức (score cao = kém tin cậy) | File |
|---|---|---|
| `fisher` | kiểu FisherRF: `Tr(F_i)` trên tất cả tham số của Gaussian; score = −log Tr(F_i) (ít thông tin = kém tin cậy) | `fisher.py` |
| `pup` | kiểu PUP 3D-GS: `log det(F_i + εI)` trên khối (μ, scale) 6×6; score = −log det (PUP cắt Gaussian độ nhạy thấp) | `pup_sensitivity.py` |
| `curvature` | `u_i = Tr((F_i^{μ} + εI)^{-1})`, khối 3×3 vị trí; score = u_i | `s_geom.py` |
| `s_geom` | kết hợp `u_i` với `v_i` (gradient inconsistency của Quỳnh Anh) | `s_geom.py` |

Phải đọc lại paper gốc để chỉnh công thức cho đúng (tham số nào, chuẩn hoá thế nào, hướng score)
và ghi rõ chỗ nào khác paper. FisherRF và PUP dùng CUDA kernel riêng; mình xấp xỉ bằng estimator ở
1.1 — đây là điểm phải nói rõ trong tài liệu.

### 1.3. `s_geom` và ablation

- `v_i` do Quỳnh Anh làm (`frecon/signals/gradient_inconsistency.py`). Chưa có thì dùng bản tạm,
  nhưng **không** sửa file của Quỳnh Anh — trao đổi qua issue.
- Các biến thể: `u` riêng, `v` riêng, tích `u·v`, tổng hạng (rank-sum). Mọi metric D1–D3 dựa trên
  thứ hạng → `exp(−λ·)` và λ không ảnh hưởng; ghi rõ điều này.
- Độ nhạy theo ε (damping) và S (số mẫu).

### Kết quả kỳ vọng

- Signal chạy qua `compute_signals(...)`, đúng hợp đồng trong `docs/data_format.md`.
- Ghi được vào `scores.npz` để Tuấn Khải chạy D1 mà không sửa code.
- Thời gian chạy và bộ nhớ trên GTX 1650 cho cảnh synthetic (~28k Gaussian).

---

## 2. D3 — held-out predictive validity (hạn: 15/11, trước mốc W8)

Câu hỏi: signal có dự báo được sai số trên các view **không** dùng để huấn luyện không?

### 2.1. Sai số held-out gán cho từng Gaussian

```text
e_i = Σ_views Σ_p w_{i,p} · err_p  /  Σ_views Σ_p w_{i,p}
```

`w_{i,p}` = trọng số blending của Gaussian *i* tại pixel *p*. Không cần sửa renderer: render với
`features` là một cột hằng số `requires_grad`, lấy `Σ_p err_p · out_p` rồi backward → gradient theo
feature của Gaussian *i* chính là `Σ_p w_{i,p} err_p`. Mẫu số là `contrib` (đã có,
`return_contrib=True`). `err_p`: L1 hoặc 1−SSIM theo pixel so với GT.

### 2.2. Metric

| Mức | Metric | Cách tính |
|---|---|---|
| Gaussian | Spearman(score, e_i) | trên Gaussian có contrib > 0 ở held-out |
| Gaussian | AUROC(score, f_i) | f_i từ Trường Thịnh (đang có bản tạm) |
| Pixel | AUSE | render score thành bản đồ (`features=score[:,None]`), sắp xếp pixel theo score, đường sparsification so với oracle (sắp theo sai số thật) |
| Tuỳ chọn | calibration | isotonic regression score → sai số, fit trên một nửa view held-out, đánh giá trên nửa còn lại |

Báo cáo kèm **random baseline** cùng scene/seed (giống D1).

### 2.3. Output

```text
results/d3/<scene>/<condition>/d3_metrics.csv     scene,condition,seed,signal,spearman_e,auroc_f,ause,ause_random
results/d3/<scene>/<condition>/figures/           sparsification curves, score vs e_i
```

Lệnh mục tiêu:

```bash
python scripts/run_d3.py --scene synthetic --condition v8_az360 --config configs/phase0.yaml
```

### 2.4. Liên hệ RQ4

D3 xong là có thể đặt cạnh D2 (Huy) và D1 (Tuấn Khải): signal nào dự báo tốt (D3) nhưng can thiệp
không cục bộ / ngược chiều (D2)? Phát chuẩn bị bảng ghép D1–D2–D3 theo signal.

---

## 3. Viết

- Related work: FisherRF, PUP 3D-GS, các phương pháp uncertainty cho radiance field
  (Manifold Sampling, Bayes' Rays, CF-NeRF, ...). Mỗi paper: signal là gì, dùng để làm gì,
  được đánh giá thế nào → khác FaithfulRecon ở đâu. Không claim "chưa ai làm" nếu chưa kiểm tra kỹ.
- Method: phần signal bậc hai và D3.
- File: `docs/research/Phat_second_order_and_d3.md`.

---

## 4. Lịch

| Tuần | Thời gian | Việc | Mốc |
|---|---|---|---|
| W2 | 09/10–11/10 | Đọc FisherRF + PUP (phần công thức), đọc `frecon/signals/`, `frecon/render/torch_rast.py` | — |
| W3 | 12/10–18/10 | `fisher_blocks` + test so với Jacobian chính xác | estimator đúng |
| W4 | 19/10–25/10 | `fisher`, `pup`, `curvature`, `s_geom` (bản tạm v_i); xuất `scores.npz` | **signal zoo đủ cho D1** |
| W5 | 26/10–01/11 | Ablation `s_geom`; hỗ trợ Huy đưa signal bậc hai vào D2 | — |
| W6–W7 | 02/11–15/11 | D3: gán sai số, Spearman, AUSE, random baseline, `run_d3.py` | **D3 chạy end-to-end** |
| W8 | 16/11–22/11 | D3 trên tất cả signal; bảng ghép D1–D2–D3 (RQ4) | Gate 5 |
| W9+ | 23/11– | Robustness theo số view; viết Related work + Method | — |

## 5. Tiêu chí hoàn thành

- [ ] Estimator Fisher khớp Jacobian chính xác trên cảnh nhỏ (có test).
- [ ] `fisher`, `pup`, `curvature`, `s_geom` đúng hợp đồng output, có test.
- [ ] Ghi rõ chỗ khác paper gốc (FisherRF, PUP).
- [ ] D3 chạy một lệnh, ra `d3_metrics.csv` + hình, có random baseline.
- [ ] Bảng ghép D1–D2–D3.
- [ ] Tài liệu research + lệnh tái lập.

Quy trình git, commit, Definition of Done, mẫu báo cáo: như 3 bạn còn lại (`CONTRIBUTING.md`).
