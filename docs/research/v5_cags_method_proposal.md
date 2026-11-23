# Đề xuất phương pháp CAGS: phát hiện và kìm hãm floater do bất nhất trắc quang trong quá trình huấn luyện 3DGS

*Tên tạm: CAGS (Compensation-Aware Gaussian Suppression). Cập nhật ngày 19/11/2026. Tài liệu nội bộ của nhóm FaithfulRecon (HCMUT, DACN).*

Tài liệu gồm bốn phần:
- tóm tắt các kết quả đã có, kèm mức độ tin cậy;
- đặc tả phương pháp ở dạng chốt phạm vi;
- cách hiện thực trong mã nguồn;
- giao thức thực nghiệm, cùng tiêu chí để dừng hoặc chuyển hướng.

Mọi số liệu đều đo trên DTU với 8 view huấn luyện, 10k vòng lặp và khởi tạo từ SfM. Chỉ số chính là **ΔW_cov**: phần tăng của tỉ lệ trọng số kết xuất do các Gaussian có khoảng cách d > 0,1 tới bề mặt tham chiếu, chỉ tính trên vùng có tham chiếu, so với mức không nhiễu.

---

## 1. Tóm tắt kết quả hiện tại

### 1.1 Kết quả đã vững (đã đo trên 15 scan hoặc lặp lại nhất quán)

| # | Kết quả | Số liệu chính | Phạm vi |
|---|---|---|---|
| R1 | Bất nhất trắc quang giữa các view làm tăng floater | ΔW_cov run gốc: P-A 0,084; P-AC 0,043; P-C 0,044; P-B 0,024. W tăng ở 30/30, 29/30, 30/30 và 25/30 cặp (scan, seed) | 15 scan × 2 seed |
| R2 | Kèm theo đó là suy giảm hình học và ảnh | F-score@0,1: 0,70 khi không nhiễu, 0,51 với P-A, 0,60 với P-C; PSNR: 23,3 → 18,6 dB (P-A) và 20,0 dB (P-C) | 15 scan × 2 seed |
| R3 | Gaussian bù trừ hình thành sớm | Oracle bù trừ bật từ vòng 100 giảm ΔW 63%; từ vòng 250 giảm 46%; từ vòng 500 chỉ còn 17% | 2 scan, mức P-C |
| R4 | Bộ tối ưu "lách" mọi tiêu chí cứng | Oracle hình học xoá được 100% W(0,1) nhưng chỉ 14% W(0,05); với nhãn d/σ_n, phần bù chuyển sang các Gaussian rộng | 2 scan |
| R5 | Bề mặt STL của DTU bị khuyết (scan63: 13,4%; scan83: 9,1% diện tích vật thể) | W tuyệt đối bị thổi phồng, nhưng ΔW ≈ ΔW_cov (0,076 so với 0,076; 0,032 so với 0,034) | 2 scan |
| R6 | Oracle hình học không dùng được làm trần trên | Xoá cả bề mặt thật ở vùng STL khuyết; PSNR giảm từ 26,5 xuống 23,7 dB | 2 scan |
| R7 | Có thể học được việc phát hiện Gaussian ngoài bề mặt từ trạng thái huấn luyện | AUROC 0,925, AP 0,59 (AGPD); heuristic tốt nhất AP 0,46; chỉ dùng đặc trưng thuộc tính: AP 0,39 | 15 scan, 5 fold leave-scan-out |
| R8 | Thông tin của detector chủ yếu đến từ phân bố hỗ trợ đa view (nhóm G) | Thêm nhóm G: AP 0,39 → 0,58; thêm nhóm P (phần dư) và D (động học): +0,01–0,02, nằm trong mức nhiễu giữa các fold | 15 scan |

### 1.2 Kết quả chưa được ủng hộ hoặc chưa kết luận được

| # | Nội dung | Tình trạng |
|---|---|---|
| N1 | H5: đặc trưng trắc quang (P) và động học (D) bổ sung thông tin | **Không được ủng hộ.** AP_joint của AGPD so với AG chênh 0,002 |
| N2 | Nhận diện riêng "Gaussian bù trừ" (nhãn joint) | **Yếu.** AP_joint ≈ 0,12 (heuristic 0,01–0,02). Muốn precision 0,8 thì recall chỉ còn 0,4–2,3% |
| N3 | Can thiệp trong huấn luyện bằng detector (pilot D đầu tiên) | **Không đo được.** Rủi ro chưa hiệu chỉnh nên chỉ ≤ 0,02% Gaussian bị tác động; mọi chênh lệch đo được là nhiễu |
| N4 | Oracle bù trừ trong cấu hình D (đủ ba hành động) | Giảm ΔW ở 3/4 cặp (scan, mức), tăng mạnh ở scan83 mức P-AC. Cần thêm seed |
| N5 | GC-ILA (mô hình ngoại quan) kết hợp CAGS | Chưa đo được, vì CAGS gần như không tác động trong pilot đầu |

### 1.3 Hai lưu ý phương pháp luận

1. **Dao động giữa các seed rất lớn.** ΔW_cov của run gốc ở scan63, mức P-C là 0,025 với seed 0 và 0,083 với seed 1. Mọi kết luận về can thiệp cần ít nhất 2 seed, so sánh theo cặp (scan, seed) và kiểm định Wilcoxon.
2. **Quá trình huấn luyện tất định theo seed nhưng rất nhạy với nhiễu loạn nhỏ.** Khi CAGS không tác động, kết quả trùng run gốc tới 3 chữ số. Chỉ cần khoảng 10 Gaussian bị tác động từ vòng 2000 là quỹ đạo đã khác hẳn. Hiệu ứng của phương pháp vì vậy phải được đo trên tập lớn, không đọc từ một vài run.

### 1.4 Vì sao hướng đi có vẻ "ngày càng xa"

Ba vấn đề khác nhau đã bị trộn lẫn:
- Sai số của **thước đo**: STL khuyết, phép so sánh khác tập scan.
- Giả thuyết về **tín hiệu**: đặc trưng trắc quang (N1, N2) không đứng vững, nhưng tín hiệu hình học thì vững (R7, R8).
- **Cơ chế can thiệp** (N3): thực tế chưa được thử nghiệm đúng nghĩa.

Phần 2 chốt phạm vi: giữ lại những gì đã vững, chỉ còn một câu hỏi mở cần trả lời, và đặt sẵn tiêu chí dừng ở mục 4.

---

## 2. Phương pháp đề xuất (phạm vi chốt)

### 2.1 Ý tưởng

Floater do bất nhất trắc quang là các Gaussian **chỉ phục vụ một số ít view** để tái tạo phần màu sai lệch của các view đó. Chúng hình thành **sớm** (R3), và bộ tối ưu sẽ tìm đường vòng quanh mọi luật xoá cứng (R4).

Từ đó, phương pháp gồm ba phần:
1. **Mô hình ngoại quan theo view** hấp thụ phần sai lệch toàn cục, để Gaussian không phải gánh việc này.
2. **Detector học được** ước lượng rủi ro "ngoài bề mặt" của từng Gaussian, dựa trên cách các view nhìn thấy nó (nhóm G là chủ đạo). Detector được huấn luyện ngoại tuyến trên trạng thái huấn luyện có nhãn GT.
3. **Can thiệp mềm, sớm và liên tục** theo rủi ro: hạn chế độ mờ, giảm biên độ cập nhật, chặn nhân bản. Không xoá cứng, để bộ tối ưu không có lý do tìm đường vòng.

### 2.2 Sơ đồ

```
 ảnh huấn luyện I_j (có bất nhất trắc quang)
        │
        ▼
 ┌──────────────────────┐   loss L1/SSIM trên  g_j(Î_j)  (GC-ILA: biến đổi màu theo view, miền log)
 │  3DGS + GC-ILA       │◄──────────────────────────────────────────────────────────────┐
 └──────────┬───────────┘                                                               │
            │ mỗi K = 100 vòng, từ T_ON = 100                                           │
            ▼                                                                            │
 ┌──────────────────────┐   1 lần kết xuất / view, kênh probe → w_ij (đóng góp của      │
 │  Probe đa view       │   Gaussian i cho view j), phần dư, độ sâu, xung đột gradient  │
 └──────────┬───────────┘                                                               │
            ▼                                                                            │
 ┌──────────────────────┐   31 đặc trưng (A thuộc tính, G hỗ trợ đa view, P trắc quang, │
 │  Detector f_θ (MLP)  │   D động học) → p_geo, δ̂  → rủi ro q_i → hiệu chỉnh → EMA r_i  │
 └──────────┬───────────┘                                                               │
            ▼                                                                            │
 ┌──────────────────────┐   chặn clone/split nếu r_i > 0,5; giới hạn độ mờ;             │
 │  Can thiệp mềm       │   điều biến cập nhật Adam; ngân sách ≤ 20% số Gaussian ───────┘
 └──────────────────────┘
```

### 2.3 Mô hình ngoại quan (GC-ILA)

Mỗi view huấn luyện j có một biến đổi màu theo kênh trong miền log (`appearance='gc_log'`). Biến đổi này được áp lên ảnh kết xuất trước khi tính loss, và loại các điểm ảnh bão hòa (`sat_mask`). Khi kết xuất view kiểm định, không áp biến đổi nào. GC-ILA mô hình hoá chính xác nhiễu P-A (exposure, white balance, gamma đều là biến đổi log-affine), còn P-C (vệt sáng cục bộ) thì chỉ được mô hình hoá một phần.

**Vai trò trong CAGS.** Pilot cho thấy oracle bù trừ khi dùng *không có* GC ở mức P-AC có thể làm ΔW tăng (N4). Giả thuyết: khi còn sai lệch toàn cục, gần như mọi Gaussian đều phải bù, nên kìm một nhóm thì nhóm khác mọc ra thay. Do đó **phương pháp chính là GC-ILA + CAGS**, và trần trên tương ứng là GC-ILA + oracle bù trừ.

### 2.4 Probe đa view

Mỗi lần probe thực hiện V = 8 lần kết xuất (một lần cho mỗi view huấn luyện). Mỗi lần kết xuất có thêm 6 kênh đặc trưng "probe" được khởi tạo bằng 0 và có `requires_grad`. Với ảnh kết xuất R_k(p) của kênh k và một bản đồ trọng số c_k(p) tuỳ chọn, ta có:

$$\frac{\partial}{\partial z_{ik}} \sum_p c_k(p)\, R_k(p) = \sum_p c_k(p)\, T_i(p)\,\alpha_i(p),$$

tức là tổng của c_k trên footprint của Gaussian i, có trọng số theo đóng góp blending. Như vậy chỉ một lần backward là có đủ các đại lượng sau:
- tổng đóng góp w_ij (với c = 1);
- phần dư tuyệt đối và phần dư có dấu (với c = |r| và c = log-ratio);
- log φ của footprint (với c = log φ_j, chỉ dùng cho nhãn).

Việc này đã được kiểm tra: kết quả khớp chính xác với đóng góp blending tính trực tiếp.

### 2.5 Đặc trưng (31 chiều)

| Nhóm | Đặc trưng | Vai trò (theo kết quả 7b) |
|---|---|---|
| A – thuộc tính (10) | opacity, ls_max, ls_min, aniso, màu RGB, năng lượng SH, khoảng cách tới tâm, tỉ lệ độ sâu | nền; một mình đạt AP 0,39 |
| G – hỗ trợ đa view (9) | n_vis, entropy và max_share của w_i·, front_frac, log tổng đóng góp; trung bình k-NN của khoảng cách, opacity, front, share | **tín hiệu chính** (AP 0,58) |
| P – trắc quang (5) | \|phần dư\|, độ lệch chuẩn phần dư có dấu theo view, xung đột gradient DC/SH giữa các view, xung đột k-NN | bổ sung không đáng kể |
| D – động học (7) | tuổi, thế hệ, Δopacity, Δscale, gradient tích luỹ, rủi ro của cha, thời điểm chuẩn hoá | bổ sung không đáng kể |

Đặc trưng không bị chặn được chuẩn hoá theo thứ hạng trong từng cảnh, để bất biến với tỉ lệ cảnh và số Gaussian.

### 2.6 Nhãn (chỉ dùng khi huấn luyện detector)

- **Nhãn ngoài bề mặt (LABEL_V2):**
  - y_geo = 1 nếu d > 0,1;
  - y_geo = 0 nếu d < 0,05, hoặc nếu d ≤ 0,1 và d/σ_n < 1;
  - các trường hợp còn lại là mơ hồ (−1) và bị loại khỏi loss.
  - Trong đó d là khoảng cách tới STL, σ_n là độ lệch chuẩn của Gaussian theo pháp tuyến PCA của STL.
- **Mức bù trừ:** δ_i = ‖Σ_j ŵ_ij · log φ_j(footprint) − log φ̄_i‖, trong đó φ_j là trường sai lệch trắc quang đã biết của view j, và φ̄_i là trung bình của φ_j tại tâm Gaussian trên các view nhìn thấy nó. τ_δ là phân vị 90 của δ trên các Gaussian nằm trên bề mặt. y_joint = 1 khi y_geo = 1 và δ > τ_δ.

### 2.7 Detector

- **Kiến trúc:** MLP gồm 3 lớp, mỗi lớp 128 chiều với LayerNorm và GELU, và có hai đầu ra:
  - logit p_geo;
  - δ̂, tính theo đơn vị τ_δ.
- **Loss:**
  - focal BCE (γ = 2) cho p_geo, cân bằng lớp, có trọng số theo tần suất lấy mẫu × log(1 + đóng góp);
  - Huber cho δ̂, với hệ số 0,5.
- **Dữ liệu:**
  - trạng thái được ghi ở 10 mốc từ vòng 100 đến 10000;
  - 15 scan × 5 mức nhiễu × 2 seed, tổng khoảng 14 triệu dòng;
  - lấy mẫu phân tầng, tối đa 5000 dòng mỗi mốc.
- **Đánh giá:** 5 fold leave-scan-out. Mỗi run dùng detector của fold không chứa scan của nó.
- **Rủi ro:** có hai lựa chọn, sẽ được quyết định bằng pilot 2 (mục 4):
  - *hình học*: q_i = p_geo;
  - *joint*: q_i = p_geo · σ((δ̂ − 1)/0,25).

### 2.8 Hiệu chỉnh ngưỡng (thêm sau pilot đầu)

Đầu ra của detector không được hiệu chỉnh, nên rủi ro trung bình chỉ khoảng 0,02. Vì vậy τ được chọn trên dữ liệu huấn luyện của fold:
- *rủi ro hình học*: τ là ngưỡng thấp nhất mà ở đó precision (ngoài bề mặt) ≥ 0,8. Kết quả: τ ≈ 0,68–0,76, recall 0,59–0,86, gắn cờ 5–11% số Gaussian.
- *rủi ro joint*: τ được chọn sao cho gắn cờ 1% số Gaussian, ngang với oracle bù trừ.

Khi chạy, rủi ro được chuyển thành q̃ = min(1, 0,5·q/τ). Nhờ vậy ngưỡng τ_s = 0,5 tương ứng với q = τ.

### 2.9 Trạng thái rủi ro và kế thừa

- r_i ← 0,5·r_i + 0,5·q̃_i ở mỗi lần probe.
- Gaussian con sinh ra do clone/split thừa hưởng r và `parent_risk` từ cha, nên floater không thể "trốn" bằng cách tách đôi.
- Trạng thái phụ trợ (thời điểm sinh, thế hệ, rủi ro…) được đồng bộ khi thêm hoặc xoá Gaussian và được lưu trong checkpoint.

### 2.10 Can thiệp

Cường độ can thiệp là m_i = clip((r_i − τ_s)/(1 − τ_s), 0, 1) với τ_s = 0,5. Chỉ những Gaussian nằm trong top 20% về rủi ro mới có m_i > 0.

| Hành động | Công thức | Mục đích |
|---|---|---|
| gate | không clone/split Gaussian có r_i > 0,5 | ngăn floater sinh sôi |
| att | logit độ mờ bị giới hạn ≤ logit((1 − m_i)^γ), với γ = 2, áp sau mỗi bước tối ưu | giảm ảnh hưởng một cách mềm, có thể hồi phục |
| mod | θ ← θ_old + (1 − m_i)(θ_Adam − θ_old) cho màu, SH và scale | làm chậm việc Gaussian học cách bù màu theo view |

Lịch can thiệp: bắt đầu từ T_ON = 100 (theo R3), probe mỗi K = 100 vòng, và luôn có một lần probe ngay tại T_ON.

### 2.11 Chi phí

Mỗi lần probe tốn khoảng 8 lần kết xuất cộng 8 lần backward, cộng một lần chạy MLP trên N ≤ 300k Gaussian. Với K = 100, tổng cộng có khoảng 100 lần probe trong 10k vòng, tức khoảng 8% số lần kết xuất của quá trình huấn luyện. Ablation `cags_k500` dùng K = 500.

---

## 3. Hiện thực trong mã nguồn

### 3.1 Tệp

| Tệp | Nội dung |
|---|---|
| `frecon/cags.py` | `FEATS` và `GROUPS`; trạng thái phụ trợ (`ensure_aux`, `on_append`, `on_prune`); nhãn (`_gt`, `gt_labels`); `probe`; `Detector`; `joint_risk`; `load_detector` (đọc thêm `tau`); `_strength`; các hook `gate`, `pre_step`, `post_step`, `tick`; ghi log (`_log`, `write_log` → `cags_log.npz`, `cags_hist.csv`) |
| `scripts/cags_train.py` | đọc log và tính lại nhãn LABEL_V2; `train_one` (focal + Huber); `calibrate` (theo precision hoặc tỉ lệ gắn cờ); `scores` (AUROC, AP, AP_joint, Spearman δ); baseline heuristic; `--calibrate_only` |
| `scripts/fm3.py` | đánh giá: W_float, PSNR, Chamfer/F-score; `covered_wfloat` (W_cov và `ref_cover`) |
| `notebooks/colab_v4_v5_cags.ipynb` | toàn bộ pipeline trên Colab: vá mã (PATCH 1–8), sinh cấu hình, hàng đợi huấn luyện song song, đánh giá, phân tích và trực quan hoá |

### 3.2 Các hook trong `train.py` (PATCH 8)

| Vị trí | Hook |
|---|---|
| khởi tạo Trainer | lưu danh sách camera huấn luyện (`_cams`) cho probe |
| trước / sau `optimizer.step()` | `pre_step` lưu giá trị cũ của các tham số được điều biến; `post_step` trộn lại theo (1 − m) và áp giới hạn độ mờ |
| sau bước tối ưu và densification | `tick`: probe khi đến hạn, ghi log, cập nhật rủi ro |
| densification | `gate` trả về mặt nạ Gaussian bị cấm clone/split; ghi nhận chỉ số cha để kế thừa trạng thái |
| thêm / xoá Gaussian | `on_append`, `on_prune` giữ trạng thái phụ trợ đồng bộ |
| lưu / nạp checkpoint | lưu và nạp `_aux` |

### 3.3 Cấu hình (`TrainConfig`)

`cags_mode` (none, log, oracle_geo, oracle_cmp, det, rule), `cags_actions` (full, gate, att, mod), `cags_from` = 100, `cags_every` = 100, `cags_ema` = 0,5, `cags_tau_s` = 0,5, `cags_tau_gate` = 0,5, `cags_gamma` = 2, `cags_budget` = 0,2, `cags_model` (đường dẫn detector), `cags_log_steps`, `cags_log_max`.

### 3.4 Quy trình chạy (notebook)

1. **A:** huấn luyện run gốc ở chế độ `log` trên 15 scan × 5 mức × 2 seed để ghi đặc trưng và nhãn.
2. **B:** chạy oracle trên 2 scan để chốt T_ON và bộ hành động.
3. **C:** huấn luyện detector 5 fold; chạy **5C-cal** để hiệu chỉnh τ.
4. **D:** chạy CAGS trên 15 scan × 3 mức (none, P-C, P-AC) × 2 seed; chạy **pilot 2 scan trước**.
5. **E:** ablation trên 4 scan.
6. **F:** DAgger, tức huấn luyện lại detector trên trạng thái của chính các run CAGS.
7. Đánh giá bằng cell 6; phân tích bằng các cell 7a–7d; trực quan hoá bằng 7v.

---

## 4. Giao thức thực nghiệm và tiêu chí quyết định

### 4.1 Câu hỏi còn mở (duy nhất)

> Khi GC-ILA đã hấp thụ phần sai lệch toàn cục, can thiệp mềm và sớm theo rủi ro của detector có giảm ΔW_cov nhiều hơn mức dao động giữa các seed mà không làm giảm chất lượng ở mức không nhiễu hay không?

### 4.2 Pilot 2 (đang chạy)

- **Phạm vi:** scan63 và scan83 × {none, P-C, P-AC} × seed {0, 1}.
- **Biến thể:** gc, cags, gccags (rủi ro joint), cags_nocmp, gccags_nocmp (rủi ro hình học), orcmp, gcorcmp.
- **Kiểm tra cơ chế** (7c-bis):
  - tỉ lệ Gaussian bị tác động phải đạt khoảng 1–10%;
  - precision so với nhãn GT phải ≥ 0,6.
- **So sánh:** gccags* so với gc, theo từng cặp (scan, seed).

### 4.3 Cây quyết định

| Kết quả pilot 2 | Bước tiếp theo |
|---|---|
| gccags_nocmp hoặc gccags giảm ΔW_cov so với gc ở ≥ 6/8 cặp (P-C, P-AC), và PSNR mức none giảm ≤ 0,3 dB | Chạy D đầy đủ với biến thể thắng (15 scan × 2 seed), kiểm định Wilcoxon theo cặp; sau đó chạy E và F |
| Chỉ biến thể hình học thắng | Trình bày đóng góp là "detector học được, huấn luyện trên dữ liệu có nhiễu trắc quang, can thiệp sớm"; ghi nhận trung thực rằng tín hiệu bù trừ trắc quang (N1, N2) chưa khai thác được |
| Có tác động (đúng tỉ lệ, precision cao) nhưng ΔW không giảm hoặc tăng | Đây là hiện tượng lách tiêu chí (R4). Thử hai cách trong một lượt nữa: (a) chỉ dùng `mod` + `gate`, bỏ `att`; (b) kéo dài chặn nhân bản sang toàn bộ dòng con cháu. Nếu vẫn không giảm thì dừng phần can thiệp |
| gcorcmp cũng không tốt hơn gc | Trên nền GC-ILA, phần bù trừ còn lại không thể khắc phục bằng can thiệp lên từng Gaussian. Dừng phần can thiệp |

### 4.4 Định vị bài báo theo từng nhánh

- **Nhánh thuận:**
  - C1: phân tích cơ chế (R1–R6, gồm thời điểm hình thành, hiện tượng lách tiêu chí và lưu ý về thước đo trên DTU);
  - C2: detector học được dựa trên hỗ trợ đa view (R7, R8);
  - C3: can thiệp trong huấn luyện, có kiểm định thống kê trên 15 scan.
- **Nhánh dừng can thiệp:** giữ C1 và C2. C3 được trình bày như một kết quả âm có phân tích: oracle cho thấy giới hạn trên, và hiện tượng lách tiêu chí giải thích vì sao can thiệp lên từng Gaussian khó thành công. Hướng này phù hợp với dạng bài phân tích hơn là bài phương pháp.

### 4.5 Tiêu chí báo cáo

- Chỉ số chính là ΔW_cov. Các chỉ số phụ gồm F-score@0,1, Chamfer và PSNR (ở mức none và ở mức có nhiễu).
- Báo cáo trung vị theo scan, kèm số scan cải thiện trên tổng số.
- Kiểm định Wilcoxon một phía theo cặp (scan, seed), với n ≥ 6.
- Không dùng các từ "chứng minh" hay "đầu tiên". Kết quả trên 2 scan chỉ được gọi là "sơ bộ".

---

## 5. Hạn chế và rủi ro đã biết

1. **Nhãn phụ thuộc STL bị khuyết.** Bề mặt thật ở vùng khuyết bị gán nhãn "ngoài bề mặt", nên detector có thể học kìm cả bề mặt thật. Cần theo dõi PSNR ở mức none và cột "rủi ro CAGS" trong hình 7v.
2. **Precision hiệu chỉnh trên dữ liệu huấn luyện** cao hơn thực tế. Precision thật được đo qua 7c-bis trên run kiểm định.
3. **Phân phối dịch chuyển.** Detector được huấn luyện trên trạng thái của run gốc nhưng lại được dùng trên run đang bị can thiệp. DAgger (phần F) xử lý vấn đề này.
4. **Phạm vi dữ liệu.** Hiện chỉ có DTU (vật thể, nền đã che) và nhiễu tổng hợp. Khả năng tổng quát hoá sang ảnh chụp thật có sai lệch trắc quang tự nhiên chưa được kiểm tra.

## 6. Hướng mở rộng (chỉ làm khi nhánh thuận thành công)

- **Đặc trưng P' không cần GT.** Ước lượng trường phơi sáng cục bộ của từng view từ phần dư log đã làm mượt, rồi tính δ̂ theo đúng định nghĩa của δ. Mục tiêu là khai thác lại tín hiệu bù trừ trắc quang. Việc này cần ghi log lại phần A.
- **Nhãn có xét vùng phủ.** Gán −1 (mơ hồ) cho các Gaussian mà footprint chủ yếu rơi vào vùng vật thể không có tham chiếu.
- **Mở rộng dữ liệu.** Thêm một bộ dữ liệu có sai lệch trắc quang tự nhiên, ví dụ ảnh chụp ngoài trời với phơi sáng tự động.
