# Reliability signals trong FaithfulRecon — Quỳnh Anh

Ngày cập nhật: 16/10/2026. Phạm vi: static 3DGS, geometric failure do thiếu quan sát đa góc nhìn.

Code: `frecon/signals/`; exporter: `scripts/export_signals.py`; kiểm tra: `tests/test_signals.py`, `tests/test_signal_export.py`.
Tất cả score có shape `(N,)`, dtype `float32`, hữu hạn, theo thứ tự checkpoint. **Score cao = kém tin cậy hơn** là quy ước ranking, chưa phải kết luận đã được kiểm chứng.

## 0. Hợp đồng và phạm vi bằng chứng

- `gaussian_id = i` là hàng i trong checkpoint; không sort lại mảng score trước khi bàn giao.
- Opacity chỉ cần checkpoint. Ba signal còn lại cần model, renderer, background, training cameras và ảnh training tương ứng.
- Loss: `L_c = (1-lambda) L1 + lambda (1-SSIM)`, mặc định lambda=0.2, lấy từ train config khi export.
- `C_ic = sum_p T_icp * a_icp` là contribution của Gaussian i lên view c ở torch backend. `a_icp` gồm opacity nhân footprint; T là transmittance của các splat phía trước.
- `V_i = {c thuộc training views: C_ic > 1e-3}`. Contribution được tính lại tại checkpoint, không dùng held-out views để tạo score.
- Renderer `gsplat` hiện xấp xỉ contribution bằng radius² × opacity; không tương đương torch backend. Run báo cáo này dùng torch.
- Export Top-K dùng score giảm dần, exact tie theo Gaussian ID tăng dần; K=`max(1, round(frac*N))`. Quy tắc `torch.topk` của intervention Phase 0 được giữ nguyên. Khi tie ở cutoff, exporter có thể chọn tập khác Phase 0.
- Chỉ so Gaussian ID giữa các artifact cùng checkpoint hash. Các seed train độc lập không có cùng primitive identity.

## 1. Opacity

### 1. Signal definition

Opacity là tham số biểu diễn độ opaque được học của mỗi Gaussian. Signal hiện tại chọn các Gaussian có opacity nhỏ trước. Đây là heuristic về mức đóng góp tiềm năng, không phải xác suất Gaussian sai hình học.

### 2. Mathematical formulation

`alpha_i = sigmoid(opacity_logit_i)`; `score_i = 1-alpha_i`.

Opacity được kích hoạt trong (0,1) về mặt toán học; float32 có thể bão hòa ở biên. Khi render, alpha ở pixel còn nhân Gaussian footprint và chịu clamp/threshold. Vì vậy alpha_i không bằng tổng contribution, và không thể suy visibility chỉ từ alpha_i.

### 3. Implementation location in 3DGS

`GaussianModel.opacities` trong `frecon/gaussians.py`; `opacity.compute()` đọc activated opacity rồi detach. `Trainer.reset_opacity()` giới hạn opacity về tối đa 0.01; `densify_and_prune()` loại Gaussian dưới ngưỡng. Opacity ở checkpoint phụ thuộc lịch reset, pruning, số bước train và tối ưu tương quan với scale/color.

3DGS gốc cũng dùng sigmoid cho opacity, prune primitive gần trong suốt và reset opacity định kỳ [R1, mục 5]. Việc chúng là ứng viên pruning không tự chứng minh chúng là primitive lỗi hình học.

### 4. Input / output

Input tối thiểu: opacity logits của N Gaussian, cùng thứ tự checkpoint. Không cần camera hoặc backward. Output: N giá trị float32, hữu hạn. Độ phức tạp O(N), không normalize theo scene.

### 5. Expected interpretation

Opacity nhỏ có thể xuất hiện ở primitive ít được dữ liệu hỗ trợ hoặc dư thừa. Tuy nhiên, hướng dự đoán trực tiếp nhất là **xóa chúng ít ảnh hưởng rendering**. Kỳ vọng xóa chúng cải thiện geometric error cần kiểm chứng độc lập.

### 6. Potential confounders

- Reset opacity hoặc pruning gần thời điểm save có thể thay đổi phân phối mạnh.
- Occlusion, footprint, texture và redundancy làm opacity khác contribution thực.
- Gaussian sai hình học vẫn có thể opaque để giải thích ảnh training.
- Nhãn floater Phase 0 yêu cầu opacity > 0.05; primitive opacity thấp bị loại khỏi positive label ngay bởi định nghĩa. AUROC thấp có liên hệ với gate này.

### 7. Limitations

Không quan sát trực tiếp geometry, coverage hoặc uncertainty. Signal chọn nhiều primitive gần trong suốt có thể cho response D2 ở noise floor. Khi response chưa measurable, locality/direction gain không đủ để kết luận.

### 8. Relation to FaithfulRecon

Là baseline trong D1 và D2. Phase 0 có SNR_in 1.10 ± 0.05, chưa measurable. Trong run tính lại, precision@5% với nhãn tạm bằng 0 ở cả ba seed; cần đọc cùng opacity gate của nhãn. Phải tách câu hỏi “xóa ít gây hại” khỏi “xóa sửa lỗi”.

### 9. References

[R1], code `frecon/signals/opacity.py`, `frecon/gaussians.py`, `frecon/train.py`; nhãn trong `frecon/measure.py::floater_labels`.

## 2. Gradient magnitude

### 1. Signal definition

Độ lớn gradient photometric loss theo vị trí chiếu 2D của Gaussian. Đo mức nhạy bậc nhất tại checkpoint đối với dịch chuyển screen-space, lấy trung bình trên các view có đóng góp.

### 2. Mathematical formulation

Với mean chiếu bằng pixel `u_ic=(u,v)`:

`h_ic = (W_c/2 * dL_c/du, H_c/2 * dL_c/dv)`.

Đây là gradient theo tọa độ NDC bởi chain rule. `score_i = sum_{c in V_i} ||h_ic||_2 / |V_i|`. Nếu không có view hợp lệ, score=0; count được export để phân biệt thiếu bằng chứng với gradient thấp thật.

Đây là **mean of norms**, không phải norm of mean; các gradient đối nhau không triệt tiêu trước khi lấy norm.

### 3. Implementation location in 3DGS

`frecon/signals/common.py::view_statistics`, gọi bởi `gradmag.compute`. `autograd.grad(loss, means2d)` lấy đạo hàm, không cập nhật checkpoint hoặc ghi đè `.grad` của parameter.

`Trainer.train_step` có cùng phép đổi pixel→NDC, nhưng dùng `out['visible']` (footprint/frustum) thay vì contribution threshold, tích lũy dọc training khi parameters thay đổi và reset sau densification. Signal hiện tại là **statistic hậu kiểm có liên hệ với densification**, không phải bản sao chính xác accumulator lịch sử.

3DGS gốc dùng average view-space positional gradient để quyết định clone/split, với cả vùng thiếu biểu diễn lẫn Gaussian phủ quá lớn [R1, mục 5.2].

### 4. Input / output

Checkpoint, training cameras với ảnh, renderer, background, contribution threshold và loss weight. Một render/backward theo từng training view; output N float32. Opacity và geometry được giữ nguyên; không optimizer step. Statistics chia sẻ với visibility trong cùng SignalContext; cache được phân biệt theo threshold/loss.

### 5. Expected interpretation

Score lớn cho thấy loss đang tạo áp lực dịch chuyển primitive trên ảnh. Có thể liên quan misplaced geometry hoặc vùng biểu diễn chưa đủ. Tuy nhiên, primitive hợp lệ trên cạnh, cấu trúc mảnh và vùng texture cũng có thể cần thiết và nhạy mạnh. Densification utility không đồng nhất removal utility.

### 6. Potential confounders

Opacity, projected footprint, contrast/texture, silhouette, background, loss reduction, resolution và số view hợp lệ. Chuyển sang NDC không làm gradient bất biến hoàn toàn với resolution. Mean/max/sum giữa views trả lời các câu hỏi khác nhau; đổi aggregation là ablation riêng.

### 7. Limitations

Không đo trực tiếp khoảng cách đến reference surface hoặc độ bất định. Near-zero gradient có thể do hội tụ, bị che hoặc không đóng góp. Kết quả compute CPU và score gốc có vài sai khác giá trị gradient mặc dù ranking gần như giữ nguyên; chưa xác minh được nguồn của từng sai khác số học.

### 8. Relation to FaithfulRecon

Phase 0: SNR_in 4.71 ± 0.38, gain locality 3D +0.0943 ± 0.0187; gain direction local Chamfer -0.00013 ± 0.00007. AUROC khoảng 0.654 và precision@5% 0.108 so random 0.083 cho thấy có failure association, nhưng removal vẫn gây hại geometry hơn random.

Diễn giải: primitive có ảnh hưởng và có association với failure chưa chắc là tập nên xóa. Giả thuyết mixed selection giữa primitive lỗi và cấu trúc hữu ích cần kiểm tra qua visualization, accuracy/completeness và structure labels.

### 9. References

[R1, mục 5.2]; code `gradmag.py`, `common.py`, `train.py`; `results/phase0/report.md`.

## 3. Visibility

### 1. Signal definition

Số training view mà Gaussian có tổng contribution vượt threshold. Đây là effective rendering support của trained model; không đồng nhất số camera có line-of-sight đến reference surface.

### 2. Mathematical formulation

`n_i = sum_c 1[C_ic > eps_contrib]`.

`score_i = -n_i + eta_i`, với `eta_i = 1e-3 * Uniform[0,1)`.

Số view ít hơn luôn xếp trước; jitter chỉ phá tie giữa các integer count. Jitter dùng generator chung có seed trong SignalContext, giữ tương thích hành vi cũ. Thứ tự gọi random/visibility có thể làm thay đổi tie-break. Metadata ghi signal order.

### 3. Implementation location in 3DGS

Torch rasterizer tích lũy `contrib` từ blending weights đã xét occlusion. `common.view_statistics` đếm view và `visibility.compute` đảo dấu, thêm jitter. Không dùng đơn thuần radius>0 như accumulator training.

Backend gsplat dùng proxy footprint×opacity nên semantic khác; không trộn backend mà vẫn gọi đây là cùng phép đo visibility.

### 4. Input / output

Cùng context với GradMag. Hiện implementation dùng chung pass có gradient với GradMag, dù riêng visibility về bản chất chỉ cần forward. Output N float32; lưu count nguyên riêng trong diagnostics. Không exclude Gaussian count=0 khỏi baseline.

### 5. Expected interpretation

Ít view hỗ trợ có thể tạo mơ hồ độ sâu và geometric failure. Tuy nhiên n view gần nhau không có cùng ràng buộc hình học như n view có baseline/góc triangulation lớn. Độ phủ view là điều kiện có thể gây failure, không phải failure đã xảy ra.

### 6. Potential confounders

Opacity thấp, occlusion, footprint nhỏ, resolution/threshold và camera layout đều ảnh hưởng count. Count=0 có thể tạo selection gần vô tác dụng trong D2. Ties nhiều làm selection phụ thuộc tie seed. Trong dữ liệu này, khoảng 31% Top 5% visibility có count=0 khi tính lại.

Condition label m_i có liên quan coverage nhưng được định nghĩa từ reference geometry và cameras độc lập trained model. Cần ghi rõ sự gần nhau về construct để tránh overclaim từ D1; hiện m_i chưa có trong dữ liệu.

### 7. Limitations

Không mô tả góc quan sát hoặc mức đóng góp trong mỗi view sau khi vượt threshold. Integer count có độ phân giải thấp; jitter không phải bằng chứng hình học. Giữa score gốc và CPU recompute, Top5 overlap khoảng 82–86%; integer counts khớp toàn bộ seed0/1, seed2 lệch một Gaussian.

### 8. Relation to FaithfulRecon

Phase 0 SNR_in 1.77 ± 0.57; seed0 vượt 2 nhưng seed1/2 không vượt. Theo gate đã công bố, chưa diễn giải gain trung bình thành bằng chứng signal faithful. Có thể đề xuất contribution threshold sweep và tie-seed sweep; giữ riêng khỏi protocol chính.

### 9. References

[R1] cho alpha blending; định nghĩa count/jitter là heuristic của repo, không phải một uncertainty estimator từ paper. Code `visibility.py`, `common.py`, `render/torch_rast.py`, `render/gsplat_rast.py`.

## 4. Cross-view gradient inconsistency

### 1. Signal definition

Độ phân tán giữa các gradient loss theo vị trí 3D world của cùng Gaussian trên các training views có contribution. Đây là candidate hypothesis; chưa phải một phương pháp đã được chứng minh hoặc một estimate calibrated uncertainty.

### 2. Mathematical formulation

`g_ic = dL_c/dmu_i`, `gbar_i = mean_{c in V_i} g_ic`.

Population covariance trace: `var_i = mean ||g_ic - gbar_i||²`.

`score_i = var_i / (||gbar_i||² + delta_eff)`.

Baseline: delta=1e-12, n_min=2, delta_rel=0. Optional ablation: `delta_eff = delta + delta_rel * median_j ||gbar_j||²`, với j có ít nhất n_min view. Nếu ít hơn n_min view, score=0 và ghi count riêng. Zero có nghĩa thiếu bằng chứng bất đồng, không có nghĩa geometry tin cậy.

Dùng population covariance (chia n), không Bessel correction (n-1). Gradient bao gồm cả đường đạo hàm qua projected center lẫn projected covariance. Không so trực tiếp gradient 2D của các camera có hệ trục khác nhau.

### 3. Implementation location in 3DGS

`gradient_inconsistency.position_gradient_statistics` render từng training view, dùng `autograd.grad(loss, model.means)`. Welford streaming float64 lưu count, mean (N,3), M2 (N,); bộ nhớ O(N) theo số view. Variance âm do roundoff được clamp về 0. Score cuối float32; nonfinite gây lỗi rõ ràng thay vì lặng lẽ thay score.

Không gọi optimizer, không densify/prune, không thay checkpoint hoặc `.grad` có sẵn. Cache moments theo loss weight và contribution threshold; đổi delta dùng lại moments đúng.

### 4. Input / output

Giống GradMag, thêm delta, delta_rel và n_min. Output N score theo checkpoint. Diagnostics riêng gồm `gi_count`, `gi_mean_norm_sq`, `gi_variance_trace`. Metadata ghi effective delta, covariance convention và fallback.

### 5. Expected interpretation

Hai view tạo gradient đối nhau có mean nhỏ, variance lớn, score cao. Có thể phản ánh xung đột giữa các ràng buộc photometric và geometric ambiguity. Đây là lý do để thử D1/D2, không phải suy luận rằng Gaussian chắc chắn sai hoặc removal sẽ có ích.

**Công thức cũng đo biến thiên độ lớn**: gradient (1,0,0) và (3,0,0) cùng hướng vẫn cho variance dương. Nếu muốn đo direction-only, phải xây variant khác, định nghĩa xử lý norm gần 0 và chạy ablation riêng.

### 6. Potential confounders

- Mean gradient gần 0 làm score lớn; delta ảnh hưởng ranking và đơn vị world.
- Ít view cho covariance không ổn định; fallback có thể tạo association với coverage.
- Texture, loss, camera calibration, appearance mismatch, occlusion và opacity làm gradient khác nhau dù geometry đúng.
- Scale và orientation của hệ tọa độ: trace ratio bất biến với quay trực chuẩn (đã test); đổi world length scale cần đổi delta tương ứng.
- Gradient directions khác nhau có thể do projection geometry bình thường; cần matched/stratified diagnostics.

### 7. Limitations

Baseline delta cố định chưa được chọn bằng sensitivity study. Không phân biệt epistemic/aleatoric uncertainty [R6]; không phải Fisher information hoặc curvature [R4,R5]. Không xem score này là inverse Hessian hay influence function.

Ba seed có 666/800/638 Gaussian dưới hai contributing views. Fallback được báo riêng, không dùng nhãn ground truth để điền score. Các Gaussian này vẫn ở output đủ N.

### 8. Relation to FaithfulRecon

Đã tính được trên ba checkpoint: AUROC với f_i tạm 0.6508 ± 0.0250; precision@5% 0.0991 ± 0.0064. Chưa có D2 intervention của signal này, nên chưa kết luận faithfulness. Đây là thành phần v_i có thể bàn giao cho Phát để kết hợp s_geom; không tự sửa module của track khác.

Spearman với visibility count khoảng +0.40 và với opacity score khoảng -0.20 đến -0.24. Cần stratify theo coverage/opacity trước khi kết luận signal độc lập với các baseline. Khoảng 2.6–3.8% Top5 có mean-gradient-squared dưới delta.

### 9. References

Công thức là candidate trong thiết kế FaithfulRecon và module stub ban đầu. [R1] cung cấp differentiable 3DGS; [R4,R5,R6] cung cấp bối cảnh uncertainty/sensitivity, **không được trích như nguồn chứng minh chính xác công thức này**.

## 5. Kết quả, refinement và reproducibility

Phân tích từ dữ liệu thật: `results/signals/QuynhAnh_signal_audit.md`; các bảng per-seed, reproduction và confounder đi kèm. Báo cáo tách score tính lại khỏi metrics D2 gốc; không gán D2 cũ cho selection mới.

| Problem | Why it matters | Proposed check | Điều kiện thay đổi protocol |
|---|---|---|---|
| Nhiều selection gần không đóng góp | Response có thể ở noise floor | Diagnostic contribution, active-only variant và random trong cùng eligible pool | Trao đổi lead/D2 owner trước |
| Visibility nhiều tie | Top-K thay đổi dù count không đổi | Giữ checkpoint, sweep tie seed và signal order | Không đổi baseline jitter khi chưa trao đổi |
| GradMag ảnh hưởng opacity/footprint | Có thể chọn cấu trúc cần thiết | Stratify theo alpha, footprint, count; inspection thin structures | Chạy ablation riêng |
| GI mean gần 0 và ít view | Score lớn hoặc fallback làm lệch ranking | Delta sweep và n_min sweep, báo count/mean²/variance | Ghi tham số trước, không chọn theo test labels |
| Image locality mask rộng | Random cũng có LR cao | Báo mask area, đề xuất mask hẹp và 3D locality | Thuộc protocol D2, cần trao đổi |
| Chỉ đo sau reoptimization | Khó tách xóa trực tiếp khỏi thích nghi | Đề xuất đo ngay sau removal và sau T bước | Thuộc protocol D2, cần trao đổi |

Từ root repo, trong môi trường có dependencies:

```powershell
./scripts/reproduce_quynhanh.ps1 -Python 'D:/Programs/anaconda3/envs/chess-rl/python.exe'
```

Đường dẫn data mặc định là `../HKDN-261-20261016T025052Z-1-001/HKDN-261/data`. Override bằng `-DataRoot 'D:/.../data'`; chạy một seed bằng `-Seeds 0`.

Lệnh đơn cho recompute (seed0):

```powershell
python scripts/export_signals.py --run_dir '../HKDN-261-20261016T025052Z-1-001/HKDN-261/data/d1/synthetic/v8_az360/seed_0' --checkpoint '../HKDN-261-20261016T025052Z-1-001/HKDN-261/data/seed_0/base.pt' --source checkpoint --config results/phase0/config_used.yaml --signals random opacity gradmag visibility gradient_inconsistency --out_dir runs/signals/recomputed/synthetic/v8_az360/seed_0 --figure_dir runs/signals/figures/recomputed
```

Để export score gốc, dùng `--source cached`, bỏ gradient_inconsistency khỏi `--signals`, đổi output/figure sang `cached`. Không cần deserialize checkpoint trong cached mode; vẫn kiểm hash nội dung file.

Outputs cho mỗi seed nằm trong `runs/signals/{cached,recomputed}/synthetic/v8_az360/seed_<s>/`:

- `checkpoint.json`, `scores.npz`, `signal_results.csv` (đúng 6 cột yêu cầu, thứ tự Gaussian giữ nguyên).
- `rankings.npz` (mỗi signal chứa danh sách Gaussian ID theo rank), `topk.npz` (ID Top1/5/10).
- `diagnostics.npz`, `signal_summary.json` (counts, phân phối, cutoff ties, metadata).
- Bộ đầy đủ 81 hình: `runs/signals/figures/{cached,recomputed}/*_top{1,5,10}.png` (gitignored); đây là 3D scatter tâm Gaussian, không phải rendered-opacity mask.
- PR chỉ giữ 8 hình seed 0 / Top 5% trong `results/signals/{cached,recomputed}/`. Script reproduce chỉ copy 8 hình đại diện này sang results; các ảnh còn lại để trong runs và gói ZIP để chia sẻ qua Drive.
- Trạng thái upload và link Drive được ghi trong `results/signals/README.md`; chưa có link thật thì không coi bước chia sẻ đã hoàn tất.

Checkpoint hash phải khớp manifest/score/xyz. Override path cũ bằng `--checkpoint`, không sửa file data gốc. Config dùng để recompute là bản committed `config_used.yaml`; gói data không kèm config gốc hay môi trường chạy gốc nên provenance vẫn có giới hạn.

Test:

```powershell
python -m pytest -q
```

Trong sandbox Windows nếu temp mặc định không ghi được, dùng một đường dẫn mới trong workspace: `python -m pytest -q --basetemp runs/pytest_signal_check_01`.

## 6. References

- [R1] Kerbl et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. Local `../2308.04079v1.pdf`; https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ . Chính: alpha blending, activated parameters, positional gradients, adaptive density control.
- [R2] Adebayo et al. (2018). *Sanity Checks for Saliency Maps*. Local `../1810.03292v3.pdf`. Lý do không dùng visual plausibility làm bằng chứng duy nhất.
- [R3] Hooker et al. (2019). *A Benchmark for Interpretability Methods in Deep Neural Networks*. Local `../9167-a-benchmark-for-interpretability-methods-in-deep-neural-networks.pdf`. Bối cảnh đánh giá bằng removal/retraining và random baseline; không chứng minh trực tiếp cho Gaussian intervention.
- [R4] Jiang et al. *FisherRF: Active View Selection and Mapping with Radiance Fields using Fisher Information*. Local `../2311.17874v2.pdf`. Bối cảnh Fisher information/view selection, phối hợp track Phát.
- [R5] Hanson et al. *PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting*. Local `../2406.10219v3.pdf`. Bối cảnh pruning dựa sensitivity; mục tiêu compression khác geometric-failure identification.
- [R6] Kendall & Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* Local `../1703.04977v2.pdf`. Phân biệt loại uncertainty; các heuristic ở đây chưa phải uncertainty calibrated.

Local paper paths ở mục này được tính từ root repo. Những tham chiếu ngoài R1 hiện chỉ dùng làm bối cảnh qua phần giới thiệu/abstract; cần đọc kỹ phần phương pháp trước khi bổ sung claim cụ thể từ chúng.
