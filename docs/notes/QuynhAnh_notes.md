# Nhật ký triển khai — Quỳnh Anh

## 2026-10-17 — Giảm dung lượng ảnh trong PR

- Giữ 8 ảnh seed 0 / Top 5% trong results/signals: 4 cached (gồm random) và 4 research signal tính lại. Tổng dung lượng ảnh còn 1527940 bytes (~1.53 MB), thay cho 81 ảnh / 16423129 bytes.
- Bảo toàn cả 81 ảnh trong runs/signals/figures/ và runs/signals/archives/QuynhAnh_signal_figures_full.zip, đều gitignored. figure_manifest.csv ghi hash/kích thước và ảnh nào còn trong PR.
- Exporter mặc định ghi ảnh vào runs; reproduce chỉ copy 8 ảnh đại diện vào results. Thêm .gitignore whitelist để tránh đưa lại toàn bộ ảnh vào commit.
- Đã nhận và ghi link thư mục Drive vào README. Mở được thư mục faithfulrecon nhưng phiên trình duyệt chưa đăng nhập, không có control upload; chưa upload ZIP. Cần upload bằng account có quyền write và xác nhận reviewer tải được trước merge.
- Không khôi phục hoặc đưa vào thay đổi này hai file docs/pr đã được người dùng xóa, hay thay đổi ngoài track signals đang có trong working tree.

## 2026-10-16

### Đã làm

- Đọc README, data contract, training/rendering path và code signal; giữ nguyên Phase 0 config, labels và intervention rule.
- Xác nhận ba checkpoint của data HKDN-261 khớp SHA256 manifest; N=27733/27984/28691.
- Implement cross-view gradient inconsistency bằng world-position loss gradients, population covariance, Welford float64 O(N), output float32.
- Thêm test gradient giống nhau/đối nhau/zero, thiếu view, amplitude variation cùng hướng, rotation invariance, covariance tường minh, threshold cache, relative delta và bảo toàn parameter/.grad.
- GradMag lấy autograd.grad thay backward ghi đè parameter gradients; đối chiếu với cách backward cũ bằng test. Cache view statistics phân biệt threshold/loss.
- Cho phép truyền signal_params khi compute_signals; kiểm dtype float32 trong output contract.
- Hoàn thiện exporter cached/checkpoint, kiểm hash/N, CSV, rankings, Top1/5/10, diagnostics và scatter 3D.
- Xuất cached score nguyên giá trị và recompute bốn signal + random trên cả ba checkpoint.
- Viết audit riêng, đọc metrics D2 gốc và sanity với f_i tạm; không sửa file analysis của track khác.
- Có script PowerShell reproduce toàn bộ: scripts/reproduce_quynhanh.ps1.

### Assumptions / quyết định hiện tại

- Gói data không có config.yaml gốc; dùng results/phase0/config_used.yaml committed làm cấu hình tái dựng scene. Config source và hash được ghi trong output; chưa xác minh provenance với người chạy gốc.
- Background trắng; scene synthetic/v8_az360; dùng toàn bộ 8 training views; không dùng held-out để tạo score.
- Máy có RTX3050 nhưng environment sẵn có là torch 2.4.1+cpu; chạy CPU, 4 threads. Không thay đổi môi trường Python của người dùng.
- GI baseline: population covariance, delta=1e-12, n_min=2, fallback score=0. Fallback là thiếu bằng chứng disagreement, không phải đáng tin cậy. Count tách riêng trong diagnostics.
- Delta_rel mặc định 0; tùy chọn relative delta là ablation, chưa chọn tham số dựa trên nhãn.
- Giữ legacy visibility jitter và shared RNG để không thay đổi Phase 0. Export sorting exact tie theo ID, có ghi metadata; Phase 0 torch.topk vẫn giữ nguyên.
- m_i_available=false và provisional=true ở nhãn được cấp. Không tính metric với m_i.
- D2 metrics cũ thuộc original selections; không gán chúng cho CPU selections hoặc GI mới.

### Kết quả và vấn đề thực tế

- GradMag Top5 overlap với score gốc: 99.928%, 99.857%, 99.930%; Spearman > 0.99998. Một số giá trị gradient lệch đáng kể, chưa bitwise reproduce.
- Opacity Top5 overlap 100%; max activation difference 1.19e-7.
- Visibility Top5 overlap 83.706%, 86.133%, 81.812%. Integer counts khớp seed0/1; seed2 lệch 1 Gaussian. RNG/ties giải thích phần lớn selection drift; nguyên nhân count mismatch cần kiểm threshold/precision ở runtime gốc.
- Random score không trùng score gốc. Khả năng do device RNG (config gốc yêu cầu CUDA), nhưng runtime gốc không được ghi; chưa xác nhận nguyên nhân. Cached export giữ nguyên random gốc nếu cần replay artifact.
- Có 435/488/395 Gaussian không contributing view; 666/800/638 dưới hai view. Đừng fill missing GI bằng score rất cao rồi coi đó là disagreement.
- GI AUROC f_i tạm 0.6508 ± 0.0250; precision@5% 0.0991 ± 0.0064. Chưa chạy GI removal/D2.
- GI liên hệ vừa phải với visibility count (~+0.40 Spearman). Cần stratified analysis, chưa chứng minh signal độc lập coverage.
- Opacity AUROC bị ảnh hưởng cách định nghĩa f_i có opacity gate; không tự đảo orientation để tăng AUROC.
- Image mask Phase 0 và locality rule giữ nguyên; các refinement chỉ được đề xuất trong research/audit.

### Failed attempts / môi trường

- Python mặc định ở Anaconda base không có torch. Dùng D:/Programs/anaconda3/envs/chess-rl/python.exe, đã có đủ dependencies CPU.
- pytest tmp_path không ghi được vào temp mặc định của sandbox. Chạy lại với --basetemp trỏ một thư mục mới trong runs/; đây là lỗi môi trường, không phải test toán học thất bại.
- Lượt chạy exporter đầu tiên phát hiện path override tương đối bị resolve thêm theo run_dir. Đã sửa: --checkpoint tương đối resolve từ cwd; manifest path tương đối resolve từ run_dir. CLI test kiểm path override và hash rejection.
- Không sửa checkpoint.json hay dữ liệu nguồn; local checkpoint.json trong output chứa đường dẫn mới.

### Vấn đề cần thảo luận trước refinement

1. Chốt n_min/fallback cho GI: zero giữ meaning “không có bằng chứng”, nhưng che nhóm thiếu quan sát nếu đọc score như reliability calibrated.
2. Có chạy active-only baseline không? Nếu có, random cần cùng eligible pool và format phải vẫn đủ N, kèm mask rõ ràng.
3. Có tách RNG theo signal không? Đây sẽ đổi visibility tie selection; cần phiên bản ablation, không thay legacy âm thầm.
4. Có dùng direction-only gradient statistic không? Công thức hiện tại phản ánh cả amplitude; variant cần tên/định nghĩa riêng.
5. Đề xuất với lead/D2 owner: immediate-removal measurement, mask hẹp, locality3D trong decision rule, tách sign expected/opposite.
6. Xin xác nhận runtime/config gốc để audit CPU-vs-CUDA gradient/count drift.

### Outputs / kiểm chứng

- runs/signals/cached/synthetic/v8_az360/seed_{0,1,2}/: scores gốc, CSV, rank và selections.
- runs/signals/recomputed/synthetic/v8_az360/seed_{0,1,2}/: bốn research signal + random, CSV, rank, selections và diagnostics.
- runs/signals/figures/recomputed/: Top1/5/10 của cả năm signal qua ba seed (45 hình); results chỉ giữ 4 hình đại diện.
- runs/signals/figures/cached/: Top1/5/10 của bốn score gốc qua ba seed (36 hình); results chỉ giữ 4 hình đại diện.
- results/signals/QuynhAnh_signal_audit.md và các CSV nhỏ: checkpoint audit, reproduction, sanity, confounders, D2 per-seed.
- Tất cả file per-Gaussian để trong runs/ (gitignored), không commit checkpoint hoặc CSV lớn.
- Lệnh reproduce và test xem docs/research/QuynhAnh_signal_research.md. Bản test cuối ghi trong results/signals/validation.md.
