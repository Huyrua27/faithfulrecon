## Mục đích

Hoàn thiện pipeline reliability signals của Quỳnh Anh từ checkpoint đến score per Gaussian, ranking và Top-K visualization. Trước thay đổi này, gradient inconsistency và exporter còn là stub; sau thay đổi có thể tính và xuất bốn signal trên checkpoint Phase 0, kèm metadata và bằng chứng kiểm tra.

Track: **signals / Reliability Signal Analysis**.

## Thay đổi

- Implement cross-view gradient inconsistency bằng gradient photometric loss theo vị trí 3D world; dùng population covariance và Welford float64 với bộ nhớ O(N).
- Giữ baseline delta=1e-12, n_min=2; Gaussian thiếu view nhận score=0 và có count riêng để thể hiện thiếu bằng chứng, không diễn giải zero là reliable.
- Hoàn thiện exporter với hai chế độ: dùng score đã lưu hoặc tính lại từ checkpoint. Kiểm SHA256, Gaussian count, shape, float32 và finiteness.
- Xuất scores.npz, CSV đúng data contract, ranking, Top 1/5/10%, diagnostics và scatter 3D. Score arrays giữ thứ tự checkpoint; ranking chứa Gaussian ID được sắp xếp.
- GradMag dùng autograd.grad để bảo toàn parameter/.grad; cache statistics phân biệt contribution threshold và loss weight. Giữ cách tổng hợp mean và visibility jitter của baseline.
- Bổ sung tests, script reproduce, tài liệu giải thích từng signal, implementation notes và audit kết quả ba seed.

Protocol D2, labels và configs Phase 0 không được thay đổi trong phần signals này. Quy tắc phá exact tie bằng Gaussian ID chỉ áp dụng cho ranking/export; intervention Phase 0 vẫn dùng quy tắc cũ.

## Lệnh tái lập

Từ root repo, dùng Python đã cài requirements.txt:

```powershell
./scripts/reproduce_quynhanh.ps1 -Python 'D:/Programs/anaconda3/envs/chess-rl/python.exe' -DataRoot '../HKDN-261-20261016T025052Z-1-001/HKDN-261/data'
```

Đổi `-Python` và `-DataRoot` theo máy của reviewer. Gói data cần có `seed_{0,1,2}/base.pt`, `metrics.json` và `d1/synthetic/v8_az360/seed_{0,1,2}/`.

Tests:

```powershell
python -m pytest -q
git diff --check
```

Trong sandbox Windows nếu temp mặc định không ghi được, truyền một thư mục mới vào `--basetemp runs/<unique-test-directory>`.

## Kết quả / bằng chứng

Các lần chạy experiment đã có trên ba checkpoint:

| Seed | Số Gaussian | Số dòng CSV tính lại, 5 signal gồm random |
|---|---:|---:|
| 0 | 27733 | 138665 |
| 1 | 27984 | 139920 |
| 2 | 28691 | 143455 |

- Tổng **422040 dòng CSV**, không tính header; **45 hình tính lại** và **36 hình từ score gốc**.
- Test kiểm lại ngày 17/10/2026: **28 passed, 19 skipped**. Các skip thuộc phần benchmark/evaluation/second-order/D3 chưa triển khai sẵn trong repo.
- Đã kiểm từng dòng CSV với score và Gaussian ID, ranking permutation, Top-K prefix, checkpoint hash, shape/dtype/finiteness và cached-score equality.
- Opacity Top5 overlap với score gốc 100%; GradMag khoảng 99.86–99.93%; visibility khoảng 82–86%. Visibility integer counts khớp seed0/1 và lệch một Gaussian ở seed2; tie-breaking/runtime arithmetic còn cần audit.
- GI AUROC với nhãn floater tạm: **0.6508 ± 0.0250**; precision@5% **0.0991 ± 0.0064**. Đây là association ban đầu; **chưa chạy D2 removal cho GI**.
- Phân tích Phase 0 dùng metrics gốc: GradMag tạo response measurable và locality 3D tốt hơn random nhưng local geometry xấu hơn random sau removal. Không gán các metrics này cho selection tính lại hoặc GI mới.

Tài liệu và bảng nhỏ trong PR:

- `docs/research/QuynhAnh_signal_research.md`
- `docs/notes/QuynhAnh_notes.md`
- `results/signals/README.md`
- `results/signals/QuynhAnh_signal_audit.md`
- `results/signals/validation.md`
- `results/signals/{checkpoint_audit,reproduction_comparison,signal_sanity,confounder_checks,phase0_per_seed}.csv`

Checkpoint, NPZ và per-Gaussian CSV chỉ nằm trong `runs/` đã gitignore. Các link tới `runs/` trong tài liệu chỉ dùng sau khi reviewer có data và chạy reproduce.

## Giới hạn và điểm cần review

1. Nhãn được cấp có `provisional=true`, `m_i_available=false`; chưa đánh giá m_i.
2. Gói data không kèm config/runtime gốc đầy đủ; recompute dùng `results/phase0/config_used.yaml` đã commit và ghi lại config hash.
3. GI đo cả biến thiên độ lớn lẫn hướng gradient. Default delta và fallback thiếu view cần được review trước khi dùng trong intervention chính.
4. Visibility phụ thuộc contribution threshold, opacity, occlusion và backend. Jitter còn dùng shared RNG; signal order được ghi trong metadata.
5. Đề xuất active-only selection, delta sweep, tie-seed sweep, mask locality hẹp và immediate-removal measurement mới là các hướng kiểm tra; cần trao đổi trước khi đổi protocol.

## Checklist (Definition of Done)

- [x] Research explanation cập nhật.
- [x] Implementation chạy trên checkpoint thật.
- [x] Tests pass; có tests cho phần mới.
- [x] Experimental outputs và visualization Top 1/5/10%.
- [x] Documentation và lệnh reproduce.
- [x] Giữ format scores/CSV theo `docs/data_format.md`; diagnostics/ranking là artifact bổ sung riêng.
- [x] Observation và Interpretation tách riêng; chưa claim GI faithful.
- [ ] Kiểm danh sách staged files trước commit; chỉ gồm track signals và tài liệu PR.
- [ ] Kiểm commit không chứa checkpoint, dataset, NPZ hoặc per-Gaussian CSV.

<!-- Bạn có thể sửa nội dung phía trên trước khi mở PR. Nếu muốn thêm ghi chú cho
reviewer, viết thành đoạn Markdown bình thường ở đây. Nội dung trong HTML comment
không hiển thị khi GitHub render. -->
