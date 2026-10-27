# Nhật ký — Tấn Phát

Branch: `feature/phat-second-order, feature/phat-d3`

Ghi theo ngày, mới nhất ở trên. Gồm: implementation issues, assumptions, failed attempts,
câu hỏi cần thảo luận. Ngắn gọn, có link tới commit / file / log.

## 2026-10-27 — PR evaluation sau khi cập nhật main

- Main mới `02a1536` đã merge code signals của Quỳnh Anh và D1 của Tuấn Khải;
  merge main vào feature branch trước khi mở PR theo CONTRIBUTING.
- Phát hiện thay đổi convention v: official dùng 0 cho <n_min view, bản tạm dùng 1.
  Wrapper giờ delegate cả custom threshold/loss weight, ghi fallback/provider/count/
  effective delta; test fallback cũ isolate bằng NotImplementedError.
- Full suite sau tích hợp: **132 passed, 7 skipped**. Export lại 3 checkpoint sang
  `runs/pr_review/d1`, giữ baseline; D1 trên main chạy 30 rows và xác minh hash checkpoint.
- Package official-v và evidence mới được tách khỏi package temporary-v cũ để giữ
  provenance. Không diễn giải AUROC thấp thành lỗi estimator hoặc tự đảo hướng score.
- Evaluation cuối: `docs/analysis/Phat_W4_PR_review_2026-10-27.md`.
  GTX 1650 tiếp tục là task user-deferred; D3 vẫn ngoài scope PR W4.

## 2026-10-27

- **Handoff:** tạo ZIP portable gồm base checkpoint, scores, labels cho 3 seed,
  config, manifest SHA256 và hướng dẫn. `checkpoint.json` trong package dùng path
  tương đối; manifest gốc giữ riêng. Không đổi source arrays/labels/baseline.
- **Consumer thực tế:** chạy nguyên code Tuấn Khải tại `e4ba397` từ snapshot riêng,
  không sửa/merge file track khác. D1 exit 0, 30 rows (10 signals × 3 seed), hash
  checkpoint kiểm chứng được; chỉ warning provisional labels và m_i unavailable.
  AUROC khớp diagnostic trước đó, input files không đổi.
- **Delivery:** upload ZIP vào folder W4 mới bên trong data Drive của team. Đọc
  metadata lại và `rclone check --download`: 1 file khớp, 0 differences. Receipt,
  manifest và consumer evidence nằm ở `results/second_order/w4/handoff/`.
- **Test:** full suite 54 passed, 14 skipped, gồm 5 test mới cho portable package:
  relocation/source preservation, corruption, row order, unavailable m_i và baseline.
  Snapshot consumer được giữ trong thư mục ẩn để pytest không collect nhầm tests.
- **Remaining:** user cho phép để GTX 1650 profiling thành task mở ngày 27/10;
  không thay bằng số liệu RTX 4060. Convention v tạm vẫn cần Huy/Quỳnh Anh review.
- **Git:** refresh main vẫn tại `3aef207`; nộp code/docs/evidence trên
  `feature/phat-second-order`, không đưa checkpoint/NPZ/package vào Git.
- **Tài liệu handoff:** `docs/analysis/Phat_W4_handoff_2026-10-27.md`.

## 2026-10-23

- **Git:** kết nối GitHub đã hoạt động; fetch và fast-forward branch
  `feature/phat-second-order` tới `origin/main` `3aef207`, giữ lại phần W3 đang stage.
- **Đã làm:** Fisher diagonal/all-parameter trace, PUP với basis activated scale
  hoặc log-scale, curvature, v tạm, tích và rank-sum. Export riêng của Phát kiểm tra
  SHA256/N/config trước khi ghép file atomically; giữ nguyên baseline, không sửa
  exporter hoặc gradient_inconsistency của Quỳnh Anh.
- **Dữ liệu:** đã có quyền Drive; tải 39 file, checksum khớp. Audit cả 3 checkpoint:
  hash, N, thứ tự xyz, baseline signals, khoảng cách/floater labels và image metrics.
  Xuất 6 score mới cùng 4 baseline vào `runs/d1/synthetic/v8_az360/seed_{0,1,2}/scores.npz`.
- **Kiểm chứng:** full suite 49 passed, 14 skipped. S=16 trên checkpoint seed 0:
  rank correlation với reference S=256 đạt 0.99799 trở lên cho 5 score có Fisher block.
  Sweep 3 probe seed và eps=1e-8/1e-6/1e-4; reference là xấp xỉ, không phải Jacobian exact.
- **Profile:** RTX 4060 Laptop, trained N=27,733/27,984/28,691, 8 view 160px,
  S=16: 4.18–4.78 giây, peak allocated 354–372 MiB. Chưa đo GTX 1650.
- **Lỗi đã sửa:** CUDA eigvalsh với 28k matrix nhỏ dùng khoảng 15 GiB workspace;
  chia batch 512 và thêm regression cho memory. SciPy/MKL ở test DTU mới trên main
  bị native abort trong môi trường này; suite pass với `MKL_THREADING_LAYER=SEQUENTIAL`.
- **Observation:** AUROC theo f_i tạm: v=0.6514, GradMag=0.6536; Fisher/PUP/
  curvature/product <0.5. Không đảo chiều theo label, không kết luận faithfulness.
  v dùng fallback=1 cho <2 view; metadata ghi provider và số Gaussian thiếu evidence.
- **Tiếp theo:** review/handoff W4, đo GTX 1650; W5 ablation visibility/basis/u-v,
  phối hợp D2. D3 và bảng D1–D2–D3 vẫn là mốc W6–W8.
- **Tài liệu:** `results/second_order/w4/report.md`,
  `docs/analysis/Phat_W3_W4_status_2026-10-23.md`.

## 2026-10-17

- **Đã làm:** tạo branch `feature/phat-second-order` từ local `main` (`f49ac9b`);
  implement `fisher_blocks` cho nhóm tham số bất kỳ, gồm block vị trí 3x3 và
  vị trí/log-scale 6x6. Không đổi core hoặc file thuộc track của bạn khác.
- **Kiểm chứng:** exact image Jacobian trên 20 Gaussian, 2 view 16x16; sweep
  S=4/64/1024 với 4 probe seed. S=1024: relative Frobenius error trung bình
  4.084% (3x3), 4.145% (6x6). Xem `results/second_order/report.md`.
- **Test:** targeted 19 passed, 6 skipped (W4); full suite 26 passed, 19 skipped.
  CUDA smoke và analytical normalization test chạy được với Anaconda/PyTorch CUDA.
- **Assumption:** SignalContext gắn với checkpoint/camera/backend không thay đổi;
  cache không tự phát hiện thay đổi model. Block dùng log-scale gốc, chưa damping.
- **Môi trường:** PATH hiện trỏ tới Python 3.14/PyTorch CPU; dùng
  `C:/Users/phatl/anaconda3/python.exe` (3.12.7, torch 2.8.0+cu129) để kiểm chứng.
- **Thử nhưng không được:** `git fetch --no-tags origin` không kết nối được GitHub
  cổng 443; branch dựa trên local `main`, chưa xác nhận remote mới hơn ở lần này.
- **Tiếp theo:** W4: `fisher`, `pup`, `curvature`; quyết định basis scale/log-scale
  cho PUP, damping/S và tích hợp `v_i` của Quỳnh Anh; chưa có checkpoint để profile
  cảnh ~28k Gaussian. D3 vẫn để giai đoạn sau.

## YYYY-MM-DD

- **Đã làm:**
- **Assumption:**
- **Thử nhưng không được (và vì sao):**
- **Câu hỏi cần thảo luận:**
- **Tiếp theo:**
