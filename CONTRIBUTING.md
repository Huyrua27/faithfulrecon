# Quy trình làm việc

## Git

- **Không commit trực tiếp vào `main`.** Mỗi người làm trên branch riêng, merge qua Pull Request,
  Huy review.

  | Thành viên | Branch |
  |---|---|
  | Quỳnh Anh | `feature/quynhanh-signals` |
  | Trường Thịnh | `feature/truongthinh-benchmark` |
  | Tuấn Khải | `feature/tuankhai-evaluation` |

  Việc nhỏ tách riêng thì tạo branch con: `feature/quynhanh-gradient-inconsistency`, `fix/<mô-tả>`.
- Trước khi mở PR: `git pull origin main` (hoặc rebase) để cập nhật, chạy `pytest`.
- PR nhỏ, một mục đích. Mô tả theo template (`.github/pull_request_template.md`).
- Chỉ sửa file thuộc track của mình. Cần sửa `frecon/` lõi hoặc `docs/data_format.md` → nói trước,
  PR riêng.

## Commit message

```text
feat: implement gradient magnitude signal
feat: add geometric failure labels
feat: add D1 AUPRC evaluation
fix: correct reference coordinate transform
docs: document signal implementation
test: add sklearn agreement test for auprc
exp: add phase0 analysis for seed 0-2
```

Loại: `feat`, `fix`, `docs`, `test`, `refactor`, `exp` (kết quả/phân tích thí nghiệm), `chore`.

## Không commit

- Checkpoint, render, file per-Gaussian (`*.pt`, `*.npz`, `signal_results.csv`) → để trong `runs/`.
- Dataset → `data/` (đã ignore).
- Được commit: code, docs, `results/**` dạng nhỏ (`metrics.csv` tổng hợp, `.png`, `.md`).
  File > 5 MB phải hỏi trước.

## Mỗi lần nộp task phải có 3 phần

| Phần | Trả lời câu hỏi | Ở đâu |
|---|---|---|
| A. Research / technical document | Tôi đang làm gì và tại sao? | `docs/research/<Tên>_*.md` |
| B. Code | Tôi đã implement như thế nào? | `frecon/<track>/`, `scripts/` |
| C. Experimental evidence | Kết quả cho thấy điều gì? | `results/<track>/`, `docs/analysis/` |

## Definition of Done

Research explanation + Implementation + Test + Experimental output + Documentation +
Reproducible command (ghi đúng lệnh trong tài liệu, người khác chạy lại được).

## Ghi kết quả thí nghiệm

Dùng `docs/templates/experiment_report.md`. **Observation ≠ Interpretation**:

> Observation: GradMag có SNR_in = 4.71.
> ✗ Therefore GradMag is faithful.
> ✓ Interpretation: GradMag tạo intervention response cao hơn noise floor trong Phase 0 dưới protocol hiện tại.

## Khi bị kẹt

Mở issue theo `.github/ISSUE_TEMPLATE/blocker.md` (vấn đề, log, đã thử gì, nguyên nhân dự kiến,
hướng xử lý). Không để blocker quá 2 ngày mà không báo.
