# Quy trình làm việc

## Ai làm gì

| Thành viên | Track | Code | Tài liệu |
|---|---|---|---|
| Quỳnh Anh | Reliability signals | `frecon/signals/`, `scripts/export_signals.py`, `tests/test_signals.py` | `docs/research/QuynhAnh_signal_research.md`, `docs/notes/QuynhAnh_notes.md` |
| Trường Thịnh | Geometric failure & intervention | `frecon/benchmark/`, `scripts/make_labels.py`, `tests/test_benchmark_labels.py` | `docs/research/TruongThinh_geometric_failure_research.md`, `docs/analysis/phase0_analysis.md`, `docs/notes/TruongThinh_notes.md` |
| Tuấn Khải | D1 evaluation & infrastructure | `frecon/eval/`, `scripts/run_d1.py`, `tests/test_d1_metrics.py`, `configs/d1.yaml` | `docs/research/TuanKhai_evaluation.md`, `docs/notes/TuanKhai_notes.md` |
| Tấn Phát | Second-order signals & D3 | `frecon/signals/{second_order,fisher,pup_sensitivity,s_geom}.py`, `frecon/eval/d3.py`, `scripts/run_d3.py`, `tests/test_second_order.py`, `tests/test_d3.py` | `docs/research/Phat_second_order_and_d3.md`, `docs/notes/Phat_notes.md` |
| Minh Huy | Lead, lõi, D2, review | `frecon/` (lõi), `scripts/phase0*`, `scripts/prepare_d1_run.py` | — |

Phân công chi tiết: `docs/tasks/assignment.md` (Quỳnh Anh, Trường Thịnh, Tuấn Khải), `docs/tasks/Phat_tasks.md` (Phát). Tìm việc của mình: `grep -rn "TODO(QuynhAnh)" .`
(hoặc `TruongThinh`, `TuanKhai`). Định dạng file giữa các track: `docs/data_format.md`.

Dữ liệu dùng chung (không có trên git): xin Huy `runs/phase0/seed_*/base.pt` + `runs/d1/`, hoặc tự tạo:

```bash
python scripts/prepare_d1_run.py --phase0_dir runs/phase0 --seeds 0 1 2 --scene synthetic --condition v8_az360
```

## Git

- **Không commit trực tiếp vào `main`.** Mỗi người làm trên branch riêng, merge qua Pull Request,
  Huy review.

  | Thành viên | Branch |
  |---|---|
  | Quỳnh Anh | `feature/quynhanh-signals` |
  | Trường Thịnh | `feature/truongthinh-benchmark` |
  | Tuấn Khải | `feature/tuankhai-evaluation` |
  | Tấn Phát | `feature/phat-second-order`, `feature/phat-d3` |

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
