# FaithfulRecon

**An intervention-based protocol for evaluating spatial reliability signals in 3D Gaussian Splatting.**

Đề tài DACN / HKDN — Khoa KH&KT Máy tính, ĐH Bách Khoa ĐHQG-HCM.
GVHD: TS. Nguyễn Đức Dũng, ThS. Nguyễn Tuấn Khôi · CVDN: Mr. Ngô Hoàng Anh.

Câu hỏi chính: một reliability signal nội tại của 3DGS có xác định được đúng các Gaussian
liên quan tới geometric failure không, và can thiệp lên các Gaussian đó có tạo ra phản ứng
cục bộ, đúng chiều như kỳ vọng không?
Đánh giá theo 3 chiều: **D1** failure association · **D2** intervention response ·
**D3** held-out predictive validity.

## Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Renderer PyTorch + huấn luyện 3DGS + cảnh synthetic có GT hình học | xong |
| Phase 0 (pilot D2) | xong — **GO** ([docs/phase0.md](docs/phase0.md), `results/phase0/`) |
| Reliability signals (opacity, gradmag, visibility) | có bản Phase 0, đang kiểm tra — Quỳnh Anh |
| Cross-view gradient inconsistency | TODO — Quỳnh Anh |
| Geometric labels `m_i`, `f_i` | `f_i` tạm từ Phase 0; `m_i` TODO — Trường Thịnh |
| D1 evaluation pipeline | TODO — Tuấn Khải |
| Phase 1 (D1–D3 trên nhiều scene) | chưa bắt đầu |

## Cấu trúc repo

```text
faithfulrecon/
├── frecon/                       # thư viện chính (import: `from frecon...`)
│   ├── camera.py  gaussians.py  losses.py  train.py  measure.py   ── lõi (Huy)
│   ├── render/                   # rasterizer PyTorch (+ gsplat tuỳ chọn)
│   ├── data/                     # cảnh synthetic, loader NeRF-Synthetic
│   ├── signals/                  # reliability signals ── Quỳnh Anh
│   ├── benchmark/                # reference geometry, d_i, m_i, f_i ── Trường Thịnh
│   └── eval/                     # D1 metrics, random baseline, plots ── Tuấn Khải
├── scripts/
│   ├── phase0.py  analyze_phase0.py  make_phase0_figure.py   ── Phase 0 (Huy)
│   ├── prepare_d1_run.py         # tạo dữ liệu D1 từ Phase 0 (Huy)
│   ├── export_signals.py         # ── Quỳnh Anh
│   ├── make_labels.py            # ── Trường Thịnh
│   └── run_d1.py                 # ── Tuấn Khải
├── configs/                      # phase0.yaml, d1.yaml
├── tests/                        # pytest; mỗi người có file test riêng
├── docs/
│   ├── data_format.md            # ★ hợp đồng dữ liệu giữa các task — đọc trước
│   ├── tasks/assignment.md       # phân công chi tiết
│   ├── research/                 # tài liệu nghiên cứu từng người
│   ├── notes/                    # nhật ký: assumption, lỗi, câu hỏi
│   ├── analysis/                 # phân tích kết quả (Phase 0, D1, ...)
│   └── templates/                # mẫu báo cáo experiment / blocker
├── results/                      # kết quả NHỎ được commit (csv tổng hợp, png, md)
├── data/                         # dataset (không commit) — xem data/README.md
└── runs/                         # output thí nghiệm (không commit)
```

## Ai làm gì

| Thành viên | Track | Code | Tài liệu | Branch |
|---|---|---|---|---|
| Quỳnh Anh | Reliability signals | `frecon/signals/`, `scripts/export_signals.py`, `tests/test_signals.py` | `docs/research/QuynhAnh_signal_research.md`, `docs/notes/QuynhAnh_notes.md` | `feature/quynhanh-signals` |
| Trường Thịnh | Geometric failure & intervention | `frecon/benchmark/`, `scripts/make_labels.py`, `tests/test_benchmark_labels.py` | `docs/research/TruongThinh_geometric_failure_research.md`, `docs/analysis/phase0_analysis.md`, `docs/notes/TruongThinh_notes.md` | `feature/truongthinh-benchmark` |
| Tuấn Khải | D1 evaluation & infrastructure | `frecon/eval/`, `scripts/run_d1.py`, `tests/test_d1_metrics.py` | `docs/research/TuanKhai_evaluation.md`, `docs/notes/TuanKhai_notes.md` | `feature/tuankhai-evaluation` |
| Minh Huy | Lead, lõi, Phase 0, review | `frecon/` (lõi), `scripts/phase0*` | proposal, báo cáo | — |

Tìm việc của mình: `grep -rn "TODO(QuynhAnh)" .` (hoặc `TruongThinh`, `TuanKhai`).

## Cài đặt

Python ≥ 3.10, PyTorch có CUDA (CPU cũng chạy được test). Không cần biên dịch.

```bash
pip install -r requirements.txt
python -m tests.test_rasterizer      # phải in PASS
pytest                               # test của từng track (nhiều test đang skip = TODO)
```

## Bắt đầu nhanh

```bash
# dữ liệu D1 dùng chung (cần runs/phase0 — xin Huy nếu chưa có):
python scripts/prepare_d1_run.py --phase0_dir runs/phase0 --seeds 0 1 2 --scene synthetic --condition v8_az360

# mục tiêu của từng track (hiện là TODO):
python scripts/export_signals.py --run_dir runs/d1/synthetic/v8_az360/seed_0
python scripts/make_labels.py    --run_dir runs/d1/synthetic/v8_az360/seed_0
python scripts/run_d1.py --scene synthetic --condition v8_az360
```

Quy trình git, commit, Definition of Done: [CONTRIBUTING.md](CONTRIBUTING.md).
