# D1 evaluation pipeline

**Owner:** Tuấn Khải · **Branch:** `feature/tuankhai-evaluation` · **Status:** TODO

> Mục tiêu: người khác lấy checkpoint + signal scores + geometric labels, chạy **một lệnh**
> và nhận AUROC, AUPRC, Spearman, Precision@K, Failure enrichment và figures, không sửa code.

Code: `frecon/eval/` · Entry: `scripts/run_d1.py` · Test: `tests/test_d1_metrics.py`

## 1. Input format
TODO — tóm tắt từ `docs/data_format.md` §2–4; cách xử lý `m_i_available = False` (nhãn tạm).

## 2. Output format
TODO — `metrics.csv` (§5), `figures/`, `logs/`.

## 3. Metrics
TODO — định nghĩa chính xác, xử lý tie, nhãn suy biến; K = 1/5/10 %.

## 4. Random baseline
TODO — matched (cùng scene/condition/K/seed), R lần, báo cáo mean và khoảng.

## 5. Seed handling
TODO — mỗi seed là một hàng; tổng hợp mean ± std qua seed.

## 6. Sanity checks
TODO — N khớp, scene/condition khớp, NaN, K selection, random seed, phân bố nhãn, thiếu dữ liệu.

## 7. Reproduction command
```bash
python scripts/prepare_d1_run.py --phase0_dir runs/phase0 --seeds 0 1 2 --scene synthetic --condition v8_az360
python scripts/run_d1.py --scene synthetic --condition v8_az360
```
TODO — cập nhật khi hoàn thành.
