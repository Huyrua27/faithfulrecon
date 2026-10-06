# Phase 0 analysis

**Owner:** Trường Thịnh (phân tích chính), Quỳnh Anh (góc nhìn signal) · **Status:** TODO

Số liệu: `results/phase0/report.md`, hình: `results/phase0/*.png`, cấu hình: `configs/phase0.yaml`,
cách đo: `docs/phase0.md`.

## Tóm tắt số liệu (đã có — mean ± std qua 3 seed)

| Signal | SNR_in | Gain_LR (3D) | Gain_LR (image) | Gain_Dir (1−SSIM) | Gain_Dir (local CD) | AUROC vs f_i |
|---|---|---|---|---|---|---|
| opacity | 1.10 ± 0.05 (n.m.) | — | — | — | — | 0.35 |
| gradmag | 4.71 ± 0.38 | +0.094 ± 0.019 | +0.011 ± 0.007 | +0.00007 ± 0.00007 | −0.00013 ± 0.00007 | 0.65 |
| visibility | 1.77 ± 0.57 (n.m.) | — | — | — | — | 0.36 |
| random | 2.69 ± 0.18 | — | — | — | — | — |

n.m. = not measurable (SNR_in < 2 ở ít nhất một seed) → không diễn giải gain.

## Câu hỏi cần trả lời (mục 2.4 phân công)

1. Response có vượt noise floor không? — TODO
2. Signal và random có response khác nhau không? — TODO
3. Response có local không? — TODO
4. Direction có phù hợp hypothesis không? — TODO
5. GradMag: response đo được nhưng hình học cục bộ xấu hơn random — diễn giải thế nào? — TODO
6. Metric hiện tại đã đủ để đánh giá intervention chưa? — TODO

Viết theo `docs/templates/experiment_report.md`; tách Observation / Interpretation.
Không biến kết quả thành claim mạnh hơn dữ liệu.

## Điểm cần refine

TODO — mỗi điểm: Problem → Why it matters → Proposed change → What experiment would validate it.
