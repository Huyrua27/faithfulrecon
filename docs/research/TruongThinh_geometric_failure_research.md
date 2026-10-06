# Geometric failure, labels and intervention — research notes

**Owner:** Trường Thịnh · **Branch:** `feature/truongthinh-benchmark` · **Status:** TODO

> Cần định nghĩa rõ: Gaussian geometric failure là gì, under-observed condition là gì, và
> intervention response cần được đo như thế nào để kiểm tra locality/direction.

Code: `frecon/benchmark/` · Script: `scripts/make_labels.py` · Test: `tests/test_benchmark_labels.py`
Output: `geometric_labels.npz` (format: `docs/data_format.md` §4).

```text
Gaussian → position / geometry → reference geometry → geometric error d_i → failure label f_i
under-observed condition m_i  ≠  geometric failure f_i
```

## 1. Problem definition
TODO

## 2. Under-observed condition (`m_i`)
TODO — chỉ phụ thuộc camera + bề mặt tham chiếu; xử lý che khuất; k view, góc tam giác hoá θ.

## 3. Geometric failure definition (`f_i`)
TODO

## 4. Reference geometry
TODO — nguồn, hệ toạ độ, đơn vị, kiểm tra alignment (kết quả `check_alignment`).

## 5. Continuous error (`d_i`)
TODO — center-to-surface; mật độ điểm tham chiếu so với τ; có dấu hay không.

## 6. Binary label
TODO — ngưỡng, bảng prevalence theo ngưỡng (`results/benchmark/label_stats.csv`).

## 7. Intervention hypothesis
TODO

## 8. Locality criterion
TODO — image-space mask (hiện quá rộng: LR_img ≈ 0.90 cả với random) vs 3D displacement.

## 9. Direction criterion
TODO — "đúng chiều kỳ vọng" vs "nhất quán nhưng ngược chiều".

## 10. Noise-floor criterion
TODO — null_a / null_b, SNR_in, ngưỡng 2.

## 11. Limitations
TODO

## 12. Proposed refinements
Mỗi đề xuất: Problem → Why it matters → Proposed change → What experiment would validate it.
TODO

## 13. References
TODO
