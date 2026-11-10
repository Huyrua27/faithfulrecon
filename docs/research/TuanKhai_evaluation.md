# D1 evaluation pipeline

**Owner:** Tuấn Khải · **Track:** evaluation · **Branch:** `feature/tuankhai-evaluation`

**Status:** D1 core implemented; Phase 0 provisional results checked for v8_az360 seeds 0–2. The expanded two-condition run awaits PR #1 merge and the new data package. Official m_i comparison is prepared below, with no official m_i result yet.

> Mục tiêu: người khác lấy checkpoint + signal scores + geometric labels, chạy **một lệnh**
> và nhận AUROC, AUPRC, Spearman, Precision@K, Failure enrichment và figures, không sửa code.

Code: `frecon/eval/` · Entry: `scripts/run_d1.py` · Tests: `tests/test_d1_metrics.py`, `tests/test_d1_io.py`, `tests/test_random_baseline.py`, `tests/test_d1_plots.py`, `tests/test_run_d1.py`.

## 1. Input format
Hợp đồng chính: [data_format.md](../data_format.md), §2–4. Mỗi run nằm tại `runs/d1/<scene>/<condition>/seed_<s>/` và có:

- `scores.npz`: score float32 cho từng Gaussian và metadata của signal.
- `geometric_labels.npz`: xyz/distance float32, m_i/f_i bool và metadata của nhãn.
- `checkpoint.json`: đường dẫn checkpoint, SHA-256 và số Gaussian.

Các file phải cùng checkpoint, N, scene/condition/seed và thứ tự hàng Gaussian. Hash xác nhận checkpoint identity; upstream vẫn phải bảo đảm thứ tự hàng. Score cao = kém tin cậy hơn; runner sử dụng polarity đã được exporter chuẩn hóa.

`m_i_available=False` nghĩa là chưa có nhãn m_i. Mảng False giữ chỗ không được dùng như nhãn âm thật. Nếu availability=True nhưng nhãn thật toàn False/True, giữ hàng kết quả và trả NaN cho metric phân loại theo convention D1.

Export Phase 0 hiện có `provisional=True`. Không tự đổi cờ này hoặc tạo m_i giả để đạt đủ output.

## 2. Output format
Mặc định output là `results/d1/<scene>/<condition>/`:

- `metrics.csv`: một hàng cho mỗi `(scene, condition, seed, signal, label)`, schema 18 cột theo hợp đồng §5.
- `figures/`: ROC, PR, failure enrichment và score–error cho f_i. Khi có m_i, thêm ROC/PR/condition enrichment tại `figures/m_i/`.
- `logs/run_d1.log`: nguồn input/hash, validation warnings, R và spread của matched random, số seed hợp lệ, thống kê và thời gian chạy.

ROC, PR và enrichment lấy trung bình đều qua các seed có metric xác định; legend ghi số seed. Score–error gộp Gaussian qua seed. Log hiện tổng hợp qua seed cho AUROC, AP và Precision@5%.

## 3. Metrics
| Metric | Cách tính / diễn giải |
|---|---|
| AUROC | Xác suất score của positive lớn hơn negative; tie tính 0,5. Implementation dùng tổng rank, các score bằng nhau nhận rank trung bình. |
| AUPRC / AP | Average precision: tổng Precision tại cuối mỗi nhóm score bằng nhau, có trọng số là phần recall tăng. Không dùng diện tích hình thang của đường PR nội suy tuyến tính. |
| Spearman | Tương quan rank giữa risk score và continuous distance d_i; dùng rank trung bình khi tied. |
| Precision@K | Tỷ lệ positive kỳ vọng trong top K score cao nhất. Tại ranh giới tied, chia đều số suất còn lại cho nhóm tied. |
| Enrichment@K | Precision@K chia prevalence. 1 là mức kỳ vọng của lựa chọn ngẫu nhiên khi nhãn có hai lớp. |

K dùng các mức 1/5/10%, với số phần tử thực tế `max(1, round(k_frac * N))`. Với N nhỏ, nhiều mức phần trăm có thể cho cùng K.

Nhãn phân loại toàn True hoặc toàn False trả NaN theo convention repo, kể cả Precision/AP. Spearman vẫn phụ thuộc score/distance, không phụ thuộc việc f_i/m_i suy biến; error hằng hoặc không có error thì Spearman là NaN.

## 4. Random baseline
Mỗi baseline giữ nguyên scene, condition, seed, Gaussian/nhãn, distance và các mức K; chỉ xếp hạng là ngẫu nhiên. `SeedSequence([seed, draw])` tạo score tái lập cho từng lượt. Các signal trong cùng run/label dùng chung baseline đã cache.

Mặc định R=100. Hàm trả mean/std/q05/q95; CSV lưu ba mean theo hợp đồng, log lưu spread. q05/q95 là phân vị của phân bố lượt random, không phải confidence interval của mean. Std hiện dùng ddof=0.

Sanity với nhãn có hai lớp: AUROC kỳ vọng 0,5, Precision@K kỳ vọng bằng prevalence, enrichment kỳ vọng 1. AP trung bình hữu hạn mẫu không nhất thiết bằng đúng prevalence. Signal `random` được exporter tạo một lượt là một hàng signal riêng, khác control R lượt này.

## 5. Seed handling
Một signal/label có một hàng cho mỗi seed. Thống kê qua seed loại các giá trị không xác định, đồng thời ghi `n_valid` và seed bị loại. Seed identity trùng bị từ chối; std=0 khi chỉ còn một seed không phải bằng chứng ổn định.

Mỗi condition phải chạy và báo cáo riêng. Bộ mở rộng reviewer yêu cầu có inventory:

| Condition | Seed cần có |
|---|---|
| v8_az360 | 0, 1, 2, 3, 4, 5 |
| v8_az180 | 0, 1, 2 |

Kiểm đủ inventory này trước lần cập nhật sau merge. Kết quả đã lưu tại [Phase 0 provisional](../../results/d1/phase0_provisional_2026-10-17/synthetic/v8_az360/README.md) hiện chỉ bao gồm v8_az360 seed 0–2.

## 6. Sanity checks
Validation kiểm N/shape/dtype, giá trị hữu hạn, metadata và đường dẫn, availability, provisional labels và checkpoint SHA-256. Nếu checkpoint file chưa có, log cảnh báo hash chưa được xác minh; không mô tả cảnh báo này như một kiểm chứng thành công.

Test bao gồm ca tính tay, perfect/inverted/tied scores, nhãn suy biến, guard input, đối chiếu sklearn/SciPy, random sanity, diện tích đường ROC/PR, availability, PNG và CLI integration. Fixtures kiểm m_i trong test là synthetic; chúng không tạo official m_i results.

```bash
python -m pytest -q tests/test_d1_metrics.py tests/test_d1_io.py tests/test_random_baseline.py tests/test_d1_plots.py tests/test_run_d1.py
```

## 7. Reproduction command
Sau khi PR #1 được merge và bộ input đủ 9 run đã được đặt tại `runs/d1/`, chạy từ repo root:

```bash
python scripts/run_d1.py --scene synthetic --condition v8_az360 --n_random 100
python scripts/run_d1.py --scene synthetic --condition v8_az180 --n_random 100
```

Các lệnh tạo hai bộ output riêng tại `results/d1/synthetic/v8_az360/` và `results/d1/synthetic/v8_az180/`. Đây là lệnh chuẩn bị, chưa phải bằng chứng hai cấu hình đã chạy.

NPZ có sẵn thì evaluation không cần load model hoặc train lại. Chỉ dùng `scripts/prepare_d1_run.py` khi cần tạo export từ Phase 0 và đã có checkpoint, signals, config cùng runtime của bước đó. Sửa đường dẫn checkpoint trong bản input cục bộ sau khi đối chiếu hash; không đổi label/polarity để làm số liệu tốt hơn.

## 8. Chuẩn bị so sánh f_i và m_i chính thức

**Hai câu hỏi khác nhau:** m_i là điều kiện under-observed, f_i là kết cục geometric failure. Signal dự đoán được m_i chưa tự chứng minh nó dự đoán được f_i hoặc có tính faithfulness.

Khi nhận nhãn từ owner:

1. Kiểm lại checkpoint/hash/N/thứ tự hàng và label_params. Xác nhận provenance/threshold và trạng thái provisional từ owner; không tự gán nhãn là chính thức chỉ vì availability=True.
2. Chạy cùng score cho hai label trên cùng run. Pipeline hiện đã xuất hai nhóm hàng và bộ figure riêng khi m_i có sẵn; schema metrics.csv giữ nguyên.
3. Báo cáo prevalence của từng nhãn và bảng đếm `(m_i, f_i)` theo run. Không giả định f_i phải là tập con của m_i; kiểm giả thiết quan hệ này bằng định nghĩa benchmark.
4. So sánh theo scene/condition/signal và ghi tập seed hợp lệ. Bất kỳ so sánh paired nào phải dùng giao các seed có cả hai metric xác định, không lấy hai mean từ các tập seed khác nhau.
5. Báo AUROC/AP/Precision/enrichment cùng matched random riêng của từng nhãn. AP và Precision phụ thuộc prevalence, vì vậy không diễn giải chênh lệch raw AP giữa f_i và m_i như cùng một bài toán. Enrichment và chênh lệch với random giúp mô tả baseline nhưng cũng không biến hai label thành tương đương.
6. Spearman dùng cùng distance nên giống nhau ở hai nhóm label trong cùng run/signal; chỉ tính là một endpoint, không phải hai bằng chứng độc lập.
7. Tách observation và interpretation. Báo rõ các metric NaN/seed bị loại, dữ liệu provisional còn sót và hạn chế của label gate. Không suy ra quan hệ nhân quả hoặc claim faithfulness chỉ từ association D1.

**Điểm phải theo dõi ở nhãn Phase 0:** f_i hiện yêu cầu opacity > 0,05, còn top 5% opacity-risk ở ba seed đã chạy có opacity < 0,014. Precision@5%=0 của signal này vì vậy chịu ảnh hưởng trực tiếp của label gate. Khi owner cung cấp định nghĩa mới, giữ rõ phiên bản nhãn và đánh giá lại thay vì so sánh lẫn các phiên bản.
