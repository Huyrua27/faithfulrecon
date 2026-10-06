# FaithfulRecon — Phân công công việc cho 3 sinh viên

## Mục tiêu chung

Các task dưới đây được chia theo hướng để 3 bạn vừa hoàn thành phần việc của đồ án tổng hợp, vừa tạo ra các module/dữ liệu có thể sử dụng trực tiếp cho đề tài **FaithfulRecon**..

Không yêu cầu tiếp tục viết survey tổng quan chung. Nếu đọc paper mới thì phải phục vụ trực tiếp cho task được giao.

---

# 1. Quỳnh Anh — Reliability Signals & Research Analysis

## Vai trò

**Research Track 1 — Reliability Signal Analysis**

Tập trung nghiên cứu và triển khai các reliability signals có thể dùng để xác định Gaussian primitives liên quan đến geometric failure trong 3DGS.

## Công việc

### 1.1. Nghiên cứu reliability signals

Tập trung vào:

- Opacity
- Gradient Magnitude
- Visibility
- Cross-view Gradient Inconsistency

Với mỗi signal cần trả lời:

- Signal đo đại lượng gì?
- Được tính ở đâu trong pipeline 3DGS?
- Orientation của score là gì?
- Vì sao signal có thể liên quan đến geometric reliability?
- Signal có hạn chế/confounder gì?
- Có thể dùng signal trong intervention experiment như thế nào?

### 1.2. Reproduce / kiểm tra implementation

Pipeline:

```text
3DGS checkpoint
      ↓
Gaussian parameters / rendering information
      ↓
signal score per Gaussian
      ↓
ranking
      ↓
Top-K selection
```

Cần xác định rõ input/output và assumption của implementation.

### 1.3. Phân tích kết quả Phase 0

Phân tích:

- signal nào tạo response mạnh;
- signal nào response yếu;
- signal nào khác random removal;
- locality có hợp lý không;
- response có ổn định qua seed không;
- trường hợp response measurable nhưng direction ngược kỳ vọng nên được diễn giải thế nào.

Không chỉ báo cáo số liệu; cần đưa ra technical interpretation có căn cứ.

### 1.4. Đề xuất refinement

Nếu phát hiện vấn đề về:

- normalization;
- score orientation;
- selection rule;
- aggregation giữa views;
- signal bị dominated bởi visibility/opacity;

thì ghi lại vấn đề và đề xuất cách kiểm tra.

Không tự ý thay đổi protocol chính nếu chưa trao đổi.

## Kết quả kỳ vọng

Cần trả lời được:

> Với từng reliability signal, signal thực sự đang đo cái gì, được tính như thế nào trong 3DGS, và có cơ sở nào để kỳ vọng signal đó liên quan đến geometric reliability?

Đồng thời tạo được:

```text
score_i
```

cho từng Gaussian `i`, từ đó rank và chọn Top-K Gaussian.

## File cần nộp

### `QuynhAnh_signal_research.md`

Nội dung:

```text
1. Signal definition
2. Mathematical formulation
3. Implementation location in 3DGS
4. Input / output
5. Expected interpretation
6. Potential confounders
7. Limitations
8. Relation to FaithfulRecon
9. References
```

Mỗi signal khoảng 1–2 trang, tập trung vào nội dung cần cho experiment.

### `signals/`

```text
signals/
├── opacity.py
├── gradmag.py
├── visibility.py
└── gradient_inconsistency.py
```

### `signal_results.csv`

Tối thiểu:

```text
scene
condition
seed
signal
gaussian_id
score
```

### Visualization

Tối thiểu:

```text
Top 1%
Top 5%
Top 10%
```

của từng signal trên scene.

### `QuynhAnh_notes.md`

Ghi implementation issues, assumptions, failed attempts và vấn đề cần thảo luận.

## Tiêu chí hoàn thành

- [ ] Hiểu và giải thích được từng signal.
- [ ] Có implementation chạy được.
- [ ] Output đúng số lượng Gaussian.
- [ ] Có ranking.
- [ ] Có Top-K visualization.
- [ ] Có documentation.
- [ ] Có command reproduce.
- [ ] Phân tích được hạn chế/confounder.

---

# 2. Trường Thịnh — Geometric Reliability, Failure Definition & Intervention Research

## Vai trò

**Research Track 2 — Geometric Failure & Intervention Analysis**

Tập trung vào phần research question: làm thế nào định nghĩa và kiểm tra một Gaussian có liên quan đến geometric failure, đồng thời nghiên cứu cách intervention kiểm chứng điều này.

## Công việc

### 2.1. Nghiên cứu geometric failure

Xây dựng understanding rõ về:

```text
Gaussian
   ↓
position / geometry
   ↓
reference geometry
   ↓
geometric error
   ↓
failure label
```

Phải phân biệt:

```text
under-observed condition
        ≠
geometric failure
```

Nghiên cứu và kiểm tra cách tạo:

```text
m_i
```

và:

```text
f_i
```

### 2.2. Reference geometry và geometric error

Xây dựng pipeline:

- load reference geometry;
- đưa về cùng coordinate system với Gaussian model;
- tính khoảng cách Gaussian tới reference;
- tạo continuous geometric error;
- tạo binary geometric failure label.

Output cơ bản:

```text
gaussian_id
xyz
distance
m_i
f_i
```

### 2.3. Phân tích intervention protocol

Nghiên cứu:

```text
Reliability signal
       ↓
select Top-K Gaussian
       ↓
remove
       ↓
re-optimize
       ↓
measure response
```

và matched control:

```text
Random Top-K
       ↓
remove
       ↓
re-optimize
       ↓
compare
```

Tập trung vào:

- locality;
- direction;
- matched random control;
- re-optimization noise;
- response trong 3D;
- response trong image space.

### 2.4. Phân tích Phase 0

Trả lời:

1. Response có vượt noise floor không?
2. Signal và random có response khác nhau không?
3. Response có local không?
4. Direction có phù hợp hypothesis không?
5. Nếu response measurable nhưng direction ngược kỳ vọng thì diễn giải thế nào?
6. Metric hiện tại đã đủ để đánh giá intervention chưa?

Đặc biệt phân tích trường hợp **GradMag có measurable response nhưng direction ở một số metric chưa cùng chiều với expectation**.

Không biến kết quả thành claim mạnh hơn dữ liệu.

### 2.5. Đề xuất refinement

Có thể đề xuất refinement cho:

- image region mask;
- 3D locality rule;
- direction criterion;
- geometric failure threshold;
- matched random selection;
- noise-floor measurement.

Mỗi đề xuất phải có:

```text
Problem
→ Why it matters
→ Proposed change
→ What experiment would validate it
```

## Kết quả kỳ vọng

Cần có định nghĩa rõ:

> Gaussian geometric failure là gì, under-observed condition là gì, và intervention response cần được đo như thế nào để kiểm tra locality/direction.

Đồng thời tạo benchmark labels để D1/D2 sử dụng.

## File cần nộp

### `TruongThinh_geometric_failure_research.md`

```text
1. Problem definition
2. Under-observed condition
3. Geometric failure definition
4. Reference geometry
5. Continuous error
6. Binary label
7. Intervention hypothesis
8. Locality criterion
9. Direction criterion
10. Noise-floor criterion
11. Limitations
12. Proposed refinements
13. References
```

### `benchmark/`

```text
benchmark/
├── reference_geometry.py
├── geometric_error.py
├── failure_labels.py
└── visualize_labels.py
```

### `geometric_labels.npz`

Tối thiểu:

```text
xyz
distance
m_i
f_i
```

### `phase0_analysis.md`

Phân tích:

- SNR;
- Gain_LR;
- Gain_Dir;
- random control;
- seed consistency;
- các điểm cần refine.

### Figures

```text
reference geometry + Gaussian
under-observed map
geometric failure map
geometric error
```

## Tiêu chí hoàn thành

- [ ] Phân biệt rõ `m_i` và `f_i`.
- [ ] Reference geometry align đúng.
- [ ] Có continuous geometric error.
- [ ] Có failure labels.
- [ ] Có visualization.
- [ ] Có phân tích Phase 0.
- [ ] Có refinement có thể kiểm chứng.
- [ ] Không overclaim từ Phase 0.

---

# 3. Tuấn Khải — D1 Evaluation & Experimental Infrastructure

## Vai trò

**Engineering / Evaluation Track**

Biến output của Quỳnh Anh và Trường Thịnh thành evaluation pipeline reproducible để đánh giá reliability signals.

## Công việc

### 3.1. Xây dựng data interface

Format thống nhất:

```text
signal score
+
geometric labels
+
scene metadata
+
seed
```

### 3.2. D1 evaluation

Implement:

- AUROC;
- AUPRC;
- Spearman correlation;
- Precision@K;
- Failure enrichment @K.

K:

```text
1%
5%
10%
```

### 3.3. Random baseline

Matched random:

```text
same scene
same condition
same K
same seed
```

và hỗ trợ nhiều seed.

### 3.4. Visualization

Tạo:

- ROC curve;
- Precision-Recall curve;
- score vs geometric error;
- failure enrichment curve;
- Top-K spatial visualization nếu input cho phép.

### 3.5. Reproducibility

Command mục tiêu:

```bash
python run_d1.py \
    --scene <scene> \
    --condition <condition>
```

Output:

```text
metrics.csv
figures/
logs/
```

### 3.6. Sanity checks

Tự kiểm tra:

- số Gaussian của score và label có khớp;
- scene/condition có khớp;
- NaN;
- K selection;
- random seed;
- label distribution;
- missing data.

## Kết quả kỳ vọng

Một người khác có thể lấy:

```text
checkpoint
+
signal scores
+
geometric labels
```

và chạy một command để nhận:

```text
AUROC
AUPRC
Spearman
Precision@K
Failure enrichment
figures
```

mà không sửa code thủ công.

## File cần nộp

### `d1_evaluation.py`

Các metric chính.

### `run_d1.py`

Entry point chạy experiment.

### `metrics.csv`

```text
scene
condition
seed
signal
auroc
auprc
spearman
precision_at_1
precision_at_5
precision_at_10
```

### `figures/`

```text
roc.png
pr_curve.png
failure_enrichment.png
score_vs_error.png
```

### `TuanKhai_evaluation.md`

```text
1. Input format
2. Output format
3. Metrics
4. Random baseline
5. Seed handling
6. Sanity checks
7. Reproduction command
```

### Dependency/config

Nếu có dependency riêng phải ghi rõ cách setup.

## Tiêu chí hoàn thành

- [ ] D1 pipeline end-to-end.
- [ ] AUROC.
- [ ] AUPRC.
- [ ] Spearman.
- [ ] Precision@K.
- [ ] Failure enrichment.
- [ ] Random baseline.
- [ ] Multi-seed.
- [ ] Figures.
- [ ] Sanity checks.
- [ ] One-command reproduction.
- [ ] Documentation.

---

# 4. Cách 3 task liên kết với nhau

```text
             Quỳnh Anh
        Reliability Signals
                │
                │ scores[N]
                ▼
          ┌─────────────┐
          │   D1 Eval   │
          └─────────────┘
                ▲
                │ labels
                │
             Trường Thịnh
      Geometric Benchmark
                │
                ▼
          Tuấn Khải
        Evaluation Pipeline
```

Output cuối cùng:

```text
3DGS checkpoint
       │
       ├───────────────┐
       ▼               ▼
 Reliability       Reference
   Signal           Geometry
       │               │
       ▼               ▼
   score_i        m_i / f_i / d_i
       │               │
       └───────┬───────┘
               ▼
          D1 Evaluation
               │
               ▼
      quantitative results
```

---

# 5. Quy định về file nộp

Mỗi bạn khi nộp task phải có 3 phần:

## A. Research / technical document

Trả lời:

> Tôi đang làm gì và tại sao?

## B. Code

Trả lời:

> Tôi đã implement như thế nào?

## C. Experimental evidence

Trả lời:

> Kết quả cho thấy điều gì?

Không chỉ nộp code mà không có explanation/evidence; cũng không chỉ nộp report nếu task yêu cầu implementation.

---

# 6. Format báo cáo kết quả

Mỗi experiment nên ghi:

```text
Experiment:
Scene:
Condition:
Seed:
Checkpoint:
Method:
Parameter:
Metric:

Result:

Observation:

Interpretation:

Limitation:

Next action:
```

Đặc biệt:

### Observation ≠ Interpretation

Ví dụ:

> Observation: GradMag có SNR_in = 4.71.

Không tự viết:

> Therefore GradMag is faithful.

Nên viết:

> Interpretation: GradMag tạo intervention response cao hơn noise floor trong Phase 0 dưới protocol hiện tại.

Sau đó đề xuất experiment tiếp theo để kiểm chứng.

---

# 7. Git / submission

Branch riêng:

```text
feature/quynhanh-signals
feature/truongthinh-benchmark
feature/tuankhai-evaluation
```

> Ghi chú: bản phân công gốc ghi `feature/tuankhai-benchmark` và `feature/truongthinh-evaluation`
> (đảo tên). Đã sửa theo đúng track: Trường Thịnh làm benchmark, Tuấn Khải làm evaluation.
> Vị trí file trong repo: xem bảng "Ai làm gì" trong README.md.

Commit message:

```text
feat: implement gradient magnitude signal
feat: add geometric failure labels
feat: add D1 AUPRC evaluation
fix: correct reference coordinate transform
docs: document signal implementation
```

Không commit trực tiếp vào `main`.

---

# 8. Definition of Done

Một task chỉ được coi là hoàn thành khi có:

```text
Research explanation
+
Implementation
+
Test
+
Experimental output
+
Documentation
+
Reproducible command
```

Nếu gặp blocker:

```text
1. Vấn đề
2. Error/log
3. Đã thử gì
4. Nguyên nhân dự kiến
5. Hướng xử lý đề xuất
```

Không để blocker kéo dài nhiều ngày mà không báo.
