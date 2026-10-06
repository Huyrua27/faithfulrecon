# Data format — hợp đồng giữa các task

File này là **điểm nối** giữa 3 task. Ai đổi format phải mở PR sửa file này trước và được
Huy duyệt; code chỉ được merge khi khớp với file này.

```text
checkpoint (base.pt)
   ├── Quỳnh Anh  → scores.npz            (score_i cho từng signal)
   ├── Trường Thịnh → geometric_labels.npz (xyz, distance, m_i, f_i)
   └── Tuấn Khải  → đọc 2 file trên → metrics.csv + figures/
```

## 1. Quy ước chung

| Quy ước | Giá trị |
|---|---|
| `gaussian_id` | chỉ số hàng trong checkpoint (`model.means[i]`), 0..N−1. Không sắp xếp lại. |
| Hướng của score | **score cao = KÉM tin cậy hơn** (được chọn trước khi xoá / top-K). |
| N | mọi file của cùng một run phải có cùng N, khớp với checkpoint. |
| Hệ toạ độ | world frame của Gaussian model (camera OpenCV: x phải, y xuống, z tới trước). |
| Đơn vị độ dài | đơn vị world của scene (synthetic: vật thể nằm trong khoảng [−1, 1]³). |
| Kiểu dữ liệu | score `float32`, label `bool`, xyz/distance `float32`. Không NaN/Inf. |

## 2. Thư mục một run

```text
runs/d1/<scene>/<condition>/seed_<s>/
├── checkpoint.json          # {"path": ".../base.pt", "sha256": "...", "n_gaussians": N}
├── scores.npz               # Quỳnh Anh
├── geometric_labels.npz     # Trường Thịnh
└── signal_results.csv       # (tuỳ chọn) bản CSV của scores.npz, theo yêu cầu task
```

- `<scene>`: ví dụ `synthetic`, `lego`.
- `<condition>`: cấu hình view, ví dụ `v8_az360` (8 view train, phủ 360°).
- `scripts/prepare_d1_run.py` tạo thư mục này từ một run Phase 0 có sẵn
  (`checkpoint.json` + `scores.npz` + nhãn **tạm** `geometric_labels.npz`, xem mục 4).

## 3. `scores.npz` (Quỳnh Anh)

| Key | Shape / kiểu | Nội dung |
|---|---|---|
| `<signal_name>` | (N,) float32 | một key cho mỗi signal: `opacity`, `gradmag`, `visibility`, `gradient_inconsistency`, `random` |
| `meta` | object (dict) | `scene`, `condition`, `seed`, `checkpoint_sha256`, `n_gaussians`, `signal_params` (dict theo signal) |

CSV tương đương (`signal_results.csv`, dạng long): `scene,condition,seed,signal,gaussian_id,score`.
File này lớn (N × số signal hàng) → để trong `runs/`, **không commit**.

## 4. `geometric_labels.npz` (Trường Thịnh)

| Key | Shape / kiểu | Nội dung |
|---|---|---|
| `xyz` | (N, 3) float32 | tâm Gaussian `mu_i` |
| `distance` | (N,) float32 | `d_i`: khoảng cách tới bề mặt tham chiếu |
| `m_i` | (N,) bool | điều kiện under-observed (chỉ phụ thuộc camera + bề mặt tham chiếu) |
| `f_i` | (N,) bool | kết cục geometric failure |
| `meta` | object (dict) | `scene`, `condition`, `seed`, `checkpoint_sha256`, `n_gaussians`, `label_params`, `m_i_available` (bool), `provisional` (bool) |

Nhãn tạm do `prepare_d1_run.py` tạo: `f_i` và `distance` theo code Phase 0
(`measure.floater_labels`, τ = 0.03); **chưa có `m_i`** (`m_i_available = False`, `m_i` toàn False).
Tuấn Khải phải bỏ qua `m_i` khi `m_i_available = False`. Trường Thịnh thay bằng nhãn chính thức.

## 5. `metrics.csv` (Tuấn Khải)

Một hàng cho mỗi (scene, condition, seed, signal, label):

```text
scene,condition,seed,signal,label,prevalence,auroc,auprc,spearman,
precision_at_1,precision_at_5,precision_at_10,enrichment_at_1,enrichment_at_5,enrichment_at_10,
random_auroc_mean,random_auprc_mean,random_precision_at_5_mean
```

- `label` ∈ {`f_i`, `m_i`}; `spearman` tính với `distance` (giống nhau cho cả hai label).
- `precision_at_k`, `enrichment_at_k`: k theo % (1, 5, 10).
- Các cột `random_*`: trung bình trên R lần random có cùng scene/condition/seed (matched baseline).

## 6. Trạng thái dữ liệu hiện có

| Nguồn | Có gì | Ghi chú |
|---|---|---|
| `runs/phase0/seed_{0,1,2}/` | `base.pt`, `signals.pt`, run tối ưu lại, `metrics.json` | Phase 0 (scene synthetic, 8 view, 3 seed). Không commit (máy Huy) — xin file nếu cần. |
