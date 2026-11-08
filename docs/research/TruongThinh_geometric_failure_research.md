# Geometric Failure, Benchmark Labels và Intervention Analysis

**Phụ trách:** Trường Thịnh

**Phạm vi thực nghiệm:** synthetic `v8_az360`, seeds 0–2

**Trạng thái:** hoàn thành cho phạm vi synthetic; labels được đánh dấu `provisional`

**Code:** `frecon/benchmark/`, `scripts/make_labels.py`

**Phân tích liên quan:** [Phase 0 và D2 v2.1](../analysis/phase0_analysis.md)

## 1. Bài toán

Mục tiêu là xây dựng nhãn hình học độc lập với reliability signal để trả lời hai câu hỏi:

1. Gaussian có nằm trong vùng thiếu quan sát từ camera huấn luyện hay không?
2. Gaussian đã trở thành một kết quả hình học lỗi hay chưa?

Hai câu hỏi được biểu diễn bằng condition label `m_i` và outcome label `f_i`. Việc tách hai
nhãn cho phép D1 kiểm tra signal đang theo dõi điều kiện có thể dẫn tới failure hay chỉ theo dõi
outcome đã quan sát được.

```text
camera + reference surface ──> m_i (under-observed condition)
trained Gaussian + reference ──> d_i ──> f_i (geometric-failure outcome)
```

Báo cáo sử dụng cảnh procedural synthetic gồm sphere, box và ba thin rods. Cấu hình có 8
training views phủ 360°, 16 held-out views, độ phân giải 160×160 và ba checkpoint tương ứng
seeds 0, 1, 2. Phạm vi thực nghiệm không bao gồm kết quả Blender hoặc DTU.

## 2. Under-observed condition `m_i`

Mỗi Gaussian center `mu_i` được ánh xạ tới điểm tham chiếu gần nhất `x_j`. Với mỗi điểm tham
chiếu, implementation tính số training views nhìn thấy điểm đó và góc lớn nhất giữa mọi cặp
viewing rays hợp lệ:

```text
m_i = (visible_view_count_j < k_views)
      OR (max_triangulation_angle_j < theta_deg)
```

Thực nghiệm dùng `k_views = 2` và `theta_deg = 10°`. Một điểm tham chiếu được xem là nhìn
thấy khi nằm trước camera, chiếu vào trong khung hình và, nếu có normal, mặt hướng về camera
với dot product lớn hơn `0.05`. Phép kiểm tra hiện chưa có depth test hoặc ray casting, nên
không phát hiện điểm bị bề mặt khác che khuất. Vì vậy `m_i` là visibility proxy dựa trên
frustum và normal, chưa phải occlusion-aware visibility hoàn chỉnh.

Góc giữa views được tính vector hóa. Viewing directions có tensor `(V, M, 3)`; tích vô hướng
giữa các cặp view tạo tensor `(V, V, M)`. Những cặp không đồng thời nhìn thấy điểm được mask,
sau đó cosine nhỏ nhất được đổi sang góc lớn nhất. Với tối đa 8 views, cách này tránh vòng lặp
Python qua 60.000 reference samples.

`m_i` phụ thuộc camera, reference surface và phép ánh xạ Gaussian sang điểm tham chiếu gần
nhất. Nhãn không sử dụng reliability score, opacity, held-out error hoặc `f_i`. Nó mô tả điều
kiện quan sát, không tự khẳng định Gaussian đã sai hình học.

## 3. Geometric-failure outcome `f_i`

Continuous geometric error của Gaussian `i` là khoảng cách Euclidean từ tâm Gaussian đến điểm
tham chiếu gần nhất:

```text
d_i = min_j ||mu_i - x_j||_2
```

Outcome label hiện tại là:

```text
f_i = (d_i > 0.03)
      AND (heldout_contribution_i > 0.01)
      AND (opacity_i > 0.05)
```

`heldout_contribution_i` là tổng contribution của Gaussian trên 16 held-out views. Điều kiện
contribution loại các center xa surface nhưng không ảnh hưởng ảnh held-out; opacity gate giữ
định nghĩa floater lịch sử của Phase 0.

Opacity gate tạo confound khi đánh giá signal opacity: Gaussian có opacity thấp dễ được signal
chọn nhưng lại bị loại khỏi positive set của `f_i`. Biến thể bỏ gate, `f_i_nogate`, chưa được
ghi vào archive vì [data contract](../data_format.md) hiện chỉ quy định `m_i` và `f_i`. Việc
thêm label này cần Huy duyệt contract và Tuấn Khải cập nhật D1. Báo cáo không trình bày kết quả
D1 cho `f_i_nogate`.

### Hệ quả đối với đánh giá D1

**Quan sát.** Trong kết quả D1 provisional của `v8_az360` seeds 0–2, top 5% Gaussian được
opacity-risk chọn đều có opacity dưới `0.014`, thấp hơn ngưỡng `0.05` trong định nghĩa `f_i`.
Do đó Precision@5% của opacity bằng 0 ở cả ba seed. Chi tiết được ghi trong
[README của D1 provisional](../../results/d1/phase0_provisional_2026-10-17/synthetic/v8_az360/README.md).

**Diễn giải.** Precision@5% bằng 0 chịu ảnh hưởng trực tiếp của opacity gate và không đủ để kết
luận riêng rằng opacity signal không liên quan đến geometric failure. Kết quả này cần được báo
cáo cùng định nghĩa label; việc đánh giá biến thể không gate chỉ được thực hiện sau khi
`f_i_nogate` được duyệt trong data contract và tích hợp vào D1.

## 4. Reference geometry và hệ tọa độ

Synthetic reference được dựng lại từ toàn bộ scene configuration, gồm object list, số điểm,
texture frequency, seed và camera parameters. Scene loader và reference loader sinh cùng tensor
điểm từ cùng config; script dừng nếu hai tensor khác shape hoặc khác giá trị.

| Thuộc tính | Giá trị |
|---|---|
| Scene | procedural synthetic: sphere, box, rods |
| Requested reference samples | 60.000 |
| Reference samples thực tế | 59.997, do phân bổ theo diện tích và làm tròn |
| Coordinate frame | world frame của Gaussian model |
| Đơn vị | scene world unit; vật thể nằm xấp xỉ trong `[-1, 1]^3` |
| Camera convention | OpenCV: x phải, y xuống, z hướng trước |

Không có coordinate transform bổ sung cho synthetic vì checkpoint, camera và reference được
tạo trong cùng world frame. `check_alignment` báo 59.997/59.997 reference points nằm trong
frustum của từng training view. Đây chỉ là frustum coverage; nó không độc lập kiểm chứng scale,
rotation, translation hoặc depth occlusion. Nguồn geometry đồng nhất từ procedural config cung
cấp bằng chứng provenance mạnh hơn coverage check, nhưng output vẫn được đánh dấu `provisional`.

Code có nhánh lấy Blender reference từ `ref_points_path` và lấy DTU reference từ scene loader.
Hai nhánh này không thuộc phạm vi thực nghiệm của báo cáo và chưa được xác nhận end-to-end.

## 5. Continuous geometric error

Khoảng cách được tính bằng `torch.cdist` theo hai tầng chunk: tối đa 512 Gaussian centers và
8.192 reference samples mỗi block. Với khoảng 28.000 Gaussian và 59.997 điểm tham chiếu, cách
này tránh tạo ma trận khoảng cách đầy đủ có hơn 1,6 tỷ phần tử.

`d_i` là khoảng cách tới point sample gần nhất, không phải khoảng cách giải tích tới mesh. Sai số
xấp xỉ phụ thuộc mật độ và phân bố reference samples. Threshold `tau = 0.03` lớn hơn phần lớn
nearest-sample spacing trong scene hiện tại nhưng chưa có threshold sweep để đánh giá độ nhạy.

## 6. Kết quả benchmark labels

| Condition | Seed | N Gaussians | `m_i` positive | `f_i` positive | Median `d_i` | P95 `d_i` |
|---|---:|---:|---:|---:|---:|---:|
| v8_az360 | 0 | 27.733 | 1.160 (4,18%) | 2.329 (8,40%) | 0,00850 | 0,07565 |
| v8_az360 | 1 | 27.984 | 1.401 (5,01%) | 2.531 (9,04%) | 0,00868 | 0,07643 |
| v8_az360 | 2 | 28.691 | 1.396 (4,87%) | 2.467 (8,60%) | 0,00873 | 0,07506 |
| Mean ± SD | — | — | **4,68 ± 0,44%** | **8,68 ± 0,33%** | **0,00864 ± 0,00012** | **0,07571 ± 0,00069** |

### Quan sát

- Prevalence của `m_i` nằm trong khoảng 4,18–5,01%; prevalence của `f_i` nằm trong khoảng
  8,40–9,04%.
- Hai nhãn có prevalence khác nhau ở cả ba seed. Bản đồ không gian cho thấy `m_i` tập trung
  ở các reference regions có camera geometry yếu, trong khi `f_i` phân bố rộng hơn trên các
  center xa surface nhưng còn đóng góp lên held-out views.
- Median `d_i` ổn định giữa seeds, khoảng 0,0085–0,0087 world unit; p95 khoảng 0,075–0,076.
- Figure seed 0 cho thấy nhóm `f_i=1` nằm phía trên threshold 0,03 theo định nghĩa; nhóm
  `m_i=1` có phân phối distance rộng hơn nhóm `m_i=0`.

### Diễn giải

Kết quả phù hợp với mục tiêu tách condition và outcome: under-observation không đồng nghĩa với
geometric failure, và geometric failure có thể xuất hiện ngoài positive set của condition proxy.
Chênh lệch prevalence không tự chứng minh quan hệ nhân quả giữa hai nhãn. Muốn đo quan hệ đó cần
bảng giao nhau, conditional prevalence hoặc mô hình thống kê riêng.

Figures cho mỗi seed nằm tại:

- `results/benchmark/v8_az360/figures/seed_<s>/ref_vs_gaussians.png`
- `results/benchmark/v8_az360/figures/seed_<s>/under_observed_map.png`
- `results/benchmark/v8_az360/figures/seed_<s>/geometric_failure_map.png`
- `results/benchmark/v8_az360/figures/seed_<s>/error_distribution.png`

Archive `geometric_labels.npz` giữ đúng contract gồm `xyz`, `distance`, `m_i`, `f_i`, `meta`.
Mọi array có cùng N, `xyz`/`distance` là `float32`, labels là bool và không có NaN/Inf. Hash
checkpoint trong manifest, archive và summary khớp ở cả ba seed.

## 7. Intervention hypothesis

Intervention D2 kiểm tra một signal bằng quy trình:

```text
rank signal -> chọn Top-K -> remove -> re-optimize -> đo response
random cùng K, scene, condition và seed -> remove -> re-optimize -> matched comparison
```

Giả thuyết là primitive do signal chọn có ảnh hưởng nhân quả cục bộ phù hợp với ý nghĩa signal:
response phải đo được trên noise floor, tập trung gần vùng selection và có chiều cải thiện kỳ
vọng so với matched random. Một signal chỉ có bằng chứng thuận lợi khi measurability, locality
và direction được đọc cùng nhau.

Phase 0 dùng K = 5%, removal, 1.000 re-optimization steps, hai random selections mỗi seed và
hai null runs. D2 v2.1 tăng lên năm random selections mỗi seed và đo ở step 0/200/1000.

## 8. Locality criterion

Image locality dùng `LR_image = d_in / (d_in + d_out)`, trong đó `d_in` và `d_out` là thay
đổi ảnh trung bình trong và ngoài vùng selection. Phase 0 dùng projected footprint mask; D2
v2.1 dùng mask theo blending weight, với primitive được tính vào pixel khi chiếm ít nhất 20%
blending weight.

3D locality dùng cùng dạng tỷ số trên displacement của surviving Gaussian centers nằm gần và xa
selection, với neighborhood radius 0,05 world unit.

Trong Phase 0, GradMag có `Gain_LR 3D = +0,094 ± 0,019`, còn `Gain_LR image = +0,011 ±
0,007`. LR image tuyệt đối của GradMag và random đều gần 0,9, cho thấy footprint mask quá rộng
để phân biệt tốt. D2 v2.1 đã triển khai narrower blending-weight mask; kết quả chi tiết nằm trong
[phase0_analysis.md](../analysis/phase0_analysis.md).

Locality dương chỉ cho biết response tập trung gần selection hơn random; nó không cho biết
selection là primitive lỗi hoặc removal cải thiện geometry.

## 9. Direction criterion

Direction được định nghĩa là âm của thay đổi error: giá trị dương nghĩa là removal cải thiện
error hoặc gây hại ít hơn. Paired gain là direction của signal trừ mean direction của random
trong cùng seed.

Phase 0 đo held-out L1, `1−SSIM`, PSNR và local Chamfer distance. D2 v2.1 tách local geometry
thành accuracy và completeness. Với GradMag, Phase 0 cho `Gain_Dir (1−SSIM) = +0,00007 ±
0,00007`, không nhất quán qua seed; `Gain_Dir local CD = −0,00013 ± 0,00007`, ngược chiều
cải thiện geometry. `ΔPSNR = +0,125 dB` là quan sát riêng của seed 0, không phải mean ba seed.

D2 v2.1 tiếp tục cho thấy GradMag measurable nhưng local completeness ngược kỳ vọng ở cả
`v8_az360` và `v8_az180`. Vì các metric không cùng chiều, kết quả không hỗ trợ kết luận removal
GradMag cải thiện geometry một cách nhất quán.

## 10. Noise-floor criterion

Mỗi intervention được so với `null_a`, một run re-optimization không removal có cùng view order.
`null_b`, cũng không removal nhưng dùng view order khác, ước lượng noise floor. Measurability dùng:

```text
SNR_in = response_inside_region(intervention vs null_a)
         / response_inside_region(null_b vs null_a)
```

Ngưỡng quy ước là `SNR_in >= 2` ở mọi seed. Đây là gate pilot, không phải kiểm định thống kê.
Trong Phase 0, GradMag đạt `4,71 ± 0,38`; Opacity đạt `1,10 ± 0,05`; Visibility đạt `1,77 ±
0,57`; random control đạt `2,69 ± 0,18`. Chỉ GradMag trong các signal đạt gate ở mọi seed.

Response trên noise floor cho thấy intervention tạo ảnh hưởng đo được. Nó không tự xác nhận
signal faithful hoặc primitive được chọn là geometric failure.

## 11. Tổng hợp Phase 0 và D2 v2.1

Phase 0 trả về GO vì ít nhất một signal tạo response measurable và có paired gain nhất quán theo
gate pilot. GO xác nhận protocol có khả năng đo response; nó không xác nhận GradMag faithful.

D2 v2.1 đã triển khai hai refinement từng được đề xuất: mask ảnh hẹp hơn theo blending weight và
tách local accuracy khỏi local completeness. Ở step 1000, GradMag vẫn có SNR khoảng 5,5–5,6 và
Gain_LR 3D dương, nhưng held-out direction không ổn định hoặc ngược kỳ vọng, còn completeness
âm ở cả hai camera conditions. Bằng chứng cho thấy GradMag chọn primitive có ảnh hưởng, nhưng
ý nghĩa “removal cải thiện geometric failure” chưa được ủng hộ nhất quán.

Giải thích GradMag ưu tiên cạnh, texture tần số cao hoặc thin rods vẫn là giả thuyết. DCT chỉ cho
thấy vùng GradMag nhỉnh hơn toàn vật thể khoảng 1–5%, chưa đủ hỗ trợ ưu tiên mạnh high-frequency
texture; thin-structure overlap chưa được đo trực tiếp.

## 12. Validation

Bộ test benchmark chạy trên Google Colab với Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0
và Tesla T4. Kết quả: **16 tests passed trong 14,35 giây**.

Tests bao phủ khoảng cách tới reference samples; view count, triangulation angle, normal-facing
visibility và nearest-reference mapping; threshold semantics của `f_i`; archive keys/shape/dtype;
invalid inputs; checkpoint relocation và SHA-256 mismatch; condition/config mismatch; scores
identity; full synthetic config reconstruction; và end-to-end label generation trên small scene.

Log nằm tại `results/benchmark/v8_az360/logs/benchmark_tests.txt`; môi trường tại
`results/benchmark/v8_az360/logs/environment.json`.

## 13. Giới hạn và refinement

| Vấn đề | Vì sao quan trọng | Hướng xử lý | Trạng thái |
|---|---|---|---|
| `m_i` chưa depth-occlusion-aware | Có thể đếm điểm bị che khuất là visible | Render depth reference hoặc ray-cast | Chưa triển khai; labels provisional |
| Alignment check chỉ đo frustum coverage | Không phát hiện sai scale/pose | Thêm landmark, bounding-box và depth diagnostics | Chưa triển khai độc lập |
| Distance tới sampled points | Sampling density ảnh hưởng `d_i` gần threshold | Tăng density hoặc point-to-mesh; sensitivity study | Chưa chạy sweep |
| Opacity gate trong `f_i` | Confound khi đánh giá opacity signal | Lưu thêm `f_i_nogate`, đánh giá riêng | Đã duyệt; `make_labels.py` lưu `f_i_nogate`, `frecon/eval/io.py` đọc được |
| Phase 0 chỉ có ba seed | Không đủ cho claim thống kê mạnh | Thêm seeds và uncertainty/test phù hợp | Chưa thực hiện |
| Footprint mask quá rộng | LR image cao cả với random | Dùng blending-weight mask | Đã triển khai trong D2 v2.1 |
| Local CD gộp accuracy/completeness | Che khuất trade-off | Báo cáo hai thành phần riêng | Đã triển khai trong D2 v2.1 |
| Edge/texture/thin-rod explanation | Dễ biến giả thuyết thành kết luận | Đo overlap theo structure category | Chưa kiểm chứng |

## 14. Reproduction

```bash
python -m pytest tests/test_benchmark_labels.py -q

python scripts/make_labels.py --run_dir ../data/d1/synthetic/v8_az360/seed_0 --checkpoint ../data/seed_0/base.pt --config configs/phase0.yaml --device cuda
python scripts/make_labels.py --run_dir ../data/d1/synthetic/v8_az360/seed_1 --checkpoint ../data/seed_1/base.pt --config configs/phase0.yaml --device cuda
python scripts/make_labels.py --run_dir ../data/d1/synthetic/v8_az360/seed_2 --checkpoint ../data/seed_2/base.pt --config configs/phase0.yaml --device cuda
```

| Seed | Checkpoint SHA-256 |
|---:|---|
| 0 | `95bb31a8830d1ca8ab1cadbc65b2c192014b0bbf533b3dd9a1c6b3fc4605c517` |
| 1 | `2483c9a07e7e20fd54834d289cf0afa7cee77b69dad0df2176af78e832d202e5` |
| 2 | `10bc6f76ed740546fede09d37a0e12661dfdc0a7d761422f0befe19367b857d1` |

Config SHA-256: `1deebe0220dbc484a08415ddc65e842d83c04ab19d02edd9f32ff10d082f498d`.

## 15. Kết luận

Pipeline đã tạo thành công continuous error `d_i`, condition label `m_i`, outcome label `f_i`
và bốn loại visualization cho ba checkpoint synthetic. `m_i` có prevalence `4,68 ± 0,44%`,
trong khi `f_i` có prevalence `8,68 ± 0,33%`, phù hợp với việc hai nhãn biểu diễn hai khái
niệm khác nhau. Archive và identity checks đạt yêu cầu, 16 tests pass.

Kết quả intervention cho thấy GradMag tạo response measurable và có 3D locality cao hơn random,
nhưng direction geometry không nhất quán với giả thuyết cải thiện. Bằng chứng chưa xác nhận
GradMag faithful, chưa xác minh quan hệ nhân quả giữa `m_i` và `f_i`, và chưa khái quát sang
Blender/DTU hoặc camera conditions khác.

## Tài liệu tham khảo

1. Kerbl et al., [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079), 2023.
2. [FaithfulRecon project overview](../../README.md).
3. [Phase 0 protocol](../phase0.md) và [Phase 0 results](../../results/phase0/report.md).
4. [D2 v2.1 — v8_az360](../../results/d2v21_v8_az360/report.md) và [v8_az180](../../results/d2v21_v8_az180/report.md).
5. [Data contract](../data_format.md).
