# Phân tích Phase 0 và cập nhật D2 v2.1

**Phạm vi:** Phase 0 trên cảnh synthetic, 8 training views, 16 held-out views, ảnh 160×160,
K = 5%, removal và re-optimization 1000 bước. Phase 0 có 3 seed (0–2).

**Nguồn:** [Phase 0 report](../../results/phase0/report.md),
[D2 v2.1 v8_az180](../../results/d2v21_v8_az180/report.md),
[D2 v2.1 v8_az360](../../results/d2v21_v8_az360/report.md).

Các số Phase 0 dưới đây là mean ± standard deviation qua ba seed. Tài liệu nguồn không báo
cáo kiểm định giả thuyết cho các so sánh được nêu.

## 1. Kết quả Phase 0

| Signal | SNR_in | Gain_LR (3D) | Gain_LR (ảnh) | Gain_Dir (1−SSIM) | Gain_Dir (local CD) | AUROC vs. floater |
|---|---:|---:|---:|---:|---:|---:|
| Opacity | 1.10 ± 0.05 | +0.046 ± 0.016 | +0.029 ± 0.008 | −0.00005 ± 0.00016 | +0.00005 ± 0.00004 | 0.349 ± 0.020 |
| GradMag | 4.71 ± 0.38 | +0.094 ± 0.019 | +0.011 ± 0.007 | +0.00007 ± 0.00007 | −0.00013 ± 0.00007 | 0.654 ± 0.010 |
| Visibility | 1.77 ± 0.57 | +0.002 ± 0.047 | −0.007 ± 0.017 | +0.00178 ± 0.00155 | +0.00069 ± 0.00081 | 0.355 ± 0.012 |
| Random control | 2.69 ± 0.18 | — | — | — | — | — |

Gain_Dir dương biểu thị chiều thay đổi tốt hơn theo định nghĩa báo cáo; các chỉ số chiều
trong bảng là paired gains so với random. Random Precision@5% là 8.3% ± 0.5%, còn GradMag
Precision@5% là 10.8% ± 0.9%.

### 1.1 Response và đối chứng

**Quan sát.** GradMag có SNR_in trên ngưỡng quy ước 2 ở cả ba seed. Opacity và Visibility
không đạt ngưỡng ở mọi seed. Random control cũng vượt ngưỡng nhiễu. Precision@5% trung bình
của GradMag cao hơn random trong số liệu mô tả.

**Diễn giải.** Trong các tín hiệu được thử, removal theo GradMag tạo response đo được trong
pilot này. Chênh lệch Precision@5% là mô tả trên ba seed; chưa có kiểm định thống kê, nên
chưa thể gọi là khác biệt có ý nghĩa thống kê. SNR cho biết response vượt noise floor theo
quy tắc đã chọn, không tự chứng minh signal chọn đúng primitive lỗi.

### 1.2 Locality

**Quan sát.** Gain_LR (3D param) của GradMag là +0.094 ± 0.019. Gain_LR ảnh là
+0.011 ± 0.007; LR ảnh tuyệt đối là 0.9127 ± 0.0044 với GradMag và 0.9013 ± 0.0062 với
random. Mặt nạ footprint rộng khiến LR ảnh của cả hai selection đều cao.

**Diễn giải.** Số liệu gợi ý dịch chuyển tham số tập trung gần selection GradMag hơn random
trong cấu hình pilot. Chênh lệch locality ảnh nhỏ và không đủ để kết luận locality ảnh phân
biệt tốt giữa các signal. Các metric này cũng không cho biết các primitive được chọn là
floater hay cấu trúc hợp lệ.

### 1.3 Chiều thay đổi trên ảnh held-out

**Quan sát.** Gain_Dir (1−SSIM) của GradMag có mean +0.00007 ± 0.00007. Dấu và độ lớn của
hiệu ứng không nhất quán qua ba seed. ΔPSNR +0.125 dB được báo cáo cho seed 0; đó không phải
trung bình qua seed. Gain_Dir (L1) là +0.00002 ± 0.00005.

**Diễn giải.** Các số liệu này chưa đủ để kết luận removal GradMag cải thiện chất lượng ảnh
held-out một cách nhất quán. Kết quả seed 0 có thể được nêu như một quan sát riêng của seed
đó, không dùng làm kết luận trung bình.

### 1.4 Hình học cục bộ và các giả thuyết về GradMag

**Quan sát.** Gain_Dir (local CD) của GradMag là −0.00013 ± 0.00007, theo chiều ngược với
việc cải thiện local Chamfer so với random. Metric gộp Phase 0 không tách riêng accuracy và
completeness.

**Diễn giải.** Số liệu cho thấy local CD sau removal GradMag không cải thiện so với random
trong pilot. Chúng chưa xác định nguyên nhân. Việc GradMag có thể chọn cạnh, bề mặt chi tiết,
hay các thanh mảnh là giả thuyết cần kiểm chứng bằng thống kê overlap không gian giữa
selection và từng vùng cấu trúc.

Phân tích DCT hiện có chỉ cho thấy vùng được GradMag chọn nhỉnh hơn toàn vật thể khoảng
1–5%; mức này chưa ủng hộ giả thuyết rằng signal ưu tiên rõ các vân tần số cao. Giả thuyết
về các thanh mảnh cũng chưa được kiểm chứng. Vì vậy không quy nguyên nhân local CD âm cho
texture, cạnh hoặc rods nếu chưa có phép đo trực tiếp.

### 1.5 Phạm vi kết luận Phase 0

Phase 0 cho thấy protocol có thể ghi nhận response lớn hơn noise floor cho một số selection
trong cấu hình đã thử. Kết quả chưa xác nhận GradMag là signal hình học đáng tin cậy: D1 là
sanity check với nhãn floater tạm, còn chiều local CD đi ngược kỳ vọng. Không suy rộng các
quan sát pilot thành kết luận chung về 3DGS hoặc mọi điều kiện camera.

## 2. Cập nhật từ D2 protocol v2.1

D2 v2.1 bổ sung mask theo blending weight (primitive được tính vào vùng ảnh khi đóng góp ít
nhất 20% blending weight của pixel; cấu hình đặt kích thước vùng tối thiểu 1%), năm random
selection cho mỗi seed, các snapshot tại 0/200/1000 bước, và báo cáo riêng local accuracy
và local completeness. Đây là các refinement đã chạy, không còn là đề xuất cho Phase 0.

Các số direction trong bảng dưới đây được báo cáo theo đơn vị ×10⁻³. Nhãn E/O là quy ước
nhất quán dấu của báo cáo D2 v2.1, không phải kiểm định ý nghĩa thống kê.

| Condition | Signal | SNR_in @1000 | Gain_LR ảnh | Gain_LR 3D | Gain_Dir held-out (1−SSIM) | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---:|---:|---:|---:|---:|---:|
| v8_az360 | GradMag | 5.5 ± 0.5 | +0.012 ± 0.005 (E) | +0.093 ± 0.013 (E) | −0.023 ± 0.228 (không nhất quán) | −0.015 ± 0.140 (không nhất quán) | −0.239 ± 0.074 (O) |
| v8_az180 | GradMag | 5.6 ± 0.1 | −0.028 ± 0.006 (O) | +0.112 ± 0.014 (E) | −0.254 ± 0.156 (O) | −0.164 ± 0.148 (không nhất quán) | −0.288 ± 0.143 (O) |

Trong báo cáo v8_az360, phần mô tả ghi seeds 0–5; config đi kèm liệt kê seeds 0–4. Cần
đối chiếu lại run artifacts trước khi chốt số lượng seed của aggregate này. Báo cáo v8_az180
và config đều ghi seeds 0–2.

### 2.1 Quan sát

- GradMag vượt ngưỡng SNR ở cả hai điều kiện theo báo cáo.
- Gain_LR 3D dương ở cả hai điều kiện, còn Gain_LR ảnh dương ở v8_az360 nhưng âm ở
  v8_az180.
- Ở v8_az360, held-out và local accuracy không có dấu nhất quán; local completeness mang
  dấu ngược kỳ vọng theo quy ước báo cáo.
- Ở v8_az180, held-out và local completeness mang dấu ngược kỳ vọng; local accuracy biến
  thiên qua seed.
- Báo cáo v2.1 phân loại GradMag là measurable nhưng “stated meaning not supported” ở cả
  hai điều kiện. Visibility có SNR cao nhưng báo cáo kết luận chưa đủ bằng chứng theo chiều
  response; Opacity không measurable.

### 2.2 Diễn giải

v2.1 cho thấy measurable response và các metric direction/locality trả lời những câu hỏi
khác nhau. Locality 3D dương không đồng nghĩa removal cải thiện hình học; completeness âm
cho thấy độ phủ tham chiếu cục bộ kém hơn theo metric được dùng. Kết quả giữa 180° và 360°
cũng khác nhau, nên chưa có cơ sở để mô tả một response ảnh/locality chung cho mọi mức
angular coverage.

Các metric accuracy/completeness giúp tách sai số gần bề mặt với thiếu độ phủ, nhưng chưa
chỉ ra primitive nào thuộc thanh mảnh hoặc vì sao chúng bị chọn. Cần phân tích overlap với
các vùng cấu trúc trước khi dùng thin rods như lời giải thích.

**Kết luận có thể rút ra:** D2 v2.1 đo được response GradMag trong các điều kiện báo cáo,
nhưng các metric hiện có không ủng hộ cách diễn giải removal GradMag là cải thiện hình học
nhất quán. “D2: GO” là kết luận về tính đo được của protocol theo gate đã đặt, không phải
xác nhận GradMag faithful.

## 3. Các kiểm tra còn lại

- Đối chiếu số seed v8_az360 giữa config, run files và report.
- Tuấn Khải kiểm tra D1 với f_i lịch sử. Nếu cần thêm f_i_nogate để so sánh opacity, trước
  hết Huy cần duyệt thay đổi data contract; chưa tạo key mới trong archive theo contract hiện tại.
- Giữ các giả thuyết texture/edge/thin rods ở mức giả thuyết cho đến khi có phân tích overlap
  trực tiếp.
- Nếu dùng D1 m_i, chỉ dùng file nhãn có m_i_available=true; nhãn tạm hoặc artifact chưa
  xác minh geometry phải được đánh dấu provisional.
