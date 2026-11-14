# Khảo sát dataset cho RQ1/RQ2: tiêu chí và quy trình kiểm tra

**Ngày lập:** 12/11/2026. **Người thực hiện:** [điền tên]. **Hạn:** [điền ngày].
**Bối cảnh:** `docs/research/report_problem_rw/report.pdf` (mục 2.1, 5.1).

## 1. Mục đích

Đề tài định lượng floater trong 3DGS khi số góc nhìn hạn chế và các ảnh huấn luyện sai lệch trắc quang hoặc chiếu sáng.
Thí nghiệm hiện tại dùng DTU với nhiễu ISP tổng hợp (gain, white balance, gamma). Nhiễu này **không mô phỏng thay đổi
chiếu sáng vật lý**: bóng đổ và vùng phản xạ không thay đổi theo vị trí nguồn sáng. Khảo sát nhằm chọn:

1. một nguồn **chiếu sáng vật lý** có hình học tham chiếu (ưu tiên trong DTU);
2. một dataset thứ hai để kiểm tra khả năng tổng quát của kết quả.

Mọi thông số ghi trong báo cáo khảo sát phải được **kiểm tra trực tiếp trên dữ liệu đã tải về**, không chỉ dựa vào bài báo
hoặc trang web của dataset.

## 2. Tiêu chí

### 2.1. Tiêu chí bắt buộc (không đạt một tiêu chí thì loại)

| # | Tiêu chí | Ngưỡng đạt | Cách kiểm tra |
|---|---|---|---|
| B1 | **Có hình học tham chiếu (GT):** mesh hoặc đám mây điểm quét, cùng hệ toạ độ với camera | Có GT; sai lệch khi chiếu GT lên ảnh ≤ 2 px (ở độ phân giải huấn luyện) | Chiếu GT lên 3 view bất kỳ; chồng lên mask và cạnh ảnh (mục 3, bước 3) |
| B2 | **Số góc nhìn mỗi vật thể/cảnh** | ≥ 24 view, đủ cho tối đa 8 view huấn luyện và ≥ 16 view kiểm định cố định | Đếm file ảnh theo từng vật thể; kiểm tra không trùng view |
| B3 | **Hình học bất biến giữa các điều kiện sáng:** vật và camera không bị dịch chuyển, hoặc tư thế camera của mọi điều kiện sáng nằm trong cùng một hệ toạ độ | Cùng hệ toạ độ; GT dùng chung cho mọi điều kiện | Đọc quy trình thu dữ liệu; chiếu cùng một GT lên ảnh của 2 điều kiện sáng khác nhau |
| B4 | **Biết điều kiện sáng của từng ảnh:** chỉ số đèn, tên môi trường sáng, hoặc thông số phơi sáng (EXIF) | Mỗi ảnh có nhãn điều kiện sáng | Kiểm tra cấu trúc thư mục, tên file, metadata |
| B5 | **Calibration khả dụng:** nội tham số, ngoại tham số | Có; không có lỗi căn chỉnh đã được báo cáo mà chưa sửa | Tìm issue trên GitHub và các bài báo sau có dùng dataset |

### 2.2. Tiêu chí ưu tiên (dùng để xếp hạng các dataset đạt B1–B5)

| # | Tiêu chí | Điểm 0 / 1 / 2 |
|---|---|---|
| U1 | Có mask vật thể | không / một phần / mọi ảnh |
| U2 | Mức độ điều khiển chiếu sáng | chỉ "khác nhau" / có nhãn rời rạc / có tham số liên tục (vị trí đèn, EV) |
| U3 | Độ phủ góc nhìn (phân bố camera quanh vật) | một phía / nửa vòng / toàn vòng |
| U4 | Độ hoàn chỉnh của GT (lỗ thủng, vùng bị che) | nhiều lỗ / ít lỗ / gần kín |
| U5 | Được dùng trong tài liệu sparse-view hoặc 3DGS | không / ít / phổ biến |
| U6 | Khả thi trên Colab và Drive (dung lượng, giấy phép, thời gian huấn luyện) | khó / được / dễ |
| U7 | Đa dạng vật liệu (khuếch tán, phản xạ) | một loại / vài loại / đa dạng |

## 3. Quy trình kiểm tra cho mỗi dataset

1. **Đọc tài liệu:** bài báo gốc, trang dataset, README, license. Ghi lại các thông số *theo tài liệu*.
2. **Tải một vật thể/cảnh** (không tải toàn bộ). Ghi dung lượng của một vật thể và của toàn bộ dataset.
3. **Kiểm tra căn chỉnh GT:** chiếu điểm GT lên ảnh bằng tham số camera đi kèm. Lưu hình chồng lớp cho 3 view và 2 điều kiện sáng.
   Đo sai lệch biên (px) giữa hình chiếu GT và mask.
4. **Đếm và phân loại ảnh:** số view, số điều kiện sáng mỗi view, độ phân giải, định dạng (8-bit, HDR, RAW).
5. **Kiểm tra B3:** cùng một GT có khớp ảnh của mọi điều kiện sáng không.
6. **Thử nạp vào pipeline:** viết hàm nạp tối thiểu (camera, ảnh, mask, GT) theo định dạng `frecon/data/dtu.py`; huấn luyện 1 run
   ngắn (1000 vòng) để kiểm tra không lỗi.
7. **Ghi kết quả** theo mẫu ở mục 5. Phân biệt rõ thông số *đã kiểm tra trên dữ liệu* và thông số *chỉ theo tài liệu*.

## 4. Danh sách ứng viên ban đầu

Trạng thái: **[XM]** = đã xác minh qua tài liệu ngày 12/11/2026; **[CKT]** = cần kiểm tra trên dữ liệu.

### 4.1. DTU, ảnh gốc Rectified (ưu tiên 1)

- **[XM]** 49 hoặc 64 vị trí camera; 7 điều kiện chiếu sáng, từ đèn có hướng tới ánh sáng khuếch tán; GT là đám mây điểm quét
  bằng ánh sáng cấu trúc. Nguồn: trang DTU MVS 2014; Aanæs et al., IJCV 2016.
- **Cách dùng dự kiến:** gán cho mỗi view huấn luyện một điều kiện sáng khác nhau, tạo sai lệch chiếu sáng vật lý với cùng hình
  học và cùng GT. So sánh trực tiếp với nhiễu ISP tổng hợp trên cùng scan.
- **[CKT]**
  - Bản tiền xử lý IDR/NeuS đang dùng có chứa đủ 7 điều kiện sáng không. Nhiều khả năng là không, khi đó cần tải bộ Rectified gốc.
  - Dung lượng bộ Rectified; quy ước tên file theo view và điều kiện sáng.
  - Ma trận camera của bộ Rectified có khớp với `cameras_sphere.npz` đang dùng không.
  - Mức độ khác nhau giữa 7 điều kiện sáng trên scan 24, 63, 83 (minh hoạ bằng hình).
  - Lỗ thủng của GT: completeness cần lọc theo vùng quan sát được.

### 4.2. DiLiGenT-MV

- **[XM]** 5 vật thể (BEAR, BUDDHA, COW, POT2, READING); 20 view; 96 đèn điểm đã hiệu chỉnh mỗi view; GT là mesh quét
  (`mesh_Gt.ply`); ảnh 612×512, vật thể chiếm tối đa khoảng 400×400 px; khoảng 6.85 GB. Nguồn: trang DiLiGenT-MV; Li et al. 2020.
- **Rủi ro [XM]:** NPL-MVPS (arXiv 2405.12057) báo cáo GT normal và mask không khớp với ảnh kết xuất từ GT mesh khi dùng tham số
  camera đi kèm, và có một số ảnh bị hỏng.
- **[CKT]**
  - Kiểm tra căn chỉnh GT (B1, B5) là bắt buộc.
  - 20 view không đạt B2: chỉ đủ khoảng 8 view huấn luyện và 12 view kiểm định. Cần quyết định có chấp nhận ngoại lệ không.
  - Đèn điểm trong phòng tối tạo bóng gắt; cần chọn tập đèn hợp lý.

### 4.3. Stanford-ORB (NeurIPS 2023)

- **[XM]** 14 vật thể thực trong 7 cảnh in-the-wild; GT quét 3D; ảnh đa góc nhìn; ánh sáng môi trường GT. Nguồn: arXiv 2310.16044.
- **[CKT]**
  - Số view mỗi vật thể (B2), chưa xác minh được.
  - Vật thể được đặt trong các cảnh khác nhau: tư thế camera giữa các cảnh có cùng hệ toạ độ với GT không (B3).
  - Định dạng ảnh (HDR/LDR), có mask không.

### 4.4. OpenIllumination (NeurIPS 2023)

- **[XM]** 64 vật thể; 48 camera DSLR × 13 kiểu chiếu sáng; 20 vật thể thêm 24 camera tốc độ cao × 142 đèn OLAT; có tham số
  camera, chiếu sáng GT và mask; vật thể kích thước 10–20 cm, chụp trong phòng tối. Nguồn: arXiv 2309.07921.
- **[CKT]** Có hình học GT hay không (B1). Nếu không có thì loại khỏi thí nghiệm chính.

### 4.5. Nhóm có GT nhưng không điều khiển chiếu sáng (chỉ dùng với nhiễu tổng hợp)

Tanks and Temples, ETH3D, ScanNet++. Chưa kiểm tra thông số. Chỉ xem xét nếu cần kiểm tra khả năng tổng quát của kết quả nhiễu ISP
trên cảnh lớn hơn.

### 4.6. Dữ liệu tổng hợp (tuỳ chọn)

Kết xuất cảnh có mesh GT bằng Blender với nhiều vị trí đèn. Điều khiển chiếu sáng hoàn toàn, GT chính xác tuyệt đối; nhược điểm là
khoảng cách với ảnh thực. Chỉ cân nhắc nếu không dataset thực nào đạt B1–B5.

## 5. Mẫu báo cáo cho mỗi dataset

```text
Dataset:                      Phiên bản / link tải:            Giấy phép:
Ngày kiểm tra:                Người kiểm tra:
Dung lượng: 1 vật thể ___ GB; toàn bộ ___ GB

B1 GT hình học:     [Đạt/Không]  dạng ___ ; sai lệch chiếu ___ px (view ___, ___, ___)   [đã kiểm tra / theo tài liệu]
B2 Số view:         [Đạt/Không]  ___ view/vật thể; phân bố ___                          [đã kiểm tra / theo tài liệu]
B3 Hình học bất biến:[Đạt/Không] ___                                                     [đã kiểm tra / theo tài liệu]
B4 Nhãn chiếu sáng: [Đạt/Không]  ___ điều kiện; dạng nhãn ___                            [đã kiểm tra / theo tài liệu]
B5 Calibration:     [Đạt/Không]  lỗi đã báo cáo: ___                                     [đã kiểm tra / theo tài liệu]

U1–U7: ___ / 14 điểm (ghi điểm từng mục)
Nạp vào pipeline: [Được/Lỗi] ___ ; run thử 1000 vòng: ___
Hình đính kèm: chồng GT lên ảnh (3 view × 2 điều kiện sáng); 1 lưới ảnh các điều kiện sáng của cùng một view
Kết luận: [Dùng / Dùng có điều kiện / Loại] — lý do: ___
```

## 6. Sản phẩm nộp

- `docs/analysis/dataset_survey.md`: một mục theo mẫu ở mục 5 cho mỗi dataset đã kiểm tra, kèm bảng xếp hạng cuối.
- `docs/analysis/figures/dataset_survey/`: hình chồng GT và lưới điều kiện sáng.
- Nếu chọn DTU Rectified: thêm hàm nạp ảnh theo điều kiện sáng trong `frecon/data/dtu.py`, kèm test.

## 7. Quy tắc

- Không ghi một thông số là "đã kiểm tra" nếu chỉ đọc từ tài liệu.
- Mỗi kết luận Đạt/Không phải kèm bằng chứng: hình, số liệu, hoặc dòng tài liệu (mục, trang).
- Không tải toàn bộ dataset trước khi một vật thể mẫu đạt B1–B5.
- Kiểm tra scan24 (việc P-1) cần hoàn tất trước khi kết luận việc đổi dataset, vì lỗi pipeline sẽ lặp lại trên mọi dataset.

## 8. Nguồn

- DTU MVS 2014: https://roboimagedata.compute.dtu.dk/?page_id=36 ; Aanæs et al., IJCV 2016, doi:10.1007/s11263-016-0902-9
- DiLiGenT-MV: https://sites.google.com/site/photometricstereodata/mv ; Li et al., arXiv:2001.06659
- NPL-MVPS (báo cáo lỗi căn chỉnh DiLiGenT-MV): arXiv:2405.12057
- Stanford-ORB: arXiv:2310.16044 ; https://stanfordorb.github.io
- OpenIllumination: arXiv:2309.07921 ; https://oppo-us-research.github.io/OpenIllumination
