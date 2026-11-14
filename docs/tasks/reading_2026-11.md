# Đọc paper: floater khi ít ảnh và ánh sáng không đồng nhất

**Giao ngày:** 10/11/2026. **Hạn:** thứ 6, 13/11 cho 2 bài cốt lõi; thứ 3, 17/11 cho phần còn lại và bảng tổng hợp.
Bối cảnh: đọc trước mục 1, 6, 7 của `docs/research/lit_review_floater_anh_sang.md`. Số trong ngoặc vuông là số tài liệu ở mục 8 của file đó.

Mục tiêu: trả lời chính xác **người ta làm gì, đo bằng gì, còn thiếu gì**, để nhóm chọn phương pháp đối chứng và chốt
thiết kế thí nghiệm. Không viết tổng quan chung.

## Phân công

| Người | Hướng | Bài cốt lõi (hạn 13/11) | Bài còn lại (hạn 17/11) |
|---|---|---|---|
| Trường Thịnh | Hướng 1: floater khi ít ảnh, cách đo hình học | CoR-GS [5], SparseGS [4] | DNGaussian [2], FSGS [3], Floaters No More [7], RegNeRF [21] (chỉ phần giao thức DTU) |
| Tuấn Khải | Hướng 2: mô hình hoá ánh sáng / ngoại hình | Bilateral Guided RF [11], VastGaussian [10] (chỉ phần decoupled appearance) | NeRF-W [8], WildGaussians [9], Unifying Appearance Codes and Bilateral Grids [12], Luminance-GS [13] |
| Quỳnh Anh | Giao của hai bài toán | MS-GS [17], SparseGS-W [18] | Sparse-View 3DGS in the Wild [19], WildSplatter [20], HeroGS [6] |


## Câu hỏi riêng theo hướng

**Thịnh (hướng 1)**
- Floater sinh ra do đâu theo tác giả? Họ chặn bằng ràng buộc gì (độ sâu, co-regularization, cắt tỉa, gradient)?
- Họ có đo floater **bằng hình học** không (Chamfer, accuracy/completeness, khoảng cách tới bề mặt), hay chỉ PSNR/SSIM/LPIPS và hình minh hoạ?
- Giao thức DTU: scan nào, bao nhiêu view, view nào, có dùng mask khi tính chỉ số không? (so với RegNeRF [21])
- Có giả định ánh sáng nhất quán không? Nếu ảnh lệch sáng thì phương pháp có nguy cơ gì?

**Khải (hướng 2)**
- Module bù ánh sáng có dạng gì (embedding theo ảnh, affine màu, bilateral grid, CNN), gắn ở đâu (trên Gaussian hay trên ảnh render)?
- Có cơ chế nào ngăn module **hấp thụ cả nội dung hình học** không (regularization, giới hạn bậc tự do)? Tác giả có kiểm tra không?
- Khi đánh giá trên ảnh kiểm định, họ chọn ngoại hình nào (tối ưu lại embedding trên nửa ảnh, dùng ảnh tham chiếu…)? Cách này có công bằng không?
- Chi phí cài vào code của nhóm: số tham số thêm cho mỗi ảnh, có code công khai không, phụ thuộc thư viện gì.

**Quỳnh Anh (giao của hai bài toán)**
- Thiết lập: bao nhiêu ảnh, dữ liệu gì, ánh sáng lệch là thật hay tổng hợp?
- Có thí nghiệm tách riêng tác động của **ít ảnh** và của **lệch sáng** không (ablation)?
- Có đo hình học hoặc floater không, hay chỉ chất lượng ảnh?
- Dùng prior nặng nào (mô hình sinh, depth model, feature 2D)? Chạy trên DTU 8 ảnh được không?
- Với [6], [19], [20]: kiểm tra đã được hội nghị nào nhận chưa, và ghi lại.

## Mẫu ghi chú cho mỗi bài (1–2 trang)

```text
1. Thông tin: tác giả, năm, hội nghị (đã kiểm tra trên trang hội nghị/arXiv?), link code
2. Bài toán và giả định chính
3. Ý tưởng phương pháp: 3–5 dòng + 1 công thức quan trọng nhất (ghi số phương trình)
4. Xử lý floater / ánh sáng như thế nào
5. Đánh giá: dữ liệu, số view, chỉ số ảnh, chỉ số hình học (có/không), baseline
6. Kết quả liên quan đến nhóm (vài con số, ghi bảng/trang)
7. Hạn chế: (a) tác giả tự nêu, (b) nhóm nhận thấy
8. Liên quan tới RQ1/RQ2/RQ3 của nhóm; dùng làm baseline được không?
```

## Nộp

- `docs/research/reading/<Ten>_<paper>.md`, mỗi bài một file theo mẫu trên.
- `docs/research/reading/<Ten>_summary.md`: một bảng so sánh các bài trong hướng của mình, cột theo các câu hỏi
  riêng ở trên, và **3 câu hỏi mở** mà các bài chưa trả lời.
- Thứ 3, 17/11: mỗi bạn trình bày 10 phút trong buổi họp nhóm.

## Quy tắc

- Mọi khẳng định về paper phải kèm vị trí: mục, trang, bảng hoặc phương trình.
- Tách rõ **paper nói gì** và **nhóm nhận xét gì**.
- Không chép nguyên đoạn văn; tóm tắt bằng lời của mình.
- Không dùng "chứng minh", "đầu tiên", "state-of-the-art" nếu paper không có số liệu cho điều đó.
- Không đọc thêm bài ngoài danh sách nếu chưa trả lời xong các câu hỏi trên. Nếu gặp bài quan trọng, ghi tên vào summary.
