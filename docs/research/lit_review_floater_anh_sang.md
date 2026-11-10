# Tổng quan tài liệu: Floater trong 3D Gaussian Splatting khi ít ảnh và ánh sáng không đồng nhất giữa các ảnh

> Nhóm FaithfulRecon · bản nháp ngày 03/11/2026
> Phạm vi khảo sát: các công trình về 3DGS/NeRF công bố đến tháng 11/2026, tìm theo các từ khoá
> *floater, sparse view / few-shot, appearance variation, exposure, illumination, in the wild,
> bilateral grid*. Mỗi tài liệu đều có đường dẫn gốc ở Mục 8. Những điểm mới đọc qua tóm tắt,
> chưa đọc toàn văn, được ghi rõ ở cột "Mức kiểm tra".

---

## 1. Bài toán

### 1.1. Giả định nhất quán quang trắc của 3DGS

3D Gaussian Splatting (3DGS) [1] biểu diễn cảnh bằng một tập Gaussian $G$. Mỗi Gaussian có vị
trí, hình dạng, độ mờ và màu phụ thuộc hướng nhìn. Mô hình được tối ưu để ảnh dựng lại khớp với
từng ảnh huấn luyện:

$$\min_G \sum_{k=1}^{K} \ell\big(R(G,\pi_k),\, I_k\big)$$

trong đó $R(G,\pi_k)$ là ảnh dựng từ camera $\pi_k$ và $I_k$ là ảnh chụp thứ $k$.

Công thức này **ngầm giả định** rằng một điểm 3D có cùng màu ở mọi ảnh, chỉ khác nhau do hướng
nhìn. Trong thực tế, ảnh thứ $k$ là $I_k = f_k\big(\text{cảnh dưới ánh sáng } L_k\big)$, với:

- $f_k$: pipeline xử lý của camera — phơi sáng tự động, cân bằng trắng, tone mapping;
- $L_k$: điều kiện chiếu sáng — hướng đèn, cường độ, thời điểm chụp.

Khi $f_k$ hoặc $L_k$ thay đổi giữa các ảnh, mô hình không có biến nào để "giải thích" sự khác biệt.
Cách duy nhất để giảm loss là **thay đổi chính hình học**: thêm những Gaussian chỉ khớp với một vài
ảnh cụ thể.

### 1.2. Floater và hai nguyên nhân

**Định nghĩa vận hành.** Floater là Gaussian nằm cách bề mặt thật một khoảng đáng kể nhưng vẫn
đóng góp vào ảnh dựng. Khi xem từ góc nhìn mới, chúng hiện ra thành mảng mờ "lơ lửng" trước vật
thể.

| Nguyên nhân | Cơ chế | Vì sao ít ảnh làm nặng thêm |
|---|---|---|
| **(N1) Thiếu ràng buộc hình học** | Với ít ảnh, nhiều cấu hình Gaussian khác nhau đều tái tạo đúng các ảnh đã thấy. Gaussian đặt sai độ sâu không bị phạt. | Càng ít góc nhìn, càng ít ràng buộc chéo giữa các ảnh |
| **(N2) Ánh sáng không đồng nhất giữa các ảnh** | Sai khác quang trắc giữa các ảnh bị hấp thụ bởi những Gaussian riêng cho từng góc nhìn. BilaRF [11] và VastGaussian [10] đều nêu rõ cơ chế này là nguồn gốc của floater. | Với ít ảnh, mỗi ảnh chiếm tỉ trọng lớn trong loss, và không có nhiều ảnh khác để "bỏ phiếu" chống lại sai khác của một ảnh |

Hai nguyên nhân **cộng hưởng**: N1 tạo ra không gian nghiệm rộng, N2 đẩy tối ưu vào những nghiệm sai
trong không gian đó.

### 1.3. Bằng chứng sơ bộ của nhóm

Trên 8 scan DTU với 8 ảnh huấn luyện, ánh sáng nhất quán (nghĩa là chỉ có N1):

- khoảng 51% Gaussian ở scan24 cách bề mặt thật hơn 0,1 đơn vị chuẩn hoá (7–60% giữa các scan);
- ít nhất một nửa số đó nằm **trước** bề mặt;
- chúng không phải màu nền, 97% nằm trong bóng vật thể, độ mờ tương đương Gaussian thật.

Nghĩa là ngay cả **khi chưa có N2**, floater đã chiếm tỉ lệ lớn. Câu hỏi mở là N2 làm tình trạng này
nặng thêm bao nhiêu.

---

## 2. Hướng tiếp cận 1: chống floater khi ít ảnh

### 2.1. Ràng buộc độ sâu

- **DNGaussian** (CVPR 2024) [2]
  - Cho rằng hình học suy giảm khi ít ảnh là do *vị trí* của Gaussian, và có thể giảm nhờ ràng buộc
    độ sâu.
  - Dùng độ sâu từ mạng ước lượng đơn ảnh, với ràng buộc cứng và mềm, cộng chuẩn hoá độ sâu toàn cục
    và cục bộ.
  - Thử trên LLFF, DTU, Blender.
- **FSGS** (ECCV 2024) [3]
  - Thêm Gaussian vào giữa các Gaussian lân cận theo đồ thị gần kề (proximity-guided unpooling).
  - Sinh góc nhìn ảo và dùng độ sâu đơn ảnh làm ràng buộc trên cả ảnh thật lẫn ảnh ảo.
  - Thử trên LLFF, Mip-NeRF360, Blender.
- **SparseGS** (3DV 2025) [4] — công trình duy nhất trong nhóm này **nhắm thẳng vào floater**.
  - Ngoài ràng buộc độ sâu, còn có bước cắt tỉa: với mỗi ảnh huấn luyện, xét histogram chênh lệch độ
    sâu. Ảnh nhiều floater có histogram hai đỉnh, và Gaussian ở đỉnh "độ sâu sai" bị xoá.

**Điểm chung và hạn chế:**
- Cả ba dựa vào **độ sâu đơn ảnh**, vốn có sai số riêng và chỉ đúng đến một hằng số tỉ lệ.
- Cả ba **ngầm giả định ánh sáng nhất quán** giữa các ảnh.

### 2.2. Tự nhất quán giữa nhiều mô hình

- **CoR-GS** (ECCV 2024) [5]
  - Huấn luyện **hai** trường Gaussian song song trên cùng tập ảnh.
  - Gaussian mà hai mô hình bất đồng về vị trí bị coi là sai và bị xoá (co-pruning).
  - Điểm ảnh mà hai mô hình dựng khác nhau bị phạt trên các góc nhìn giả (co-regularization).
  - Tác giả cho thấy mức bất đồng tương quan nghịch với độ chính xác tái tạo.
  - Hạn chế: chi phí huấn luyện gấp đôi. Nếu cả hai mô hình cùng bị lệch bởi một nguyên nhân hệ
    thống như N2, sự bất đồng có thể không phát hiện ra lỗi chung đó (đây là suy luận của nhóm,
    chưa được kiểm chứng).
- **HeroGS** (arXiv 2026) [6]: co-pruning dựa trên tính tự nhất quán của trường Gaussian, kết hợp chiến
  lược đóng băng.

### 2.3. Điều chỉnh gradient và tiêu chí cắt tỉa

- **Floaters No More** (EGSR 2023) [7], trên NeRF
  - Floater gần camera là do vùng gần camera được lấy mẫu dày hơn, nên nhận nhiều gradient hơn.
  - Giải pháp là thu nhỏ gradient theo khoảng cách tới camera. Chỉ vài dòng code, không tốn thêm chi
    phí.
- **Tiêu chí nhân bản và cắt tỉa của 3DGS gốc** [1]: dựa trên độ lớn gradient vị trí và độ mờ.
  - Kết quả D2 của nhóm: loại bỏ theo độ lớn gradient làm **giảm độ phủ bề mặt ở 8/8 scan DTU**, dù
    ảnh kiểm định lại tốt hơn. Tức là tiêu chí này chọn nhầm Gaussian đang mang bề mặt.
- **Cắt tỉa để nén** (LightGaussian, PUP 3D-GS): giữ chất lượng ảnh khi bớt Gaussian, không nhằm
  phát hiện floater. Thuộc khảo sát trước của nhóm, chỉ nhắc để phân biệt.

### Bảng tổng hợp hướng 1

| Công trình | Cơ chế | Cần thêm gì | Nhắm floater trực tiếp | Giả định ánh sáng |
|---|---|---|---|---|
| DNGaussian [2] | Ràng buộc độ sâu | Mạng độ sâu đơn ảnh | Gián tiếp | Nhất quán |
| FSGS [3] | Thêm Gaussian + độ sâu + góc nhìn ảo | Mạng độ sâu đơn ảnh | Gián tiếp | Nhất quán |
| SparseGS [4] | Độ sâu + cắt tỉa theo histogram độ sâu | Mạng độ sâu đơn ảnh | **Có** | Nhất quán |
| CoR-GS [5] | Hai mô hình, xoá chỗ bất đồng | Gấp đôi chi phí | **Có** | Nhất quán |
| Floaters No More [7] | Thu nhỏ gradient gần camera | Không | **Có** (NeRF) | Nhất quán |

---

## 3. Hướng tiếp cận 2: xử lý ánh sáng không đồng nhất

### 3.1. Mã ngoại hình theo từng ảnh

- **NeRF-W** (CVPR 2021) [8]
  - Mỗi ảnh có một vector ngoại hình học được, để "giải thích" sai khác về phơi sáng, ánh sáng, thời
    tiết và hậu xử lý.
  - Thêm một trường riêng cho vật thể tạm thời (người qua đường), kèm độ bất định.
  - Đây là nền tảng cho hầu hết các công trình sau.
- **WildGaussians** (NeurIPS 2024) [9]
  - Kết hợp vector ngoại hình theo ảnh với vector theo từng Gaussian.
  - Một MLP nhỏ sinh biến đổi màu affine (hệ số nhân và hệ số cộng cho từng kênh), có thể "nướng"
    ngược vào màu SH để render như 3DGS chuẩn.
  - Xử lý vật che bằng độ bất định dựa trên đặc trưng DINOv2.
  - Đánh giá bằng PSNR, SSIM, LPIPS. **Không đánh giá floater hay hình học.**
- **GS-W** (ECCV 2024): điều kiện hoá ngoại hình bằng đặc trưng CNN từ ảnh tham chiếu. Nhóm mới đọc
  qua mô tả trong các bài khác.
- **VastGaussian** (CVPR 2024) [10] — **nêu thẳng mối liên hệ ánh sáng và floater**.
  - Sai khác ngoại hình giữa các ảnh khiến 3DGS sinh ra các khối Gaussian hiện thành floater ở góc nhìn
    mới.
  - Giải pháp *tách ngoại hình*: một CNN nhỏ biến đổi ảnh dựng theo vector ngoại hình của từng ảnh.
  - Chỉ dùng khi huấn luyện, bỏ đi khi render.

### 3.2. Mô hình hoá pipeline xử lý ảnh của camera

- **BilaRF** (SIGGRAPH 2024) [11]
  - Mỗi ảnh có một **bilateral grid** — một lưới các biến đổi affine cục bộ — xấp xỉ phơi sáng và tone
    mapping, tối ưu cùng trường bức xạ.
  - Tác giả báo cáo việc này tránh được floater.
  - Lưới huấn luyện 8×8×4 hoặc 16×16×8. Tác giả thừa nhận nếu không có ràng buộc trơn (TV), lưới
    **hấp thụ cả nội dung thật của cảnh**, ví dụ bóng đổ trên tường.
  - **Floater chỉ được chứng minh bằng hình minh hoạ.** Định lượng chỉ có PSNR, SSIM, LPIPS trên 7
    cảnh ban đêm do nhóm tác giả tự thu.
- **Unifying Appearance Codes and Bilateral Grids** (arXiv 2025) [12]
  - Grid đa tỉ lệ: ở tỉ lệ thô tương đương một mã ngoại hình toàn cục, ở tỉ lệ mịn là bilateral grid
    theo điểm ảnh.
  - Tác giả báo cáo giảm floater do sai lệch quang trắc, trên dữ liệu lái xe.
- **Luminance-GS** (CVPR 2025) [13]
  - Ánh sáng khó: thiếu sáng, cháy sáng, phơi sáng thay đổi giữa các ảnh.
  - Dùng ma trận màu theo từng ảnh, cộng một đường cong chỉnh độ sáng: đường cong chung cho mọi ảnh
    và phần lệch riêng của từng ảnh. Tối ưu cùng 3DGS.

### 3.3. Ngoài phạm vi: phản xạ và bóng loáng

Spec-Gaussian (NeurIPS 2024) [14], GaussianShader (CVPR 2024) [15] và 3DGS-DR (SIGGRAPH 2024) [16] xử lý
hiệu ứng **phụ thuộc hướng nhìn** của vật liệu, như bóng loáng hay phản xạ môi trường. Đây là hiện
tượng *nhất quán* với vật lý của cảnh, không phải sai khác giữa các ảnh. Nhóm không xét hướng này,
nhưng ghi nhận hai điểm:

- Spec-Gaussian dùng huấn luyện thô-đến-tinh để giảm floater do quá khớp.
- MS-GS [17] thừa nhận phương pháp của họ làm mờ vùng bóng loáng, vì ưu tiên nhất quán đa góc nhìn.

### Bảng tổng hợp hướng 2

| Công trình | Mô hình ánh sáng | Đối tượng dữ liệu | Đánh giá floater | Đánh giá hình học |
|---|---|---|---|---|
| NeRF-W [8] | Mã ngoại hình theo ảnh (MLP) | Ảnh du lịch, nhiều ảnh | Không | Không |
| WildGaussians [9] | Affine màu từ mã ảnh + mã Gaussian | Ảnh du lịch, nhiều ảnh | Không | Không |
| VastGaussian [10] | CNN biến đổi ảnh dựng | Cảnh lớn, nhiều ảnh | Hình minh hoạ | Không |
| BilaRF [11] | Bilateral grid theo ảnh | Ảnh điện thoại ban đêm, 20–93 ảnh | Hình minh hoạ | Không |
| Luminance-GS [13] | Ma trận màu + đường cong | Thiếu sáng, cháy sáng, phơi sáng thay đổi | Chưa kiểm tra | Chưa kiểm tra |

---

## 4. Giao của hai bài toán: ít ảnh **và** ánh sáng không đồng nhất

Giao này **đã có người làm**, nhưng mới gần đây:

- **MS-GS** (NeurIPS 2025) [17]
  - Tập ảnh thưa với nhiều ngoại hình (ngày/đêm, mùa, thời tiết).
  - Tách vector ngoại hình theo ảnh khỏi đặc trưng chuẩn theo Gaussian. Căn chỉnh độ sâu đơn ảnh theo
    từng vùng do SAM phân đoạn để khởi tạo dày. Ràng buộc hình học trên góc nhìn ảo.
  - Dữ liệu: Mip-NeRF360 thưa (20 ảnh), Phototourism thưa (20 ảnh), một bộ ảnh drone tự thu (5 ảnh
    mỗi ngoại hình).
  - Đánh giá: PSNR, SSIM, LPIPS, DreamSim.
  - **Floater chỉ được bàn định tính.** Hình học chỉ có một kiểm tra độ sâu trên một cảnh ở phụ lục.
- **SparseGS-W** (arXiv 2025) [18]
  - Từ 5 ảnh. Dùng prior hình học và prior khuếch tán (diffusion) có ràng buộc để tăng cường góc nhìn
    mới, inpainting để xoá vật che.
  - Đánh giá bằng chỉ số ảnh có tham chiếu và không tham chiếu (FID, ClipIQA, MUSIQ). Phần tóm tắt
    không nhắc tới floater.
- **Sparse-View 3DGS in the Wild** (arXiv 2026) [19]: thêm góc nhìn giả và nhân bản Gaussian theo mức
  thưa, tập trung vào vật che.
- **WildSplatter** (arXiv 2026) [20]: mô hình feed-forward, dựng Gaussian kèm mã ngoại hình từ vài ảnh
  không cần pose, dưới một giây.

**Nhận xét:**
- Các công trình này dựa vào **prior nặng** (độ sâu đơn ảnh, SAM, khuếch tán) và nhắm tới **ảnh
  ngoài trời**, nơi không có bề mặt thật để đo.
- Vì vậy không thể biết floater được **loại bỏ thật** hay chỉ được **che đi** nhờ prior làm ảnh đẹp
  hơn.

---

## 5. Cách đánh giá hiện nay

| Cách đánh giá | Dùng ở đâu | Nhược điểm đối với bài toán floater |
|---|---|---|
| PSNR, SSIM, LPIPS trên ảnh kiểm định | Hầu hết các công trình | Không đo vị trí Gaussian; ảnh có thể đẹp lên trong khi bề mặt bị thủng (kết quả D2 của nhóm) |
| PSNR có che mặt nạ vật thể trên DTU (giao thức RegNeRF [21]) | Các công trình ít ảnh | Vẫn là chỉ số ảnh; các bài dùng phiên bản DTU và tập kiểm tra khác nhau [21] |
| PSNR sau căn chỉnh affine (BilaRF), LPIPS/DreamSim (MS-GS) | Ánh sáng không đồng nhất | Cố ý bỏ qua sai khác màu, nên không phân biệt được "ánh sáng đúng" với "floater bù màu" |
| Hình minh hoạ floater | VastGaussian, BilaRF, MS-GS | Định tính, khó so sánh |
| Chamfer trên DTU (2DGS, ...) | Tái tạo bề mặt, **nhiều ảnh, ánh sáng nhất quán** | Chưa được dùng cho bài toán ít ảnh + lệch sáng |

---

## 6. Khoảng trống

**G1 — Chưa đo được tác động của ánh sáng lên floater.**
- Nhiều bài *khẳng định* sai lệch quang trắc gây floater [10, 11, 12], nhưng chỉ minh hoạ bằng hình.
- Chưa có thí nghiệm **có kiểm soát** cho thấy floater tăng bao nhiêu theo mức lệch sáng, đo bằng
  hình học thật.
- Lý do: các bộ dữ liệu có ánh sáng thay đổi (Phototourism, ảnh ban đêm, ảnh drone) không có bề mặt
  tham chiếu, còn DTU thường chỉ được dùng ở một điều kiện sáng.
- Trong khi đó, DTU gốc chụp **mỗi vị trí camera dưới 7 điều kiện ánh sáng** và có bề mặt quét bằng
  ánh sáng cấu trúc [22].

**G2 — Chưa biết các mô-đun bù ánh sáng lấy đi bao nhiêu nội dung thật.**
- BilaRF thừa nhận lưới có thể hấp thụ bóng đổ thật nếu không ràng buộc [11].
- MS-GS thừa nhận làm mờ vùng bóng loáng [17].
- Sự đánh đổi giữa "hấp thụ ánh sáng" và "làm mất hình học hoặc chi tiết thật" chưa được đo trực tiếp.

**G3 — Tiêu chí loại bỏ floater chưa được kiểm chứng bằng hình học, và chưa xét ánh sáng.**
- SparseGS và CoR-GS xoá Gaussian theo độ sâu hoặc theo bất đồng giữa hai mô hình, nhưng được đánh
  giá bằng chỉ số ảnh và giả định ánh sáng nhất quán.
- Kết quả D2 của nhóm cho thấy một tiêu chí có thể làm ảnh tốt hơn mà vẫn xoá nhầm bề mặt.

**G4 — Các giải pháp hiện có cho giao hai bài toán dựa vào prior nặng.**
- Độ sâu đơn ảnh, SAM, mô hình khuếch tán [17, 18]: tốn tài nguyên, và làm khó tách bạch đóng góp của
  từng thành phần.
- Chưa có một phương án nhẹ, chỉ dùng thông tin đa góc nhìn có sẵn trong chính mô hình.

---

## 7. Định vị đề tài

**Câu hỏi nghiên cứu:**

- **RQ1 (ứng với G1).** Khi ít ảnh, ánh sáng không đồng nhất giữa các ảnh làm tăng floater bao nhiêu,
  đo bằng bề mặt thật?
- **RQ2 (ứng với G2).** Bù ánh sáng theo ảnh (affine hoặc bilateral grid) giảm được bao nhiêu floater
  do N2, và có làm mất độ phủ hay chi tiết thật không?
- **RQ3 (ứng với G3, G4).** Cắt tỉa theo mức hỗ trợ đa góc nhìn — số view nhìn thấy và tính nhất quán
  độ sâu giữa các view — có loại được floater mà không làm thủng bề mặt không? So với ràng buộc độ sâu
  đơn ảnh (DNGaussian, SparseGS) và co-pruning (CoR-GS) thì sao?

**Cơ sở để kỳ vọng (từ kết quả sơ bộ của nhóm):**
- Tín hiệu số view phát hiện Gaussian lệch bề mặt với AUROC trung bình 0,88 trên 8 scan.
- Loại bỏ theo số view cải thiện cả độ chính xác lẫn độ phủ ở 8/8 scan (p = 0,031 sau hiệu chỉnh
  Holm).
- Cả hai đo trong điều kiện ánh sáng nhất quán. Chưa có kết quả nào trong điều kiện lệch sáng.

**Thiết kế thí nghiệm dự kiến:**
- **Dữ liệu:** DTU gốc với 7 điều kiện ánh sáng. Mỗi ảnh huấn luyện được gán một điều kiện theo các
  mức lệch tăng dần, từ cùng một điều kiện đến ngẫu nhiên hoàn toàn.
- **Phương pháp đối chứng:** 3DGS; DNGaussian; CoR-GS; SparseGS; 3DGS kèm bilateral grid; 3DGS kèm
  affine màu theo ảnh.
- **Thước đo:**
  - ảnh: PSNR, SSIM, LPIPS;
  - hình học: độ chính xác và độ phủ tách riêng;
  - floater: tỉ lệ Gaussian cách bề mặt hơn τ và tỉ lệ nằm trước bề mặt.

**Việc cần làm tiếp cho phần tổng quan:**
1. Đọc toàn văn [4], [5], [10], [11], [17], ghi lại chính xác cách họ đánh giá floater.
2. Kiểm tra hội nghị công bố của [6], [12], [18], [19] và của GS-W.
3. Kiểm tra bản DTU Rectified (7 điều kiện sáng) có khớp camera với bản tiền xử lý đang dùng không.
4. Tìm thêm các công trình về phơi sáng HDR (HDR-NeRF, HDR-GS) và về giao thức đánh giá floater, nếu
   có.

---

## 8. Tài liệu tham khảo


| # | Tài liệu | Mức |
|---|---|---|
| [1] | Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG (SIGGRAPH) 2023 | C |
| [2] | Li et al., *DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization*, CVPR 2024. [link](https://openaccess.thecvf.com/content/CVPR2024/html/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.html) | B |
| [3] | Zhu et al., *FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting*, ECCV 2024. [link](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5583_ECCV_2024_paper.php) | B |
| [4] | Xiong et al., *SparseGS: Sparse View Synthesis using 3D Gaussian Splatting*, 3DV 2025. [link](https://arxiv.org/html/2312.00206v3) | B |
| [5] | Zhang et al., *CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization*, ECCV 2024. [link](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/139_ECCV_2024_paper.php) | B |
| [6] | *HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views*, arXiv 2026. [link](https://www.alphaxiv.org/abs/2603.01099) | C |
| [7] | Philip & Deschaintre, *Floaters No More: Radiance Field Gradient Scaling for Improved Near-Camera Training*, EGSR 2023. [link](https://arxiv.org/abs/2305.02756) | B |
| [8] | Martin-Brualla et al., *NeRF in the Wild*, CVPR 2021. [link](https://openaccess.thecvf.com/content/CVPR2021/papers/Martin-Brualla_NeRF_in_the_Wild_Neural_Radiance_Fields_for_Unconstrained_Photo_CVPR_2021_paper.pdf) | B |
| [9] | Kulhanek et al., *WildGaussians: 3D Gaussian Splatting in the Wild*, NeurIPS 2024 (hội nghị cần kiểm tra lại). [link](https://arxiv.org/html/2407.08447v2) | A |
| [10] | Lin et al., *VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction*, CVPR 2024. [link](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_VastGaussian_Vast_3D_Gaussians_for_Large_Scene_Reconstruction_CVPR_2024_paper.html) | B |
| [11] | Wang et al., *Bilateral Guided Radiance Field Processing*, ACM TOG (SIGGRAPH) 2024. [link](https://arxiv.org/html/2406.00448v2) | A |
| [12] | *Unifying Appearance Codes and Bilateral Grids*, arXiv 2506.05280, 2025. [link](https://arxiv.org/html/2506.05280v4) | B |
| [13] | Cui, Chu, Harada, *Luminance-GS*, CVPR 2025. [link](https://openaccess.thecvf.com/content/CVPR2025/html/Cui_Luminance-GS_Adapting_3D_Gaussian_Splatting_to_Challenging_Lighting_Conditions_with_CVPR_2025_paper.html) | B |
| [14] | Yang et al., *Spec-Gaussian*, NeurIPS 2024. [link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/708e0d691a22212e1e373dc8779cbe53-Abstract.html) | B |
| [15] | Jiang et al., *GaussianShader*, CVPR 2024. [link](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_GaussianShader_3D_Gaussian_Splatting_with_Shading_Functions_for_Reflective_Surfaces_CVPR_2024_paper.html) | B |
| [16] | Ye et al., *3D Gaussian Splatting with Deferred Reflection*, SIGGRAPH 2024. [link](https://arxiv.org/abs/2404.18454) | B |
| [17] | Li et al., *MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild*, NeurIPS 2025. [link](https://arxiv.org/html/2509.15548) | A |
| [18] | Li et al., *SparseGS-W: Sparse-View 3D Gaussian Splatting in the Wild with Generative Priors*, arXiv 2025. [link](https://arxiv.org/abs/2503.19452) | B |
| [19] | *Sparse-View 3D Gaussian Splatting in the Wild*, arXiv 2604.27422, 2026. [link](https://arxiv.org/html/2604.27422v1) | C |
| [20] | *WildSplatter: Feed-forward 3D Gaussian Splatting with Appearance Control from Unconstrained Images*, arXiv 2026. [link](https://arxiv.org/html/2604.21182) | C |
| [21] | Niemeyer et al., *RegNeRF*, CVPR 2022. [link](https://arxiv.org/pdf/2112.00724); về khác biệt giao thức DTU: *AIM 2024 Sparse Neural Rendering Challenge*. [link](https://arxiv.org/pdf/2409.15041) | B |
| [22] | Aanæs et al., *Large-Scale Data for Multiple-View Stereopsis*, IJCV 120 (2016). [link](https://dx.doi.org/10.1007/s11263-016-0902-9); [trang dữ liệu](https://roboimagedata.compute.dtu.dk/?page_id=36) | B |

