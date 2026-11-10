# data/

Dataset không được commit (đã ignore). Trên Colab, thư mục này là symlink tới Google Drive
(`notebooks/colab_runner.ipynb`). Cấu trúc:

```text
data/
├── dtu/
│   ├── scan24/{image/, mask/, cameras_sphere.npz | cameras.npz}   # IDR / NeuS / 2DGS preprocessed
│   ├── scan24/ref_points.npy                                      # tự tạo: STL đã chuẩn hoá + cắt theo mask
│   └── Points/stl/stl024_total.ply                                # DTU MVS 2014 "Points" (STL) — chính thức
├── nerf_synthetic/<scene>/                                        # transforms_{train,test}.json + ảnh RGBA
└── reference/<scene>.ply, <scene>_points.npy                      # mesh (world frame Blender) + điểm bề mặt nhìn thấy
```

Nguồn:
- DTU STL (hình học tham chiếu): https://roboimagedata.compute.dtu.dk/?page_id=36 — archive "Points".
- DTU đã tiền xử lý (ảnh, mask, camera chuẩn hoá): link trong README của 2DGS / NeuS (định dạng IDR).
- NeRF-Synthetic: bản phát hành gốc của NeRF (`nerf_synthetic.zip`). **Không kèm hình học tham chiếu**;
  cần mesh trong hệ toạ độ Blender (`scripts/blender_export_mesh.py`) rồi
  `scripts/make_reference_points.py` (lấy điểm bằng ray casting từ camera).

Cảnh `synthetic` mặc định không cần dữ liệu: được sinh trong `frecon/data/synthetic.py`.
Ghi nguồn tải, phiên bản và lệnh tiền xử lý vào đây khi thêm dataset mới.
