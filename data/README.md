# data/

Dataset không được commit (đã ignore). Đặt dữ liệu theo cấu trúc:

```text
data/
├── nerf_synthetic/<scene>/          # transforms_{train,test}.json + ảnh RGBA
└── reference/<scene>_points.ply     # điểm lấy mẫu từ mesh .blend (cho benchmark)
```

Cảnh `synthetic` mặc định không cần dữ liệu: được sinh trong `frecon/data/synthetic.py`.
Ghi nguồn tải, phiên bản và lệnh tiền xử lý vào đây khi thêm dataset mới.
