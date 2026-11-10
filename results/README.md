# results/

Kết quả **nhỏ** được commit để cả nhóm và GVHD xem mà không cần chạy lại.
File per-Gaussian, checkpoint, render → `runs/` (không commit).

```text
results/
├── phase0/       # Phase 0: report.md, hình, config đã dùng (Huy)
├── signals/      # top-K figures, thống kê signal (Quỳnh Anh)
├── benchmark/    # label_stats.csv, hình nhãn (Trường Thịnh)
└── d1/<scene>/<condition>/   # metrics.csv, figures/, logs/ (Tuấn Khải)
```

Mỗi thư mục kết quả nên có một dòng trong `docs/analysis/` hoặc `docs/research/` giải thích
nó được tạo bằng lệnh nào, ở commit nào.
