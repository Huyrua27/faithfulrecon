# Tạo Pull Request cho Quỳnh Anh signals — CMD / Anaconda Prompt

Các lệnh dưới đây dùng cho CMD hoặc Anaconda Prompt, như terminal `(recon) D:\Work\reconfaithful\faithfulrecon>` của bạn. Copy từng ô lệnh riêng và nhấn Enter; nếu có lỗi, xử lý trước khi sang ô tiếp theo.

Branch `feature/quynhanh-signals` đã được tạo. Mô tả PR để sửa/comment: [QuynhAnh_signals_PR.md](QuynhAnh_signals_PR.md).

Lỗi `fatal: pathspec` với ký tự backtick xảy ra vì CMD không dùng ký tự đó để nối dòng. Không dùng cú pháp PowerShell `$LASTEXITCODE`, `throw`, hoặc backtick trong CMD. Các lệnh Git bên dưới đều viết một dòng, dùng dấu nháy kép.

## 1. Vào repo và chọn branch

```bat
cd /d D:\Work\reconfaithful\faithfulrecon
```

```bat
git switch feature/quynhanh-signals
```

```bat
git status --short
```

Nếu ở checkout khác chưa có branch này, tạo bằng `git switch -c feature/quynhanh-signals`. Không dùng `-c` khi branch đã tồn tại.

## 2. Stage riêng phần signals

Copy nguyên dòng lệnh sau. Dòng có thể tự xuống hàng khi hiển thị, nhưng không có ký tự nối dòng hoặc câu lệnh kiểm tra PowerShell ở cuối:

```bat
git add -- docs/research/QuynhAnh_signal_research.md docs/notes/QuynhAnh_notes.md docs/pr/QuynhAnh_signals_PR.md docs/pr/QuynhAnh_PR_commands.md frecon/signals/__init__.py frecon/signals/common.py frecon/signals/opacity.py frecon/signals/gradmag.py frecon/signals/visibility.py frecon/signals/gradient_inconsistency.py scripts/export_signals.py scripts/analyze_signal_results.py scripts/reproduce_quynhanh.ps1 tests/test_signals.py tests/test_signal_export.py results/signals
```

Working tree hiện còn thay đổi ở `docs/research/TuanKhai_evaluation.md` và `scripts/prepare_d1_run.py`; danh sách trên không stage hai file đó. Không dùng `git add .` cho working tree này.

Kiểm từng lệnh:

```bat
git diff --cached --name-only
```

```bat
git diff --cached --stat
```

```bat
git diff --cached --check
```

Lệnh cuối không in gì là bình thường khi không có lỗi whitespace. Đọc danh sách staged trước khi commit; không đưa checkpoint, NPZ, dataset hoặc per-Gaussian CSV vào PR. Tick hai checkbox cuối trong mô tả PR khi đã tự kiểm xong; nếu sửa file mô tả sau khi stage, chạy lại `git add -- docs/pr/QuynhAnh_signals_PR.md`.

Nếu cần bỏ một file khỏi staging, dùng `git restore --staged -- "duong/dan/file"`; nội dung working tree vẫn được giữ.

## 3. Commit

```bat
git commit -m "feat: implement reliability signals and checkpoint export"
```

## 4. Cập nhật main và kiểm lại

```bat
git pull --rebase --autostash origin main
```

`--autostash` tạm cất rồi khôi phục các thay đổi chưa commit. Nếu báo conflict khi rebase hoặc khôi phục, dừng để xử lý trước khi push; không xóa thay đổi của track khác.

```bat
git diff --name-only --diff-filter=U
```

Lệnh trên phải không in file conflict. Chạy tests trong môi trường `(recon)` đã cài dependencies:

```bat
python -m pytest -q
```

Nếu môi trường đang dùng chưa có dependencies, dùng Python đã được kiểm tra trên máy:

```bat
"D:\Programs\anaconda3\envs\chess-rl\python.exe" -m pytest -q
```

Chọn một trong hai lệnh test phù hợp; không cần chạy cả hai. Nếu temp mặc định bị hạn chế, thêm `--basetemp runs/pytest_pr_cmd_01` và dùng tên thư mục mới cho lần kiểm tra đó.

```bat
git diff --check
```

```bat
git diff --stat origin/main...HEAD
```

```bat
git status --short
```

Nếu sửa code sau commit, stage và commit phần sửa trước khi push. PR chỉ chứa các commit trên branch, không chứa mọi thay đổi chưa commit trong working tree.

## 5. Push branch

```bat
git push -u origin feature/quynhanh-signals
```

Chỉ mở PR khi push thành công.

## 6A. Tạo PR trên web

GitHub CLI chưa được tìm thấy trong lần kiểm tra trước. Cách web dùng được mà không cần gh. Copy mô tả PR vào clipboard:

```bat
clip < docs\pr\QuynhAnh_signals_PR.md
```

Nếu tiếng Việt trong clipboard không đúng, mở file Markdown trong editor và copy trực tiếp. Mở trang tạo PR:

```bat
start "" "https://github.com/Huyrua27/faithfulrecon/compare/main...feature/quynhanh-signals?expand=1"
```

Trên GitHub, kiểm base=`main`, compare=`feature/quynhanh-signals`. Title: `feat: implement reliability signals and checkpoint export`. Dán nội dung file vào description, xem Files changed, rồi chọn Create pull request hoặc Create draft pull request.

## 6B. Tạo PR bằng gh nếu đã cài

```bat
gh auth status
```

Nếu chưa đăng nhập, chạy `gh auth login`. Khi đã đăng nhập:

```bat
gh pr create --repo Huyrua27/faithfulrecon --base main --head feature/quynhanh-signals --title "feat: implement reliability signals and checkpoint export" --body-file docs/pr/QuynhAnh_signals_PR.md
```

Thêm `--draft` nếu muốn draft. Chọn 6A hoặc 6B cho cùng branch.

## 7. Comment hoặc sửa mô tả PR sau khi tạo

```bat
gh pr view feature/quynhanh-signals --repo Huyrua27/faithfulrecon
```

Thay 123 bằng số PR thật. Để thêm comment mới từ file:

```bat
gh pr comment 123 --repo Huyrua27/faithfulrecon --body-file docs/pr/QuynhAnh_signals_PR.md
```

Hoặc để cập nhật description:

```bat
gh pr edit 123 --repo Huyrua27/faithfulrecon --body-file docs/pr/QuynhAnh_signals_PR.md
```

Chỉ chạy hành động bạn muốn dùng. Có thể comment/sửa description trực tiếp trên GitHub nếu không dùng gh.
