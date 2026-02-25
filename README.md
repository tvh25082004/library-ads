# Image to LaTeX Converter

Hệ thống tự động chuyển đổi ảnh công thức toán học sang mã LaTeX, sử dụng **pix2tex (LaTeX-OCR)** với model Vision Transformer (ViT), lưu trữ kết quả trong PostgreSQL chạy trên Docker.

## Yêu cầu hệ thống

- Docker & Docker Compose
- Python 3.9+
- pip
- make

## Cấu trúc dự án

```
├── docker-compose.yml   # PostgreSQL 16 Alpine container
├── init.sql             # Script tạo table document
├── convert.py           # Thuật toán convert ảnh -> LaTeX (OOP)
├── seed.py              # Sinh dữ liệu mẫu để test
├── run.sh               # Script chạy toàn bộ flow
├── Makefile             # Các lệnh make
├── requirements.txt     # Python dependencies
└── samples/             # Thư mục chứa ảnh mẫu (tự sinh khi chạy)
```

## Hướng dẫn chạy

### Chạy toàn bộ (1 lệnh duy nhất)

```bash
make run
```

Lệnh này sẽ tự động thực hiện:
1. Pull image `postgres:16-alpine` từ Docker Hub
2. Khởi động container PostgreSQL
3. Tạo table `document` (qua `init.sql`)
4. Cài đặt Python dependencies
5. Sinh 10 ảnh công thức toán mẫu (5 sạch + 5 nhiễu)
6. Convert tất cả ảnh sang LaTeX bằng pix2tex
7. Hiển thị bảng kết quả ra terminal

### Các lệnh khác

| Lệnh | Mô tả |
|---|---|
| `make setup` | Chỉ khởi động DB + cài dependencies |
| `make seed` | Sinh dữ liệu ảnh mẫu + insert vào DB |
| `make convert` | Convert tất cả ảnh chưa có LaTeX |
| `make insert IMAGES="a.png b.png"` | Thêm đường dẫn ảnh của bạn vào DB |
| `make result` | Hiển thị bảng kết quả convert |
| `make status` | Xem trạng thái container + dữ liệu |
| `make stop` | Dừng container |
| `make clean` | Dừng container + xoá toàn bộ data |

### Sử dụng với ảnh của bạn

```bash
# 1. Đảm bảo DB đang chạy
make setup

# 2. Thêm ảnh vào database
make insert IMAGES="/path/to/img1.png /path/to/img2.jpg"

# 3. Chạy convert
make convert

# 4. Xem kết quả
make result
```

## Database

**Table: `document`**

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | SERIAL | Primary key, tự tăng |
| images | TEXT | Đường dẫn ảnh local |
| latex | TEXT | Mã LaTeX sau convert |
| created_at | TIMESTAMP | Thời gian tạo |
| updated_at | TIMESTAMP | Tự cập nhật khi update |

**Kết nối trực tiếp:**

```bash
docker compose exec postgres psql -U admin -d docdb
```

## Thuật toán convert

Sử dụng [pix2tex (LaTeX-OCR)](https://github.com/lukas-blecher/LaTeX-OCR) - Vision Transformer chuyên biệt cho nhận dạng công thức toán.

Pipeline xử lý ảnh trước khi OCR:
1. **Validate** - Kiểm tra file tồn tại, định dạng hỗ trợ (png, jpg, bmp, tiff, webp)
2. **Resize** - Đưa ảnh về kích thước tối ưu cho model (32-1024px)
3. **Enhance** - Tăng contrast + sharpen cho ảnh rõ nét hơn
4. **Denoise** - Loại bỏ nhiễu bằng MedianFilter
5. **Fallback** - Nếu kết quả trống, thử lại với binarize (đen/trắng) hoặc ảnh gốc

`temperature=0.05` để kết quả ổn định, ít random giữa các lần chạy.

## Định dạng ảnh hỗ trợ

PNG, JPG/JPEG, BMP, TIFF, WebP
