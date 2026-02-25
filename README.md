# Image to LaTeX Converter

Hệ thống tự động chuyển đổi ảnh công thức toán học sang mã LaTeX, sử dụng **pix2tex (LaTeX-OCR)** với model Vision Transformer (ViT), lưu trữ kết quả trong PostgreSQL chạy trên Docker.

Hỗ trợ cả **file local** lẫn **URL ảnh trên internet**, tự động download song song và convert hàng loạt.

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
├── seed.py              # Sinh dữ liệu mẫu để test (local + URL)
├── run.sh               # Script chạy toàn bộ flow
├── Makefile             # Các lệnh make
├── requirements.txt     # Python dependencies
├── samples/             # Ảnh mẫu local (tự sinh khi chạy)
└── .img_cache/          # Cache ảnh download từ URL
```

## Hướng dẫn chạy

### Chạy toàn bộ (1 lệnh duy nhất)

```bash
make run
```

Lệnh này sẽ tự động:
1. Pull image `postgres:16-alpine` từ Docker Hub
2. Khởi động container PostgreSQL + tạo table `document`
3. Cài đặt Python dependencies
4. Sinh 10 ảnh local mẫu + 3 URL mẫu, insert vào DB
5. Download song song các ảnh từ URL (ThreadPoolExecutor, tối đa 8 workers)
6. Convert tất cả ảnh sang LaTeX bằng pix2tex (có progress bar)
7. Hiển thị bảng kết quả ra terminal

### Các lệnh khác

| Lệnh | Mô tả |
|---|---|
| `make setup` | Chỉ khởi động DB + cài dependencies |
| `make seed` | Sinh dữ liệu ảnh mẫu + insert vào DB |
| `make convert` | Convert tất cả ảnh chưa có LaTeX |
| `make insert IMAGES="a.png b.png"` | Thêm ảnh local vào DB |
| `make result` | Hiển thị bảng kết quả convert |
| `make status` | Xem trạng thái container + dữ liệu |
| `make stop` | Dừng container |
| `make clean` | Dừng container + xoá toàn bộ data |

### Sử dụng với ảnh local

```bash
make setup
make insert IMAGES="/path/to/img1.png /path/to/img2.jpg"
make convert
make result
```

### Sử dụng với URL ảnh

```bash
make setup
make insert IMAGES="https://example.com/formula.png https://example.com/eq2.jpg"
make convert
make result
```

Có thể trộn cả local path và URL trong cùng 1 lệnh insert.

## Database

**Table: `document`**

| Cột | Kiểu | Mô tả |
|---|---|---|
| id | SERIAL | Primary key, tự tăng |
| images | TEXT | Đường dẫn ảnh local hoặc URL |
| latex | TEXT | Mã LaTeX sau convert |
| created_at | TIMESTAMP | Thời gian tạo |
| updated_at | TIMESTAMP | Tự cập nhật khi update |

**Kết nối trực tiếp:**

```bash
docker compose exec postgres psql -U admin -d docdb
```

## Thuật toán convert

Sử dụng [pix2tex (LaTeX-OCR)](https://github.com/lukas-blecher/LaTeX-OCR) - Vision Transformer chuyên biệt cho nhận dạng công thức toán.

### Pipeline xử lý

```
Input (local path / URL)
  │
  ├── URL? ──> Download song song (ThreadPoolExecutor, 8 workers)
  │            Cache vào .img_cache/ (tránh download lại)
  │
  └── Load ảnh (xử lý RGBA -> RGB nền trắng)
        │
        ├── Strategy 1: Standard (resize + contrast + sharpen + denoise)
        ├── Strategy 2: Aggressive (binarize đen/trắng + enhance mạnh)
        └── Strategy 3: Raw (ảnh gốc, fallback cuối)
              │
              └── pix2tex OCR (temperature=0.05) ──> LaTeX output
```

### Xử lý ảnh chất lượng kém

- **Ảnh mờ**: Tăng sharpness 2-3x trước OCR
- **Ảnh nhiễu**: MedianFilter loại nhiễu + autocontrast
- **Ảnh rất xấu**: Binarize (đen/trắng) + contrast 2x để tách nét chữ
- **3 chiến lược fallback**: Nếu strategy đầu cho kết quả trống, tự chuyển sang strategy tiếp theo

### Tối ưu cho số lượng lớn

- Download URL song song (tối đa 8 workers đồng thời)
- Cache ảnh URL đã download (không download lại)
- Chỉ convert row có `latex IS NULL` (resume nếu bị crash)
- Progress bar theo dõi tiến trình

## Định dạng ảnh hỗ trợ

PNG, JPG/JPEG, BMP, TIFF, WebP
