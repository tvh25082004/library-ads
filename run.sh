#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

log "=== Bước 1: Pull PostgreSQL image ==="
docker compose pull

log "=== Bước 2: Khởi động PostgreSQL container ==="
docker compose up -d

log "=== Bước 3: Chờ database sẵn sàng ==="
RETRIES=0
MAX_RETRIES=30
until docker compose exec -T postgres pg_isready -U admin -d docdb > /dev/null 2>&1; do
    RETRIES=$((RETRIES + 1))
    if [ "$RETRIES" -ge "$MAX_RETRIES" ]; then
        err "Database không sẵn sàng sau ${MAX_RETRIES} lần thử. Dừng lại."
        exit 1
    fi
    warn "Đang chờ database... (${RETRIES}/${MAX_RETRIES})"
    sleep 2
done
log "Database đã sẵn sàng!"

log "=== Bước 4: Cài đặt Python dependencies ==="
pip install -r requirements.txt -q

log "=== Bước 5: Tạo dữ liệu mẫu (ảnh công thức toán) ==="
python3 seed.py

log "=== Bước 6: Chạy convert images -> LaTeX ==="
python3 convert.py

log "=== Bước 7: Hiển thị kết quả ==="
docker compose exec -T postgres psql -U admin -d docdb -c \
    "SELECT id, images, latex FROM document;"

log "=== Hoàn tất! ==="
