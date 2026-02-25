import os
from pathlib import Path

import psycopg2

SAMPLES_DIR = Path(__file__).parent / "samples"
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

DB_DSN = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "docdb"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD", "admin123"),
}


def scan_samples() -> list[str]:
    if not SAMPLES_DIR.exists():
        print(f"[SEED] Thư mục {SAMPLES_DIR} không tồn tại.")
        return []

    paths = sorted(
        [
            str(f.resolve())
            for f in SAMPLES_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        ],
        key=lambda p: _natural_sort_key(Path(p).stem),
    )

    print(f"[SEED] Tìm thấy {len(paths)} ảnh trong {SAMPLES_DIR}/")
    return paths


def _natural_sort_key(name: str):
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', name)]


def insert_to_db(image_paths: list[str]):
    conn = psycopg2.connect(**DB_DSN)
    cur = conn.cursor()

    cur.execute("SELECT images FROM document")
    existing = {row[0] for row in cur.fetchall()}

    new_paths = [p for p in image_paths if p not in existing]
    if not new_paths:
        print(f"[SEED] Tất cả {len(image_paths)} ảnh đã có trong database, bỏ qua.")
        cur.close()
        conn.close()
        return

    for p in new_paths:
        cur.execute("INSERT INTO document (images) VALUES (%s)", (p,))
    conn.commit()

    print(f"[SEED] Đã thêm {len(new_paths)} ảnh mới vào database (bỏ qua {len(image_paths) - len(new_paths)} ảnh đã tồn tại).")
    cur.close()
    conn.close()


if __name__ == "__main__":
    paths = scan_samples()
    if paths:
        insert_to_db(paths)
    else:
        print("[SEED] Không có ảnh nào để thêm.")
