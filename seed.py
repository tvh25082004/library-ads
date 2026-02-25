import os
import random
from pathlib import Path

import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

SAMPLES_DIR = Path(__file__).parent / "samples"

EQUATIONS = [
    (r"$E = mc^{2}$", "energy"),
    (r"$x = \frac{-b \pm \sqrt{b^{2}-4ac}}{2a}$", "quadratic"),
    (r"$\int_{0}^{\infty} e^{-x^{2}} dx = \frac{\sqrt{\pi}}{2}$", "integral"),
    (r"$\sum_{n=1}^{\infty} \frac{1}{n^{2}} = \frac{\pi^{2}}{6}$", "series"),
    (r"$\frac{\partial^{2} u}{\partial t^{2}} = c^{2} \nabla^{2} u$", "wave_eq"),
]

DB_DSN = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "docdb"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD", "admin123"),
}


def _render_equation(latex_str: str, filepath: Path, dpi: int = 150):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, latex_str, fontsize=28, ha="center", va="center")
    ax.axis("off")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)


def _add_noise(img: Image.Image, intensity: float = 30.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _degrade_image(img: Image.Image) -> Image.Image:
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    img = _add_noise(img, intensity=25.0)
    img = ImageEnhance.Contrast(img).enhance(0.6)
    return img


def generate_images():
    SAMPLES_DIR.mkdir(exist_ok=True)
    paths = []

    for latex_str, name in EQUATIONS:
        clean_path = SAMPLES_DIR / f"{name}_clean.png"
        _render_equation(latex_str, clean_path, dpi=150)
        paths.append(str(clean_path.resolve()))
        print(f"[IMG] Ảnh sạch: {clean_path}")

        noisy_path = SAMPLES_DIR / f"{name}_noisy.png"
        _render_equation(latex_str, noisy_path, dpi=100)
        noisy_img = Image.open(noisy_path)
        degraded = _degrade_image(noisy_img)
        degraded.save(noisy_path)
        paths.append(str(noisy_path.resolve()))
        print(f"[IMG] Ảnh xấu:  {noisy_path}")

    return paths


def insert_to_db(paths):
    conn = psycopg2.connect(**DB_DSN)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM document")
    if cur.fetchone()[0] > 0:
        print("[SEED] Database đã có dữ liệu, bỏ qua seed.")
        cur.close()
        conn.close()
        return

    for p in paths:
        cur.execute("INSERT INTO document (images) VALUES (%s)", (p,))
    conn.commit()
    print(f"[SEED] Đã thêm {len(paths)} ảnh mẫu vào database.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    paths = generate_images()
    insert_to_db(paths)
