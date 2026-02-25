import os
import sys
import logging
from pathlib import Path
from typing import Optional

import psycopg2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pix2tex.cli import LatexOCR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class DBConfig:
    HOST = os.getenv("DB_HOST", "localhost")
    PORT = int(os.getenv("DB_PORT", "5432"))
    NAME = os.getenv("DB_NAME", "docdb")
    USER = os.getenv("DB_USER", "admin")
    PASSWORD = os.getenv("DB_PASSWORD", "admin123")

    @classmethod
    def dsn(cls):
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "dbname": cls.NAME,
            "user": cls.USER,
            "password": cls.PASSWORD,
        }


class DocumentRepository:
    def __init__(self, conn):
        self._conn = conn

    def fetch_unconverted(self):
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, images FROM document "
                "WHERE latex IS NULL OR latex = ''"
            )
            return cur.fetchall()

    def update_latex(self, doc_id: int, latex: str):
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE document SET latex = %s WHERE id = %s",
                (latex, doc_id),
            )
        self._conn.commit()

    def insert_image(self, image_path: str):
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO document (images) VALUES (%s) RETURNING id",
                (image_path,),
            )
            doc_id = cur.fetchone()[0]
        self._conn.commit()
        return doc_id


class ImagePreprocessor:
    """Tiền xử lý ảnh để tăng độ chính xác OCR, xử lý cả ảnh xấu/mờ/nhiễu."""

    @staticmethod
    def validate(path: Path) -> bool:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Định dạng không hỗ trợ: {path.suffix}")
        return True

    @staticmethod
    def load(path: Path) -> Image.Image:
        img = Image.open(path)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert("RGB")

    @staticmethod
    def enhance(img: Image.Image) -> Image.Image:
        img = ImageOps.autocontrast(img, cutoff=1)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        return img

    @staticmethod
    def denoise(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.MedianFilter(size=3))

    @staticmethod
    def to_grayscale_binary(img: Image.Image, threshold: int = 180) -> Image.Image:
        gray = img.convert("L")
        return gray.point(lambda x: 255 if x > threshold else 0, "L").convert("RGB")

    @staticmethod
    def optimal_resize(img: Image.Image, max_dim: int = 1024) -> Image.Image:
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        if min(w, h) < 32:
            ratio = 32 / min(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img

    @classmethod
    def process(cls, path: Path) -> Image.Image:
        cls.validate(path)
        img = cls.load(path)
        img = cls.optimal_resize(img)
        img = cls.enhance(img)
        img = cls.denoise(img)
        return img

    @classmethod
    def process_aggressive(cls, path: Path) -> Image.Image:
        """Cho ảnh chất lượng rất xấu: binarize + enhance mạnh hơn."""
        cls.validate(path)
        img = cls.load(path)
        img = cls.optimal_resize(img)
        img = cls.to_grayscale_binary(img)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(3.0)
        return img


class ImageToLatexConverter:
    """
    Sử dụng pix2tex (LaTeX-OCR) với cấu hình tối ưu cho độ chính xác cao.
    - temperature thấp -> kết quả ổn định, ít random
    - Retry với nhiều mức tiền xử lý nếu kết quả trống/lỗi
    """

    def __init__(self):
        logger.info("Đang tải model LaTeX-OCR (pix2tex ViT)...")
        self._model = LatexOCR()
        self._model.args.temperature = 0.05
        logger.info("Model đã sẵn sàng (temperature=0.05).")

    def _predict(self, img: Image.Image) -> str:
        return self._model(img)

    def convert(self, image_path: str) -> str:
        path = Path(image_path)

        strategies = [
            ("standard", ImagePreprocessor.process),
            ("aggressive", ImagePreprocessor.process_aggressive),
            ("raw", lambda p: ImagePreprocessor.load(p)),
        ]

        results = []
        for name, preprocess_fn in strategies:
            try:
                img = preprocess_fn(path)
                latex = self._predict(img)
                if latex and len(latex.strip()) > 0:
                    results.append((name, latex))
                    logger.debug(f"  [{name}] -> {latex[:60]}")
                    break
            except Exception as e:
                logger.debug(f"  [{name}] lỗi: {e}")
                continue

        if not results:
            raise RuntimeError(f"Không thể OCR ảnh: {image_path}")

        return results[0][1]


class App:
    def __init__(self):
        self._conn = psycopg2.connect(**DBConfig.dsn())
        self._repo = DocumentRepository(self._conn)
        self._converter = ImageToLatexConverter()

    def insert_images(self, paths: list[str]):
        for p in paths:
            doc_id = self._repo.insert_image(p)
            logger.info(f"Đã thêm ảnh: id={doc_id}, path={p}")

    def convert_all(self):
        rows = self._repo.fetch_unconverted()
        if not rows:
            logger.info("Không có ảnh nào cần convert.")
            return

        total = len(rows)
        success = 0
        logger.info(f"Tìm thấy {total} ảnh cần convert.")

        for doc_id, image_path in rows:
            try:
                latex = self._converter.convert(image_path)
                self._repo.update_latex(doc_id, latex)
                preview = latex[:80] + "..." if len(latex) > 80 else latex
                logger.info(f"[OK] id={doc_id} | {image_path} -> {preview}")
                success += 1
            except Exception as e:
                logger.error(f"[FAIL] id={doc_id} | {image_path} | {e}")

        logger.info(f"Kết quả: {success}/{total} ảnh convert thành công.")

    def close(self):
        self._conn.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "insert":
        image_paths = sys.argv[2:]
        if not image_paths:
            print("Sử dụng: python convert.py insert <path1> <path2> ...")
            sys.exit(1)

        app = App()
        try:
            app.insert_images(image_paths)
        finally:
            app.close()
    else:
        app = App()
        try:
            app.convert_all()
        finally:
            app.close()


if __name__ == "__main__":
    main()
