import os
import sys
import re
import logging
import hashlib
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import psycopg2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pix2tex.cli import LatexOCR
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
CACHE_DIR = Path(__file__).parent / ".img_cache"


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


class ImageLoader:
    """Tải ảnh từ local path hoặc URL, cache URL đã download."""

    URL_PATTERN = re.compile(r'^https?://')
    DOWNLOAD_TIMEOUT = 30
    MAX_DOWNLOAD_WORKERS = 8

    @classmethod
    def is_url(cls, source: str) -> bool:
        return bool(cls.URL_PATTERN.match(source))

    @classmethod
    def _url_to_cache_path(cls, url: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            ext = ".png"
        return CACHE_DIR / f"{url_hash}{ext}"

    @classmethod
    def download_single(cls, url: str) -> Path:
        cache_path = cls._url_to_cache_path(url)
        if cache_path.exists():
            return cache_path

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        resp = requests.get(url, timeout=cls.DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            img = Image.open(BytesIO(resp.content))
            img.save(cache_path)
        else:
            with open(cache_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)

        return cache_path

    @classmethod
    def download_batch(cls, urls: list[str]) -> dict[str, Path]:
        """Download nhiều URL song song, trả về dict {url: local_path}."""
        results = {}
        urls_to_download = []

        for url in urls:
            cache_path = cls._url_to_cache_path(url)
            if cache_path.exists():
                results[url] = cache_path
            else:
                urls_to_download.append(url)

        if not urls_to_download:
            return results

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        workers = min(cls.MAX_DOWNLOAD_WORKERS, len(urls_to_download))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(cls.download_single, url): url
                for url in urls_to_download
            }
            for future in tqdm(
                as_completed(future_map),
                total=len(urls_to_download),
                desc="Download ảnh",
                unit="file",
            ):
                url = future_map[future]
                try:
                    results[url] = future.result()
                except Exception as e:
                    logger.error(f"[DOWNLOAD FAIL] {url} | {e}")

        return results

    @classmethod
    def load(cls, source: str) -> Image.Image:
        if cls.is_url(source):
            local_path = cls.download_single(source)
        else:
            local_path = Path(source)

        if not local_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {source}")

        img = Image.open(local_path)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert("RGB")


class ImagePreprocessor:
    """Tiền xử lý ảnh để tăng độ chính xác OCR."""

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
    def standard(cls, img: Image.Image) -> Image.Image:
        img = cls.optimal_resize(img)
        img = cls.enhance(img)
        img = cls.denoise(img)
        return img

    @classmethod
    def aggressive(cls, img: Image.Image) -> Image.Image:
        img = cls.optimal_resize(img)
        img = cls.to_grayscale_binary(img)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(3.0)
        return img


class ImageToLatexConverter:
    """
    pix2tex (LaTeX-OCR) ViT.
    - temperature thấp -> kết quả ổn định
    - Fallback: standard -> aggressive -> raw
    """

    def __init__(self):
        logger.info("Đang tải model LaTeX-OCR (pix2tex ViT)...")
        self._model = LatexOCR()
        self._model.args.temperature = 0.05
        logger.info("Model đã sẵn sàng (temperature=0.05).")

    def _predict(self, img: Image.Image) -> str:
        return self._model(img)

    def convert(self, source: str) -> str:
        raw_img = ImageLoader.load(source)

        strategies = [
            ("standard", ImagePreprocessor.standard),
            ("aggressive", ImagePreprocessor.aggressive),
            ("raw", lambda img: img),
        ]

        for name, preprocess_fn in strategies:
            try:
                img = preprocess_fn(raw_img.copy())
                latex = self._predict(img)
                if latex and latex.strip():
                    return latex
            except Exception as e:
                logger.debug(f"  [{name}] lỗi: {e}")
                continue

        raise RuntimeError(f"Không thể OCR ảnh: {source}")


class App:
    def __init__(self):
        self._conn = psycopg2.connect(**DBConfig.dsn())
        self._repo = DocumentRepository(self._conn)
        self._converter = ImageToLatexConverter()

    def insert_images(self, sources: list[str]):
        for s in sources:
            doc_id = self._repo.insert_image(s)
            tag = "URL" if ImageLoader.is_url(s) else "LOCAL"
            logger.info(f"[{tag}] Đã thêm: id={doc_id}, source={s}")

    def convert_all(self):
        rows = self._repo.fetch_unconverted()
        if not rows:
            logger.info("Không có ảnh nào cần convert.")
            return

        total = len(rows)
        logger.info(f"Tìm thấy {total} ảnh cần convert.")

        urls = [src for _, src in rows if ImageLoader.is_url(src)]
        if urls:
            logger.info(f"Đang download {len(urls)} ảnh từ URL...")
            ImageLoader.download_batch(urls)

        success = 0
        for doc_id, source in tqdm(rows, desc="Convert OCR", unit="ảnh"):
            try:
                latex = self._converter.convert(source)
                self._repo.update_latex(doc_id, latex)
                preview = latex[:80] + "..." if len(latex) > 80 else latex
                tqdm.write(f"  [OK] id={doc_id} | {source} -> {preview}")
                success += 1
            except Exception as e:
                tqdm.write(f"  [FAIL] id={doc_id} | {source} | {e}")

        logger.info(f"Kết quả: {success}/{total} ảnh convert thành công.")

    def close(self):
        self._conn.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "insert":
        sources = sys.argv[2:]
        if not sources:
            print("Sử dụng: python convert.py insert <path_or_url> ...")
            sys.exit(1)

        app = App()
        try:
            app.insert_images(sources)
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
