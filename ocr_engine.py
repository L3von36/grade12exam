#!/usr/bin/env python3
"""
OCR engine with EasyOCR primary + Tesseract fallback.

EasyOCR is a deep-learning OCR library with significantly better accuracy
than Tesseract on noisy scans, skewed pages, and exam PDFs in general.
Tesseract is kept as a fallback for environments without GPU/torch.
"""

import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

# Prevent Tesseract from spawning too many OpenMP threads per process
# since we run multiple processes in parallel.
os.environ['OMP_THREAD_LIMIT'] = '1'


@dataclass
class PageResult:
    page_num: int
    text: str
    confidence: float
    engine: str


@dataclass
class OCRResult:
    pdf_path: str
    pages: List[PageResult] = field(default_factory=list)
    engine: str = "unknown"

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def mean_confidence(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.confidence for p in self.pages) / len(self.pages)


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 150) -> List[str]:
    """Convert PDF pages to PNG images using pdftoppm."""
    cmd = ['pdftoppm', '-r', str(dpi), '-png', pdf_path,
           os.path.join(output_dir, 'page')]
    subprocess.run(cmd, capture_output=True, check=False)
    images = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith('page') and f.endswith('.png')
    )
    return [os.path.join(output_dir, img) for img in images]


_easyocr_reader = None
_rapidocr_reader = None


def _get_easyocr_reader(languages: Optional[List[str]] = None):
    """Lazy-load the EasyOCR reader. Cached for the process lifetime."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(languages or ['en'], gpu=True)
    return _easyocr_reader


def extract_text_easyocr(image_path: str,
                         languages: Optional[List[str]] = None,
                         min_confidence: float = 0.3) -> tuple[str, float]:
    """
    Extract text from a single image using EasyOCR.
    Returns (text, mean_confidence).
    """
    reader = _get_easyocr_reader(languages)
    results = reader.readtext(image_path, detail=1, paragraph=False)

    pieces = []
    confidences = []
    for _, text, conf in results:
        if conf >= min_confidence and text.strip():
            pieces.append(text)
            confidences.append(conf)

    text = ' '.join(pieces)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, mean_conf


def _get_rapidocr_reader():
    """Lazy-load the RapidOCR reader. Cached for the process lifetime."""
    global _rapidocr_reader
    if _rapidocr_reader is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapidocr_reader = RapidOCR()
    return _rapidocr_reader


def extract_text_rapidocr(image_path: str) -> tuple[str, float]:
    """
    Extract text from a single image using RapidOCR.
    Returns (text, mean_confidence).
    """
    reader = _get_rapidocr_reader()
    results, _ = reader(image_path)
    if not results:
        return '', 0.0

    pieces = []
    confidences = []
    for _, text, conf in results:
        if text.strip():
            pieces.append(text)
            confidences.append(float(conf))

    text = ' '.join(pieces)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, mean_conf


def extract_text_tesseract(image_path: str, lang: str = 'eng') -> tuple[str, float]:
    """
    Extract text from an image using Tesseract.
    Tesseract doesn't expose a usable confidence by default; we approximate
    a confidence of 0.6 when text is recovered, 0.0 otherwise.
    """
    cmd = ['tesseract', image_path, 'stdout', '-l', lang, '--psm', '6']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        text = result.stdout if result.returncode == 0 else ''
        return text, 0.6 if text.strip() else 0.0
    except Exception:
        return '', 0.0


def extract_text_from_pdf(pdf_path: str,
                          engine: str = 'easyocr',
                          dpi: int = 150,
                          min_confidence: float = 0.3,
                          temp_dir: Optional[str] = None) -> OCRResult:
    """
    Extract text from a PDF using the chosen engine.

    engine:
      - 'rapidocr': use RapidOCR (fastest high accuracy)
      - 'easyocr': use EasyOCR (best accuracy, requires torch)
      - 'tesseract': use Tesseract (fast, lower accuracy)
      - 'auto': try RapidOCR, then EasyOCR, then Tesseract
    """
    cleanup_dir = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='ocr_')
        cleanup_dir = True
    else:
        for f in os.listdir(temp_dir):
            fp = os.path.join(temp_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    selected = engine
    if engine == 'auto':
        try:
            import rapidocr_onnxruntime  # noqa: F401
            selected = 'rapidocr'
        except ImportError:
            try:
                import easyocr  # noqa: F401
                selected = 'easyocr'
            except ImportError:
                selected = 'tesseract'

    result = OCRResult(pdf_path=pdf_path, engine=selected)

    try:
        images = pdf_to_images(pdf_path, temp_dir, dpi=dpi)
        
        def process_page(args):
            i, img = args
            try:
                if selected == 'rapidocr':
                    text, conf = extract_text_rapidocr(img)
                elif selected == 'easyocr':
                    text, conf = extract_text_easyocr(img, min_confidence=min_confidence)
                else:
                    text, conf = extract_text_tesseract(img)
                return PageResult(
                    page_num=i, text=text, confidence=conf, engine=selected
                )
            finally:
                try:
                    os.remove(img)
                except OSError:
                    pass

        max_workers = min(len(images), os.cpu_count() or 4)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                pages = list(executor.map(process_page, enumerate(images, start=1)))
        else:
            pages = [process_page((i, img)) for i, img in enumerate(images, start=1)]

        result.pages = sorted(pages, key=lambda p: p.page_num)
    finally:
        if cleanup_dir:
            try:
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
            except OSError:
                pass

    return result


def clean_text(text: str) -> str:
    if not text:
        return ''
    return re.sub(r'\s+', ' ', text).strip()
