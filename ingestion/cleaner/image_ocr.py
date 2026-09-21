# ingestion/cleaner/image_ocr.py

"""
OCR for figure images.

`extract_image_text` runs Tesseract over a saved figure image and returns
cleaned text. It is deliberately defensive: OCR is a best-effort enrichment,
so any failure (missing binary, unreadable file, Tesseract crash) returns an
empty string and lets ingestion carry on.
"""

import os
import re
import logging

from config import (
    OCR_ENABLED,
    TESSERACT_CMD,
    OCR_TESSERACT_CONFIG,
    OCR_MIN_TEXT_LENGTH,
    OCR_UPSCALE_THRESHOLD_PX,
)

logger = logging.getLogger(__name__)

# Tri-state cache: None = not checked yet, True/False = availability decided.
_ocr_ready: bool | None = None


def _check_ocr() -> bool:
    """Check once whether pytesseract + the Tesseract binary are usable."""
    global _ocr_ready
    if _ocr_ready is not None:
        return _ocr_ready

    if not OCR_ENABLED:
        _ocr_ready = False
        return False

    try:
        import pytesseract
        from PIL import Image  # noqa: F401  (import checked here, used later)

        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        version = pytesseract.get_tesseract_version()
        logger.info(f"OCR enabled — Tesseract {version}.")
        _ocr_ready = True
    except Exception as e:
        logger.warning(
            f"OCR disabled — Tesseract unavailable ({e}). "
            f"Figure images will be indexed without extracted text. "
            f"Install the Tesseract binary and set TESSERACT_CMD to enable."
        )
        _ocr_ready = False

    return _ocr_ready


def _preprocess(img):
    """Grayscale, upscale small images, and stretch contrast for OCR."""
    from PIL import Image, ImageOps

    img = ImageOps.grayscale(img)

    w, h = img.size
    if min(w, h) < OCR_UPSCALE_THRESHOLD_PX:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)

    img = ImageOps.autocontrast(img)
    return img


def _clean_ocr_text(text: str) -> str:
    """Drop blank lines and border/arrow noise, collapse runs of spaces."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Lines with no alphanumeric content are almost always OCR picking up
        # box borders, arrows or connector lines from a diagram.
        if not re.search(r"[A-Za-z0-9]", line):
            continue
        line = re.sub(r"[ \t]{2,}", " ", line)
        lines.append(line)
    return "\n".join(lines).strip()


def extract_image_text(image_path: str, config: str | None = None) -> str:
    """
    Run OCR on a saved figure image and return cleaned text.

    `config` overrides OCR_TESSERACT_CONFIG for this call (e.g. a different
    page-segmentation mode for full-screen screenshots).

    Returns "" when OCR is unavailable, the file is missing, extraction
    fails, or the result is shorter than OCR_MIN_TEXT_LENGTH. Never raises.
    """
    if not image_path or not os.path.isfile(image_path):
        return ""
    if not _check_ocr():
        return ""

    try:
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as img:
            img.load()
            processed = _preprocess(img)

        raw = pytesseract.image_to_string(
            processed, config=config or OCR_TESSERACT_CONFIG
        )
    except Exception as e:
        logger.warning(f"OCR failed for '{image_path}': {e}")
        return ""

    cleaned = _clean_ocr_text(raw)
    if len(cleaned) < OCR_MIN_TEXT_LENGTH:
        return ""

    logger.debug(f"OCR recovered {len(cleaned)} chars from '{image_path}'.")
    return cleaned
