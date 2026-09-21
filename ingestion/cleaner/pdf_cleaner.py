# ingestion/cleaner/pdf_cleaner.py

import os
import re
import logging
import fitz  # PyMuPDF
import pdfplumber

from config import FIGURES_DIR
from ingestion.cleaner.image_ocr import extract_image_text

logger = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────────
# How many pages to sample to detect repeating header/footer text
HEADER_FOOTER_SAMPLE_PAGES = 5
# Vertical position threshold (% of page height) to detect header/footer zone
HEADER_ZONE_RATIO  = 0.10   # top 10% of page = header zone
FOOTER_ZONE_RATIO  = 0.10   # bottom 10% of page = footer zone
# Minimum characters for a text block to be worth keeping
MIN_BLOCK_LENGTH = 30

# ── Figure detection tuning ────────────────────────────────────────────────────
# Figures are found by scanning each page's embedded raster images directly —
# no dependence on a "Figure N" caption existing in the page text. These
# thresholds separate real figures (screenshots, diagrams, charts) from icons,
# bullets, rules, logos and watermarks.
#
# Smallest side, in source pixels, an embedded image must have to be a figure.
MIN_FIGURE_IMAGE_PX = 90
# An image qualifies if it covers at least this share of the page area...
MIN_FIGURE_IMAGE_AREA_RATIO = 0.05
# ...or is rendered at least this large on the page (points; 72pt = 1in).
MIN_FIGURE_RENDER_W_PT = 170
MIN_FIGURE_RENDER_H_PT = 120
# An image covering more than this share of a page that also carries a lot of
# body text is treated as a background/watermark, not a figure.
FULL_PAGE_IMAGE_RATIO = 0.90
FULL_PAGE_TEXT_CHARS  = 450
# An image placed on at least this many pages is a logo/watermark, not a figure.
FIGURE_REPEAT_PAGE_THRESHOLD = 4
# A text line within this vertical gap (points) of an image is taken as its
# caption; blocks within the wider window are kept as context.
FIGURE_CAPTION_MAX_GAP_PT = 60
FIGURE_CONTEXT_MAX_GAP_PT = 180
# Characters of surrounding page text to keep as a figure's context.
FIGURE_CONTEXT_MAX_CHARS = 700
# A page with no qualifying raster image but a figure caption and at least this
# many vector draw ops is rendered whole so a drawn flowchart is still captured.
VECTOR_ART_MIN_DRAWINGS = 12


def _extract_toc(doc: fitz.Document) -> list[dict]:
    """
    Extract table of contents from PDF.
    Returns list of dicts: {level, title, page}
    """
    toc = doc.get_toc()  # [[level, title, page], ...]
    if not toc:
        logger.warning("No table of contents found in PDF.")
        return []

    entries = [
        {"level": entry[0], "title": entry[1].strip(), "page": entry[2]}
        for entry in toc
    ]
    logger.info(f"TOC extracted: {len(entries)} entries.")
    return entries


def _detect_repeating_text(doc: fitz.Document) -> set[str]:
    """
    Sample the first N pages and find text blocks that appear on
    every sampled page in the header or footer zone.
    These are repeating headers/footers to be stripped.
    """
    sample_count = min(HEADER_FOOTER_SAMPLE_PAGES, len(doc))
    page_texts = []

    for page_num in range(sample_count):
        page = doc[page_num]
        page_height = page.rect.height
        header_threshold = page_height * HEADER_ZONE_RATIO
        footer_threshold = page_height * (1 - FOOTER_ZONE_RATIO)

        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, ...)
        zone_texts = set()
        for block in blocks:
            y0, y1 = block[1], block[3]
            text = block[4].strip()
            if not text:
                continue
            # In header or footer zone
            if y1 <= header_threshold or y0 >= footer_threshold:
                zone_texts.add(text)

        page_texts.append(zone_texts)

    if not page_texts:
        return set()

    # Keep only text that appears on ALL sampled pages → repeating
    repeating = page_texts[0]
    for page_text_set in page_texts[1:]:
        repeating = repeating & page_text_set

    logger.info(f"Detected {len(repeating)} repeating header/footer text blocks.")
    return repeating


def _clean_text_block(text: str) -> str:
    """
    Clean a raw text block:
    - Fix hyphenation across lines (configu-\nration → configuration)
    - Normalize whitespace
    - Remove lone page numbers
    """
    # Fix hyphenated line breaks
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove standalone page numbers (e.g. "Page 1 of 10" or just "12")
    text = re.sub(r"(?i)^page\s+\d+\s*(of\s*\d+)?$", "", text).strip()
    text = re.sub(r"^\d+$", "", text).strip()
    return text


def _get_section_for_page(toc: list[dict], page_num: int) -> str:
    """
    Given a 1-based page number, return the most recent TOC section title.
    Used to tag chunks with their section for better retrieval context.
    """
    section = "Unknown Section"
    for entry in toc:
        if entry["page"] <= page_num:
            section = entry["title"]
        else:
            break
    return section

def _find_content_start_page(toc: list[dict]) -> int:
    """
    Auto-detect where real content starts by finding the
    minimum page number referenced in the TOC.
    Falls back to page 1 if TOC is empty.
    """
    if not toc:
        logger.warning("No TOC found — starting from page 1.")
        return 1
    content_start = min(entry["page"] for entry in toc)
    logger.info(f"Content starts at page {content_start} (auto-detected from TOC).")
    return content_start


def extract_text_blocks(
    doc: fitz.Document,
    repeating_texts: set[str],
    toc: list[dict],
    content_start_page: int = 1,
) -> list[dict]:
    """
    Extract clean text blocks from all pages.
    Skips header/footer zones and repeating text.

    Returns list of dicts:
    {
        "text":        cleaned text string,
        "page":        1-based page number,
        "section":     TOC section title,
        "chunk_type":  "text",
    }
    """
    blocks_out = []


    for page_num in range(len(doc)):

        if (page_num + 1) < content_start_page:
            continue

        page = doc[page_num]
        page_height = page.rect.height
        header_threshold = page_height * HEADER_ZONE_RATIO
        footer_threshold = page_height * (1 - FOOTER_ZONE_RATIO)
        section = _get_section_for_page(toc, page_num + 1)

        blocks = page.get_text("blocks")
        for block in blocks:
            y0, y1 = block[1], block[3]
            raw_text = block[4].strip()

            if not raw_text:
                continue

            # Skip header/footer zones
            if y1 <= header_threshold or y0 >= footer_threshold:
                continue

            # Skip repeating header/footer text anywhere on page
            if raw_text in repeating_texts:
                continue

            cleaned = _clean_text_block(raw_text)

            if len(cleaned) < MIN_BLOCK_LENGTH:
                continue

            blocks_out.append({
                "text":       cleaned,
                "page":       page_num + 1,
                "section":    section,
                "chunk_type": "text",
            })

    logger.info(f"Extracted {len(blocks_out)} text blocks across {len(doc)} pages.")
    return blocks_out


# NEW signature  
def extract_tables(pdf_path: str, toc: list[dict], content_start_page: int = 1) -> list[dict]:
    """
    Extract tables from PDF using pdfplumber.
    Each table is converted to a markdown string and treated as one chunk.

    Returns list of dicts:
    {
        "text":        markdown table string,
        "page":        1-based page number,
        "section":     TOC section title,
        "chunk_type":  "table",
    }
    """
    tables_out = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            if (page_num + 1) < content_start_page:
                continue
            tables = page.extract_tables()
            if not tables:
                continue

            section = _get_section_for_page(toc, page_num + 1)

            for table in tables:
                if not table or not table[0]:
                    continue

                # Convert to markdown
                header = "| " + " | ".join(
                    str(c).strip() if c else "" for c in table[0]
                ) + " |"
                separator = "| " + " | ".join(
                    "---" for _ in table[0]
                ) + " |"
                rows_md = []
                for row in table[1:]:
                    if not any(row):  # skip empty rows
                        continue
                    rows_md.append(
                        "| " + " | ".join(
                            str(c).strip() if c else "" for c in row
                        ) + " |"
                    )

                if not rows_md:
                    continue

                markdown_table = "\n".join([header, separator] + rows_md)

                tables_out.append({
                    "text":       markdown_table,
                    "page":       page_num + 1,
                    "section":    section,
                    "chunk_type": "table",
                })

    logger.info(f"Extracted {len(tables_out)} tables.")
    return tables_out


# Caption lines usually open with one of these words.
_CAPTION_PREFIX_RE = re.compile(
    r"^\s*(fig(?:ure)?\.?|exhibit|diagram|screenshot|scr\.?|chart|plate|image)\b",
    re.IGNORECASE,
)
# "Figure 2.1: ...", "Fig 3 - ..." anywhere in a page's text.
_FIG_CAPTION_RE = re.compile(
    r"(fig(?:ure)?\.?\s*\d+[.\d]*\s*[:.\-–]?\s*[^\n]{0,120})",
    re.IGNORECASE,
)


def _collect_repeating_image_xrefs(
    doc: fitz.Document,
    content_start_page: int,
) -> set[int]:
    """
    xrefs of images that appear on many pages — logos, watermarks and
    decorative rules — so figure detection can ignore them.
    """
    counts: dict[int, int] = {}
    for page_num in range(len(doc)):
        if (page_num + 1) < content_start_page:
            continue
        seen = {img[0] for img in doc[page_num].get_images(full=True)}
        for xref in seen:
            counts[xref] = counts.get(xref, 0) + 1
    return {xref for xref, c in counts.items() if c >= FIGURE_REPEAT_PAGE_THRESHOLD}


def _page_text_blocks(page: fitz.Page) -> list[tuple]:
    """Return [(fitz.Rect, text), ...] for the page's non-empty text blocks."""
    out = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
        if text:
            out.append((fitz.Rect(x0, y0, x1, y1), text))
    return out


def _rect_already_kept(kept: list[tuple], rect: fitz.Rect, tol: float = 3.0) -> bool:
    """True if an image rect at essentially this position was already kept."""
    for _, r in kept:
        if (
            abs(r.x0 - rect.x0) < tol and abs(r.y0 - rect.y0) < tol
            and abs(r.x1 - rect.x1) < tol and abs(r.y1 - rect.y1) < tol
        ):
            return True
    return False


def _caption_for_image(text_blocks: list[tuple], img_rect: fitz.Rect) -> str:
    """
    Nearest short text block directly below (preferred) or above the image,
    within FIGURE_CAPTION_MAX_GAP_PT. Returns "" when nothing fits.
    """
    candidates = []
    for rect, text in text_blocks:
        if min(rect.x1, img_rect.x1) - max(rect.x0, img_rect.x0) <= 0:
            continue  # no horizontal overlap with the image
        gap_below = rect.y0 - img_rect.y1
        gap_above = img_rect.y0 - rect.y1
        if 0 <= gap_below <= FIGURE_CAPTION_MAX_GAP_PT:
            candidates.append((0, gap_below, text))
        elif 0 <= gap_above <= FIGURE_CAPTION_MAX_GAP_PT:
            candidates.append((1, gap_above, text))

    for _, _, text in sorted(candidates, key=lambda c: (c[0], c[1])):
        one_line = re.sub(r"\s+", " ", text).strip()
        if _CAPTION_PREFIX_RE.match(one_line) or len(one_line) <= 140:
            return one_line[:160]
    return ""


def _context_for_image(
    text_blocks: list[tuple],
    img_rect: fitz.Rect,
    page_text: str,
) -> str:
    """Text blocks vertically near the image, falling back to the page text."""
    near = []
    for rect, text in text_blocks:
        v_gap = max(rect.y0 - img_rect.y1, img_rect.y0 - rect.y1, 0.0)
        if v_gap <= FIGURE_CONTEXT_MAX_GAP_PT:
            near.append((rect.y0, text))
    joined = _clean_text_block(" ".join(t for _, t in sorted(near))) if near else ""
    if len(joined) < 40:
        joined = page_text
    return joined[:FIGURE_CONTEXT_MAX_CHARS]


def _save_image_bytes(
    doc_id: str,
    figure_index: int,
    img_bytes: bytes,
    ext: str,
) -> str | None:
    """Write image bytes under FIGURES_DIR/<doc_id>/; return the relative path."""
    try:
        out_dir = os.path.join(FIGURES_DIR, doc_id)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"fig_{figure_index}.{ext or 'png'}"
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(img_bytes)
        return f"{doc_id}/{filename}"
    except Exception as e:
        logger.warning(f"Could not save figure image {figure_index}: {e}")
        return None


def _extract_xref_image(doc: fitz.Document, xref: int) -> tuple[bytes | None, str | None]:
    """Pull the raw bytes for an embedded image by xref."""
    try:
        extracted = doc.extract_image(xref)
        return extracted["image"], extracted["ext"]
    except Exception as e:
        logger.warning(f"Embedded image extraction failed for xref {xref}: {e}")
        return None, None


def _emit_figure(
    figures_out: list[dict],
    figure_index: int,
    doc_id: str,
    page_num: int,
    section: str,
    caption: str,
    context: str,
    img_bytes: bytes,
    ext: str,
) -> int:
    """Save an image, OCR it, and append a figure chunk. Returns next index."""
    image_path = _save_image_bytes(doc_id, figure_index, img_bytes, ext)
    if not image_path:
        return figure_index

    image_text = extract_image_text(os.path.join(FIGURES_DIR, image_path))

    head = f"[Figure] {caption}".rstrip() if caption else "[Figure]"
    chunk_text = f"{head} — Page {page_num + 1}"
    if context:
        chunk_text += f" — Context: {context}"
    if image_text:
        chunk_text += f" — Text in image: {image_text}"

    figures_out.append({
        "text":         chunk_text,
        "page":         page_num + 1,
        "section":      section,
        "chunk_type":   "figure",
        "figure_index": figure_index,
        "image_path":   image_path,
        "image_text":   image_text,
        "caption":      caption,
    })
    return figure_index + 1


def extract_figures(
    doc: fitz.Document,
    repeating_texts: set[str],
    toc: list[dict],
    content_start_page: int = 1,
    doc_id: str | None = None,
) -> list[dict]:
    """
    Detect and extract figure images from the PDF.

    Figures are found by scanning each page's embedded raster images
    (PyMuPDF ``get_images``) and keeping the ones large enough to be a real
    figure — screenshots, diagrams, charts. This does NOT rely on a
    "Figure N" caption existing in the page text, so unlabelled images are
    still captured. Icons, bullets, rules, and images repeated across many
    pages (logos/watermarks) are dropped.

    For each kept image the chunk carries:
      - a caption taken from the text directly under/over the image, if any
      - the surrounding page text as context
      - text recovered from the image itself via OCR
      - the saved image path, for display during retrieval

    As a fallback, a page with a figure caption but only vector-drawn art
    (a flowchart with no raster image) is rendered whole so the diagram is
    not lost.
    """
    if not doc_id:
        logger.info("No doc_id supplied — skipping figure image extraction.")
        return []

    repeating_xrefs = _collect_repeating_image_xrefs(doc, content_start_page)

    figures_out  = []
    figure_index = 0

    for page_num in range(len(doc)):
        if (page_num + 1) < content_start_page:
            continue

        page      = doc[page_num]
        section   = _get_section_for_page(toc, page_num + 1)
        page_area = page.rect.width * page.rect.height
        page_h    = page.rect.height
        header_th = page_h * HEADER_ZONE_RATIO
        footer_th = page_h * (1 - FOOTER_ZONE_RATIO)

        raw_text = page.get_text("text")
        for rep in repeating_texts:
            raw_text = raw_text.replace(rep, "")
        page_text   = _clean_text_block(raw_text)
        text_blocks = _page_text_blocks(page)

        # ── Embedded raster images ────────────────────────────────────────────
        kept: list[tuple] = []
        for img in page.get_images(full=True):
            xref, _smask, w_px, h_px = img[0], img[1], img[2], img[3]
            if xref in repeating_xrefs:
                continue
            if min(w_px or 0, h_px or 0) < MIN_FIGURE_IMAGE_PX:
                continue

            for rect in page.get_image_rects(xref):
                if rect.width <= 1 or rect.height <= 1:
                    continue
                if rect.y1 <= header_th or rect.y0 >= footer_th:
                    continue  # header / footer band

                ratio = (rect.width * rect.height) / page_area if page_area else 0.0
                big_enough = (
                    ratio >= MIN_FIGURE_IMAGE_AREA_RATIO
                    or (rect.width >= MIN_FIGURE_RENDER_W_PT
                        and rect.height >= MIN_FIGURE_RENDER_H_PT)
                )
                if not big_enough:
                    continue
                if ratio >= FULL_PAGE_IMAGE_RATIO and len(page_text) >= FULL_PAGE_TEXT_CHARS:
                    continue  # page-covering image behind body text = background
                if _rect_already_kept(kept, rect):
                    continue

                kept.append((xref, rect))

        for xref, rect in kept:
            img_bytes, ext = _extract_xref_image(doc, xref)
            if img_bytes is None:
                continue
            figure_index = _emit_figure(
                figures_out, figure_index, doc_id, page_num, section,
                caption=_caption_for_image(text_blocks, rect),
                context=_context_for_image(text_blocks, rect, page_text),
                img_bytes=img_bytes, ext=ext,
            )

        # ── Fallback: caption + vector-drawn diagram, no raster image ─────────
        if not kept:
            m = _FIG_CAPTION_RE.search(page_text)
            if m and len(page.get_drawings()) >= VECTOR_ART_MIN_DRAWINGS:
                pix = page.get_pixmap(dpi=200)
                figure_index = _emit_figure(
                    figures_out, figure_index, doc_id, page_num, section,
                    caption=re.sub(r"\s+", " ", m.group(1)).strip()[:160],
                    context=page_text[:FIGURE_CONTEXT_MAX_CHARS],
                    img_bytes=pix.tobytes("png"), ext="png",
                )

    logger.info(
        f"Extracted {len(figures_out)} figure image(s); "
        f"ignored {len(repeating_xrefs)} repeated image(s) as logos/watermarks."
    )
    return figures_out


def clean_pdf(pdf_path: str, doc_id: str | None = None) -> dict:
    """
    Main entry point for PDF cleaning.
    Auto-detects content start page from TOC.
    Orchestrates text, table, and figure extraction.

    doc_id is used to namespace saved figure images on disk
    (data/figures/<doc_id>/...); pass None to skip image extraction.
    """
    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)

    toc                = _extract_toc(doc)
    content_start_page = _find_content_start_page(toc)
    repeating_texts    = _detect_repeating_text(doc)

    text_blocks = extract_text_blocks(doc, repeating_texts, toc, content_start_page)
    tables      = extract_tables(pdf_path, toc, content_start_page)
    figures     = extract_figures(doc, repeating_texts, toc, content_start_page, doc_id)

    doc.close()

    total = len(text_blocks) + len(tables) + len(figures)
    logger.info(
        f"PDF clean complete — "
        f"text: {len(text_blocks)}, "
        f"tables: {len(tables)}, "
        f"figures: {len(figures)}, "
        f"total: {total}"
    )

    return {
        "toc":                toc,
        "text":               text_blocks,
        "tables":             tables,
        "figures":            figures,
        "content_start_page": content_start_page,
    }