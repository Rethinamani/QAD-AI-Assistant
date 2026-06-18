# ingestion/cleaner/pdf_cleaner.py

import re
import logging
import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────────
# How many pages to sample to detect repeating header/footer text
HEADER_FOOTER_SAMPLE_PAGES = 5
# Vertical position threshold (% of page height) to detect header/footer zone
HEADER_ZONE_RATIO  = 0.10   # top 10% of page = header zone
FOOTER_ZONE_RATIO  = 0.10   # bottom 10% of page = footer zone
# Minimum characters for a text block to be worth keeping
MIN_BLOCK_LENGTH = 30


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


def extract_figures(
    doc: fitz.Document,
    repeating_texts: set[str],
    toc: list[dict],
    content_start_page: int = 1,
) -> list[dict]:
    """
    Detect figures by scanning page text for captions like
    'Fig. 2.1', 'Figure 3', 'Fig 4.2' etc.

    Extracts the caption line + surrounding paragraph as the chunk text.
    This is more reliable than alt text for PDFs with labelled figures.
    """
    # Regex to detect figure captions like Fig. 2.1, Figure 3, Fig 4
    fig_pattern = re.compile(r"(Fig\.?\s*\d+[\.\d]*.*)", re.IGNORECASE)

    figures_out  = []
    figure_index = 0

    for page_num in range(len(doc)):
        # Skip pre-content pages
        if (page_num + 1) < content_start_page:
            continue

        page     = doc[page_num]
        section  = _get_section_for_page(toc, page_num + 1)
        raw_text = page.get_text("text").strip()

        # Remove repeating header/footer text
        for rep in repeating_texts:
            raw_text = raw_text.replace(rep, "")

        cleaned = _clean_text_block(raw_text)

        # Check if this page contains a figure caption
        match = fig_pattern.search(cleaned)
        if not match:
            continue

        # Extract the caption line as the primary identifier
        caption = match.group(1).strip()

        # Build chunk: caption + full page context
        chunk_text = f"[Figure] {caption} — Context: {cleaned}"

        figures_out.append({
            "text":         chunk_text,
            "page":         page_num + 1,
            "section":      section,
            "chunk_type":   "figure",
            "figure_index": figure_index,
        })

        figure_index += 1

    logger.info(f"Detected {len(figures_out)} labelled figures (Fig. caption pattern).")
    return figures_out


def clean_pdf(pdf_path: str) -> dict:
    """
    Main entry point for PDF cleaning.
    Auto-detects content start page from TOC.
    Orchestrates text, table, and figure extraction.
    """
    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)

    toc                = _extract_toc(doc)
    content_start_page = _find_content_start_page(toc)
    repeating_texts    = _detect_repeating_text(doc)

    text_blocks = extract_text_blocks(doc, repeating_texts, toc, content_start_page)
    tables      = extract_tables(pdf_path, toc, content_start_page)
    figures     = extract_figures(doc, repeating_texts, toc, content_start_page)

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