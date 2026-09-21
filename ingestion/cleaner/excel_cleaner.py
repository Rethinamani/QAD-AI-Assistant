# ingestion/cleaner/excel_cleaner.py

"""
Generic single-sheet Excel cleaner.

No assumptions about column names. The first row is the header; every
later row becomes a dict of {header: cleaned string}. HTML is stripped,
whitespace collapsed, fully-empty rows dropped.
"""

import re
import logging

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _strip_html(text: str) -> str:
    """Drop HTML tags (ServiceNow work-notes / comments often carry them)."""
    if not text or "<" not in text:
        return text or ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)


def _clean_value(value) -> str:
    """Cell → clean single-line string."""
    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() in ("nan", "nat", "none"):
        return ""
    text = _strip_html(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_headers(columns: list) -> list[str]:
    """
    Strip/collapse header names; give blank or pandas-placeholder headers a
    positional name (Column A, Column B, …); de-duplicate collisions.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, col in enumerate(columns):
        name = re.sub(r"\s+", " ", str(col)).strip()
        if not name or name.lower().startswith("unnamed"):
            name = f"Column {_col_letter(i)}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 0
        out.append(name)
    return out


def _col_letter(idx: int) -> str:
    """0 → A, 1 → B, … 26 → AA."""
    letters = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def clean_excel(file_path: str) -> tuple[str, list[str], list[dict]]:
    """
    Read the first sheet of an Excel file.

    Returns (sheet_name, headers, rows) where rows is a list of dicts:
        {"_row_number": <1-based Excel row, header is row 1>,
         "<Header>": "<clean value>", ...}

    Raises ValueError if the sheet is empty or has no header.
    """
    logger.info(f"Reading Excel: {file_path}")

    xl = pd.ExcelFile(file_path)
    if not xl.sheet_names:
        raise ValueError("Workbook has no sheets.")
    sheet_name = xl.sheet_names[0]
    if len(xl.sheet_names) > 1:
        logger.warning(
            f"'{file_path}' has {len(xl.sheet_names)} sheets "
            f"{xl.sheet_names} — ingesting only the first ('{sheet_name}')."
        )

    df = pd.read_excel(xl, sheet_name=sheet_name, header=0, dtype=str)
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"Sheet '{sheet_name}' has no data rows.")

    headers = _normalize_headers(list(df.columns))
    df.columns = headers

    rows: list[dict] = []
    for pos, (_, series) in enumerate(df.iterrows()):
        values = {h: _clean_value(series[h]) for h in headers}
        if not any(values.values()):
            continue  # fully-empty row
        # +2: pandas row 0 is Excel row 2 (row 1 is the header)
        values["_row_number"] = pos + 2
        rows.append(values)

    logger.info(
        f"Cleaned sheet '{sheet_name}': {len(rows)} rows, "
        f"{len(headers)} columns."
    )
    return sheet_name, headers, rows
