# ingestion/cleaner/excel_cleaner.py

import re
import logging
from openpyxl import load_workbook

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SHEET_NAME = "Errors"
HEADER_SEARCH_COLUMN = "A"          # Column to scan for the header row
HEADER_KEYWORD = "Message Type Affected"

REQUIRED_COLUMNS = [
    "Message Type Affected",
    "Error # / Error Grp,",
    "Error Message",
    "Description of error",
    "Resolution",
]


def _find_header_row(sheet) -> int:
    """
    Scan column A top-to-bottom until we find the cell containing
    HEADER_KEYWORD. Returns the 1-based row number of the header.
    Raises ValueError if not found.
    """
    for row in sheet.iter_rows():
        for cell in row:
            if cell.column == 1:  # Column A only
                if cell.value and str(cell.value).strip() == HEADER_KEYWORD:
                    logger.info(f"Header row found at row {cell.row}")
                    return cell.row

    raise ValueError(
        f"Could not find header row. "
        f"Expected '{HEADER_KEYWORD}' in column A."
    )


def _normalize_error_number(raw: str) -> str:
    """
    Standardize error number format.
    Examples:
        'ERR-4023'  → '4023'
        'Error 4023' → '4023'
        '4023'      → '4023'
        'GRP-12'    → 'GRP-12'  (keep group codes as-is)
    """
    if not raw:
        return ""
    raw = str(raw).strip()
    # Remove common prefixes like 'ERR-', 'Error ', 'error '
    raw = re.sub(r"(?i)^err(or)?[-\s]*", "", raw)
    return raw.strip()


def _clean_cell(value) -> str:
    """
    Convert a cell value to a clean string.
    - None / empty → ""
    - Strip whitespace
    - Collapse multiple spaces
    """
    if value is None:
        return ""
    text = str(value).strip()
    # Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text)
    return text


def _is_row_empty(row_data: dict) -> bool:
    """Return True if all fields in a row are empty."""
    return all(v == "" for v in row_data.values())


def _has_resolution(row_data: dict) -> bool:
    """Check if the Resolution field is populated."""
    return row_data.get("Resolution", "") != ""


def clean_excel(file_path: str) -> list[dict]:
    """
    Main entry point. Loads the Excel file, finds the header row,
    extracts and cleans rows A-E only.

    Returns a list of cleaned row dicts. Each dict looks like:
    {
        "message_type_affected": "...",
        "error_number":          "...",
        "error_message":         "...",
        "description":           "...",
        "resolution":            "...",
        "resolution_available":  True | False,
        "row_number":            12,        ← original Excel row for traceability
    }

    Rows that are completely empty are skipped.
    Duplicate error numbers are flagged in logs (latest kept).
    """
    logger.info(f"Loading Excel file: {file_path}")

    try:
        wb = load_workbook(filename=file_path, data_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load Excel file: {e}")

    # ── Sheet validation ───────────────────────────────────────────────────────
    if SHEET_NAME not in wb.sheetnames:
        raise ValueError(
            f"Sheet '{SHEET_NAME}' not found. "
            f"Available sheets: {wb.sheetnames}"
        )
    sheet = wb[SHEET_NAME]
    logger.info(f"Opened sheet: '{SHEET_NAME}'")

    # ── Find header row ────────────────────────────────────────────────────────
    header_row_num = _find_header_row(sheet)

    # ── Build column index map from header row ─────────────────────────────────
    # Maps column name → zero-based column index
    header_row = list(sheet.iter_rows(
        min_row=header_row_num,
        max_row=header_row_num,
        values_only=True
    ))[0]

    col_index = {}
    for idx, cell_value in enumerate(header_row):
        if cell_value and str(cell_value).strip() in REQUIRED_COLUMNS:
            col_index[str(cell_value).strip()] = idx

    # Validate all required columns were found
    missing = [c for c in REQUIRED_COLUMNS if c not in col_index]
    if missing:
        raise ValueError(f"Missing required columns in header: {missing}")

    logger.info(f"Column map: {col_index}")

    # ── Extract data rows ──────────────────────────────────────────────────────
    cleaned_rows = []
    seen_error_numbers = {}

    data_start_row = header_row_num + 1

    # Pre-capture column indices to avoid closure bug
    idx_message_type = col_index["Message Type Affected"]
    idx_error_num    = col_index["Error # / Error Grp,"]
    idx_error_msg    = col_index["Error Message"]
    idx_description  = col_index["Description of error"]
    idx_resolution   = col_index["Resolution"]

    try:
        for row in sheet.iter_rows(min_row=data_start_row, values_only=False):
            raw_values = [cell.value for cell in row]

            def safe_get(idx):
                val = raw_values[idx] if idx < len(raw_values) else None
                return val

            row_data = {
                "affected_message_type": _clean_cell(safe_get(idx_message_type)),
                "error_number":          _normalize_error_number(safe_get(idx_error_num)),
                "error_message":         _clean_cell(safe_get(idx_error_msg)),
                "description":           _clean_cell(safe_get(idx_description)),
                "resolution":            _clean_cell(safe_get(idx_resolution)),
            }

            # Skip completely empty rows
            if _is_row_empty(row_data):
                continue

            # Check resolution availability AFTER cleaning
            raw_resolution = safe_get(idx_resolution)
            row_data["resolution_available"] = (
                raw_resolution is not None
                and str(raw_resolution).strip() != ""
            )

            if not row_data["resolution_available"]:
                logger.warning(
                    f"Row {row[0].row}: No resolution for "
                    f"error '{row_data['error_number']}' — still ingesting."
                )

            # Duplicate error number detection
            err_num = row_data["error_number"]
            if err_num and err_num in seen_error_numbers:
                logger.warning(
                    f"Duplicate error number '{err_num}' at row {row[0].row}. "
                    f"Previous at row {seen_error_numbers[err_num]}. Keeping latest."
                )
            if err_num:
                seen_error_numbers[err_num] = row[0].row

            row_data["row_number"] = row[0].row
            cleaned_rows.append(row_data)

    except Exception as e:
        logger.error(f"Error processing row: {e}", exc_info=True)
        raise

    logger.info(
        f"Cleaning complete: {len(cleaned_rows)} valid rows extracted "
        f"({len([r for r in cleaned_rows if not r['resolution_available']])} "
        f"without resolution)"
    )

    wb.close()
    return cleaned_rows