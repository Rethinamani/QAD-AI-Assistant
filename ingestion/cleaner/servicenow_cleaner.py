# ingestion/cleaner/servicenow_cleaner.py

import re
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Expected column name mappings — handles variations in ServiceNow exports
COLUMN_MAP = {
    "ticket_id":          ["number", "ticket_id", "incident_id", "id"],
    "short_description":  ["short_description", "title", "summary"],
    "description":        ["description", "details", "body"],
    "resolution":         ["close_notes", "resolution", "resolution_notes", "resolved"],
    "category":           ["category", "type"],
    "status":             ["state", "status", "incident_state"],
}


def _find_column(row: dict, candidates: list[str]) -> str:
    """Find a value from a row using candidate column names."""
    for key in candidates:
        for row_key in row:
            if row_key.lower().strip() == key:
                val = row[row_key]
                if val is not None and str(val).strip():
                    return str(val).strip()
    return ""


def _strip_html(text: str) -> str:
    """Remove HTML tags from ServiceNow description fields."""
    if not text:
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and newlines."""
    return re.sub(r"\s+", " ", text).strip()


def clean_servicenow_row(row: dict) -> dict | None:
    """
    Clean a single ServiceNow ticket row.

    Returns a cleaned dict or None if the row should be skipped.
    """
    cleaned = {}

    for field, candidates in COLUMN_MAP.items():
        raw = _find_column(row, candidates)
        cleaned[field] = _normalize_whitespace(_strip_html(raw))

    # Skip rows with no meaningful content
    if not cleaned["short_description"] and not cleaned["description"]:
        return None

    return cleaned