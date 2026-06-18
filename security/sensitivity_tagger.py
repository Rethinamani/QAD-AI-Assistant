# security/sensitivity_tagger.py

import logging
from config import SENSITIVITY_HIGH_KEYWORDS

logger = logging.getLogger(__name__)

# ── Sensitivity levels ─────────────────────────────────────────────────────────
LEVEL_HIGH   = "high"
LEVEL_MEDIUM = "medium"
LEVEL_LOW    = "low"

# Keywords that bump a chunk to medium sensitivity
MEDIUM_KEYWORDS = [
    "internal",
    "restricted",
    "proprietary",
    "do not distribute",
    "not for distribution",
    "draft",
    "preliminary",
]


def tag_sensitivity(chunk: dict) -> dict:
    """
    Assign a sensitivity level to a chunk based on its text content.

    Rules (evaluated in order — first match wins):
    - HIGH:   text contains any SENSITIVITY_HIGH_KEYWORDS (from .env)
    - MEDIUM: text contains any MEDIUM_KEYWORDS
    - LOW:    everything else

    Adds 'sensitivity_level' key to the chunk dict.
    Returns the modified chunk.
    """
    text = chunk.get("text", "").lower()

    # Check high sensitivity keywords (from .env config)
    for keyword in SENSITIVITY_HIGH_KEYWORDS:
        if keyword.lower() in text:
            chunk["sensitivity_level"] = LEVEL_HIGH
            logger.warning(
                f"HIGH sensitivity chunk detected — "
                f"keyword: '{keyword}' | "
                f"chunk_type: {chunk.get('chunk_type')} | "
                f"page: {chunk.get('page', '?')}"
            )
            return chunk

    # Check medium sensitivity keywords
    for keyword in MEDIUM_KEYWORDS:
        if keyword in text:
            chunk["sensitivity_level"] = LEVEL_MEDIUM
            logger.info(
                f"MEDIUM sensitivity chunk — "
                f"keyword: '{keyword}' | "
                f"page: {chunk.get('page', '?')}"
            )
            return chunk

    # Default — low sensitivity
    chunk["sensitivity_level"] = LEVEL_LOW
    return chunk


def filter_high_sensitivity(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split chunks into safe (low/medium) and blocked (high) groups.

    Returns:
        safe_chunks:    list of chunks safe to ingest
        blocked_chunks: list of high-sensitivity chunks that are excluded
    """
    safe    = [c for c in chunks if c.get("sensitivity_level") != LEVEL_HIGH]
    blocked = [c for c in chunks if c.get("sensitivity_level") == LEVEL_HIGH]

    if blocked:
        logger.warning(
            f"Blocked {len(blocked)} high-sensitivity chunks from ingestion."
        )

    return safe, blocked