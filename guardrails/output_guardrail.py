# guardrails/output_guardrail.py

import re
import logging
from config import MAX_RESPONSE_TOKENS

logger = logging.getLogger(__name__)

# ── PII patterns to catch in LLM output ───────────────────────────────────────
OUTPUT_PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                    "[PHONE]"),
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",               "[IP_ADDRESS]"),
    (r"\b\d{3}-\d{2}-\d{4}\b",                                 "[SSN]"),
]

# ── Internal system terms that should never appear in responses ────────────────
INTERNAL_TERMS = [
    "chroma_db", "chromadb", "vectorstore",
    "bm25_index", "bm25",
    "pdf_chunks", "excel_errors", "servicenow_tickets",
    "doc_id", "chunk_id",
    "./data/", "./vectorstore/",
]

# ── Hallucination check — error numbers ───────────────────────────────────────
ERROR_NUMBER_PATTERN = re.compile(r"\bError\s*#?\s*(\d{3,6})\b", re.IGNORECASE)


def _scan_pii(text: str) -> tuple[str, list[str]]:
    """
    Scan and redact PII patterns from LLM output.
    Returns (cleaned_text, list of pii types found).
    """
    found = []
    for pattern, replacement in OUTPUT_PII_PATTERNS:
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text)
            found.append(replacement)
    return text, found


def _scan_internal_terms(text: str) -> tuple[str, bool]:
    """
    Check if response leaks internal system terms.
    Returns (text, leaked: bool).
    """
    lower   = text.lower()
    leaked  = False
    for term in INTERNAL_TERMS:
        if term in lower:
            logger.warning(f"Internal term leaked in response: '{term}'")
            leaked = True
            # Replace with generic term
            text = re.sub(re.escape(term), "[internal]", text, flags=re.IGNORECASE)
    return text, leaked


def _check_hallucinated_error_numbers(
    response_text: str,
    retrieved_chunks: list[dict],
) -> list[str]:
    """
    Check if the LLM mentions error numbers not present in retrieved chunks.
    Returns list of potentially hallucinated error numbers.
    """
    # Extract error numbers mentioned in response
    response_errors = set(ERROR_NUMBER_PATTERN.findall(response_text))

    # Extract error numbers present in retrieved context
    context_errors = set()
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        err  = meta.get("error_number", "")
        if err and err != "None":
            context_errors.add(str(err))

        # Also scan chunk text for error numbers
        doc_errors = ERROR_NUMBER_PATTERN.findall(chunk.get("document", ""))
        context_errors.update(doc_errors)

    # Flag errors in response not found in context
    hallucinated = [e for e in response_errors if e not in context_errors]

    if hallucinated:
        logger.warning(f"Potentially hallucinated error numbers: {hallucinated}")

    return hallucinated


def _truncate_response(text: str, max_tokens: int) -> str:
    """
    Simple token-approximate truncation.
    Uses word count as proxy (1 token ≈ 0.75 words).
    """
    max_words = int(max_tokens * 0.75)
    words     = text.split()

    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    logger.warning(
        f"Response truncated from {len(words)} to {max_words} words."
    )
    return truncated + "..."


def validate_output(
    response_text: str,
    retrieved_chunks: list[dict] = None,
) -> dict:
    """
    Run all output guardrails on the LLM response.
    """
    if retrieved_chunks is None:
        retrieved_chunks = []

    if not response_text:
        return {
            "text":                  "",
            "pii_found":             [],
            "internal_terms_leaked": False,
            "hallucinated_errors":   [],
            "was_truncated":         False,
            "safe":                  True,
        }

    try:
        # ── Check 1: PII ───────────────────────────────────────────────────────
        text, pii_found = _scan_pii(response_text)
        if pii_found:
            logger.warning(f"PII found in LLM output: {pii_found}")

        # ── Check 2: Internal terms ────────────────────────────────────────────
        text, leaked = _scan_internal_terms(text)

        # ── Check 3: Hallucinated error numbers ────────────────────────────────
        hallucinated = _check_hallucinated_error_numbers(text, retrieved_chunks)

        # ── Check 4: Truncate ──────────────────────────────────────────────────
        original_len  = len(text.split())
        text          = _truncate_response(text, MAX_RESPONSE_TOKENS)
        was_truncated = len(text.split()) < original_len

        safe = not leaked and not hallucinated

        return {
            "text":                  text,
            "pii_found":             pii_found,
            "internal_terms_leaked": leaked,
            "hallucinated_errors":   hallucinated,
            "was_truncated":         was_truncated,
            "safe":                  safe,
        }

    except Exception as e:
        logger.error(f"Output guardrail error: {e}", exc_info=True)
        raise