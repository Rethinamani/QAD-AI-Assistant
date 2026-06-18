# guardrails/input_guardrail.py

import re
import logging
from config import MAX_QUERY_LENGTH

logger = logging.getLogger(__name__)

# ── Prompt injection patterns ──────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+.*instructions",
    r"disregard\s+(previous|prior|above|all)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are|a)",
    r"forget\s+(everything|all|prior|previous)",
    r"new\s+instructions\s*:",
    r"system\s*:\s*you",
    r"<\s*system\s*>",
    r"\[system\]",
]

# ── Abuse keywords ─────────────────────────────────────────────────────────────
ABUSE_KEYWORDS = [
    "fuck", "shit", "bastard", "asshole",
    "idiot", "stupid", "moron",
]


def validate_query(query: str) -> dict:
    """
    Validate an incoming user query against all input guardrails.

    Checks (in order):
    1. Empty query
    2. Query too long
    3. Prompt injection attempt
    4. Abusive language

    Returns:
    {
        "valid":   bool,
        "reason":  str | None,   — why it was rejected (if invalid)
        "cleaned": str,          — sanitized query (if valid)
    }
    """
    # ── Check 1: Empty ─────────────────────────────────────────────────────────
    if not query or not query.strip():
        return {
            "valid":   False,
            "reason":  "empty_query",
            "cleaned": "",
        }

    cleaned = query.strip()

    # ── Check 2: Too long ──────────────────────────────────────────────────────
    if len(cleaned) > MAX_QUERY_LENGTH:
        logger.warning(
            f"Query too long: {len(cleaned)} chars "
            f"(max {MAX_QUERY_LENGTH})"
        )
        return {
            "valid":   False,
            "reason":  "query_too_long",
            "cleaned": cleaned,
        }

    # ── Check 3: Prompt injection ──────────────────────────────────────────────
    lower = cleaned.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            logger.warning(f"Prompt injection detected: '{pattern}' in query.")
            return {
                "valid":   False,
                "reason":  "prompt_injection",
                "cleaned": cleaned,
            }

    # ── Check 4: Abusive language ──────────────────────────────────────────────
    for word in ABUSE_KEYWORDS:
        if word in lower:
            logger.warning(f"Abusive language detected in query.")
            return {
                "valid":   False,
                "reason":  "abusive_language",
                "cleaned": cleaned,
            }

    return {
        "valid":   True,
        "reason":  None,
        "cleaned": cleaned,
    }


def get_rejection_message(reason: str) -> str:
    """Return a user-friendly rejection message for each reason."""
    messages = {
        "empty_query":      "Please enter a question so I can help you.",
        "query_too_long":   f"Your query is too long. Please keep it under {MAX_QUERY_LENGTH} characters.",
        "prompt_injection": "I'm not able to process that request. Please ask a question about QAD.",
        "abusive_language": "Please keep the conversation respectful and I'll do my best to help.",
    }
    return messages.get(reason, "I'm unable to process that request. Please try again.")