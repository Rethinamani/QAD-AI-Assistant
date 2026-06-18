# security/pii_scrubber.py

import logging
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)

# ── Presidio setup ─────────────────────────────────────────────────────────────
# Both engines are heavy to initialize — create once and reuse
_analyzer  = None
_anonymizer = None

def _get_engines():
    """Lazy initialization of Presidio engines."""
    global _analyzer, _anonymizer
    if _analyzer is None:
        logger.info("Initializing Presidio engines (first time only)...")
        _analyzer   = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
        logger.info("Presidio engines ready.")
    return _analyzer, _anonymizer


# PII entity types to detect
PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "LOCATION",
    "URL",
]

# What to replace each entity type with
REDACTION_MAP = {
    "PERSON":        "[PERSON]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER":  "[PHONE]",
    "IP_ADDRESS":    "[IP_ADDRESS]",
    "US_SSN":        "[SSN]",
    "CREDIT_CARD":   "[CREDIT_CARD]",
    "IBAN_CODE":     "[IBAN]",
    "LOCATION":      "[LOCATION]",
    "URL":           "[URL]",
}


def scrub_text(text: str, language: str = "en") -> dict:
    """
    Detect and redact PII from a text string.

    Returns:
    {
        "scrubbed_text":   str,   — text with PII replaced by placeholders
        "pii_detected":    bool,  — whether any PII was found
        "pii_types_found": list,  — list of PII entity types detected
        "original_length": int,
        "scrubbed_length": int,
    }
    """
    if not text or not text.strip():
        return {
            "scrubbed_text":   text,
            "pii_detected":    False,
            "pii_types_found": [],
            "original_length": 0,
            "scrubbed_length": 0,
        }

    analyzer, anonymizer = _get_engines()

    try:
        # Detect PII
        results = analyzer.analyze(
            text=text,
            entities=PII_ENTITIES,
            language=language,
        )

        if not results:
            return {
                "scrubbed_text":   text,
                "pii_detected":    False,
                "pii_types_found": [],
                "original_length": len(text),
                "scrubbed_length": len(text),
            }

        # Build operator config — replace each entity type with its placeholder
        operators = {
            entity: OperatorConfig(
                "replace",
                {"new_value": REDACTION_MAP.get(entity, "[REDACTED]")}
            )
            for entity in set(r.entity_type for r in results)
        }

        # Redact
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )

        pii_types = list(set(r.entity_type for r in results))

        logger.debug(
            f"PII detected: {pii_types} | "
            f"Original length: {len(text)} | "
            f"Scrubbed length: {len(anonymized.text)}"
        )

        return {
            "scrubbed_text":   anonymized.text,
            "pii_detected":    True,
            "pii_types_found": pii_types,
            "original_length": len(text),
            "scrubbed_length": len(anonymized.text),
        }

    except Exception as e:
        # If scrubbing fails, log and return original text
        # Never block ingestion due to scrubber failure
        logger.error(f"PII scrubbing failed: {e}", exc_info=True)
        return {
            "scrubbed_text":   text,
            "pii_detected":    False,
            "pii_types_found": [],
            "original_length": len(text),
            "scrubbed_length": len(text),
        }


def scrub_chunk(chunk: dict) -> dict:
    """
    Scrub PII from a chunk dict (as produced by pdf_cleaner or excel_ingester).
    Modifies the 'text' field in place and adds PII metadata fields.

    Special handling for figure chunks:
    - Scrubs caption/context text as normal
    - Adds 'image_may_contain_pii: true' flag since we cannot
      inspect embedded screenshot images on CPU

    Returns the modified chunk dict.
    """
    chunk_type = chunk.get("chunk_type", "text")
    original_text = chunk.get("text", "")

    result = scrub_text(original_text)

    # Update chunk text with scrubbed version
    chunk["text"] = result["scrubbed_text"]

    # Add PII metadata
    chunk["pii_detected"]    = result["pii_detected"]
    chunk["pii_types_found"] = ", ".join(result["pii_types_found"])

    # Flag figure chunks — image content cannot be scrubbed
    chunk["image_may_contain_pii"] = (chunk_type == "figure")

    if result["pii_detected"]:
        logger.warning(
            f"PII found in {chunk_type} chunk "
            f"(page {chunk.get('page', '?')}): "
            f"{result['pii_types_found']}"
        )

    return chunk