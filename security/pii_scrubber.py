# security/pii_scrubber.py

import re
import logging
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config import PII_ENTITIES, PII_SPACY_MODEL

logger = logging.getLogger(__name__)

# ── Presidio setup ─────────────────────────────────────────────────────────────
# All three engines are heavy to initialize — create once and reuse.
# BatchAnalyzerEngine wraps the same AnalyzerEngine but runs a single
# batched spaCy NLP pass over many texts instead of one pass per text,
# which is what scrub_chunks_batch() uses for ingestion.
_analyzer       = None
_anonymizer     = None
_batch_analyzer = None


class _LeanSpacyNlpEngine(SpacyNlpEngine):
    """
    spaCy NLP engine that loads a small model and keeps only the components
    Presidio actually needs (tok2vec + ner). Dropping tagger, parser,
    attribute_ruler and lemmatizer roughly halves the per-text cost with no
    measurable hit to the entity types we redact. Falls back to
    en_core_web_lg if the configured small model is not installed.
    """

    def __init__(self, model_name: str):
        super().__init__(models=[{"lang_code": "en", "model_name": model_name}])
        self._model_name = model_name

    def load(self) -> None:
        import spacy

        exclude = ["tagger", "parser", "attribute_ruler", "lemmatizer"]
        try:
            nlp = spacy.load(self._model_name, exclude=exclude)
            logger.info(f"Presidio spaCy model: {self._model_name} (lean pipeline).")
        except OSError:
            logger.warning(
                f"spaCy model '{self._model_name}' not installed — "
                f"falling back to en_core_web_lg. Install the small model "
                f"with:  python -m spacy download {self._model_name}"
            )
            nlp = spacy.load("en_core_web_lg", exclude=exclude)
        self.nlp = {"en": nlp}


def _get_engines():
    """Lazy initialization of Presidio engines."""
    global _analyzer, _anonymizer, _batch_analyzer
    if _analyzer is None:
        logger.info("Initializing Presidio engines (first time only)...")
        nlp_engine = _LeanSpacyNlpEngine(PII_SPACY_MODEL)
        nlp_engine.load()
        _analyzer       = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en"],
        )
        _anonymizer     = AnonymizerEngine()
        _batch_analyzer = BatchAnalyzerEngine(analyzer_engine=_analyzer)
        logger.info(f"Presidio engines ready — entities: {PII_ENTITIES}")
    return _analyzer, _anonymizer, _batch_analyzer


# PII entity types to detect are configured in config.PII_ENTITIES.

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


# Ingesters prefix a chunk's text with a "[source_file · locator]" header
# line (see video_ingester / excel_ingester). That line only ever repeats
# the filename and timestamps/row already stored verbatim in the chunk
# metadata, but Presidio's NER happily flags words in it ("Intro" as a
# PERSON, a "0:52-1:10" span, ...) and mangles the citation. Hold the
# header out of the PII pass and stitch it back on unchanged.
_HEADER_RE = re.compile(r"\A(\[[^\]\n]*\]\n)(.*)\Z", re.DOTALL)


def _split_header(text: str) -> tuple[str, str]:
    """(header_line_incl_newline, body) — or ('', text) when there's no header."""
    m = _HEADER_RE.match(text or "")
    if m:
        return m.group(1), m.group(2)
    return "", text or ""


def _anonymize(anonymizer: AnonymizerEngine, text: str, results: list) -> tuple:
    """
    Redact detected PII entities in text. Shared by scrub_text() and
    scrub_chunks_batch() — anonymization itself is cheap (no NLP), only
    the detection pass (analyze) benefits from batching.

    Returns (anonymized_result, pii_types_found_list).
    """
    operators = {
        entity: OperatorConfig(
            "replace",
            {"new_value": REDACTION_MAP.get(entity, "[REDACTED]")}
        )
        for entity in set(r.entity_type for r in results)
    }
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    pii_types = list(set(r.entity_type for r in results))
    return anonymized, pii_types


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

    analyzer, anonymizer, _ = _get_engines()

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

        anonymized, pii_types = _anonymize(anonymizer, text, results)

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

    header, body = _split_header(original_text)
    result = scrub_text(body)

    # Update chunk text with scrubbed version (header restored unchanged)
    chunk["text"] = header + result["scrubbed_text"]

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


def scrub_chunks_batch(chunks: list[dict], language: str = "en") -> list[dict]:
    """
    Scrub PII from a list of chunk dicts in one batched pass.

    Equivalent to calling scrub_chunk() on each chunk, but runs a single
    batched spaCy NLP pass (via Presidio's BatchAnalyzerEngine) instead of
    one pass per chunk — meaningfully faster for the hundreds/thousands
    of chunks produced by large PDFs. Modifies and returns the same list.
    """
    if not chunks:
        return chunks

    analyzer, anonymizer, batch_analyzer = _get_engines()
    # Analyse only the body — the "[file · locator]" header is held out and
    # re-attached verbatim so citations keep the real filename / timestamps.
    headers, bodies = zip(*(_split_header(c.get("text", "")) for c in chunks))
    headers, bodies = list(headers), list(bodies)

    try:
        all_results = batch_analyzer.analyze_iterator(
            bodies,
            language=language,
            entities=PII_ENTITIES,
            batch_size=64,
        )
    except Exception as e:
        # Never block ingestion due to scrubber failure — fall back to
        # treating every chunk as PII-free, same failure behavior as
        # scrub_text()'s except branch.
        logger.error(f"Batch PII scrubbing failed: {e}", exc_info=True)
        all_results = [[] for _ in bodies]

    for chunk, header, body, results in zip(chunks, headers, bodies, all_results):
        chunk_type = chunk.get("chunk_type", "text")

        if body and body.strip() and results:
            try:
                anonymized, pii_types = _anonymize(anonymizer, body, results)
                chunk["text"]            = header + anonymized.text
                chunk["pii_detected"]    = True
                chunk["pii_types_found"] = ", ".join(pii_types)
                logger.warning(
                    f"PII found in {chunk_type} chunk "
                    f"(page {chunk.get('page', '?')}): "
                    f"{chunk['pii_types_found']}"
                )
            except Exception as e:
                logger.error(f"PII anonymization failed: {e}", exc_info=True)
                chunk["pii_detected"]    = False
                chunk["pii_types_found"] = ""
        else:
            chunk["pii_detected"]    = False
            chunk["pii_types_found"] = ""

        # Flag figure chunks — image content cannot be scrubbed
        chunk["image_may_contain_pii"] = (chunk_type == "figure")

    return chunks