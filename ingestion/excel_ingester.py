# ingestion/excel_ingester.py

"""
Generic Excel ingester.

One row → one chunk, plus a single sheet-overview chunk. Works for any
single-sheet spreadsheet; the target ChromaDB collection is chosen from
the file name (see resolve_excel_target).
"""

import re
import os
import time
import logging

from ollama import Client

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    EMBED_BATCH_SIZE,
    CHROMA_COLLECTION_INCIDENTS,
    CHROMA_COLLECTION_DEFECTS,
)
from vectorstore.chroma_store import chroma_store
from registry.document_registry import (
    initialize_registry,
    register_document,
    update_status,
    mark_failed,
    get_by_filename,
)
from ingestion.cleaner.excel_cleaner import clean_excel
from security.pii_scrubber import scrub_chunks_batch
from security.sensitivity_tagger import tag_sensitivity

logger = logging.getLogger(__name__)

ollama_client = Client(host=OLLAMA_BASE_URL)

# How many columns to fan out into col_* metadata, and how long each value.
MAX_METADATA_COLUMNS = 25
MAX_METADATA_VALUE_LEN = 200
# A column with at most this many distinct values is summarised in the
# sheet-overview chunk.
SUMMARY_MAX_DISTINCT = 12


class DuplicateFileError(Exception):
    """Raised when a file with the same name was already ingested."""


# ── Routing ───────────────────────────────────────────────────────────────────
def resolve_excel_target(filename: str) -> tuple[str, str]:
    """
    Choose (collection_name, record_label) from the file name.

    'incident' in the name  → incidents collection
    'defect' or 'bug'       → defects collection
    anything else           → ValueError (caller rejects the upload)
    """
    name = filename.lower()
    if "incident" in name:
        return CHROMA_COLLECTION_INCIDENTS, "incident"
    if "defect" in name or "bug" in name:
        return CHROMA_COLLECTION_DEFECTS, "defect"
    raise ValueError(
        f"Cannot route '{filename}': the file name must contain "
        f"'incident' or 'defect' so I know which collection it belongs to."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sanitize_key(header: str) -> str:
    key = re.sub(r"\W+", "_", header.strip().lower()).strip("_")
    return f"col_{key}" if key else "col_"


def _row_to_text(filename: str, sheet: str, row: dict, headers: list[str]) -> str:
    lines = [f"[{filename} · {sheet} · row {row['_row_number']}]"]
    for h in headers:
        val = row.get(h, "")
        if val:
            lines.append(f"{h}: {val}")
    return "\n".join(lines)


def _row_metadata(
    filename: str,
    sheet: str,
    row: dict,
    headers: list[str],
    doc_id: str,
    record_label: str,
) -> dict:
    meta = {
        "source_type":  record_label,          # "incident" | "defect"
        "source_file":  filename,
        "sheet_name":   sheet,
        "doc_id":       doc_id,
        "chunk_type":   "excel_row",
        "row_number":   int(row["_row_number"]),
        "row_key":      row.get(headers[0], "") if headers else "",
    }
    for h in headers[:MAX_METADATA_COLUMNS]:
        val = row.get(h, "")
        if val:
            meta[_sanitize_key(h)] = val[:MAX_METADATA_VALUE_LEN]
    return meta


def _summary_chunk(
    filename: str,
    sheet: str,
    headers: list[str],
    rows: list[dict],
    doc_id: str,
    record_label: str,
) -> tuple[str, dict]:
    noun = "incident" if record_label == "incident" else "defect"
    parts = [
        f"[{filename} · {sheet} · overview]",
        f"The file '{filename}' (sheet '{sheet}') contains {len(rows)} "
        f"{noun} records in total.",
        f"Columns: {', '.join(headers)}.",
    ]
    for h in headers:
        distinct = sorted({r.get(h, "") for r in rows if r.get(h, "")})
        if 0 < len(distinct) <= SUMMARY_MAX_DISTINCT:
            parts.append(
                f"{h} values ({len(distinct)}): {', '.join(distinct)}."
            )
    meta = {
        "source_type":  record_label,
        "source_file":  filename,
        "sheet_name":   sheet,
        "doc_id":       doc_id,
        "chunk_type":   "sheet_summary",
        "row_number":   0,
        "row_key":      "",
    }
    return "\n".join(parts), meta


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    return ollama_client.embed(model=OLLAMA_EMBED_MODEL, input=texts)["embeddings"]


# ── Main ──────────────────────────────────────────────────────────────────────
def ingest_excel(file_path: str) -> dict:
    """
    Ingest a single-sheet Excel file into the collection its name selects.

    Raises DuplicateFileError if a file with this name is already ingested,
    ValueError if the name does not indicate incidents or defects.
    """
    initialize_registry()
    filename = os.path.basename(file_path)

    collection_name, record_label = resolve_excel_target(filename)

    existing = get_by_filename(filename)
    if existing and existing["status"] == "ready":
        raise DuplicateFileError(
            f"'{filename}' was already ingested "
            f"({existing['chunk_count']} chunks, {existing['ingested_at']}). "
            f"Upload it under a different name to re-ingest."
        )

    logger.info(f"Excel ingestion: {filename} → {collection_name}")
    doc_id = register_document(filename, source_type=record_label)

    try:
        sheet, headers, rows = clean_excel(file_path)
        if not rows:
            raise ValueError("No non-empty data rows found.")

        # Build chunks: one per row + one sheet overview.
        chunks: list[dict] = []
        for row in rows:
            chunks.append({
                "text":       _row_to_text(filename, sheet, row, headers),
                "chunk_type": "excel_row",
                "page":       row["_row_number"],   # tag_sensitivity/logging use 'page'
                "_meta":      _row_metadata(
                    filename, sheet, row, headers, doc_id, record_label
                ),
            })
        s_text, s_meta = _summary_chunk(
            filename, sheet, headers, rows, doc_id, record_label
        )
        chunks.append({
            "text": s_text, "chunk_type": "sheet_summary", "page": 0, "_meta": s_meta,
        })

        # PII scrub (mutates chunk["text"], adds pii_* keys).
        logger.info(f"Scrubbing PII for {len(chunks)} chunks...")
        t0 = time.time()
        chunks = scrub_chunks_batch(chunks)
        logger.info(f"PII scrub done in {time.time() - t0:.1f}s.")

        # Sensitivity tag + assemble final records.
        documents, metadatas, ids = [], [], []
        blocked = 0
        for i, chunk in enumerate(chunks):
            chunk = tag_sensitivity(chunk)
            if chunk["sensitivity_level"] == "high":
                blocked += 1
                continue
            meta = dict(chunk["_meta"])
            meta["sensitivity_level"] = chunk["sensitivity_level"]
            meta["pii_detected"]      = str(chunk.get("pii_detected", False))
            meta["pii_types_found"]   = chunk.get("pii_types_found", "")
            documents.append(chunk["text"])
            metadatas.append(meta)
            ids.append(f"{doc_id}_row_{meta['row_number']}")

        if not documents:
            raise ValueError("All rows were blocked by the sensitivity filter.")

        # Embed + upsert in batches.
        for b in range(0, len(documents), EMBED_BATCH_SIZE):
            sl = slice(b, b + EMBED_BATCH_SIZE)
            embeddings = _get_embeddings_batch(documents[sl])
            chroma_store.add_documents(
                collection_name=collection_name,
                documents=documents[sl],
                embeddings=embeddings,
                metadatas=metadatas[sl],
                ids=ids[sl],
            )

        update_status(doc_id, status="ready", chunk_count=len(documents))
        summary = {
            "doc_id":       doc_id,
            "filename":     filename,
            "collection":   collection_name,
            "record_type":  record_label,
            "sheet":        sheet,
            "row_chunks":   len(documents) - 1,
            "total_chunks": len(documents),
            "blocked":      blocked,
            "status":       "ready",
        }
        logger.info(f"Excel ingestion complete: {summary}")
        return summary

    except Exception as e:
        mark_failed(doc_id, reason=str(e))
        logger.error(f"Excel ingestion failed for {filename}: {e}", exc_info=True)
        raise
