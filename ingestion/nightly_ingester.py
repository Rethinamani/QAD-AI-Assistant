# ingestion/nightly_ingester.py

import os
import json
import logging
import pandas as pd
from datetime import datetime

from config import (
    NIGHTLY_DROP_DIR,
    CHROMA_COLLECTION_SN,
)
from vectorstore.chroma_store import chroma_store
from registry.document_registry import (
    initialize_registry,
    register_document,
    update_status,
    mark_failed,
)
from ingestion.cleaner.servicenow_cleaner import clean_servicenow_row
from security.pii_scrubber import scrub_text
from security.sensitivity_tagger import tag_sensitivity
from retrieval.bm25_index import build_bm25_index
from ollama import Client
from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

logger     = logging.getLogger(__name__)
ollama_client = Client(host=OLLAMA_BASE_URL)

# Ledger to track already-ingested ticket IDs
LEDGER_PATH = "./registry/servicenow_ledger.json"


def _load_ledger() -> set:
    """Load the set of already-ingested ticket IDs."""
    if not os.path.exists(LEDGER_PATH):
        return set()
    with open(LEDGER_PATH, "r") as f:
        data = json.load(f)
    return set(data.get("ingested_ids", []))


def _save_ledger(ingested_ids: set):
    """Persist the updated set of ingested ticket IDs."""
    with open(LEDGER_PATH, "w") as f:
        json.dump({"ingested_ids": list(ingested_ids)}, f, indent=2)


def _get_embedding(text: str) -> list[float]:
    response = ollama_client.embeddings(
        model=OLLAMA_EMBED_MODEL,
        prompt=text,
    )
    return response["embedding"]


def _build_ticket_text(row: dict) -> str:
    """Convert a ServiceNow ticket row to embeddable text."""
    return (
        f"Ticket ID: {row.get('ticket_id', '')}. "
        f"Short Description: {row.get('short_description', '')}. "
        f"Description: {row.get('description', '')}. "
        f"Resolution: {row.get('resolution', '')}. "
        f"Category: {row.get('category', '')}."
    ).strip()


def _read_servicenow_file(file_path: str) -> list[dict]:
    """
    Read a ServiceNow CSV or Excel export into a list of row dicts.
    Handles both .csv and .xlsx/.xls formats.
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Normalize column names — lowercase, replace spaces with underscores
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df.to_dict(orient="records")

    except Exception as e:
        raise RuntimeError(f"Failed to read ServiceNow file: {e}")


def run_nightly_ingestion(file_path: str = None) -> dict:
    """
    Main nightly ingestion pipeline for ServiceNow tickets.

    Steps:
    1. Find latest file in nightly_drop/ (or use provided path)
    2. Read and clean rows
    3. Deduplicate against ledger
    4. PII scrub + sensitivity tag
    5. Embed and upsert new tickets into ChromaDB
    6. Update ledger and registry
    7. Rebuild BM25 index

    Returns summary dict.
    """
    initialize_registry()
    os.makedirs(NIGHTLY_DROP_DIR, exist_ok=True)

    # ── Find file to process ───────────────────────────────────────────────────
    if file_path is None:
        # Auto-find latest file in nightly_drop/
        supported = (".csv", ".xlsx", ".xls")
        files = [
            os.path.join(NIGHTLY_DROP_DIR, f)
            for f in os.listdir(NIGHTLY_DROP_DIR)
            if os.path.splitext(f)[1].lower() in supported
        ]
        if not files:
            logger.warning("No files found in nightly_drop/ — nothing to ingest.")
            return {"status": "skipped", "reason": "no_files"}

        # Pick the most recently modified file
        file_path = max(files, key=os.path.getmtime)

    filename = os.path.basename(file_path)
    logger.info(f"Nightly ingestion started: {filename}")

    doc_id = register_document(filename, source_type="servicenow")

    try:
        # ── Read ───────────────────────────────────────────────────────────────
        rows = _read_servicenow_file(file_path)
        logger.info(f"Read {len(rows)} rows from {filename}")

        # ── Load ledger ────────────────────────────────────────────────────────
        ingested_ids = _load_ledger()
        logger.info(f"Ledger has {len(ingested_ids)} previously ingested tickets.")

        # ── Process rows ───────────────────────────────────────────────────────
        documents  = []
        embeddings = []
        metadatas  = []
        ids        = []
        new_ids    = set()
        skipped    = 0
        blocked    = 0

        for row in rows:
            # Clean row
            cleaned = clean_servicenow_row(row)
            if not cleaned:
                continue

            ticket_id = str(cleaned.get("ticket_id", "")).strip()

            # Skip already-ingested tickets
            if ticket_id and ticket_id in ingested_ids:
                skipped += 1
                continue

            # Skip cancelled/duplicate tickets
            status = str(cleaned.get("status", "")).lower()
            if status in ("cancelled", "duplicate", "canceled"):
                skipped += 1
                continue

            # Build text
            text = _build_ticket_text(cleaned)

            # PII scrubbing
            scrub_result = scrub_text(text)
            clean_text   = scrub_result["scrubbed_text"]

            # Sensitivity tagging
            temp_chunk = {"text": clean_text, "chunk_type": "servicenow", "page": 0}
            tagged     = tag_sensitivity(temp_chunk)

            if tagged["sensitivity_level"] == "high":
                blocked += 1
                continue

            # Embed
            embedding = _get_embedding(clean_text)
            chunk_id  = f"{doc_id}_ticket_{ticket_id or len(ids)}"

            documents.append(clean_text)
            embeddings.append(embedding)
            metadatas.append({
                "source_type":      "servicenow",
                "source_file":      filename,
                "doc_id":           doc_id,
                "ticket_id":        ticket_id,
                "category":         str(cleaned.get("category", "")),
                "sensitivity_level": tagged["sensitivity_level"],
                "pii_detected":     str(scrub_result["pii_detected"]),
                "ingested_at":      datetime.utcnow().isoformat(),
            })
            ids.append(chunk_id)

            if ticket_id:
                new_ids.add(ticket_id)

        # ── Upsert ─────────────────────────────────────────────────────────────
        if documents:
            chroma_store.add_documents(
                collection_name=CHROMA_COLLECTION_SN,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Upserted {len(documents)} new tickets.")
        else:
            logger.info("No new tickets to ingest.")

        # ── Update ledger + registry ───────────────────────────────────────────
        ingested_ids.update(new_ids)
        _save_ledger(ingested_ids)
        update_status(doc_id, status="ready", chunk_count=len(documents))

        # ── Rebuild BM25 ───────────────────────────────────────────────────────
        if documents:
            build_bm25_index(CHROMA_COLLECTION_SN, force_rebuild=True)

        summary = {
            "filename":       filename,
            "total_rows":     len(rows),
            "new_ingested":   len(documents),
            "skipped":        skipped,
            "blocked":        blocked,
            "ledger_total":   len(ingested_ids),
            "status":         "ready",
        }

        logger.info(f"Nightly ingestion complete: {summary}")
        return summary

    except Exception as e:
        mark_failed(doc_id, reason=str(e))
        logger.error(f"Nightly ingestion failed: {e}", exc_info=True)
        raise