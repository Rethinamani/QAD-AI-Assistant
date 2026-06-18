# ingestion/excel_ingester.py

import uuid
import logging
from ollama import Client

from security.pii_scrubber import scrub_text
from security.sensitivity_tagger import tag_sensitivity, filter_high_sensitivity

from config import (
    CHROMA_COLLECTION_EXCEL,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)
from vectorstore.chroma_store import chroma_store
from registry.document_registry import (
    initialize_registry,
    register_document,
    update_status,
    mark_failed,
)
from ingestion.cleaner.excel_cleaner import clean_excel

logger = logging.getLogger(__name__)

# Ollama client for embeddings
ollama_client = Client(host=OLLAMA_BASE_URL)


def _build_chunk_text(row: dict) -> str:
    """
    Convert a cleaned Excel row into a single natural language string
    for embedding. This is what gets semantically searched later.

    Format is explicit so the LLM understands each field's role.
    """
    resolution_text = (
        row["resolution"]
        if row["resolution_available"]
        else "No resolution available yet."
    )

    return (
        f"Message Type Affected: {row['affected_message_type']}. "
        f"Error Number: {row['error_number']}. "
        f"Error Message: {row['error_message']}. "
        f"Description: {row['description']}. "
        f"Resolution: {resolution_text}"
    ).strip()


def _get_embedding(text: str) -> list[float]:
    """
    Generate embedding vector for a text string using Ollama.
    """
    response = ollama_client.embeddings(
        model=OLLAMA_EMBED_MODEL,
        prompt=text,
    )
    return response["embedding"]


def _build_metadata(row: dict, doc_id: str, filename: str) -> dict:
    """
    Build the metadata envelope for a ChromaDB document.
    Metadata is used for filtering and source citation.

    ChromaDB only supports str, int, float, bool values in metadata.
    """
    return {
        "source_type":          "excel",
        "source_file":          filename,
        "doc_id":               doc_id,
        "error_number":         str(row["error_number"]),
        "affected_message_type": str(row["affected_message_type"]),
        "error_message":        str(row["error_message"]),
        "resolution_available": str(row["resolution_available"]),
        "row_number":           int(row["row_number"]),
        "sensitivity_level":    "low",
    }


def ingest_excel(file_path: str) -> dict:
    """
    Full ingestion pipeline for an Excel error file.

    Steps:
        1. Register document in registry
        2. Clean and extract rows
        3. Build chunk text per row
        4. Generate embeddings via Ollama
        5. Upsert into ChromaDB excel_errors collection
        6. Update registry with final status

    Returns a summary dict with counts and doc_id.
    """
    initialize_registry()

    filename = file_path.split("\\")[-1].split("/")[-1]
    logger.info(f"Starting Excel ingestion: {filename}")

    # ── Step 1: Register ───────────────────────────────────────────────────────
    doc_id = register_document(filename, source_type="excel")

    try:
        # ── Step 2: Clean ──────────────────────────────────────────────────────
        logger.info("Cleaning Excel file...")
        rows = clean_excel(file_path)

        if not rows:
            raise ValueError("No valid rows found after cleaning.")

        logger.info(f"Cleaned {len(rows)} rows. Generating embeddings...")

        # ── Steps 3 + 4 + 5: Scrub, tag, embed and upsert ────────────────────
        documents  = []
        embeddings = []
        metadatas  = []
        ids        = []
        blocked_count = 0

        for i, row in enumerate(rows):
            chunk_text = _build_chunk_text(row)
            logger.info(f"  Processing row {i+1}/{len(rows)} — Error #{row['error_number']}")

            # PII scrubbing
            scrub_result = scrub_text(chunk_text)
            clean_text   = scrub_result["scrubbed_text"]

            # Sensitivity tagging
            temp_chunk = {
                "text":       clean_text,
                "chunk_type": "excel",
                "page":       row["row_number"],
            }
            tagged = tag_sensitivity(temp_chunk)

            # Skip high sensitivity chunks
            if tagged["sensitivity_level"] == "high":
                logger.warning(
                    f"Skipping high-sensitivity row {row['row_number']} "
                    f"— Error #{row['error_number']}"
                )
                blocked_count += 1
                continue

            embedding = _get_embedding(clean_text)
            chunk_id  = f"{doc_id}_row_{row['row_number']}"

            # Add PII and sensitivity info to metadata
            meta = _build_metadata(row, doc_id, filename)
            meta["pii_detected"]       = str(scrub_result["pii_detected"])
            meta["pii_types_found"]    = ", ".join(scrub_result["pii_types_found"])
            meta["sensitivity_level"]  = tagged["sensitivity_level"]

            documents.append(clean_text)
            embeddings.append(embedding)
            metadatas.append(meta)
            ids.append(chunk_id)

        if not documents:
            raise ValueError("All rows were blocked by sensitivity filter.")

        chroma_store.add_documents(
            collection_name=CHROMA_COLLECTION_EXCEL,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        update_status(doc_id, status="ready", chunk_count=len(documents))

        summary = {
            "doc_id":             doc_id,
            "filename":           filename,
            "total_rows":         len(rows),
            "ingested":           len(documents),
            "blocked_high_sens":  blocked_count,
            "with_resolution":    len([r for r in rows if r["resolution_available"]]),
            "without_resolution": len([r for r in rows if not r["resolution_available"]]),
            "status":             "ready",
        }

        # ── Step 6: Update registry ────────────────────────────────────────────
        update_status(doc_id, status="ready", chunk_count=len(rows))

        summary = {
            "doc_id":               doc_id,
            "filename":             filename,
            "total_rows":           len(rows),
            "with_resolution":      len([r for r in rows if r["resolution_available"]]),
            "without_resolution":   len([r for r in rows if not r["resolution_available"]]),
            "status":               "ready",
        }

        logger.info(f"Excel ingestion complete: {summary}")
        return summary

    except Exception as e:
        mark_failed(doc_id, reason=str(e))
        logger.error(f"Excel ingestion failed: {e}", exc_info=True)
        raise