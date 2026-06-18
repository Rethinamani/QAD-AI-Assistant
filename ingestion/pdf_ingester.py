# ingestion/pdf_ingester.py

import uuid
import logging
from tqdm import tqdm
from ollama import Client

from security.pii_scrubber import scrub_chunk
from security.sensitivity_tagger import tag_sensitivity, filter_high_sensitivity

from config import (
    CHROMA_COLLECTION_PDF,
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
from ingestion.cleaner.pdf_cleaner import clean_pdf

logger = logging.getLogger(__name__)

# Ollama client for embeddings
ollama_client = Client(host=OLLAMA_BASE_URL)

# ── Figure filter constants ────────────────────────────────────────────────────

BATCH_SIZE          = 50          # number of chunks to upsert per ChromaDB batch


def _get_embedding(text: str) -> list[float]:
    """Generate embedding for a text string via Ollama."""
    response = ollama_client.embeddings(
        model=OLLAMA_EMBED_MODEL,
        prompt=text,
    )
    return response["embedding"]


def _build_metadata(
    chunk: dict,
    doc_id: str,
    filename: str,
) -> dict:
    """
    Build ChromaDB metadata envelope for a PDF chunk.
    All values must be str, int, float, or bool.
    """
    return {
        "source_type":  "pdf",
        "source_file":  filename,
        "doc_id":       doc_id,
        "page":         int(chunk["page"]),
        "section":      str(chunk.get("section", "Unknown Section")),
        "chunk_type":   str(chunk["chunk_type"]),   # text | table | figure
        "sensitivity_level": "low",
    }


def _upsert_batch(
    documents:  list[str],
    embeddings: list[list[float]],
    metadatas:  list[dict],
    ids:        list[str],
):
    """Upsert a batch of chunks into ChromaDB."""
    chroma_store.add_documents(
        collection_name=CHROMA_COLLECTION_PDF,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def ingest_pdf(file_path: str) -> dict:
    """
    Full ingestion pipeline for a PDF file.

    Steps:
        1.  Register document in registry
        2.  Clean PDF → extract text blocks, tables, figures
        3.  Filter figures using alt text + size rules
        4.  Combine all chunks into one list
        5.  Embed each chunk via Ollama (with progress bar)
        6.  Upsert into ChromaDB in batches
        7.  Update registry with final status

    Returns a summary dict.
    """
    initialize_registry()

    filename = file_path.replace("\\", "/").split("/")[-1]
    logger.info(f"Starting PDF ingestion: {filename}")

    # ── Step 1: Register ───────────────────────────────────────────────────────
    doc_id = register_document(filename, source_type="pdf")

    try:
        # ── Step 2: Clean ──────────────────────────────────────────────────────
        logger.info("Cleaning PDF...")
        cleaned = clean_pdf(file_path)

        text_blocks = cleaned["text"]
        tables      = cleaned["tables"]
        figures = cleaned["figures"]

        # ── Step 4: Combine all chunks ─────────────────────────────────────────
        all_chunks = text_blocks + tables + figures
        total      = len(all_chunks)
        logger.info(
            f"Total chunks to embed: {total} "
            f"(text={len(text_blocks)}, "
            f"tables={len(tables)}, "
            f"figures={len(figures)})"
        )

        if total == 0:
            raise ValueError("No chunks extracted from PDF.")

        # ── Steps 5 + 6: Embed + upsert in batches ────────────────────────────
        batch_docs   = []
        batch_embeds = []
        batch_metas  = []
        batch_ids    = []
        chunk_count  = 0

        blocked_count = 0

        with tqdm(
            total=total,
            desc="Embedding chunks",
            unit="chunk",
            ncols=80,
        ) as pbar:
            for i, chunk in enumerate(all_chunks):

                # PII scrubbing
                chunk = scrub_chunk(chunk)

                # Sensitivity tagging
                chunk = tag_sensitivity(chunk)

                # Skip high sensitivity chunks
                if chunk["sensitivity_level"] == "high":
                    logger.warning(
                        f"Skipping high-sensitivity chunk — "
                        f"type: {chunk['chunk_type']} | page: {chunk['page']}"
                    )
                    blocked_count += 1
                    pbar.update(1)
                    continue

                text      = chunk["text"]
                embedding = _get_embedding(text)
                chunk_id  = f"{doc_id}_chunk_{i}"

                # Build metadata with PII and sensitivity info
                metadata = _build_metadata(chunk, doc_id, filename)
                metadata["pii_detected"]          = str(chunk.get("pii_detected", False))
                metadata["pii_types_found"]        = chunk.get("pii_types_found", "")
                metadata["sensitivity_level"]      = chunk["sensitivity_level"]
                metadata["image_may_contain_pii"]  = str(
                    chunk.get("image_may_contain_pii", False)
                )

                batch_docs.append(text)
                batch_embeds.append(embedding)
                batch_metas.append(metadata)
                batch_ids.append(chunk_id)
                chunk_count += 1

                if len(batch_docs) >= BATCH_SIZE:
                    _upsert_batch(batch_docs, batch_embeds, batch_metas, batch_ids)
                    batch_docs   = []
                    batch_embeds = []
                    batch_metas  = []
                    batch_ids    = []

                pbar.update(1)
                pbar.set_postfix({
                    "type":    chunk["chunk_type"],
                    "page":    chunk["page"],
                    "section": chunk.get("section", "")[:20],
                })

            if batch_docs:
                _upsert_batch(batch_docs, batch_embeds, batch_metas, batch_ids)

        update_status(doc_id, status="ready", chunk_count=chunk_count)

        summary = {
            "doc_id":            doc_id,
            "filename":          filename,
            "total_chunks":      chunk_count,
            "blocked_high_sens": blocked_count,
            "text_chunks":       len(text_blocks),
            "table_chunks":      len(tables),
            "figure_chunks":     len(figures),
            "status":            "ready",
        }

        logger.info(f"PDF ingestion complete: {summary}")
        return summary

    except Exception as e:
        mark_failed(doc_id, reason=str(e))
        logger.error(f"PDF ingestion failed: {e}", exc_info=True)
        raise