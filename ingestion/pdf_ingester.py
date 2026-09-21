# ingestion/pdf_ingester.py

import time
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from ollama import Client

from security.pii_scrubber import scrub_chunks_batch
from security.sensitivity_tagger import tag_sensitivity, filter_high_sensitivity

from config import (
    CHROMA_COLLECTION_PDF,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_CONCURRENCY,
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

# Chunks per embed request, and how many requests to keep in flight.
BATCH_SIZE = EMBED_BATCH_SIZE
# ChromaDB upserts are serialized behind this lock — the embed calls are what
# run concurrently, not the writes.
_upsert_lock = threading.Lock()


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts in a single Ollama call.
    Far fewer HTTP round trips than embedding one chunk at a time —
    ~4-5x faster in practice for large PDFs.
    """
    response = ollama_client.embed(
        model=OLLAMA_EMBED_MODEL,
        input=texts,
    )
    return response["embeddings"]


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
        "image_path":   str(chunk.get("image_path") or ""),
        "has_image_text": bool(chunk.get("image_text")),
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
        4.5 Batch PII scrub (one NLP pass over all chunks)
        5.  Embed chunks in batches via Ollama (with progress bar)
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
        cleaned = clean_pdf(file_path, doc_id)

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

        # ── Step 4.5: Batch PII scrub ──────────────────────────────────────────
        # One batched spaCy NLP pass over all chunks (via Presidio's
        # BatchAnalyzerEngine) instead of one pass per chunk — the per-chunk
        # version re-pays spaCy's per-call overhead every time, which adds up
        # fast on the thousands of chunks a large PDF produces.
        logger.info(f"Scrubbing PII for {total} chunks (batched)...")
        t0 = time.time()
        all_chunks = scrub_chunks_batch(all_chunks)
        logger.info(f"PII scrub complete in {time.time() - t0:.1f}s.")

        # ── Step 5: Sensitivity tag + build the work list ─────────────────────
        # Tagging is cheap keyword matching, done one chunk at a time.
        # High-sensitivity chunks are dropped here and never embedded.
        blocked_count = 0
        prepared = []  # list of (chunk_id, text, metadata)

        for i, chunk in enumerate(all_chunks):
            chunk = tag_sensitivity(chunk)

            if chunk["sensitivity_level"] == "high":
                logger.warning(
                    f"Skipping high-sensitivity chunk — "
                    f"type: {chunk['chunk_type']} | page: {chunk['page']}"
                )
                blocked_count += 1
                continue

            metadata = _build_metadata(chunk, doc_id, filename)
            metadata["pii_detected"]          = str(chunk.get("pii_detected", False))
            metadata["pii_types_found"]        = chunk.get("pii_types_found", "")
            metadata["sensitivity_level"]      = chunk["sensitivity_level"]
            metadata["image_may_contain_pii"]  = str(
                chunk.get("image_may_contain_pii", False)
            )

            prepared.append((f"{doc_id}_chunk_{i}", chunk["text"], metadata))

        chunk_count = len(prepared)

        # ── Step 6: Embed + upsert ───────────────────────────────────────────
        # Each batch is one Ollama embed call. EMBED_CONCURRENCY batches are
        # kept in flight — default 1 (serial), because a CPU-only Ollama gets
        # no benefit from parallel requests. Raise it only on GPU. Upserts to
        # ChromaDB are serialized behind a lock regardless.
        batches = [
            prepared[b : b + BATCH_SIZE]
            for b in range(0, len(prepared), BATCH_SIZE)
        ]

        def _embed_and_upsert(batch):
            ids   = [row[0] for row in batch]
            texts = [row[1] for row in batch]
            metas = [row[2] for row in batch]
            embeddings = _get_embeddings_batch(texts)
            with _upsert_lock:
                _upsert_batch(texts, embeddings, metas, ids)
            return len(batch)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as pool:
            futures = [pool.submit(_embed_and_upsert, b) for b in batches]
            with tqdm(
                total=len(batches),
                desc="Embedding batches",
                unit="batch",
                ncols=80,
            ) as pbar:
                for fut in as_completed(futures):
                    fut.result()  # re-raise any embed/upsert error
                    pbar.update(1)
        logger.info(
            f"Embedded {chunk_count} chunks in {len(batches)} batches "
            f"({EMBED_CONCURRENCY}-way) in {time.time() - t0:.1f}s."
        )

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