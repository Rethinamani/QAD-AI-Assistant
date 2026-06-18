# retrieval/hybrid_search.py

import re
import logging
from ollama import Client
from sentence_transformers import CrossEncoder

# Minimum raw reranker score to be considered meaningful
# CrossEncoder scores below this indicate irrelevant results
RERANKER_MIN_RAW_SCORE = -10.5

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    CHROMA_COLLECTION_PDF,
    CHROMA_COLLECTION_EXCEL,
    CHROMA_COLLECTION_SN,
    TOP_K_DENSE,
    TOP_K_BM25,
    TOP_K_AFTER_FUSION,
    TOP_K_FINAL,
)
from vectorstore.chroma_store import chroma_store
from retrieval.bm25_index import search_bm25, load_bm25_index

logger = logging.getLogger(__name__)

# ── Model clients ──────────────────────────────────────────────────────────────
ollama_client = Client(host=OLLAMA_BASE_URL)

# CrossEncoder reranker — loads once, reused for all queries
_reranker = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading CrossEncoder reranker (first time only)...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker ready.")
    return _reranker


# ── Error number detection ─────────────────────────────────────────────────────
ERROR_NUMBER_PATTERN = re.compile(r"\b(\d{3,6})\b")


def _extract_error_number(query: str) -> str | None:
    """
    Check if the query contains a standalone error number (3-6 digits).
    Returns the error number string or None.
    """
    match = ERROR_NUMBER_PATTERN.search(query)
    return match.group(1) if match else None


def _get_query_embedding(query: str) -> list[float]:
    """Generate embedding for the query."""
    response = ollama_client.embeddings(
        model=OLLAMA_EMBED_MODEL,
        prompt=query,
    )
    return response["embedding"]


def _dense_search(
    collection_name: str,
    query_embedding: list[float],
    top_k: int,
    where: dict = None,
) -> list[dict]:
    """
    Run dense vector search on a ChromaDB collection.
    Returns list of result dicts with real chunk IDs.
    """
    try:
        results = chroma_store.query(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        output    = []
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids       = results.get("ids", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            output.append({
                "id":       chunk_id,   # real ChromaDB chunk ID
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "source":   "dense",
            })
        return output

    except Exception as e:
        logger.error(f"Dense search failed for '{collection_name}': {e}")
        return []

def _enrich_bm25_with_metadata(
    bm25_results: list[dict],
    collection_name: str,
) -> list[dict]:
    """
    Look up metadata for BM25 results from ChromaDB by document text
    when ID lookup fails.
    """
    if not bm25_results:
        return bm25_results

    collection = chroma_store.get_collection(collection_name)
    ids        = [r["id"] for r in bm25_results]

    try:
        fetched    = collection.get(ids=ids, include=["metadatas", "documents"])
        id_to_meta = dict(zip(fetched["ids"], fetched["metadatas"]))
        doc_to_meta = dict(zip(fetched["documents"], fetched["metadatas"]))

        for r in bm25_results:
            # Try ID lookup first
            if r["id"] in id_to_meta:
                r["metadata"] = id_to_meta[r["id"]]
            # Fall back to document text lookup
            elif r["document"] in doc_to_meta:
                r["metadata"] = doc_to_meta[r["document"]]
                logger.debug(f"Metadata resolved via document text for id: {r['id']}")
            else:
                logger.warning(f"Could not resolve metadata for id: {r['id']}")
                r["metadata"] = {"source_type": "pdf", "source_file": "qad_manual.pdf"}

    except Exception as e:
        logger.warning(f"Could not enrich BM25 metadata for '{collection_name}': {e}")

    return bm25_results

def _reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results:  list[dict],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """
    Merge dense and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF score = 1/(k + rank_dense) + 1/(k + rank_bm25)
    Higher score = better combined rank.

    Returns merged list sorted by RRF score, top_k items.
    """
    scores = {}  # doc_text → rrf_score
    docs   = {}  # doc_text → result dict

    # Score dense results
    for rank, result in enumerate(dense_results, start=1):
        key = result["document"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        docs[key]   = result

    # Score BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        key = result["document"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        if key not in docs:
            docs[key] = {
                "id":       result["id"],
                "document": result["document"],
                "metadata": {},
                "source":   "bm25",
            }

    # Sort by RRF score descending
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    fused = []
    for rank, key in enumerate(sorted_keys, start=1):
        result = docs[key].copy()
        result["rrf_score"] = scores[key]
        result["rrf_rank"]  = rank
        fused.append(result)

    return fused


def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Use CrossEncoder to rerank candidate chunks by relevance to query.
    Returns top_k results with normalized confidence scores.
    """
    reranker = _get_reranker()

    # Build (query, document) pairs for CrossEncoder
    pairs = [(query, c["document"]) for c in candidates]

    # Get raw scores
    raw_scores = reranker.predict(pairs)

    # Attach scores to candidates
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(raw_scores[i])

    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]


def _normalize_scores(results: list[dict]) -> list[dict]:
    """
    Normalize rerank scores to 0.0-1.0 range.
    Scores of 999.0 are sentinel exact-match scores → always map to 1.0.
    """
    if not results:
        return results

    # Separate exact match sentinels from normal scores
    normal  = [r for r in results if r.get("rerank_score", 0) < 999.0]
    pinned  = [r for r in results if r.get("rerank_score", 0) >= 999.0]

    # Normalize pinned to 1.0
    for r in pinned:
        r["confidence"] = 1.0

    if not normal:
        return results

    scores = [r["rerank_score"] for r in normal]
    top_raw = max(scores)
    min_s   = min(scores)
    max_s   = max(scores)
    rng     = max_s - min_s

    for r in normal:
        if rng == 0:
            raw_confidence = 0.5
        else:
            raw_confidence = (r["rerank_score"] - min_s) / rng

        if top_raw < RERANKER_MIN_RAW_SCORE:
            raw_confidence = raw_confidence * 0.35

        r["confidence"] = round(raw_confidence, 4)

    return results


def hybrid_search(
    query: str,
    collections: list[str] = None,
) -> list[dict]:
    """
    Full hybrid search pipeline:
    1. Check for exact error number → metadata filter on Excel first
    2. Dense vector search across all collections
    3. BM25 keyword search across all collections
    4. Reciprocal Rank Fusion to merge results
    5. CrossEncoder reranking
    6. Confidence score normalization

    Args:
        query:       User's natural language query
        collections: List of collection names to search.
                     Defaults to all three collections.

    Returns top-ranked chunks with confidence scores.
    """
    if collections is None:
        collections = [
            CHROMA_COLLECTION_PDF,
            CHROMA_COLLECTION_EXCEL,
            CHROMA_COLLECTION_SN,
        ]

    logger.info(f"Hybrid search: '{query}'")

    # ── Step 1: Exact error number pre-filter ──────────────────────────────────
    error_number = _extract_error_number(query)
    exact_results = []

    if error_number and CHROMA_COLLECTION_EXCEL in collections:
        logger.info(f"Error number detected: {error_number} — running exact filter.")
        query_embedding = _get_query_embedding(query)
        exact_results = _dense_search(
            collection_name=CHROMA_COLLECTION_EXCEL,
            query_embedding=query_embedding,
            top_k=5,
            where={"error_number": error_number},
        )
        if exact_results:
            logger.info(f"Exact match found for error #{error_number}.")

    # ── Step 2: Dense search across all collections ───────────────────────────
    query_embedding = _get_query_embedding(query)

    all_dense = []
    for collection in collections:
        results = _dense_search(
            collection_name=collection,
            query_embedding=query_embedding,
            top_k=TOP_K_DENSE,
        )
        all_dense.extend(results)

    # ── Step 3: BM25 search across all collections ────────────────────────────
    all_bm25 = []
    for collection in collections:
        try:
            results = search_bm25(collection, query, top_k=TOP_K_BM25)
            # Enrich with metadata from ChromaDB
            results = _enrich_bm25_with_metadata(results, collection)
            all_bm25.extend(results)
        except FileNotFoundError:
            logger.warning(f"No BM25 index for '{collection}' — skipping.")

        # ── Step 4: Reciprocal Rank Fusion ────────────────────────────────────
        fused = _reciprocal_rank_fusion(
            dense_results=all_dense,
            bm25_results=all_bm25,
            top_k=TOP_K_AFTER_FUSION,
        )

        # ── Step 5: Rerank ────────────────────────────────────────────────────
        reranked = _rerank(query, fused, top_k=TOP_K_FINAL)

        # ── Step 6: Inject exact error match at position 0 ────────────────────
        # Do this AFTER reranking so the exact match is never dropped or demoted
        if exact_results:
            exact_doc = exact_results[0]["document"]

            # Remove any existing occurrence of error 944 from reranked
            reranked = [r for r in reranked if r.get("document") != exact_doc]

            # Force exact match to position 0 with a fixed high rerank score
            exact_result              = exact_results[0].copy()
            exact_result["rerank_score"] = 999.0  # guaranteed top score
            reranked.insert(0, exact_result)

            # Trim back to TOP_K_FINAL
            reranked = reranked[:TOP_K_FINAL]

        # ── Step 7: Normalize confidence scores ───────────────────────────────
        final = _normalize_scores(reranked)

    # ── Clean up unknown sources ───────────────────────────────────────────
    final = [
        r for r in final
        if r.get("metadata", {}).get("source_type") not in (None, "", "unknown")
    ]

    # If filtering removed all results, return what we have unfiltered
    if not final:
        final = _normalize_scores(reranked)

    return final