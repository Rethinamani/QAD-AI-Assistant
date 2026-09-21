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
    CHROMA_COLLECTION_INCIDENTS,
    CHROMA_COLLECTION_DEFECTS,
    CHROMA_COLLECTION_VIDEO,
    ALL_COLLECTIONS,
    TOP_K_DENSE,
    TOP_K_BM25,
    TOP_K_AFTER_FUSION,
    TOP_K_FINAL,
)

# Excel-backed collections — where a bare record id (row_key) is looked up.
_EXCEL_COLLECTIONS = (CHROMA_COLLECTION_INCIDENTS, CHROMA_COLLECTION_DEFECTS)
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


# ── Record-id detection ───────────────────────────────────────────────────────
# A bare identifier token: optional letter prefix + >= 3 digits (+ trailing
# alnum). Matches "944", "4023", "SNOW8729267", "INC0012345", "202601".
ID_TOKEN_PATTERN = re.compile(r"\b([A-Za-z]{0,6}\d{3,}[A-Za-z0-9]*)\b")
# Back-compat alias (chatbot.chain imports this name for follow-up rewriting).
ERROR_NUMBER_PATTERN = re.compile(r"\b(\d{3,6})\b")

# Queries that are explicitly about a picture — make sure a figure chunk with a
# saved image survives ranking so the UI has something to show.
FIGURE_INTENT_PATTERN = re.compile(
    r"\b(diagrams?|figures?|flow\s?charts?|screenshots?|screen\s?shots?|charts?|"
    r"graphics?|illustrations?|pictures?|images?|visuals?)\b",
    re.IGNORECASE,
)

# Queries asking about the spreadsheet as a whole (counts, listings, columns) —
# make sure the per-file sheet_summary chunk is in the results.
AGGREGATE_INTENT_PATTERN = re.compile(
    r"\b(how many|how much|number of|count of|total number|list all|list every|"
    r"all the (incidents?|defects?|bugs?|rows?|records?)|what columns|"
    r"which columns|how many (incidents?|defects?|rows?|records?))\b",
    re.IGNORECASE,
)

# Queries about a recording as a whole ("what is the video about", "summarise
# the recording") — make sure the transcript overview chunk is in the results.
TRANSCRIPT_INTENT_PATTERN = re.compile(
    r"\b(video|recording|audio|clip|transcript|transcription|"
    r"(what|who).{0,20}(said|say|talk|speak|mention)|"
    r"summar(y|ise|ize).{0,20}(video|recording|clip|call|meeting))\b",
    re.IGNORECASE,
)


def _extract_id_token(query: str) -> str | None:
    """Return the first identifier-looking token in the query, or None."""
    match = ID_TOKEN_PATTERN.search(query)
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
                # Infer the source type from the collection rather than
                # fabricating a document name.
                inferred = {
                    CHROMA_COLLECTION_PDF:       "pdf",
                    CHROMA_COLLECTION_INCIDENTS: "incident",
                    CHROMA_COLLECTION_DEFECTS:   "defect",
                    CHROMA_COLLECTION_VIDEO:     "video",
                }.get(collection_name, "unknown")
                r["metadata"] = {"source_type": inferred, "source_file": "unknown"}

    except Exception as e:
        logger.warning(f"Could not enrich BM25 metadata for '{collection_name}': {e}")

    return bm25_results

def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 10,
    per_list_cap: int = 25,
) -> list[dict]:
    """
    Fuse several independently-ranked result lists with Reciprocal Rank
    Fusion.

    Each list contributes 1/(k + rank_within_that_list) — rank is the
    position *inside its own list*, never the position in a concatenation.
    That is what keeps a large collection (thousands of PDF chunks) from
    out-weighting a small one (a few dozen spreadsheet rows) purely by
    volume: every list's #1 hit gets the same 1/(k+1).

    Returns the merged list sorted by fused score, top_k items.
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for lst in ranked_lists:
        for rank, result in enumerate(lst[:per_list_cap], start=1):
            key = result["document"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, result)

    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]

    fused = []
    for rank, key in enumerate(ordered, start=1):
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
    1. Bare record id in the query → exact row_key filter on the Excel collections
    2. Dense vector search across all collections
    3. BM25 keyword search across all collections
    4. Reciprocal Rank Fusion to merge results
    5. CrossEncoder reranking
    6. Confidence score normalization

    Args:
        query:       User's natural language query
        collections: List of collection names to search.
                     Defaults to every collection.

    Returns top-ranked chunks with confidence scores.
    """
    if collections is None:
        collections = list(ALL_COLLECTIONS)

    logger.info(f"Hybrid search: '{query}'")

    # ── Step 1: Exact record-id pre-filter ────────────────────────────────────
    # If the query carries a bare id (944, SNOW8729267, INC0012345), look it
    # up by row_key in the Excel collections so it can be pinned to the top.
    id_token = _extract_id_token(query)
    exact_results = []

    if id_token:
        query_embedding = _get_query_embedding(query)
        for coll in _EXCEL_COLLECTIONS:
            if coll not in collections:
                continue
            hits = _dense_search(
                collection_name=coll,
                query_embedding=query_embedding,
                top_k=5,
                where={"row_key": id_token},
            )
            if hits:
                logger.info(f"Exact row_key match for '{id_token}' in {coll}.")
                exact_results = hits
                break

    # ── Step 2+3: Per-collection dense + BM25, each kept as its own list ──────
    query_embedding = _get_query_embedding(query)

    ranked_lists: list[list[dict]] = []
    per_collection_best: list[dict] = []   # #1 hit from every list

    for collection in collections:
        dense = _dense_search(collection, query_embedding, top_k=TOP_K_DENSE)
        if dense:
            ranked_lists.append(dense)
            per_collection_best.append(dense[0])

        try:
            bm25 = _enrich_bm25_with_metadata(
                search_bm25(collection, query, top_k=TOP_K_BM25), collection
            )
        except FileNotFoundError:
            logger.warning(f"No BM25 index for '{collection}' — skipping.")
            bm25 = []
        if bm25:
            ranked_lists.append(bm25)
            per_collection_best.append(bm25[0])

    if not ranked_lists:
        return []

    # ── Step 4: Fuse the lists (rank is per-list, so size doesn't dominate) ──
    fused = _reciprocal_rank_fusion(ranked_lists, top_k=TOP_K_AFTER_FUSION)

    # Guarantee every collection's single best candidate reaches the reranker,
    # so a large collection can't crowd a small one out before it is judged.
    seen = {r["document"] for r in fused}
    for cand in per_collection_best:
        if cand["document"] not in seen:
            fused.append(cand)
            seen.add(cand["document"])

    # ── Step 5: Rerank once (CrossEncoder scores query↔doc, size-agnostic) ──
    reranked = _rerank(query, fused, top_k=TOP_K_FINAL)

    # ── Step 6: Pin an exact record-id match to position 0 ───────────────────
    if exact_results:
        exact_doc = exact_results[0]["document"]
        reranked = [r for r in reranked if r.get("document") != exact_doc]
        exact_result = exact_results[0].copy()
        exact_result["rerank_score"] = 999.0
        reranked.insert(0, exact_result)
        reranked = reranked[:TOP_K_FINAL]

    # ── Step 6.5: Guarantee a figure result for "show me the diagram" ────────
    if FIGURE_INTENT_PATTERN.search(query) and not any(
        r.get("metadata", {}).get("chunk_type") == "figure" for r in reranked
    ):
        fig_hits = []
        for coll in collections:
            fig_hits.extend(_dense_search(
                coll, query_embedding, top_k=3, where={"chunk_type": "figure"},
            ))
        fig_hits = [f for f in fig_hits if f.get("metadata", {}).get("image_path")]
        if fig_hits:
            reranker   = _get_reranker()
            fig_scores = reranker.predict([(query, f["document"]) for f in fig_hits])
            best = fig_hits[max(range(len(fig_scores)), key=lambda i: fig_scores[i])].copy()
            best["rerank_score"] = float(max(fig_scores))
            reranked = sorted(
                reranked[: max(TOP_K_FINAL - 1, 1)] + [best],
                key=lambda x: x.get("rerank_score", 0.0), reverse=True,
            )
            logger.info(
                f"Figure-intent query — promoted figure chunk "
                f"(page {best.get('metadata', {}).get('page', '?')})."
            )

    # ── Step 6.6: Surface the sheet overview for count / "list all" asks ─────
    if AGGREGATE_INTENT_PATTERN.search(query) and not any(
        r.get("metadata", {}).get("chunk_type") == "sheet_summary" for r in reranked
    ):
        sum_hits = []
        for coll in _EXCEL_COLLECTIONS:
            if coll in collections:
                sum_hits.extend(_dense_search(
                    coll, query_embedding, top_k=2,
                    where={"chunk_type": "sheet_summary"},
                ))
        if sum_hits:
            reranker   = _get_reranker()
            sum_scores = reranker.predict([(query, s["document"]) for s in sum_hits])
            best = sum_hits[max(range(len(sum_scores)), key=lambda i: sum_scores[i])].copy()
            best["rerank_score"] = float(max(sum_scores))
            reranked = sorted(
                reranked[: max(TOP_K_FINAL - 1, 1)] + [best],
                key=lambda x: x.get("rerank_score", 0.0), reverse=True,
            )
            logger.info("Aggregate-intent query — promoted sheet overview chunk.")

    # ── Step 6.7: Surface the transcript overview for "about the video" asks ─
    if (
        TRANSCRIPT_INTENT_PATTERN.search(query)
        and CHROMA_COLLECTION_VIDEO in collections
        and not any(
            r.get("metadata", {}).get("chunk_type") == "transcript_summary"
            for r in reranked
        )
    ):
        vid_hits = _dense_search(
            CHROMA_COLLECTION_VIDEO, query_embedding, top_k=3,
            where={"chunk_type": "transcript_summary"},
        )
        if vid_hits:
            reranker   = _get_reranker()
            vid_scores = reranker.predict([(query, v["document"]) for v in vid_hits])
            best = vid_hits[max(range(len(vid_scores)), key=lambda i: vid_scores[i])].copy()
            best["rerank_score"] = float(max(vid_scores))
            reranked = sorted(
                reranked[: max(TOP_K_FINAL - 1, 1)] + [best],
                key=lambda x: x.get("rerank_score", 0.0), reverse=True,
            )
            logger.info("Transcript-intent query — promoted transcript overview chunk.")

    # ── Step 7: Normalize confidence scores ─────────────────────────────────
    final = _normalize_scores(reranked)

    # Drop chunks whose source could not be identified.
    filtered = [
        r for r in final
        if r.get("metadata", {}).get("source_type") not in (None, "", "unknown")
    ]
    return filtered or final