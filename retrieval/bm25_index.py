# retrieval/bm25_index.py

import os
import pickle
import logging
from rank_bm25 import BM25Okapi
from config import (
    CHROMA_COLLECTION_PDF,
    CHROMA_COLLECTION_INCIDENTS,
    CHROMA_COLLECTION_DEFECTS,
    CHROMA_COLLECTION_VIDEO,
)
from vectorstore.chroma_store import chroma_store

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BM25_INDEX_DIR = "./vectorstore/bm25_indexes"

# Maps collection name → pickle file path
COLLECTION_INDEX_MAP = {
    CHROMA_COLLECTION_PDF:       os.path.join(BM25_INDEX_DIR, "bm25_pdf.pkl"),
    CHROMA_COLLECTION_INCIDENTS: os.path.join(BM25_INDEX_DIR, "bm25_incidents.pkl"),
    CHROMA_COLLECTION_DEFECTS:   os.path.join(BM25_INDEX_DIR, "bm25_defects.pkl"),
    CHROMA_COLLECTION_VIDEO:     os.path.join(BM25_INDEX_DIR, "bm25_video.pkl"),
}


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.
    Keeps punctuation removal minimal to preserve error codes like '4023'.
    """
    return text.lower().split()


def _fetch_all_documents(collection_name: str) -> tuple[list[str], list[str]]:
    """
    Fetch all documents and their IDs from a ChromaDB collection.
    Returns (ids, documents) tuple.
    """
    collection = chroma_store.get_collection(collection_name)
    count = collection.count()

    if count == 0:
        logger.warning(f"Collection '{collection_name}' is empty — skipping.")
        return [], []

    # ChromaDB get() with no filter returns all documents
    results = collection.get(
        limit=count,
        include=["documents"],
    )

    ids       = results.get("ids", [])
    documents = results.get("documents", [])

    logger.info(f"Fetched {len(documents)} documents from '{collection_name}'.")
    return ids, documents


def build_bm25_index(collection_name: str, force_rebuild: bool = False) -> dict:
    """
    Build a BM25 index for a ChromaDB collection and persist it to disk.

    Args:
        collection_name: Name of the ChromaDB collection.
        force_rebuild:   If True, rebuild even if index already exists.

    Returns dict with index metadata:
    {
        "collection":   str,
        "doc_count":    int,
        "index_path":   str,
        "rebuilt":      bool,
    }
    """
    os.makedirs(BM25_INDEX_DIR, exist_ok=True)
    index_path = COLLECTION_INDEX_MAP.get(collection_name)

    if not index_path:
        raise ValueError(f"No index path configured for collection: {collection_name}")

    # Skip rebuild if index exists and force_rebuild is False
    if os.path.exists(index_path) and not force_rebuild:
        logger.info(
            f"BM25 index already exists for '{collection_name}' — skipping rebuild. "
            f"Use force_rebuild=True to regenerate."
        )
        return {
            "collection": collection_name,
            "doc_count":  None,
            "index_path": index_path,
            "rebuilt":    False,
        }

    # Fetch all documents from ChromaDB
    ids, documents = _fetch_all_documents(collection_name)

    if not documents:
        logger.warning(f"No documents to index for '{collection_name}'.")
        return {
            "collection": collection_name,
            "doc_count":  0,
            "index_path": index_path,
            "rebuilt":    False,
        }

    # Tokenize
    logger.info(f"Tokenizing {len(documents)} documents...")
    tokenized = [_tokenize(doc) for doc in documents]

    # Build BM25 index
    logger.info("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)

    # Persist index + ids + raw documents to disk
    # We store ids and documents alongside so we can map
    # BM25 scores back to chunk IDs during retrieval
    payload = {
        "bm25":      bm25,
        "ids":       ids,
        "documents": documents,
    }

    with open(index_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(
        f"BM25 index saved: {index_path} "
        f"({len(documents)} documents)"
    )

    return {
        "collection": collection_name,
        "doc_count":  len(documents),
        "index_path": index_path,
        "rebuilt":    True,
    }


def load_bm25_index(collection_name: str) -> dict:
    """
    Load a persisted BM25 index from disk.

    Returns:
    {
        "bm25":      BM25Okapi instance,
        "ids":       list of chunk IDs,
        "documents": list of raw document strings,
    }

    Raises FileNotFoundError if index doesn't exist yet.
    """
    index_path = COLLECTION_INDEX_MAP.get(collection_name)

    if not index_path:
        raise ValueError(f"No index path configured for collection: {collection_name}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"BM25 index not found for '{collection_name}'. "
            f"Run build_bm25_index('{collection_name}') first."
        )

    with open(index_path, "rb") as f:
        payload = pickle.load(f)

    logger.info(
        f"BM25 index loaded: '{collection_name}' "
        f"({len(payload['ids'])} documents)"
    )
    return payload


def search_bm25(
    collection_name: str,
    query: str,
    top_k: int = 20,
) -> list[dict]:
    """
    Run a BM25 keyword search against a collection's index.

    Returns list of top_k results sorted by BM25 score (descending):
    [
        {
            "id":       chunk_id,
            "document": raw text,
            "score":    float BM25 score,
            "rank":     1-based rank,
        },
        ...
    ]
    """
    payload   = load_bm25_index(collection_name)
    bm25      = payload["bm25"]
    ids       = payload["ids"]
    documents = payload["documents"]

    tokenized_query = _tokenize(query)
    scores          = bm25.get_scores(tokenized_query)

    # Pair each doc with its score and sort descending
    scored = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []
    for rank, (idx, score) in enumerate(scored, start=1):
        results.append({
            "id":       ids[idx],
            "document": documents[idx],
            "score":    float(score),
            "rank":     rank,
        })

    return results


def build_all_indexes(force_rebuild: bool = False):
    """
    Build BM25 indexes for all three collections.
    Call this at app startup or after any new ingestion.
    """
    logger.info("Building BM25 indexes for all collections...")
    results = []
    for collection_name in COLLECTION_INDEX_MAP:
        result = build_bm25_index(collection_name, force_rebuild=force_rebuild)
        results.append(result)
    logger.info("All BM25 indexes ready.")
    return results