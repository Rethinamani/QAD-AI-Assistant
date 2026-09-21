# debug_scores.py
from sentence_transformers import CrossEncoder
from retrieval.bm25_index import search_bm25, load_bm25_index
from vectorstore.chroma_store import chroma_store
from config import CHROMA_COLLECTION_PDF, CHROMA_COLLECTION_EXCEL

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Check raw reranker scores ──────────────────────────────────────────────────
print("\n── Raw Reranker Scores ──\n")

queries = [
    "What happens when tax environment is missing?",
    "purple elephant dancing",
]

# Grab a few PDF chunks to test against
collection = chroma_store.get_collection(CHROMA_COLLECTION_PDF)
sample = collection.get(limit=5, include=["documents"])
sample_docs = sample["documents"]

for query in queries:
    pairs  = [(query, doc) for doc in sample_docs]
    scores = reranker.predict(pairs)
    print(f"Query: {query}")
    print(f"  Raw scores: {[round(float(s), 4) for s in scores]}")
    print(f"  Max score : {round(float(max(scores)), 4)}")
    print()

# ── Check BM25 ID format vs ChromaDB ID format ─────────────────────────────────
print("\n── BM25 ID vs ChromaDB ID format ──\n")

bm25_payload = load_bm25_index(CHROMA_COLLECTION_PDF)
bm25_ids = bm25_payload["ids"][:3]
print(f"BM25 IDs (first 3): {bm25_ids}")

chroma_results = collection.get(limit=3, include=["documents"])
chroma_ids = chroma_results["ids"][:3]
print(f"ChromaDB IDs (first 3): {chroma_ids}")