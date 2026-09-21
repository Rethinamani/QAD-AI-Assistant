# test_step5.py
import logging
from retrieval.bm25_index import build_all_indexes, search_bm25
from config import CHROMA_COLLECTION_PDF, CHROMA_COLLECTION_EXCEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print("\n── Building BM25 Indexes ──\n")
results = build_all_indexes(force_rebuild=True)

for r in results:
    print(f"  Collection : {r['collection']}")
    print(f"  Doc count  : {r['doc_count']}")
    print(f"  Index path : {r['index_path']}")
    print(f"  Rebuilt    : {r['rebuilt']}")
    print()

print("\n── Test Search 1: Excel — Error keyword ──\n")
hits = search_bm25(CHROMA_COLLECTION_EXCEL, "tax environment error", top_k=3)
for hit in hits:
    print(f"  Rank {hit['rank']} | Score: {hit['score']:.4f}")
    print(f"  Text: {hit['document'][:120]}")
    print()

print("\n── Test Search 2: PDF — Requisition process ──\n")
hits = search_bm25(CHROMA_COLLECTION_PDF, "requisition approval process", top_k=3)
for hit in hits:
    print(f"  Rank {hit['rank']} | Score: {hit['score']:.4f}")
    print(f"  Text: {hit['document'][:120]}")
    print()

print("\n── Test Search 3: Error number exact match ──\n")
hits = search_bm25(CHROMA_COLLECTION_EXCEL, "4415", top_k=3)
for hit in hits:
    print(f"  Rank {hit['rank']} | Score: {hit['score']:.4f}")
    print(f"  Text: {hit['document'][:120]}")
    print()

print("✅ Step 5 complete.")