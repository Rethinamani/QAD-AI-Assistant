# debug_grs.py
from retrieval.bm25_index import search_bm25
from vectorstore.chroma_store import chroma_store
from config import CHROMA_COLLECTION_PDF

print("\n── BM25 search for GRS full form ──\n")
results = search_bm25(CHROMA_COLLECTION_PDF, "full form GRS Global Requisition System", top_k=5)
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['document'][:150]}")

print("\n── ChromaDB keyword search for 'Global Requisition System' ──\n")
collection = chroma_store.get_collection(CHROMA_COLLECTION_PDF)
all_data   = collection.get(include=["documents"])

matches = [
    doc for doc in all_data["documents"]
    if "global requisition system" in doc.lower()
]
print(f"  Chunks containing 'Global Requisition System': {len(matches)}")
for m in matches[:3]:
    print(f"\n  → {m[:200]}")