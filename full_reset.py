# full_reset.py
import os
import shutil
from config import REGISTRY_DB_PATH, CHROMA_PERSIST_DIR

# ── 1. Delete ChromaDB ──────────────────────────────────────────────────────
if os.path.exists(CHROMA_PERSIST_DIR):
    shutil.rmtree(CHROMA_PERSIST_DIR)
    print(f"Deleted ChromaDB folder: {CHROMA_PERSIST_DIR}")
else:
    print("No ChromaDB folder found — skipping.")

# ── 2. Delete BM25 indexes ───────────────────────────────────────────────────
BM25_PATH = "./vectorstore/bm25_indexes"
if os.path.exists(BM25_PATH):
    shutil.rmtree(BM25_PATH)
    print(f"Deleted BM25 index folder: {BM25_PATH}")
else:
    print("No BM25 index folder found — skipping.")

# ── 3. Delete registry DB ────────────────────────────────────────────────────
if os.path.exists(REGISTRY_DB_PATH):
    os.remove(REGISTRY_DB_PATH)
    print(f"Deleted registry DB: {REGISTRY_DB_PATH}")
else:
    print("No registry DB found — skipping.")

print("\n✅ Full reset complete.")
print("Restart your API — it will reinitialize the registry and rebuild empty BM25 indexes on startup.")