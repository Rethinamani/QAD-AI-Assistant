# test_step2b.py
import logging
import os
from ingestion.excel_ingester import ingest_excel
from vectorstore.chroma_store import chroma_store
from config import CHROMA_COLLECTION_EXCEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EXCEL_FILE = os.path.join("data", "uploads", "qad_errors.xlsx")

print("\n── Running Excel Ingestion ──\n")
summary = ingest_excel(EXCEL_FILE)

print("\n── Ingestion Summary ──")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n── ChromaDB Collection Stats ──")
stats = chroma_store.collection_stats(CHROMA_COLLECTION_EXCEL)
print(f"  Collection: {stats['collection']}")
print(f"  Total chunks: {stats['count']}")

print("\n── Spot Check — Query ChromaDB directly ──")
# Quick raw query to verify data is in there
collection = chroma_store.get_collection(CHROMA_COLLECTION_EXCEL)
results = collection.get(limit=2, include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n  Chunk {i+1}:")
    print(f"    Text: {doc[:120]}...")
    print(f"    Error #: {meta['error_number']}")
    print(f"    Source: {meta['source_file']}")
    print(f"    Row: {meta['row_number']}")

print("\n✅ Step 2 complete.")