# test_step3b.py
import logging
import os
from ingestion.pdf_ingester import ingest_pdf
from vectorstore.chroma_store import chroma_store
from config import CHROMA_COLLECTION_PDF

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PDF_FILE = os.path.join("data", "uploads", "qad_manual.pdf")

print("\n── Running PDF Ingestion ──")
print("This will take a while — watch the progress bar.\n")

summary = ingest_pdf(PDF_FILE)

print("\n── Ingestion Summary ──")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n── ChromaDB Collection Stats ──")
stats = chroma_store.collection_stats(CHROMA_COLLECTION_PDF)
print(f"  Collection : {stats['collection']}")
print(f"  Total chunks: {stats['count']}")

print("\n── Spot Check — Query ChromaDB ──")
collection = chroma_store.get_collection(CHROMA_COLLECTION_PDF)
results = collection.get(limit=3, include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n  Chunk {i+1}:")
    print(f"    Type    : {meta['chunk_type']}")
    print(f"    Page    : {meta['page']}")
    print(f"    Section : {meta['section']}")
    print(f"    Text    : {doc[:120]}...")

print("\n✅ Step 3 complete.")