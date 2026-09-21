# test_step1.py
from registry.document_registry import (
    initialize_registry,
    register_document,
    update_status,
    list_all
)
from vectorstore.chroma_store import chroma_store

print("\n── Testing Document Registry ──")
initialize_registry()

doc_id = register_document("qad_manual.pdf", "pdf")
update_status(doc_id, "ready", chunk_count=42)

doc_id2 = register_document("qad_errors.xlsx", "excel")
update_status(doc_id2, "ready", chunk_count=150)

print("\nAll registered documents:")
for doc in list_all():
    print(doc)

print("\n── Testing ChromaDB ──")
for stat in chroma_store.all_stats():
    print(stat)

print("\n✅ Step 1 complete.")