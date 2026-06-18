# Add to reset_pdf_collection.py or run separately
from vectorstore.chroma_store import chroma_store
from config import CHROMA_COLLECTION_PDF

# Reset PDF collection
collection = chroma_store.get_collection(CHROMA_COLLECTION_PDF)
all_ids = collection.get()["ids"]
if all_ids:
    collection.delete(ids=all_ids)
    print(f"🗑️  Deleted {len(all_ids)} chunks from pdf_collection")