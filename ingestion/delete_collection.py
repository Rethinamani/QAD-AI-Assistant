# cleanup_incidents.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("qad_knowledge")

before = collection.count()
collection.delete(where={"source": "incidents_excel"})
after = collection.count()

print(f"Deleted incidents. Before: {before}, After: {after}")