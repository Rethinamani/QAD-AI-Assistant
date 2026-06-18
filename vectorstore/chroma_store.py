# vectorstore/chroma_store.py
import chromadb
from chromadb.config import Settings
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_PDF,
    CHROMA_COLLECTION_EXCEL,
    CHROMA_COLLECTION_SN,
)


class ChromaStore:
    """
    Manages all ChromaDB collections for the QAD Support Assistant.
    Single instance shared across the application.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collections = {}
        self._initialize_collections()

    def _initialize_collections(self):
        """Create all 3 collections if they don't already exist."""
        for name in [
            CHROMA_COLLECTION_PDF,
            CHROMA_COLLECTION_EXCEL,
            CHROMA_COLLECTION_SN,
        ]:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Collection ready: {name}")

    def get_collection(self, name: str):
        """Return a collection by name."""
        if name not in self._collections:
            raise ValueError(f"Unknown collection: {name}")
        return self._collections[name]

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ):
        """
        Add documents with their embeddings to a collection.
        All lists must be the same length.
        """
        if not (len(documents) == len(embeddings) == len(metadatas) == len(ids)):
            raise ValueError(
                "documents, embeddings, metadatas and ids must all be the same length."
            )

        collection = self.get_collection(collection_name)
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"✅ Upserted {len(documents)} chunks into '{collection_name}'")

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 20,
        where: dict = None,
    ) -> dict:
        """
        Dense vector search on a collection.
        Returns ChromaDB result dict with documents, metadatas, distances.
        """
        collection = self.get_collection(collection_name)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return collection.query(**kwargs)

    def delete_by_doc_id(self, collection_name: str, doc_id: str):
        """Delete all chunks belonging to a specific document."""
        collection = self.get_collection(collection_name)
        results = collection.get(where={"doc_id": doc_id})
        ids_to_delete = results.get("ids", [])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"🗑️  Deleted {len(ids_to_delete)} chunks for doc_id={doc_id}")
        else:
            print(f"⚠️  No chunks found for doc_id={doc_id} in '{collection_name}'")

    def collection_stats(self, collection_name: str) -> dict:
        """Return count and name of a collection."""
        collection = self.get_collection(collection_name)
        return {
            "collection": collection_name,
            "count": collection.count(),
        }

    def all_stats(self) -> list[dict]:
        """Return stats for all 3 collections."""
        return [self.collection_stats(name) for name in self._collections]


# Singleton instance — import this everywhere
chroma_store = ChromaStore()