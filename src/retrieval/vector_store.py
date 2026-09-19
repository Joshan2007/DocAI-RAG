"""
Dense vector store manager for DocAI using ChromaDB and Sentence-Transformers.
Handles embedding generation, persistent collection indexing, and cosine similarity querying.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, TOP_K_DENSE
from src.ingestion.chunker import DocumentChunk


class VectorStore:
    """Manages the local ChromaDB vector store collection for dense semantic search with auto-healing resilience."""

    def __init__(self, persist_directory: Optional[str] = CHROMA_PERSIST_DIR, collection_name: Optional[str] = None):
        self.persist_directory = persist_directory

        # Initialize client: in-memory if persist_directory is None or ':memory:', otherwise persistent
        if self.persist_directory and self.persist_directory != ":memory:":
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection_name = collection_name or "docai_knowledge_base"
        else:
            import uuid
            self.client = chromadb.EphemeralClient()
            self.collection_name = collection_name or f"docai_kb_{uuid.uuid4().hex}"

        # Use ChromaDB's high-speed ONNX all-MiniLM-L6-v2 embedding function
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Ensure collection is created and reachable
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensures the collection exists and is fresh. Handles recreation on NotFoundError."""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            _ = self.collection.count()
        except Exception:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collection

    @property
    def valid_collection(self):
        """Returns active collection, auto-recovering if invalidated."""
        try:
            _ = self.collection.count()
            return self.collection
        except Exception:
            return self._ensure_collection()

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Adds or updates a batch of document chunks in the vector collection."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata_dict() for chunk in chunks]

        # Chroma upsert ensures idempotency
        try:
            self.valid_collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        except Exception:
            self._ensure_collection().upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        return len(chunks)

    def query(self, query_text: str, top_k: int = TOP_K_DENSE) -> List[Dict[str, Any]]:
        """Queries vector collection for top-k semantically relevant chunks."""
        try:
            coll = self.valid_collection
            count = coll.count()
        except Exception:
            coll = self._ensure_collection()
            count = coll.count()

        if count == 0:
            return []

        actual_k = min(top_k, count)
        try:
            results = coll.query(
                query_texts=[query_text],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            coll = self._ensure_collection()
            results = coll.query(
                query_texts=[query_text],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"]
            )

        formatted_results: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted_results

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            dist = distances[i]
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            formatted_results.append({
                "chunk_id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "score": float(similarity),
                "distance": float(dist),
                "source": "dense",
            })

        return formatted_results

    def delete_document(self, doc_identifier: str) -> None:
        """Removes all chunks associated with a specific document (by filename or doc_id)."""
        coll = self.valid_collection
        # Try deleting by source_file first
        try:
            coll.delete(where={"source_file": doc_identifier})
        except Exception:
            pass
        # Try deleting by doc_id as well
        try:
            coll.delete(where={"doc_id": doc_identifier})
        except Exception:
            pass

    def count(self) -> int:
        """Returns total chunk count in the collection."""
        try:
            return self.valid_collection.count()
        except Exception:
            return self._ensure_collection().count()

    def clear(self) -> None:
        """Clears all entries from the current collection completely."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            try:
                coll = self.valid_collection
                all_ids = coll.get()["ids"]
                if all_ids:
                    coll.delete(ids=all_ids)
            except Exception:
                pass
        self._ensure_collection()
