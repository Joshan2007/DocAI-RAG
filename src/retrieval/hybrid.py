"""
Hybrid Retrieval Engine for DocAI.
Merges Dense Vector Search (ChromaDB) and Sparse Lexical Search (BM25)
using Reciprocal Rank Fusion (RRF) to maximize both semantic and keyword recall.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.config import TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FINAL, RRF_K
from src.ingestion.chunker import DocumentChunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever


@dataclass
class RetrievedChunk:
    """Represents a unified retrieved chunk enriched with retrieval telemetry."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    rrf_score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    retrieval_source: str = "hybrid"  # "both", "dense_only", or "bm25_only"


class HybridRetriever:
    """Orchestrates hybrid retrieval over both VectorStore and BM25Retriever."""

    def __init__(self, vector_store: Optional[VectorStore] = None, bm25_retriever: Optional[BM25Retriever] = None):
        self.vector_store = vector_store or VectorStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()

    def index_documents(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Indexes document chunks into both dense vector store and BM25 index."""
        dense_count = self.vector_store.add_chunks(chunks)
        sparse_count = self.bm25_retriever.index_chunks(chunks)
        return {"dense_indexed": dense_count, "sparse_indexed": sparse_count}

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        k_dense: int = TOP_K_DENSE,
        k_sparse: int = TOP_K_SPARSE,
        rrf_constant: int = RRF_K,
    ) -> List[RetrievedChunk]:
        """
        Executes hybrid retrieval:
        1. Queries ChromaDB for top dense candidates.
        2. Queries BM25 for top lexical candidates.
        3. Applies Reciprocal Rank Fusion (RRF) to combine candidate rankings.
        4. Sorts and returns top-K fused chunks.
        """
        dense_results = self.vector_store.query(query, top_k=k_dense)
        sparse_results = self.bm25_retriever.query(query, top_k=k_sparse)

        # Fallback to single mode if one retriever has no results
        if not dense_results and not sparse_results:
            return []

        # Map to store fused candidate info by chunk_id
        candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Results
        for rank, res in enumerate(dense_results, start=1):
            cid = res["chunk_id"]
            rrf_contribution = 1.0 / (rrf_constant + rank)
            candidates[cid] = {
                "chunk_id": cid,
                "text": res["text"],
                "metadata": res["metadata"],
                "rrf_score": rrf_contribution,
                "dense_score": res["score"],
                "dense_rank": rank,
                "bm25_score": None,
                "bm25_rank": None,
                "in_dense": True,
                "in_sparse": False,
            }

        # 2. Process Sparse BM25 Results
        for rank, res in enumerate(sparse_results, start=1):
            cid = res["chunk_id"]
            rrf_contribution = 1.0 / (rrf_constant + rank)
            if cid in candidates:
                candidates[cid]["rrf_score"] += rrf_contribution
                candidates[cid]["bm25_score"] = res["score"]
                candidates[cid]["bm25_rank"] = rank
                candidates[cid]["in_sparse"] = True
            else:
                candidates[cid] = {
                    "chunk_id": cid,
                    "text": res["text"],
                    "metadata": res["metadata"],
                    "rrf_score": rrf_contribution,
                    "dense_score": None,
                    "dense_rank": None,
                    "bm25_score": res["score"],
                    "bm25_rank": rank,
                    "in_dense": False,
                    "in_sparse": True,
                }

        # 3. Classify retrieval source & construct RetrievedChunk objects
        fused_chunks: List[RetrievedChunk] = []
        for cid, data in candidates.items():
            if data["in_dense"] and data["in_sparse"]:
                source_type = "both"
            elif data["in_dense"]:
                source_type = "dense_only"
            else:
                source_type = "bm25_only"

            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=data["text"],
                    metadata=data["metadata"],
                    rrf_score=data["rrf_score"],
                    dense_score=data["dense_score"],
                    bm25_score=data["bm25_score"],
                    dense_rank=data["dense_rank"],
                    bm25_rank=data["bm25_rank"],
                    retrieval_source=source_type,
                )
            )

        # 4. Sort by RRF score descending
        fused_chunks.sort(key=lambda x: x.rrf_score, reverse=True)

        return fused_chunks[:top_k]

    def clear(self) -> None:
        """Clears both vector store and BM25 index."""
        self.vector_store.clear()
        self.bm25_retriever.clear()
