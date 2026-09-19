"""
Sparse lexical BM25 retriever for DocAI.
Performs exact-keyword and technical entity matching using BM25Okapi.
"""

import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.config import TOP_K_SPARSE
from src.ingestion.chunker import DocumentChunk


class BM25Retriever:
    """Manages an in-memory lexical index of document chunks using BM25Okapi."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text into normalized lowercase alphanumeric terms."""
        text = text.lower()
        # Keep alphanumeric words and common symbols like underscores
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Indexes or appends document chunks into the BM25 corpus."""
        if not chunks:
            return 0

        # Avoid re-adding duplicate chunk IDs
        existing_ids = {c.chunk_id for c in self.chunks}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)
        new_tokens = [self.tokenize(c.text) for c in new_chunks]
        self.corpus_tokens.extend(new_tokens)

        # Build / rebuild BM25 model
        self.bm25 = BM25Okapi(self.corpus_tokens)
        return len(new_chunks)

    def query(self, query_text: str, top_k: int = TOP_K_SPARSE) -> List[Dict[str, Any]]:
        """Queries the lexical index and returns the top-k scoring chunks."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query_text)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Sort indices by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        actual_k = min(top_k, len(sorted_indices))
        top_indices = sorted_indices[:actual_k]

        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            # Only include chunks that have non-zero lexical overlap
            if raw_score <= 0.0:
                continue

            chunk = self.chunks[idx]
            normalized_score = min(1.0, raw_score / max_score)

            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.to_metadata_dict(),
                "score": normalized_score,
                "raw_bm25_score": raw_score,
                "source": "sparse_bm25",
            })

        return results

    def delete_document(self, source_file: str) -> int:
        """Removes all chunks belonging to a document and rebuilds BM25 model."""
        initial_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source_file != source_file]
        deleted_count = initial_count - len(self.chunks)

        if deleted_count > 0:
            if self.chunks:
                self.corpus_tokens = [self.tokenize(c.text) for c in self.chunks]
                self.bm25 = BM25Okapi(self.corpus_tokens)
            else:
                self.corpus_tokens = []
                self.bm25 = None

        return deleted_count

    def clear(self) -> None:
        """Clears all indexed chunks."""
        self.chunks = []
        self.corpus_tokens = []
        self.bm25 = None
