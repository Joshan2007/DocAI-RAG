"""
DocAI Pipeline Orchestrator.
Glues ingestion, hybrid indexing, multi-turn memory, grounded generation, and evaluation.
"""

import time
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Generator

from src.ingestion.parser import DocumentParser, ParsedDocument
from src.ingestion.chunker import DocumentChunker, DocumentChunk
from src.retrieval.hybrid import HybridRetriever, RetrievedChunk
from src.generation.llm_client import LLMClient
from src.generation.prompts import RAGPromptManager
from src.generation.memory import ConversationMemory
from src.evaluation.evaluator import RAGEvaluator, EvaluationResult


class DocAIPipeline:
    """End-to-end pipeline orchestrating document processing, hybrid retrieval, and answer generation."""

    def __init__(self, api_key: Optional[str] = None):
        self.parser = DocumentParser()
        self.chunker = DocumentChunker()
        self.retriever = HybridRetriever()
        self.llm = LLMClient(api_key=api_key)
        self.memory = ConversationMemory()
        self.indexed_files: Dict[str, Dict[str, Any]] = {}

    def ingest_file(self, file_source: Union[str, Path, io.BytesIO], filename: Optional[str] = None) -> Dict[str, Any]:
        """Parses, chunks, and indexes a single document."""
        start_t = time.time()
        parsed_doc: ParsedDocument = self.parser.parse(file_source, filename=filename)
        chunks: List[DocumentChunk] = self.chunker.chunk_document(parsed_doc)

        index_stats = self.retriever.index_documents(chunks)
        duration_ms = (time.time() - start_t) * 1000

        file_meta = {
            "filename": parsed_doc.filename,
            "file_type": parsed_doc.file_type,
            "total_pages": parsed_doc.total_pages,
            "total_chunks": len(chunks),
            "ingest_time_ms": round(duration_ms, 1),
        }
        self.indexed_files[parsed_doc.filename] = file_meta

        return {
            "status": "success",
            "metadata": file_meta,
            "index_stats": index_stats,
        }

    def ask(self, user_query: str) -> Dict[str, Any]:
        """Processes a question through contextualization, hybrid retrieval, and grounded generation."""
        # 1. Multi-turn Query Contextualization
        recent_history = self.memory.get_recent_history()
        search_query = self.llm.contextualize_query(recent_history, user_query)

        # 2. Hybrid Retrieval
        t_ret_start = time.time()
        retrieved_chunks = self.retriever.retrieve(search_query)
        t_ret_ms = (time.time() - t_ret_start) * 1000

        # 3. Prompt Construction & Generation
        prompt = RAGPromptManager.build_rag_prompt(user_query, retrieved_chunks)
        t_gen_start = time.time()
        answer = self.llm.generate(prompt)
        t_gen_ms = (time.time() - t_gen_start) * 1000

        # 4. Evaluation Heuristics
        evaluation = RAGEvaluator.evaluate(
            query=user_query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            retrieval_ms=t_ret_ms,
            generation_ms=t_gen_ms,
        )

        # 5. Update Memory
        citations_data = [
            {
                "source": c.metadata.get("source_file", "Unknown"),
                "page": c.metadata.get("page_number", 1),
                "chunk_id": c.chunk_id,
                "score": round(c.rrf_score, 4),
                "text_snippet": c.text[:180] + "..." if len(c.text) > 180 else c.text,
                "retrieval_method": c.retrieval_source,
            }
            for c in retrieved_chunks
        ]

        self.memory.add_user_message(user_query)
        self.memory.add_assistant_message(
            content=answer,
            citations=citations_data,
            metrics={
                "retrieval_ms": round(t_ret_ms, 1),
                "generation_ms": round(t_gen_ms, 1),
                "relevance_score": evaluation.retrieval_relevance_score,
                "groundedness_score": evaluation.groundedness_score,
            }
        )

        return {
            "query": user_query,
            "search_query": search_query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "citations": citations_data,
            "evaluation": evaluation,
        }

    def ask_stream(self, user_query: str) -> Generator[Dict[str, Any], None, None]:
        """Streams answer generation for interactive UIs, yielding retrieved context first then tokens."""
        recent_history = self.memory.get_recent_history()
        search_query = self.llm.contextualize_query(recent_history, user_query)

        t_ret_start = time.time()
        retrieved_chunks = self.retriever.retrieve(search_query)
        t_ret_ms = (time.time() - t_ret_start) * 1000

        citations_data = [
            {
                "source": c.metadata.get("source_file", "Unknown"),
                "page": c.metadata.get("page_number", 1),
                "chunk_id": c.chunk_id,
                "score": round(c.rrf_score, 4),
                "text_snippet": c.text[:180] + "..." if len(c.text) > 180 else c.text,
                "retrieval_method": c.retrieval_source,
            }
            for c in retrieved_chunks
        ]

        # First yield the metadata & citations
        yield {
            "type": "retrieval_complete",
            "search_query": search_query,
            "citations": citations_data,
            "retrieval_ms": round(t_ret_ms, 1),
            "chunks": retrieved_chunks,
        }

        # Stream generation
        prompt = RAGPromptManager.build_rag_prompt(user_query, retrieved_chunks)
        t_gen_start = time.time()
        accumulated_text = ""

        for token in self.llm.stream_generate(prompt):
            accumulated_text += token
            yield {"type": "token", "token": token}

        t_gen_ms = (time.time() - t_gen_start) * 1000

        evaluation = RAGEvaluator.evaluate(
            query=user_query,
            answer=accumulated_text,
            retrieved_chunks=retrieved_chunks,
            retrieval_ms=t_ret_ms,
            generation_ms=t_gen_ms,
        )

        self.memory.add_user_message(user_query)
        self.memory.add_assistant_message(
            content=accumulated_text,
            citations=citations_data,
            metrics={
                "retrieval_ms": round(t_ret_ms, 1),
                "generation_ms": round(t_gen_ms, 1),
                "relevance_score": evaluation.retrieval_relevance_score,
                "groundedness_score": evaluation.groundedness_score,
            }
        )

        yield {
            "type": "generation_complete",
            "full_answer": accumulated_text,
            "evaluation": evaluation,
        }

    def delete_document(self, filename: str) -> bool:
        """Removes a specific document from ChromaDB vector store, BM25 index, and registry."""
        if filename in self.indexed_files:
            doc_meta = self.indexed_files[filename]
            from pathlib import Path
            doc_id = doc_meta.get("doc_id", Path(filename).stem)

            # 1. Delete from ChromaDB vector store
            try:
                self.retriever.vector_store.delete_document(doc_id)
            except Exception:
                pass

            # 2. Delete from BM25 index and rebuild
            try:
                self.retriever.bm25_retriever.delete_document(filename)
            except Exception:
                pass

            # 3. Remove from indexed_files registry
            del self.indexed_files[filename]
            return True
        return False

    def clear_all(self) -> None:
        """Clears all indexed documents, vector collections, BM25 indices, and conversation memory."""
        self.retriever.clear()
        self.memory.clear()
        self.indexed_files.clear()

    def clear(self) -> None:
        """Alias for clear_all."""
        self.clear_all()
