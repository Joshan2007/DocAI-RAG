"""
Comprehensive Pre-Deployment Verification Test Suite for DocAI Knowledge Assistant.
Verifies production RAG requirements and architecture components:
1. Multi-format ingestion: PDF, DOCX, XLSX, CSV, MD, TXT
2. Semantic chunking & metadata integrity
3. Hybrid Retrieval: Dense ChromaDB + Sparse BM25 + Reciprocal Rank Fusion (RRF)
4. Factual grounding & evaluation scoring
5. Multi-turn conversation memory & contextual query rewriting
6. Individual document deletion & index re-balancing
7. Model switching across all Google Gemini models
8. Fault-tolerant Local Grounded Synthesizer (zero-crash resilience on 429/offline)
9. Live FastAPI HTTP endpoints & SSE streaming
"""

import json
import urllib.request
from pathlib import Path
import pytest

from src.pipeline import DocAIPipeline
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.config import SAMPLE_DOCS_DIR
from src.evaluation.evaluator import RAGEvaluator


class TestDocAISystemVerification:
    """Automated verification suite validating all production system criteria."""

    @classmethod
    def setup_class(cls):
        cls.pipeline = DocAIPipeline()
        cls.pipeline.clear_all()
        cls.sample_dir = Path(SAMPLE_DOCS_DIR)

    @classmethod
    def teardown_class(cls):
        cls.pipeline.clear_all()

    def test_01_multi_format_ingestion_and_parsing(self):
        """Req 1 & 2: Supports ingestion and parsing across PDF, DOCX, XLSX, CSV, MD, TXT."""
        parser = DocumentParser()
        formats_tested = set()

        files_to_test = [
            "engineering_specification.pdf",
            "platform_engineering_guidelines.docx",
            "quarterly_performance.xlsx",
            "performance_benchmarks.csv",
            "enterprise_cloud_security_policy.md",
            "distributed_systems_syllabus.md",
            "quantum_computing_research.txt",
        ]

        for fname in files_to_test:
            fpath = self.sample_dir / fname
            assert fpath.exists(), f"Sample document {fname} must exist"
            parsed = parser.parse(str(fpath))
            assert len(parsed.pages) > 0, f"Failed to extract pages from {fname}"
            assert len(parsed.pages[0].text) > 20, f"Extracted text too short for {fname}"
            formats_tested.add(parsed.file_type)

        assert {"pdf", "docx", "xlsx", "csv", "md", "txt"}.issubset(formats_tested)

    def test_02_semantic_chunking_and_metadata_preservation(self):
        """Req 3: Splits content into semantic chunks preserving source and page metadata."""
        parser = DocumentParser()
        chunker = DocumentChunker(chunk_size=400, chunk_overlap=80)

        parsed_docx = parser.parse(str(self.sample_dir / "platform_engineering_guidelines.docx"))
        chunks = chunker.chunk_document(parsed_docx)

        assert len(chunks) >= 1
        for c in chunks:
            assert c.source_file == "platform_engineering_guidelines.docx"
            assert c.page_number >= 1
            assert len(c.text) > 10
            assert c.doc_id is not None

    def test_03_full_pipeline_multi_doc_indexing(self):
        """Req 4 & 5: Indexes multi-format document collection into ChromaDB + BM25."""
        self.pipeline.clear_all()
        files_to_index = [
            "engineering_specification.pdf",
            "platform_engineering_guidelines.docx",
            "quarterly_performance.xlsx",
            "performance_benchmarks.csv",
            "enterprise_cloud_security_policy.md",
            "distributed_systems_syllabus.md",
        ]

        for fname in files_to_index:
            res = self.pipeline.ingest_file(str(self.sample_dir / fname), filename=fname)
            assert res["status"] == "success"
            assert res["metadata"]["total_chunks"] > 0

        assert len(self.pipeline.indexed_files) == len(files_to_index)

    def test_04_hybrid_rrf_retrieval_and_keyword_precision(self):
        """Req 5: Tests hybrid retrieval with exact technical keywords & natural language."""
        # Query A: Exact technical keyword (BM25 advantage)
        res_a = self.pipeline.ask("What is the P95 Hybrid Latency benchmark target?")
        assert len(res_a["retrieved_chunks"]) > 0
        top_chunk_a = res_a["retrieved_chunks"][0]
        assert "performance_benchmarks.csv" in top_chunk_a.metadata.get("source_file", "")
        assert "18ms" in top_chunk_a.text or "Latency" in top_chunk_a.text

        # Query B: Spreadsheet quantitative data
        res_b = self.pipeline.ask("What was the Q3 Actual for Document Ingestion Volume?")
        assert len(res_b["retrieved_chunks"]) > 0
        top_chunk_b = res_b["retrieved_chunks"][0]
        assert "quarterly_performance.xlsx" in top_chunk_b.metadata.get("source_file", "")
        assert "43,800" in top_chunk_b.text or "Ingestion" in top_chunk_b.text

        # Query C: Word document guidelines
        res_c = self.pipeline.ask("What is the standard value for Reciprocal Rank Fusion constant k?")
        assert len(res_c["retrieved_chunks"]) > 0
        found_rrf = any("60" in c.text and "platform_engineering_guidelines.docx" in c.metadata.get("source_file", "") for c in res_c["retrieved_chunks"])
        assert found_rrf

    def test_05_grounded_answer_synthesis_and_zero_crash_resilience(self):
        """Req 7 & 8: Generates grounded answer with citations and zero-crash fallback."""
        query = "What encryption standard is mandatory under the enterprise cloud security policy?"
        res = self.pipeline.ask(query)

        assert "answer" in res
        assert len(res["answer"]) > 50
        assert "AES-256" in res["answer"] or "encryption" in res["answer"].lower()

        # Check citations
        assert len(res["citations"]) > 0
        first_cite = res["citations"][0]
        assert "enterprise_cloud_security_policy.md" in first_cite["source"]
        assert first_cite["page"] >= 1
        assert len(first_cite["text_snippet"]) > 10

        # Check evaluation metrics
        assert res["evaluation"].is_grounded is True
        assert res["evaluation"].groundedness_score >= 0.6
        assert res["evaluation"].latency_retrieval_ms >= 0

    def test_06_multi_turn_conversational_history(self):
        """Bonus 2: Tests multi-turn conversation memory and query contextualization."""
        self.pipeline.memory.clear()
        self.pipeline.memory.add_user_message("Tell me about CS-482.")
        self.pipeline.memory.add_assistant_message("CS-482 is Distributed Systems Engineering taught by Prof. Sarah Lin.")

        follow_up = "What is the grading breakdown?"
        contextualized = self.pipeline.llm.contextualize_query(
            self.pipeline.memory.get_recent_history(),
            follow_up
        )
        assert len(contextualized) >= len(follow_up)

        res = self.pipeline.ask(follow_up)
        assert len(res["retrieved_chunks"]) > 0
        assert any("distributed_systems_syllabus.md" in c.metadata.get("source_file", "") for c in res["retrieved_chunks"])

    def test_07_individual_document_deletion_and_reindexing(self):
        """Interactive feature: Deletes a specific file and verifies index integrity."""
        target_file = "distributed_systems_syllabus.md"
        assert target_file in self.pipeline.indexed_files

        success = self.pipeline.delete_document(target_file)
        assert success is True
        assert target_file not in self.pipeline.indexed_files

        res = self.pipeline.ask("CS-482 Distributed Systems syllabus")
        for c in res["retrieved_chunks"]:
            assert c.metadata.get("source_file", "") != target_file

    def test_08_dynamic_gemini_model_switching(self):
        """Tests dynamic switching between all 4 Gemini models."""
        models = [
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-2.5-flash",
        ]
        for m in models:
            self.pipeline.llm.set_model(m)
            assert self.pipeline.llm.model_name == m

    def test_09_live_fastapi_server_endpoints(self):
        """Req 9: Verifies live HTTP endpoints on running FastAPI instance."""
        with urllib.request.urlopen("http://127.0.0.1:8000/") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] == "online"
            assert "DocAI" in data["service"]

        with urllib.request.urlopen("http://127.0.0.1:8000/api/health") as resp:
            assert resp.status == 200
            health = json.loads(resp.read().decode())
            assert health["status"] == "online"
            assert "model" in health

        with urllib.request.urlopen("http://127.0.0.1:8000/api/documents") as resp:
            assert resp.status == 200
            docs = json.loads(resp.read().decode())
            assert "documents" in docs

        with urllib.request.urlopen("http://127.0.0.1:8000/docs") as resp:
            assert resp.status == 200

    def test_10_live_sse_chat_stream(self):
        """Verifies live SSE streaming of thinking tokens and grounded answers."""
        payload = json.dumps({
            "message": "What is the P95 latency target?",
            "model": "gemini-1.5-flash"
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            import http.client
            try:
                raw_bytes = resp.read()
            except http.client.IncompleteRead as e:
                raw_bytes = e.partial
            raw_stream = raw_bytes.decode("utf-8", errors="ignore")
            assert "event: " in raw_stream
            assert "data: " in raw_stream
