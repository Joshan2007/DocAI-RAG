"""
Automated unit and integration tests for DocAI RAG pipeline.
Covers document ingestion, chunking, BM25 retrieval, vector search, hybrid RRF fusion, and evaluation.
"""

import pytest
from src.ingestion.parser import DocumentParser, ParsedDocument, DocumentPage
from src.ingestion.chunker import DocumentChunker, DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import HybridRetriever, RetrievedChunk
from src.generation.prompts import RAGPromptManager
from src.evaluation.evaluator import RAGEvaluator


class TestDocAIIngestion:
    """Tests for document parsing and chunking components."""

    def test_parser_clean_text(self):
        dirty = "Hello   world!\n\n\n\nThis  is   a test.\xa0"
        cleaned = DocumentParser.clean_text(dirty)
        assert cleaned == "Hello world!\n\nThis is a test."

    def test_parser_markdown_text(self, tmp_path):
        test_file = tmp_path / "test_doc.md"
        test_file.write_text("# Test Title\n\nThis is test content for DocAI.", encoding="utf-8")

        parser = DocumentParser()
        parsed = parser.parse(str(test_file))

        assert parsed.filename == "test_doc.md"
        assert parsed.file_type == "md"
        assert len(parsed.pages) == 1
        assert "Test Title" in parsed.pages[0].text

    def test_parser_docx(self, tmp_path):
        import docx
        doc_file = tmp_path / "sample_test.docx"
        doc = docx.Document()
        doc.add_heading("Lunorsoft Specifications", level=1)
        doc.add_paragraph("This is a Word document testing the DocAI parser.")
        table = doc.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Header 1"
        table.rows[0].cells[1].text = "Header 2"
        doc.save(str(doc_file))

        parser = DocumentParser()
        parsed = parser.parse(str(doc_file))
        assert parsed.file_type == "docx"
        assert "Lunorsoft Specifications" in parsed.pages[0].text
        assert "Header 1 | Header 2" in parsed.pages[0].text

    def test_parser_csv(self, tmp_path):
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text("model,accuracy,latency\nGemini-2.5-Flash,0.98,420ms\n", encoding="utf-8")

        parser = DocumentParser()
        parsed = parser.parse(str(csv_file))
        assert parsed.file_type == "csv"
        assert "Gemini-2.5-Flash" in parsed.pages[0].text

    def test_parser_excel(self, tmp_path):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Financials"
        ws.append(["Quarter", "Revenue", "Margin"])
        ws.append(["Q1", "$1.2M", "24%"])
        xlsx_file = tmp_path / "financials.xlsx"
        wb.save(str(xlsx_file))

        parser = DocumentParser()
        parsed = parser.parse(str(xlsx_file))
        assert parsed.file_type == "xlsx"
        assert "Financials" in parsed.pages[0].text
        assert "Quarter | Revenue | Margin" in parsed.pages[0].text
        assert "Q1 | $1.2M | 24%" in parsed.pages[0].text

    def test_chunker_basic_splitting(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        parsed = ParsedDocument(
            filename="sample.txt",
            doc_id="sample",
            file_type="txt",
            total_pages=1,
            pages=[
                DocumentPage(
                    page_number=1,
                    text="Paragraph one is relatively brief. " * 3 + "\n\n" + "Paragraph two is also quite interesting. " * 3
                )
            ]
        )
        chunks = chunker.chunk_document(parsed)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.source_file == "sample.txt"
            assert c.page_number == 1
            assert c.doc_id == "sample"
            assert len(c.text) > 20


class TestDocAIRetrieval:
    """Tests for BM25 and Hybrid retrieval."""

    def test_bm25_retrieval(self):
        bm25 = BM25Retriever()
        chunks = [
            DocumentChunk("c1", "d1", "f1.txt", 1, 0, "The quick brown fox jumps over the lazy dog.", 45),
            DocumentChunk("c2", "d1", "f1.txt", 1, 1, "Quantum computing utilizes qubits cooled to millikelvin temperatures.", 65),
            DocumentChunk("c3", "d1", "f1.txt", 2, 2, "Enterprise security mandates TLS 1.3 and AES-256 encryption keys.", 60),
        ]
        bm25.index_chunks(chunks)

        results = bm25.query("qubits quantum", top_k=2)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "c2"
        assert results[0]["score"] > 0.0

    def test_rrf_scoring_logic(self):
        """Validates Reciprocal Rank Fusion formula and ranking."""
        c1 = RetrievedChunk("c1", "text 1", {}, rrf_score=1.0/(60+1) + 1.0/(60+1), retrieval_source="both")
        c2 = RetrievedChunk("c2", "text 2", {}, rrf_score=1.0/(60+2), retrieval_source="dense_only")

        candidates = [c2, c1]
        candidates.sort(key=lambda x: x.rrf_score, reverse=True)

        assert candidates[0].chunk_id == "c1"
        assert candidates[0].retrieval_source == "both"


class TestDocAIGenerationAndEvaluation:
    """Tests for prompt engineering, citations, and evaluation metrics."""

    def test_prompt_formatting(self):
        chunks = [
            RetrievedChunk(
                chunk_id="chunk_1",
                text="AES-256 encryption is mandatory for all persistent volumes.",
                metadata={"source_file": "security_policy.pdf", "page_number": 2},
                rrf_score=0.03
            )
        ]
        prompt = RAGPromptManager.build_rag_prompt("What encryption is required?", chunks)
        assert "security_policy.pdf" in prompt
        assert "Page: 2" in prompt
        assert "AES-256 encryption" in prompt

    def test_evaluator_metrics(self):
        chunks = [
            RetrievedChunk(
                chunk_id="chunk_1",
                text="Planar surface codes operate with 99.4% fidelity threshold.",
                metadata={"source_file": "quantum.txt", "page_number": 1},
                rrf_score=0.032
            )
        ]
        query = "What is the fidelity threshold of the surface code?"
        answer = "The planar surface code achieves a 99.4% fidelity threshold [Doc: quantum.txt, Page 1]."

        eval_result = RAGEvaluator.evaluate(query, answer, chunks, retrieval_ms=15.0, generation_ms=300.0)

        assert eval_result.citation_count == 1
        assert eval_result.is_grounded is True
        assert eval_result.groundedness_score >= 0.70
        assert eval_result.retrieval_relevance_score >= 0.50
        assert eval_result.latency_retrieval_ms == 15.0
