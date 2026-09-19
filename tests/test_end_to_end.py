"""
End-to-End integration test for DocAI pipeline.
Tests full multi-document ingestion (PDF, MD, TXT) and hybrid retrieval.
"""

from pathlib import Path
from src.pipeline import DocAIPipeline
from src.config import SAMPLE_DOCS_DIR


def test_end_to_end_pipeline():
    # Initialize fresh pipeline instance
    pipeline = DocAIPipeline()
    pipeline.clear_all()

    sample_dir = Path(SAMPLE_DOCS_DIR)
    assert sample_dir.exists(), "sample_docs directory must exist"

    # Ingest sample files
    files = list(sample_dir.glob("*.*"))
    # Filter only doc files
    doc_files = [f for f in files if f.suffix in [".pdf", ".md", ".txt"]]
    assert len(doc_files) >= 3, f"Expected at least 3 sample documents, found {len(doc_files)}"

    for doc_path in doc_files:
        res = pipeline.ingest_file(str(doc_path), filename=doc_path.name)
        assert res["status"] == "success"
        assert res["metadata"]["total_chunks"] > 0

    assert len(pipeline.indexed_files) >= 3

    # Query 1: Exact technical keyword matching (BM25 + Dense)
    q1 = "What are the requirements for CMEK and encryption keys?"
    res1 = pipeline.ask(q1)
    assert len(res1["retrieved_chunks"]) > 0
    top_chunk1 = res1["retrieved_chunks"][0]
    assert "enterprise_cloud_security_policy.md" in top_chunk1.metadata["source_file"]
    assert "AES-256" in top_chunk1.text

    # Query 2: PDF extraction test
    q2 = "What does the engineering specification say about target response latency?"
    res2 = pipeline.ask(q2)
    assert len(res2["retrieved_chunks"]) > 0
    # Check that engineering_specification.pdf was retrieved
    pdf_found = any("engineering_specification.pdf" in c.metadata.get("source_file", "") for c in res2["retrieved_chunks"])
    assert pdf_found, "Should retrieve chunk from engineering_specification.pdf"

    # Query 3: Multi-turn conversational context
    pipeline.memory.add_user_message("Tell me about CS-482.")
    pipeline.memory.add_assistant_message("CS-482 is Distributed Systems Engineering taught by Prof. Sarah Lin.")
    follow_up = "What is the late policy?"
    res3 = pipeline.ask(follow_up)
    assert len(res3["retrieved_chunks"]) > 0
    top_chunk3 = res3["retrieved_chunks"][0]
    assert "distributed_systems_syllabus.md" in top_chunk3.metadata["source_file"]

    # Clean up
    pipeline.clear_all()
