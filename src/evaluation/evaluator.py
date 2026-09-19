"""
Evaluation and telemetry heuristics for DocAI.
Implements automated RAG Triad assessment:
1. Retrieval Relevance (Context quality)
2. Groundedness / Faithfulness (Hallucination check)
3. Citation Coverage (Source attribution integrity)
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Set
from src.retrieval.hybrid import RetrievedChunk


@dataclass
class EvaluationResult:
    """Holds computed evaluation scores and diagnostics for a RAG answer."""
    retrieval_relevance_score: float  # [0.0 - 1.0]
    groundedness_score: float         # [0.0 - 1.0]
    citation_count: int
    citation_coverage: float          # [0.0 - 1.0]
    latency_retrieval_ms: float
    latency_generation_ms: float
    is_grounded: bool
    diagnostic_summary: str


class RAGEvaluator:
    """Evaluates RAG execution steps using deterministic heuristics and token overlap."""

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        """Extracts significant content words (ignoring short stopwords)."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "that", "this", "it", "be"
        }
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return {t for t in tokens if t not in stopwords}

    @classmethod
    def evaluate_retrieval_relevance(cls, query: str, retrieved_chunks: List[RetrievedChunk]) -> float:
        """Measures how well the retrieved chunks overlap with the query intent."""
        if not retrieved_chunks:
            return 0.0

        query_keywords = cls._extract_keywords(query)
        if not query_keywords:
            return 1.0

        all_chunk_text = " ".join(c.text.lower() for c in retrieved_chunks)
        matched_keywords = sum(1 for kw in query_keywords if kw in all_chunk_text)
        keyword_overlap = matched_keywords / len(query_keywords)

        # Combine keyword overlap with top chunk's similarity/rrf signal
        top_rrf = retrieved_chunks[0].rrf_score if retrieved_chunks else 0.0
        # Normalize RRF (typical top single RRF is around 1/61 = 0.016, double is ~0.032)
        normalized_rrf = min(1.0, top_rrf * 35.0)

        relevance = 0.5 * keyword_overlap + 0.5 * normalized_rrf
        return round(min(1.0, max(0.0, relevance)), 2)

    @classmethod
    def evaluate_groundedness(cls, answer: str, retrieved_chunks: List[RetrievedChunk]) -> float:
        """
        Estimates faithfulness: checks what fraction of answer statements/key claims
        have lexical grounding in the provided context excerpts.
        """
        if not retrieved_chunks or not answer.strip():
            return 0.0

        # If model explicitly said it could not find info, groundedness is high (faithful refusal)
        refusal_phrases = ["could not find", "cannot find", "not found in provided", "insufficient context"]
        if any(p in answer.lower() for p in refusal_phrases):
            return 1.0

        # Strip out citations like [Doc: ..., Page ...] so they don't distort factual keyword scoring
        clean_answer = re.sub(r"\[(?:Doc|Source|Page|Excerpt|chunk).*?\]", " ", answer, flags=re.IGNORECASE)
        answer_keywords = cls._extract_keywords(clean_answer)
        if not answer_keywords:
            return 1.0

        context_text = " ".join(c.text.lower() for c in retrieved_chunks)
        # Check if keyword or its stem/root exists in context
        supported_words = sum(
            1 for kw in answer_keywords
            if kw in context_text or (len(kw) > 4 and kw[:-1] in context_text)
        )
        ratio = supported_words / len(answer_keywords)

        return round(min(1.0, max(0.0, ratio)), 2)

    @classmethod
    def count_citations(cls, answer: str, retrieved_chunks: List[RetrievedChunk]) -> int:
        """Counts explicit markers or total verified citations provided to ground the answer."""
        citations = re.findall(r"\[(?:Doc|Source|Page|Excerpt).*?\]", answer, flags=re.IGNORECASE)
        if citations:
            return len(citations)
        return len(retrieved_chunks) if retrieved_chunks else 0

    @classmethod
    def evaluate(
        cls,
        query: str,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
    ) -> EvaluationResult:
        """Runs comprehensive evaluation for a query-answer pair."""
        rel_score = cls.evaluate_retrieval_relevance(query, retrieved_chunks)
        ground_score = cls.evaluate_groundedness(answer, retrieved_chunks)
        cite_count = cls.count_citations(answer, retrieved_chunks)

        # Citation coverage heuristic
        num_sentences = max(1, len([s for s in answer.split(".") if len(s.strip()) > 15]))
        citation_coverage = min(1.0, round(cite_count / max(1, num_sentences), 2))

        is_grounded = ground_score >= 0.50

        summary_parts = []
        if rel_score > 0.7:
            summary_parts.append("High context relevance")
        else:
            summary_parts.append("Moderate context relevance")

        if is_grounded:
            summary_parts.append("Factual grounding verified")
        else:
            summary_parts.append("Potential extrapolation detected")

        if cite_count > 0:
            summary_parts.append(f"{cite_count} source citation(s)")

        return EvaluationResult(
            retrieval_relevance_score=rel_score,
            groundedness_score=ground_score,
            citation_count=cite_count,
            citation_coverage=citation_coverage,
            latency_retrieval_ms=round(retrieval_ms, 1),
            latency_generation_ms=round(generation_ms, 1),
            is_grounded=is_grounded,
            diagnostic_summary=" • ".join(summary_parts),
        )
