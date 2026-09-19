"""
Prompt templates and formatting for DocAI.
Enforces strict factual grounding and query contextualization.
"""

from typing import List, Dict
from src.retrieval.hybrid import RetrievedChunk


class RAGPromptManager:
    """Constructs prompts for RAG generation and multi-turn query contextualization."""

    SYSTEM_PROMPT = (
        "You are DocAI, an advanced, highly reliable, production-grade AI Knowledge Assistant. "
        "Your mission is to provide direct, comprehensive, and accurately grounded answers based on the provided document excerpts.\n\n"
        "REASONING PROTOCOL:\n"
        "Before generating your final response, briefly analyze the query and retrieved context enclosed in <thought>...</thought> tags.\n"
        "Keep your <thought> concise (3 to 5 sentences): identify the question intent, note which document pages or sections contain the relevant facts, and plan a clear answer structure.\n\n"
        "RESPONSE RULES:\n"
        "1. Direct Answer First: Immediately after the </thought> closing tag, provide your final direct response. Do NOT repeat or explain your thinking process in the answer.\n"
        "2. Clean Text (No Inline Citations): Do NOT insert clumsy bracketed citations (like [Doc: ..., Page ...]) inside sentences or paragraphs. Write smooth, readable, professional prose.\n"
        "3. Source Formatting: Do not add a '### Sources' section or repeat the document metadata; the application displays retrieved sources separately when requested.\n"
        "4. Thoroughness: Directly answer all parts of the user's question with depth, structured formatting (clear headings, bullet points, and bold key terms).\n"
        "5. If a topic is covered in the excerpts (e.g. assignment tasks, problem statements, requirements), provide a complete and detailed breakdown of what the document specifies."
    )

    QUERY_CONTEXTUALIZER_SYSTEM = (
        "You are an expert search query reformulator. Given the chat history and the user's latest question, "
        "rewrite the latest question into a single, clear, self-contained search query. "
        "Resolve any pronouns (e.g. 'it', 'they', 'this', 'that') using the prior conversation context. "
        "Do NOT answer the question. Return ONLY the reformulated search query text."
    )

    @staticmethod
    def format_context_block(retrieved_chunks: List[RetrievedChunk]) -> str:
        """Formats retrieved chunks into a numbered, structured context block with clear metadata."""
        if not retrieved_chunks:
            return "No relevant context found in uploaded documents."

        formatted_passages = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source_name = chunk.metadata.get("source_file", "Unknown Document")
            page_num = chunk.metadata.get("page_number", 1)
            formatted_passages.append(
                f"--- [EXCERPT {i}] (Source: {source_name}, Page: {page_num}) ---\n"
                f"{chunk.text}\n"
            )

        return "\n".join(formatted_passages)

    @classmethod
    def build_rag_prompt(cls, query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Constructs the complete prompt sent to the LLM."""
        context_block = cls.format_context_block(retrieved_chunks)
        source_pages = {}
        for chunk in retrieved_chunks:
            source = chunk.metadata.get("source_file", "Unknown Document")
            page = chunk.metadata.get("page_number", 1)
            source_pages.setdefault(source, set()).add(page)

        source_summary = "; ".join(
            f"{source} (page(s): {', '.join(str(page) for page in sorted(pages))})"
            for source, pages in source_pages.items()
        ) or "no identified source"

        user_content = (
            f"SOURCE SCOPE: The retrieved excerpts come from exactly {len(source_pages)} document(s): {source_summary}. "
            "Do not claim that the material covers multiple documents or unrelated topics unless that is explicitly supported by these excerpts. "
            "Do not invent additional documents, topics, or facts.\n\n"
            f"CONTEXT EXCERPTS:\n{context_block}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER (Ground your answer strictly in the excerpts above. Do not add a Sources section or inline citation markers):"
        )
        return user_content

    @classmethod
    def build_query_reformulation_prompt(cls, history: List[Dict[str, str]], latest_query: str) -> str:
        """Builds a prompt for rewriting follow-up questions into standalone queries."""
        history_str = ""
        for msg in history[-4:]:  # Use up to last 4 messages
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"

        prompt = (
            f"Chat History:\n{history_str}\n"
            f"Latest User Question: {latest_query}\n\n"
            f"Reformulated Standalone Search Query:"
        )
        return prompt
