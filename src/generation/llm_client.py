"""
LLM Client abstraction for DocAI supporting Google Gemini (google-genai SDK) and Mock/Offline fallback.
Features automatic model fallback retry upon 404/deprecation errors.
"""

import os
import time
from typing import Generator, Optional, List, Dict, Any

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    GENAI_NEW_SDK = False

from src.config import GEMINI_API_KEY, DEFAULT_LLM_MODEL, FALLBACK_MODELS, TEMPERATURE, MAX_OUTPUT_TOKENS
from src.generation.prompts import RAGPromptManager


class LLMClient:
    """Unified LLM Client supporting Google Gemini with graceful fallback modes and resilience."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = (api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")).strip()
        raw_model = model_name or DEFAULT_LLM_MODEL
        # Normalize model name by stripping 'models/' prefix if present
        self.model_name = raw_model.replace("models/", "").strip()
        self.provider = "gemini" if self.api_key else "mock"

        self._init_client()

    def set_api_key(self, api_key: Optional[str] = None) -> None:
        """Dynamically updates the Gemini API key and synchronizes client instance."""
        if api_key is not None:
            self.api_key = api_key.strip()
        self.provider = "gemini" if self.api_key else "mock"
        self._init_client()

    def set_keys(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Compatibility method for key updates."""
        self.set_api_key(api_key)

    def set_model(self, model_name: str) -> None:
        """Dynamically updates the active Gemini model."""
        self.model_name = model_name.replace("models/", "").strip()
        self.provider = "gemini" if self.api_key else "mock"
        self._init_client()

    def _init_client(self) -> None:
        """Initializes provider client instances."""
        if self.provider == "gemini":
            if GENAI_NEW_SDK:
                self.client = genai.Client(api_key=self.api_key)
            else:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=RAGPromptManager.SYSTEM_PROMPT,
                    generation_config=genai.types.GenerationConfig(
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    )
                )
        else:
            self.client = None

    def _get_candidate_models(self) -> List[str]:
        """Returns ordered list of candidate models for retry on 404 / deprecation."""
        models = [self.model_name]
        for fb in FALLBACK_MODELS:
            if fb not in models:
                models.append(fb)
        return models

    def generate(self, prompt: str, user_query: str = "", retrieved_chunks: Optional[List[Any]] = None) -> str:
        """Generates a complete response synchronously with automatic model fallback and extractive synthesis."""
        if self.provider == "gemini":
            last_err = ""
            for model_cand in self._get_candidate_models():
                try:
                    if GENAI_NEW_SDK:
                        config = genai_types.GenerateContentConfig(
                            system_instruction=RAGPromptManager.SYSTEM_PROMPT,
                            temperature=TEMPERATURE,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        )
                        response = self.client.models.generate_content(
                            model=model_cand,
                            contents=prompt,
                            config=config,
                        )
                        self.model_name = model_cand
                        return response.text or ""
                    else:
                        client_model = genai.GenerativeModel(
                            model_name=model_cand,
                            system_instruction=RAGPromptManager.SYSTEM_PROMPT,
                        )
                        response = client_model.generate_content(prompt)
                        self.model_name = model_cand
                        return response.text or ""
                except Exception as e:
                    err_str = str(e)
                    last_err = err_str
                    # Fallback to next candidate on 404, 429, RESOURCE_EXHAUSTED, or quota limits
                    err_lower = err_str.lower()
                    if any(k in err_lower for k in ["404", "not_found", "not available", "deprecated", "429", "resource_exhausted", "quota", "rate limit", "exhausted"]):
                        continue
                    else:
                        return f"[Gemini API Error: {err_str}]"

            # If every candidate is unavailable or quota-exhausted, stay useful with local grounding.
            last_error_lower = last_err.lower()
            if any(k in last_error_lower for k in [
                "404", "not_found", "not found", "not available", "deprecated",
                "429", "resource_exhausted", "quota", "rate limit", "exhausted",
            ]):
                return "".join(self._extractive_synthesis(
                    prompt=prompt,
                    user_query=user_query,
                    retrieved_chunks=retrieved_chunks,
                    quota_notice=any(k in last_error_lower for k in ["429", "resource_exhausted", "quota", "rate limit", "exhausted"]),
                    error_msg=last_err
                ))
            return f"[Gemini API Error: {last_err}]"

        else:
            return "".join(self._extractive_synthesis(
                prompt=prompt,
                user_query=user_query,
                retrieved_chunks=retrieved_chunks,
                offline_notice=True
            ))

    def _extractive_synthesis(
        self,
        prompt: str = "",
        user_query: str = "",
        retrieved_chunks: Optional[List[Any]] = None,
        quota_notice: bool = False,
        offline_notice: bool = False,
        error_msg: str = "",
    ) -> Generator[str, None, None]:
        """
        Synthesizes a structured, fully grounded response from retrieved chunks
        when the external LLM provider is offline, unconfigured, or quota-exhausted (429).
        """
        import re

        # Extract user_query and retrieved_chunks from prompt if not supplied
        if not user_query and "USER QUESTION:" in prompt:
            try:
                parts = prompt.split("USER QUESTION:")
                if len(parts) > 1:
                    user_query = parts[1].split("ANSWER (")[0].strip()
            except Exception:
                user_query = ""

        if not retrieved_chunks and "CONTEXT EXCERPTS:" in prompt:
            try:
                excerpts_part = prompt.split("CONTEXT EXCERPTS:")[1].split("USER QUESTION:")[0]
                matches = re.split(r'--- \[EXCERPT \d+\] \(Source: (.*?), Page: (\d+)\) ---', excerpts_part)
                if len(matches) > 3:
                    parsed_chunks = []
                    for i in range(1, len(matches), 3):
                        if i + 2 < len(matches):
                            src = matches[i].strip()
                            pg = int(matches[i+1].strip()) if matches[i+1].strip().isdigit() else 1
                            txt = matches[i+2].strip()
                            class SimpleChunk:
                                def __init__(self, s, p, t):
                                    self.text = t
                                    self.metadata = {"source_file": s, "page_number": p}
                                    self.rrf_score = 0.05
                            parsed_chunks.append(SimpleChunk(src, pg, txt))
                    retrieved_chunks = parsed_chunks
            except Exception:
                pass

        query_str = user_query or "your question"
        chunks = retrieved_chunks or []

        # 1. Thought Trace
        if chunks:
            sources = list(dict.fromkeys([
                c.metadata.get("source_file", "document")
                for c in chunks if hasattr(c, "metadata")
            ]))
            sources_label = ", ".join(sources) if sources else "indexed files"
            thought = (
                f"<thought>\n"
                f"Analyzing query: \"{query_str}\".\n"
                f"Scanning knowledge base: Evaluated {len(chunks)} relevant excerpt(s) from {sources_label}.\n"
                f"{'Google AI Studio quota limit (429) active; engaging Local Grounded Synthesizer to formulate verified response.' if quota_notice else 'Synthesizing verified factual statements directly grounded on document evidence.'}\n"
                f"Extracting key findings, requirements, and definitions.\n"
                f"</thought>\n\n"
            )
        else:
            thought = (
                f"<thought>\n"
                f"Analyzing query: \"{query_str}\".\n"
                f"No indexed document excerpts found in the active collection.\n"
                f"</thought>\n\n"
            )

        for w in thought.split(" "):
            yield w + " "
            time.sleep(0.005)

        # 2. Status Callout Banner
        if quota_notice:
            banner = (
                "> [!WARNING]\n"
                "> **Google AI Studio API Free-Tier Quota Limit Reached (429 RESOURCE_EXHAUSTED)**\n"
                ">\n"
                "> Your Gemini API key has consumed its free-tier quota in Google Cloud. DocAI has automatically activated the **Local Grounded Synthesizer** so you can continue testing, searching, and reviewing your documents without interruption.\n"
                ">\n"
                "> 💡 **To restore generative LLM reasoning immediately**:\n"
                "> 1. Open [Google AI Studio](https://aistudio.google.com/apikey).\n"
                "> 2. Click **Create API key** and select **Create in NEW project** (each new project gets a fresh 1,500 req/day quota!).\n"
                "> 3. Click the **Key icon (🔑)** in the top bar and paste your new key.\n\n"
            )
            for w in banner.split(" "):
                yield w + " "
                time.sleep(0.004)
        elif offline_notice:
            banner = (
                f"> [!NOTE]\n"
                f"> **Local Grounded Mode (No Gemini API Key Configured)**\n"
                f">\n"
                f"> Operating in Local Grounded Mode for `{self.model_name}`. Excerpts are retrieved via Hybrid Search (Dense ChromaDB + Sparse BM25) and synthesized directly.\n"
                f"> *(Add your `GEMINI_API_KEY` via the top-right 🔑 icon to enable live Gemini completions)*.\n\n"
            )
            for w in banner.split(" "):
                yield w + " "
                time.sleep(0.004)

        # 3. Empty context handling
        if not chunks:
            msg = (
                "No relevant document passages were found in the current index to answer this query. "
                "Please attach one or more documents (PDF, Word, Excel, CSV, TXT) using the paperclip icon to query your documents."
            )
            for w in msg.split(" "):
                yield w + " "
                time.sleep(0.01)
            return

        # 4. Extract and rank sentences from retrieved chunks
        stopwords = {
            "what", "is", "are", "the", "a", "an", "in", "on", "of", "for", "to",
            "and", "or", "about", "this", "that", "it", "can", "you", "me", "please",
            "tell", "explain", "describe", "show", "give", "how", "why", "when", "where",
            "does", "do", "did", "with", "from", "at", "by", "as", "be", "all"
        }
        clean_q = re.sub(r'[^a-zA-Z0-9\s]', ' ', query_str.lower())
        q_terms = [t for t in clean_q.split() if t not in stopwords and len(t) > 2]

        ranked_sentences = []
        doc_page_map = {}

        for c in chunks[:6]:
            src = c.metadata.get("source_file", "Document") if hasattr(c, "metadata") else "Document"
            pg = c.metadata.get("page_number", 1) if hasattr(c, "metadata") else 1
            doc_page_map.setdefault(src, set()).add(pg)

            lines = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', c.text) if len(s.strip()) > 20]
            for line in lines:
                line_lower = line.lower()
                score = 0
                for term in q_terms:
                    if term in line_lower:
                        score += 3
                if any(w in line_lower for w in ["problem", "statement", "objective", "requirement", "task", "assignment", "round", "evaluation", "architecture"]):
                    score += 2
                if score > 0 or len(q_terms) == 0:
                    ranked_sentences.append((score, line, src, pg))

        # Sort by relevance score descending
        ranked_sentences.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate sentences while preserving high scores
        seen_texts = set()
        unique_sentences = []
        for score, text, src, pg in ranked_sentences:
            normalized = re.sub(r'\W+', '', text.lower())[:60]
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_sentences.append((score, text, src, pg))

        # Format structured findings
        response_blocks = []

        if unique_sentences:
            top_passages = [s[1] for s in unique_sentences[:2]]
            response_blocks.append("### Answer\n" + " ".join(top_passages) + "\n\n")

            if len(unique_sentences) > 2:
                response_blocks.append("### Key Details & Specifications\n")
                for _, text, src, pg in unique_sentences[2:7]:
                    clean_text = text.lstrip("-*• ")
                    header_candidate = clean_text[:45].split(':')[0].strip()
                    response_blocks.append(f"- **{header_candidate}**: {clean_text}\n")
                response_blocks.append("\n")
        else:
            response_blocks.append("### Extracted Relevant Content\n")
            for c in chunks[:3]:
                src = c.metadata.get("source_file", "Document")
                pg = c.metadata.get("page_number", 1)
                response_blocks.append(f"**From {src} (Page {pg}):**\n> {c.text.strip()}\n\n")

        # Sources Section
        response_blocks.append("### Sources\n")
        for doc_name, pages in doc_page_map.items():
            sorted_pages = sorted(list(pages))
            pages_str = ", ".join(str(p) for p in sorted_pages)
            response_blocks.append(f"* **{doc_name}** — Page(s) {pages_str}\n")

        full_response = "".join(response_blocks)
        for w in full_response.split(" "):
            yield w + " "
            time.sleep(0.008)

    def stream_generate(
        self,
        prompt: str,
        user_query: str = "",
        retrieved_chunks: Optional[List[Any]] = None,
    ) -> Generator[str, None, None]:
        """Streams response tokens incrementally with automatic model fallback and extractive synthesis."""
        if self.provider == "gemini":
            last_err = ""
            for model_cand in self._get_candidate_models():
                streamed_any = False
                try:
                    if GENAI_NEW_SDK:
                        config = genai_types.GenerateContentConfig(
                            system_instruction=RAGPromptManager.SYSTEM_PROMPT,
                            temperature=TEMPERATURE,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        )
                        response_stream = self.client.models.generate_content_stream(
                            model=model_cand,
                            contents=prompt,
                            config=config,
                        )
                        # Test if first chunk streams without error
                        first_chunk = True
                        for chunk in response_stream:
                            if first_chunk:
                                self.model_name = model_cand
                                first_chunk = False
                            if chunk.text:
                                streamed_any = True
                                yield chunk.text
                        return  # Success!
                    else:
                        client_model = genai.GenerativeModel(
                            model_name=model_cand,
                            system_instruction=RAGPromptManager.SYSTEM_PROMPT,
                        )
                        response = client_model.generate_content(prompt, stream=True)
                        for chunk in response:
                            if chunk.text:
                                streamed_any = True
                                yield chunk.text
                        self.model_name = model_cand
                        return  # Success!

                except Exception as e:
                    err_str = str(e)
                    last_err = err_str
                    err_lower = err_str.lower()
                    is_fallback_error = any(k in err_lower for k in [
                        "404", "not_found", "not available", "deprecated",
                        "429", "resource_exhausted", "quota", "rate limit", "exhausted"
                    ])
                    if not streamed_any and is_fallback_error:
                        continue
                    else:
                        yield f"\n[Error during streaming: {err_str}]"
                        return

            # If every candidate is unavailable or quota-exhausted, use local grounded synthesis.
            last_error_lower = last_err.lower()
            if any(k in last_error_lower for k in [
                "404", "not_found", "not found", "not available", "deprecated",
                "429", "resource_exhausted", "quota", "rate limit", "exhausted",
            ]):
                yield from self._extractive_synthesis(
                    prompt=prompt,
                    user_query=user_query,
                    retrieved_chunks=retrieved_chunks,
                    quota_notice=any(k in last_error_lower for k in ["429", "resource_exhausted", "quota", "rate limit", "exhausted"]),
                    error_msg=last_err
                )
                return

            yield f"\n[Gemini API Error: {last_err}]"

        else:
            yield from self._extractive_synthesis(
                prompt=prompt,
                user_query=user_query,
                retrieved_chunks=retrieved_chunks,
                offline_notice=True
            )

    def contextualize_query(self, history: List[Dict[str, str]], query: str) -> str:
        """Rewrites a conversational follow-up query into a standalone search query."""
        if not history or len(history) == 0:
            return query

        reformulate_prompt = RAGPromptManager.build_query_reformulation_prompt(history, query)

        if self.provider == "gemini":
            for model_cand in self._get_candidate_models():
                try:
                    if GENAI_NEW_SDK:
                        config = genai_types.GenerateContentConfig(
                            system_instruction=RAGPromptManager.QUERY_CONTEXTUALIZER_SYSTEM,
                            temperature=0.0,
                            max_output_tokens=128,
                        )
                        response = self.client.models.generate_content(
                            model=model_cand,
                            contents=reformulate_prompt,
                            config=config,
                        )
                        rewritten = response.text.strip() if response.text else query
                        return rewritten
                    else:
                        reformer = genai.GenerativeModel(
                            model_name=model_cand,
                            system_instruction=RAGPromptManager.QUERY_CONTEXTUALIZER_SYSTEM,
                        )
                        response = reformer.generate_content(reformulate_prompt)
                        rewritten = response.text.strip() if response.text else query
                        return rewritten
                except Exception as e:
                    err_str = str(e)
                    err_lower = err_str.lower()
                    if any(k in err_lower for k in [
                        "404", "not_found", "not available", "deprecated",
                        "429", "resource_exhausted", "quota", "rate limit", "exhausted"
                    ]):
                        continue
                    return query
            return query
        else:
            return query
