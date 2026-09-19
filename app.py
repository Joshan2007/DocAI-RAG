"""DocAI Streamlit application.

Run locally with:
    streamlit run app.py
"""

import io
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import DocAIPipeline


AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
SUPPORTED_EXTENSIONS = ["pdf", "docx", "xlsx", "xls", "csv", "tsv", "md", "txt", "py", "json"]



def get_secret_api_key() -> str:
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        return os.getenv("GEMINI_API_KEY", "").strip()


def split_thought(text: str) -> tuple[str, str]:
    """Separate optional model reasoning from the user-facing answer."""
    thought_match = re.search(r"<thought>\s*(.*?)\s*</thought>\s*", text, re.IGNORECASE | re.DOTALL)
    if not thought_match:
        if re.search(r"<thought>\s*", text, re.IGNORECASE):
            return "", ""
        return "", text.strip()

    thought = thought_match.group(1).strip()
    answer = (text[:thought_match.start()] + text[thought_match.end():]).strip()
    answer = re.split(r"\n#{1,3}\s*Sources\s*:?.*", answer, maxsplit=1, flags=re.IGNORECASE | re.DOTALL)[0].strip()
    return thought, answer


def render_answer(text: str, placeholder: Any = None) -> tuple[str, str]:
    """Render only the clean answer and return the extracted reasoning and answer."""
    thought, answer = split_thought(text)
    target = placeholder or st
    target.markdown(answer or "_Preparing an answer..._")
    return thought, answer


def render_reasoning(thought: str) -> None:
    if thought:
        with st.expander("Reasoning", expanded=False):
            st.caption(thought)


def render_citations(citations: Iterable[Dict[str, Any]]) -> None:
    citations = list(citations)
    if not citations:
        return

    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for index, citation in enumerate(citations, start=1):
            source = citation.get("source", "Unknown source")
            page = citation.get("page", 1)
            score = citation.get("score", 0)
            excerpt = re.sub(r"\s+", " ", citation.get("text_snippet", "")).strip()
            st.markdown(f"**{index}. {source}** · page {page} · score {score}")
            st.caption(citation.get("retrieval_method", "hybrid retrieval"))
            st.write(excerpt)


def render_source_summary(citations: Iterable[Dict[str, Any]]) -> None:
    """Show compact source attribution without expanding excerpt cards."""
    grouped_sources = {}
    for citation in citations:
        source = citation.get("source", "Unknown source")
        grouped_sources.setdefault(source, set()).add(citation.get("page", 1))

    if grouped_sources:
        summary = "; ".join(
            f"{source} (page(s): {', '.join(str(page) for page in sorted(pages))})"
            for source, pages in grouped_sources.items()
        )
        st.caption(f"Source: {summary}")


def render_metrics(evaluation: Any) -> None:
    if not evaluation:
        return
    columns = st.columns(4)
    columns[0].metric("Relevance", f"{evaluation.retrieval_relevance_score:.2f}")
    columns[1].metric("Groundedness", f"{evaluation.groundedness_score:.2f}")
    columns[2].metric("Citations", evaluation.citation_count)
    columns[3].metric("Grounded", "Yes" if evaluation.is_grounded else "Review")


# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocAI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("DocAI Knowledge Assistant")
st.caption("Upload documents, search their contents, and get grounded answers with citations.")

# ── Pipeline Lifecycle in Session State (In-Memory for Zero Disk Leaks) ───────
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = DocAIPipeline(
        api_key=None,
        persist_directory=None,  # Ephemeral in-memory ChromaDB: 0 disk leaks, complete isolation
    )
pipeline: DocAIPipeline = st.session_state["pipeline"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Gemini API key",
        value=get_secret_api_key(),
        type="password",
        help="For Streamlit Cloud, add GEMINI_API_KEY in App settings > Secrets.",
    ).strip()

    model_name = st.selectbox("Generation model", AVAILABLE_MODELS, index=0)

    # Sync API key and model dynamically without recreating pipeline
    pipeline.llm.set_api_key(api_key)
    pipeline.llm.set_model(model_name)

    show_sources = st.checkbox(
        "Show source excerpts",
        value=False,
        help="Keep this off for a cleaner conversation. Sources remain available when enabled.",
    )

    doc_mode = st.radio(
        "Document mode",
        options=["Single document (replaces previous)", "Multi-document (combine documents)"],
        index=0,
        help="Single document mode automatically purges previous documents and chat history when you upload a new document.",
    )
    is_multi_doc = "Multi-document" in doc_mode

    if api_key:
        st.success(f"Gemini active ({model_name})")
    else:
        st.info("Local grounded mode (enter API key above for Gemini reasoning)")

    st.divider()
    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Add files to the knowledge base",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
    )

    # ── Synchronize Uploaded Files with Pipeline ──────────────────────────────
    current_uploaded_dict = {f.name: f for f in uploaded_files} if uploaded_files else {}
    current_names = set(current_uploaded_dict.keys())
    indexed_names = set(pipeline.indexed_files.keys())

    if not is_multi_doc:
        # Single document mode: only index the latest uploaded file
        if uploaded_files:
            latest_file = uploaded_files[-1]
            if list(pipeline.indexed_files.keys()) != [latest_file.name]:
                # Completely purge previous documents and reset conversation
                pipeline.clear_all()
                st.session_state["messages"] = []
                try:
                    result = pipeline.ingest_file(
                        io.BytesIO(latest_file.getvalue()),
                        filename=latest_file.name,
                    )
                    st.sidebar.success(
                        f"Loaded **{latest_file.name}** ({result['metadata']['total_chunks']} chunks)"
                    )
                except Exception as error:
                    st.sidebar.error(f"Could not index {latest_file.name}: {error}")
        else:
            if pipeline.indexed_files:
                pipeline.clear_all()
                st.session_state["messages"] = []
    else:
        # Multi-document mode:
        # 1. Remove files that were removed from the uploader widget
        for removed in indexed_names - current_names:
            pipeline.delete_document(removed)
            st.sidebar.info(f"Removed **{removed}**")

        # 2. Ingest newly added files
        for name, f in current_uploaded_dict.items():
            if name not in pipeline.indexed_files:
                try:
                    result = pipeline.ingest_file(
                        io.BytesIO(f.getvalue()),
                        filename=name,
                    )
                    st.sidebar.success(
                        f"Indexed **{name}** ({result['metadata']['total_chunks']} chunks)"
                    )
                except Exception as error:
                    st.sidebar.error(f"Could not index {name}: {error}")

    # Display active documents in knowledge base
    if pipeline.indexed_files:
        st.caption(f"**Active Documents ({len(pipeline.indexed_files)}):**")
        for fname, meta in pipeline.indexed_files.items():
            st.write(f"- 📄 `{fname}` ({meta.get('total_chunks', 0)} chunks)")
    else:
        st.caption("No documents currently indexed.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Docs", use_container_width=True):
            pipeline.clear_all()
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            pipeline.clear_memory()
            st.session_state["messages"] = []
            st.rerun()

# ── Main Conversation Area ────────────────────────────────────────────────────
if not api_key:
    st.info(
        "💡 **Gemini API Key Needed for AI Answers**: Please enter your **Gemini API key** in the sidebar on the left. "
        "Without an API key, the system can only perform basic keyword matching instead of reading and understanding your questions."
    )

st.subheader("Conversation")
messages = st.session_state.setdefault("messages", [])

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_reasoning(message.get("thinking", ""))
        if message.get("citations"):
            render_source_summary(message["citations"])
        if show_sources and message.get("citations"):
            render_citations(message["citations"])
        if message.get("evaluation"):
            render_metrics(message["evaluation"])

question = st.chat_input("Ask a question about your documents")
if question:
    if not pipeline.indexed_files:
        st.warning("⚠️ No documents are currently loaded. Please upload a document in the sidebar first.")
    else:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_parts = []
            thought = ""
            citations = []
            evaluation = None

            try:
                for event in pipeline.ask_stream(question):
                    if event["type"] == "retrieval_complete":
                        citations = event["citations"]
                    elif event["type"] == "token":
                        answer_parts.append(event["token"])
                        thought, _ = render_answer("".join(answer_parts), answer_placeholder)
                    elif event["type"] == "generation_complete":
                        evaluation = event["evaluation"]
            except Exception as error:
                st.error(f"Unable to answer this question: {error}")

            answer = "".join(answer_parts)
            if answer:
                thought, answer = render_answer(answer, answer_placeholder)
                render_reasoning(thought)
                render_source_summary(citations)
                if show_sources and citations:
                    render_citations(citations)
                render_metrics(evaluation)
                messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "thinking": thought,
                        "citations": citations,
                        "evaluation": evaluation,
                    }
                )