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
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
]
SUPPORTED_EXTENSIONS = ["pdf", "docx", "xlsx", "xls", "csv", "tsv", "md", "txt", "py", "json"]


@st.cache_data(show_spinner=False, ttl=180)
def check_api_key_and_models(key: str) -> tuple[bool, str, list[str]]:
    """Dynamically discover available models on the provided Gemini API key."""
    from src.generation.llm_client import LLMClient
    return LLMClient.validate_api_key_and_get_models(key)


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
    """Renders prominent verified source citations with exact page numbers and excerpts."""
    citations = list(citations)
    if not citations:
        return

    with st.expander(f"📚 Verified Sources & Citations ({len(citations)})", expanded=True):
        for index, citation in enumerate(citations, start=1):
            source = citation.get("source", "Unknown source")
            page = citation.get("page", 1)
            score = citation.get("score", 0)
            excerpt = re.sub(r"\s+", " ", citation.get("text_snippet", "")).strip()
            method = citation.get("retrieval_method", "hybrid retrieval")
            st.markdown(f"**[{index}] {source}** · Page {page} · *{method}*")
            if excerpt:
                st.caption(f'"{excerpt}"')


def render_source_summary(citations: Iterable[Dict[str, Any]]) -> None:
    """Show compact source attribution."""
    render_citations(citations)


def render_metrics(evaluation: Any, citations: Optional[Iterable[Dict[str, Any]]] = None) -> None:
    if not evaluation:
        return
    cite_count = len(list(citations)) if citations is not None else getattr(evaluation, "citation_count", 0)
    columns = st.columns(4)
    columns[0].metric("Relevance", f"{evaluation.retrieval_relevance_score:.2f}")
    columns[1].metric("Groundedness", f"{evaluation.groundedness_score:.2f}")
    columns[2].metric("Citations", cite_count)
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

    models_to_display = AVAILABLE_MODELS
    if api_key:
        is_valid, msg, discovered = check_api_key_and_models(api_key)
        if is_valid and discovered:
            st.success(f"Gemini connected ({len(discovered)} models available)")
            models_to_display = discovered
        elif not is_valid:
            err_summary = msg.split("\n")[0].strip()
            if "'message':" in err_summary:
                m_match = re.search(r"'message':\s*['\"](.*?)['\"]", err_summary)
                if m_match:
                    err_summary = m_match.group(1)
            st.error(f"❌ API Key Notice: {err_summary}")
        else:
            st.warning("⚠️ No models returned by Google for this key.")
    else:
        st.info("Local grounded mode (enter API key above for Gemini reasoning)")

    model_name = st.selectbox("Generation model", models_to_display, index=0)

    allow_custom = st.checkbox("Custom model name", value=False)
    if allow_custom:
        custom_input = st.text_input("Model ID", value=model_name).strip()
        if custom_input:
            model_name = custom_input

    # Sync API key and model dynamically without recreating pipeline
    pipeline.llm.set_api_key(api_key)
    pipeline.llm.set_model(model_name)

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
                    f"Loaded **{name}** ({result['metadata']['total_chunks']} chunks)"
                )
            except Exception as error:
                st.sidebar.error(f"Could not index {name}: {error}")

    if not current_names and pipeline.indexed_files:
        pipeline.clear_all()
        st.session_state["messages"] = []

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
        if message.get("evaluation"):
            render_metrics(message["evaluation"], message.get("citations"))

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
                render_metrics(evaluation, citations)
                messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "thinking": thought,
                        "citations": citations,
                        "evaluation": evaluation,
                    }
                )