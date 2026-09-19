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
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-2.5-flash",
]
SUPPORTED_EXTENSIONS = ["pdf", "docx", "xlsx", "xls", "csv", "tsv", "md", "txt", "py", "json"]


@st.cache_resource(show_spinner="Loading DocAI retrieval engine...")
def get_pipeline(api_key: str, model_name: str, session_id: str) -> DocAIPipeline:
    session_directory = Path(tempfile.gettempdir()) / "docai_streamlit" / session_id
    pipeline = DocAIPipeline(
        api_key=api_key or None,
        persist_directory=str(session_directory),
    )
    pipeline.llm.set_model(model_name)
    return pipeline


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


st.set_page_config(page_title="DocAI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("DocAI Knowledge Assistant")
st.caption("Upload documents, search their contents, and get grounded answers with citations.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Gemini API key",
        value=get_secret_api_key(),
        type="password",
        help="For Streamlit Cloud, add GEMINI_API_KEY in App settings > Secrets.",
    ).strip()
    model_name = st.selectbox("Generation model", AVAILABLE_MODELS)
    show_sources = st.checkbox(
        "Show source excerpts",
        value=False,
        help="Keep this off for a cleaner conversation. Sources remain available when enabled.",
    )

    if api_key:
        st.success("Gemini enabled")
    else:
        st.info("Local grounded mode")

    st.divider()
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Add files to the knowledge base",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
    )
    clear_documents = st.button("Clear indexed documents", use_container_width=True)

session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)
pipeline = get_pipeline(api_key, model_name, session_id)

if clear_documents:
    pipeline.clear_all()
    st.session_state.pop("messages", None)
    st.session_state.pop("uploaded_names", None)
    st.success("Knowledge base and conversation memory cleared.")

if uploaded_files:
    indexed_names = st.session_state.setdefault("uploaded_names", set())
    new_files = [file for file in uploaded_files if file.name not in indexed_names]
    if new_files:
        with st.status("Indexing documents...", expanded=True) as status:
            for uploaded_file in new_files:
                try:
                    result = pipeline.ingest_file(
                        io.BytesIO(uploaded_file.getvalue()),
                        filename=uploaded_file.name,
                    )
                    indexed_names.add(uploaded_file.name)
                    st.write(
                        f"Indexed **{uploaded_file.name}**: "
                        f"{result['metadata']['total_chunks']} chunks"
                    )
                except Exception as error:
                    st.error(f"Could not index {uploaded_file.name}: {error}")
            status.update(label="Document indexing complete", state="complete")

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