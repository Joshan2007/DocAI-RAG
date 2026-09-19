# 🧠 DocAI: Streamlit AI Knowledge Assistant

[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-purple.svg?style=for-the-badge)](https://www.trychroma.com/)
[![Hybrid Retrieval](https://img.shields.io/badge/Retrieval-BM25%20%2B%20Dense%20(RRF)-green.svg?style=for-the-badge)]()
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini%20Flash%20%26%20Pro-4285F4.svg?style=for-the-badge&logo=google)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**DocAI** is an advanced, production-grade AI Knowledge Assistant engineered to ingest, index, and query complex multi-domain documents (enterprise policies, scientific research papers, financial spreadsheets, technical specifications, and student coursework) with zero hallucination, verifiable citations, dynamic real-time reasoning, and real-time retrieval evaluation.

---

## 🏗️ Detailed System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Universal Document Ingestion & Indexing"]
        DOCS["Multi-Format Source Documents\n(PDF, DOCX, XLSX, CSV, MD, TXT)"] --> PARSER["DocumentParser\n(pypdf, python-docx, openpyxl, utf-8)"]
        PARSER --> CHUNKER["DocumentChunker\n(600 chars, 120-char sliding overlap,\nmetadata preservation: page, sheet, chunk_id)"]
        CHUNKER --> DENSE_EMB["ONNX all-MiniLM-L6-v2 Embeddings\n(384-dimensional dense vectors)"]
        CHUNKER --> SPARSE_TOK["BM25Okapi Tokenizer\n(Alphanumeric keyword index)"]
        DENSE_EMB --> CHROMA[("Persistent ChromaDB Vector Store")]
        SPARSE_TOK --> BM25_IDX[("In-Memory BM25 Lexical Index")]
    end

    subgraph QUERY_PROCESSING["2. Query & Contextualization Pipeline"]
        USER_Q["User Question / Follow-up"] --> MEMORY["Conversational Memory Buffer\n(Multi-turn history)"]
        MEMORY --> REFORMULATOR["Query Contextualizer\n(Resolves coreferences & pronouns)"]
    end

    subgraph HYBRID_RETRIEVAL["3. Hybrid Retrieval & Fusion Engine"]
        REFORMULATOR --> D_SEARCH["Dense Semantic Search\n(Cosine distance top-K)"]
        REFORMULATOR --> S_SEARCH["Sparse Lexical Search\n(BM25 exact match top-K)"]
        CHROMA -.-> D_SEARCH
        BM25_IDX -.-> S_SEARCH
        D_SEARCH --> RRF["Reciprocal Rank Fusion (RRF, k=60)\nRRF_score = 1/(60 + rank_dense) + 1/(60 + rank_sparse)"]
        S_SEARCH --> RRF
        RRF --> TOP_CHUNKS["Top Reranked Candidate Chunks\n(Cross-document context)"]
    end

    subgraph GENERATION_EVAL["4. LLM Synthesis & Telemetry Engine"]
        TOP_CHUNKS --> PROMPT_ENG["Strict Grounded Prompt Engineer\n(Forces <thought> reasoning + [Doc: Page] citations)"]
        USER_Q --> PROMPT_ENG
        PROMPT_ENG --> GEMINI{"Google Gemini Engine\n(1.5 Flash / 2.0 Flash / 1.5 Pro / 2.5 Flash)"}
        GEMINI -- "Rate Limit 429" --> ROTATE["Automatic Model Rollover / Local Synthesizer"]
        GEMINI -- "Success" --> STREAM["Streamed Generation\n(retrieval_complete -> token -> generation_complete)"]
        ROTATE --> STREAM
        STREAM --> EVAL["RAG Triad Automated Evaluator\n- Retrieval Relevance (0.0 - 1.0)\n- Groundedness / Faithfulness (0.0 - 1.0)\n- Citation Coverage & Latency (ms)"]
    end

    subgraph PRESENTATION["5. Streamlit Application"]
      STREAM --> UI["Streamlit UI\n- Document uploader\n- Chat interface\n- Model selector\n- Source citations\n- RAG quality metrics"]
        EVAL --> UI
    end
```

DocAI was engineered to overcome the core vulnerabilities of traditional Retrieval-Augmented Generation: keyword blindness, hallucinations, clumsy citations, and catastrophic failures during API rate-limits.

### 1. Multi-Format Ingestion & Semantic Chunking Strategy
- **Universal Document Parsing**: A unified abstraction layer ([`src/ingestion/parser.py`](src/ingestion/parser.py)) parses heterogeneous file formats into structured text objects while preserving document geometry:
  - **PDF**: Page-by-page text extraction with footer/header stripping via `pypdf`.
  - **Microsoft Word (`.docx`)**: Structural extraction preserving heading hierarchies and nested table cells via `python-docx`.
  - **Excel Spreadsheets (`.xlsx`, `.xls`)**: Tabular serialization preserving workbook sheet names, row indices, and cell formatting via `openpyxl`.
  - **Delimited Files (`.csv`, `.tsv`)**: Normalizes rows into readable, column-annotated natural language records.
  - **Plain Text & Markdown (`.md`, `.txt`, `.py`, `.json`)**: Robust multi-encoding UTF-8/Latin-1 fallback decoding.
- **Context-Aware Sliding Window Chunking**: Rather than splitting blindly at arbitrary character counts (which severs sentences and compounds errors), [`src/ingestion/chunker.py`](src/ingestion/chunker.py) uses a sliding window of **600 characters with 120-character overlap** split along linguistic boundaries (paragraphs $\to$ sentences $\to$ clauses). Every chunk retains immutable metadata: `source_file`, `page_number`, `sheet_name`, and `chunk_index`.

---

### 2. Dual-Index Hybrid Retrieval (Dense Vector + BM25Okapi)
Single-mode retrieval architectures suffer from fundamental trade-offs:
- **Dense Vector Search** (`all-MiniLM-L6-v2` / ChromaDB) excels at semantic understanding and conversational synonyms, but struggles with exact alphanumeric strings, error codes, and unique identifiers.
- **Sparse Lexical Search** (BM25Okapi) provides exact term-matching precision for acronyms, function names, and technical codes (`AES-256`, `CS-482`, `P95`), but fails when queries use paraphrased phrasing.

DocAI implements a **parallel dual-index architecture**:

1. **Dense Vector Store** ([`src/retrieval/vector_store.py`](src/retrieval/vector_store.py)): Fast, local ONNX embeddings indexed into persistent ChromaDB collections with cosine distance metric.
2. **Sparse Lexical Index** ([`src/retrieval/bm25_retriever.py`](src/retrieval/bm25_retriever.py)): An in-memory BM25Okapi inverted index tokenized on alphanumeric boundaries.

---

### 3. Reciprocal Rank Fusion (RRF, $k=60$)
To merge the disjoint ranked candidate lists from Dense and Sparse search without scale bias, DocAI utilizes **Reciprocal Rank Fusion (RRF)**:

$$RRF\_score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{k + rank_m(d)}$$

Where:
- $rank_m(d)$ is the 1-based rank position of document $d$ within retrieval method $m$.
- $k$ is the smoothing constant set to **60** (industry standard).

**Why RRF is superior to Linear Score Weighting:**
Score normalization across different retrieval modalities (e.g. cosine similarity range $[0, 1]$ vs. unbounded BM25 scores $[0, \infty)$) is notoriously sensitive to document length and query specificity. RRF operates solely on **relative rank order**, ensuring that items retrieved highly by both methods receive a dominant boost, while preserving high-confidence unique hits from either retriever.

---

### 4. Dynamic Step-by-Step Thinking & Streamed Generation
- **Dynamic Reasoning Extraction**: Modern models produce higher quality, better grounded answers when allowed to reason through evidence before speaking. DocAI instructs Gemini via system prompts to formulate an explicit thought trace inside `<thought>...</thought>` tags.
- **Streaming Generation**: The pipeline exposes retrieval metadata, answer tokens, and final evaluation directly to Streamlit while keeping document retrieval and answer generation in one process.
- **Pure Google Gemini Architecture**: Configured for high-throughput, low-latency reasoning across four specialized models:
  - **Gemini 1.5 Flash** (Default • Recommended general assistant)
  - **Gemini 2.0 Flash** (Next-Gen high-speed multimodal reasoning)
  - **Gemini 1.5 Pro** (Deep reasoning for complex multi-document synthesis)
  - **Gemini 2.5 Flash** (Experimental preview reasoning engine)

---

### 5. Anti-Hallucination Guardrails & Zero-Crash Resilience
- **Strict Factual Containment**: The generation prompt ([`src/generation/prompts.py`](src/generation/prompts.py)) establishes strict guardrails: answers must be derived *exclusively* from retrieved passages. When facts are absent or ambiguous, the model is explicitly constrained to state that insufficient information is available.
- **Dedicated Non-Intrusive Citations**: In-text superscript interruptions fragment reading comprehension. DocAI formats citations cleanly *after* the synthesized answer, providing structured citation cards with source filenames, page numbers, relevance confidence scores, and verbatim excerpt quotes.
- **Graceful Quota Degradation**: If Google AI Studio returns `429 RESOURCE_EXHAUSTED`, DocAI automatically cascades across fallback models. If completely offline or unauthenticated, the **Local Grounded Synthesizer** takes over, extracting and presenting verified facts deterministically with zero system crashes.

---

### 6. Automated RAG Triad Telemetry & Quality Assessment
Every synthesized answer is evaluated in real-time by [`src/evaluation/evaluator.py`](src/evaluation/evaluator.py) against the **RAG Triad**:
1. **Retrieval Relevance**: Measures token-level intent overlap between the user's query and the top retrieved passages.
2. **Groundedness / Faithfulness**: Computes the ratio of synthesized claims that are verified by retrieved passage content (preventing hallucinations).
3. **Citation Coverage**: Verifies that assertions link directly back to indexed documents.
4. **Latency Profiling**: Reports separate latency metrics for retrieval phase ($t_{retrieval}$) versus LLM token synthesis ($t_{generation}$) in milliseconds.

---

### 7. Granular Document Lifecycle & Self-Healing Index
Unlike primitive RAG demos where the entire database must be wiped to update documents, DocAI features an interactive document manager:
- Each uploaded file is indexed directly from Streamlit's file uploader.
- The sidebar can clear the indexed knowledge base and conversation memory without restarting the app.

---

## 📊 Comparison: Naive RAG vs. DocAI

| Dimension | Naive Tutorial RAG | **DocAI Production Engine** |
| :--- | :--- | :--- |
| **Retrieval Mode** | Dense Vector Only (Cosine distance) | **Hybrid Search (Dense ChromaDB + Sparse BM25 via RRF, $k=60$)** |
| **Domain Terminology** | Often misses exact acronyms, codes, and IDs | **BM25 captures exact lexical tokens (`AES-256`, `CS-482`, `P95`)** |
| **Chunking Logic** | Blind character slicing (cuts words/sentences) | **Context-Aware Semantic Chunking with sliding overlap** |
| **Source Attribution** | Clumsy inline tags or completely missing | **Dedicated Post-Answer Citations with page, score & excerpts** |
| **Thinking Mode** | Static canned text | **Dynamic step-by-step `<thought>` reasoning streamed in real-time** |
| **Document Management** | All-or-nothing wipe | **Granular individual document removal (✕) with live re-indexing** |
| **Hallucination Control**| Prone to creative extrapolation | **Strict Grounding Policy & Out-of-Domain Refusal** |
| **Conversational Memory**| Single turn or naive concatenation | **Follow-up Query Contextualizer & Reformulation** |
| **Quality Evaluation** | None (Anecdotal inspection) | **Automated RAG Triad Telemetry (Groundedness, Relevance, Latency)** |
| **Quota Resilience** | Crashes on 429 or quota limit | **Automatic Gemini model rollover + Local Grounded Synthesizer** |

---

## 📁 Repository Structure

```
RAG/
├── app.py                        # Self-contained Streamlit application
├── src/                          # Modular Production RAG Engine
│   ├── config.py                 # Hyperparameters, paths & Gemini model list
│   ├── pipeline.py               # Orchestrator connecting all RAG components
│   ├── ingestion/
│   │   ├── parser.py             # Universal document parser (PDF, DOCX, XLSX, CSV, MD, TXT)
│   │   └── chunker.py            # Semantic chunker with sliding overlap & metadata
│   ├── retrieval/
│   │   ├── vector_store.py       # ChromaDB persistent dense vector store
│   │   ├── bm25_retriever.py     # BM25Okapi lexical index with dynamic reindexing
│   │   └── hybrid.py             # Reciprocal Rank Fusion (RRF, k=60) engine
│   ├── generation/
│   │   ├── prompts.py            # Strict grounding, anti-hallucination & thought tags
│   │   ├── llm_client.py         # Gemini Flash & Pro client with automatic rollover
│   │   └── memory.py             # Multi-turn conversational memory & query reformulator
│   └── evaluation/
│       └── evaluator.py          # RAG Triad automated metrics (Relevance, Groundedness, Latency)
├── sample_docs/                  # Multi-domain sample knowledge portfolio
│   ├── engineering_specification.pdf
│   ├── platform_engineering_guidelines.docx
│   ├── quarterly_performance.xlsx
│   ├── performance_benchmarks.csv
│   ├── enterprise_cloud_security_policy.md
│   ├── distributed_systems_syllabus.md
│   └── quantum_computing_research.txt
├── tests/                        # Pipeline and integration test suite
│   ├── test_rag_pipeline.py              # Ingestion, chunking, retrieval & evaluation tests
│   └── test_end_to_end.py                # Full pipeline integration tests
├── requirements.txt              # Clean Python dependencies
└── run_app.py                    # One-click Streamlit launcher
```

---

## 🚀 Local Quickstart (One Command)

**Requirements**: Python 3.10+.

```bash
git clone https://github.com/Joshan2007/DocAI-RAG.git
cd DocAI-RAG
python -m pip install -r requirements.txt
streamlit run app.py
```

The optional launcher does everything automatically on first run:

| Step | What happens |
|------|-------------|
| **1** | Creates a Python virtual environment (`.venv`) |
| **2** | Installs all Python dependencies from `requirements.txt` |
| **3** | Prompts you for your **Gemini API key** (one time only — saved to `.env`) |
| **4** | Starts Streamlit |

> Get a free Gemini API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

On every subsequent run, installed dependencies and the saved key are reused.

Once running, open [http://localhost:8501](http://localhost:8501). Enter the Gemini key in the sidebar; no separate backend is needed.

Press `Ctrl+C` to stop Streamlit cleanly.

### Run Automated Tests
```bash
# Activate the venv first, then:
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS / Linux

pytest tests/ -v
```



---

## 🌐 Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/) and choose **New app**.
3. Select the repository, branch, and set the main file to `app.py`.
4. Deploy. No Render service, Vercel project, backend URL, or separate server is required.

Visitors paste their own Gemini API key into the sidebar. To provide a default key, add `GEMINI_API_KEY` under the app's Streamlit Cloud secrets.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
