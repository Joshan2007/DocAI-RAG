"""
Configuration settings for DocAI Knowledge Assistant.
Centralizes hyper-parameters, model selections, and storage paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
SAMPLE_DOCS_DIR = str(BASE_DIR / "sample_docs")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Document Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))  # Characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))  # Character overlap

# Retrieval Configuration
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", 8))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", 8))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 8))
RRF_K = int(os.getenv("RRF_K", 60))  # Standard constant for Reciprocal Rank Fusion

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"]
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))  # Low temperature for strict factual grounding
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 1024))
