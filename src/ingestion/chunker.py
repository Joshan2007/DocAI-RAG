"""
Context-aware chunker for DocAI.
Splits parsed document pages into semantically cohesive passages while preserving
rich metadata (document name, page number, chunk index) for precise citations.
"""

from dataclasses import dataclass
from typing import List
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.parser import ParsedDocument


@dataclass
class DocumentChunk:
    """Represents a chunk of text extracted from a document with full attribution metadata."""
    chunk_id: str
    doc_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    char_count: int

    def to_metadata_dict(self) -> dict:
        """Serializes chunk metadata for vector store storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
        }


class DocumentChunker:
    """Chunks documents using hierarchical separator splitting with overlap."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Splits a single block of text using recursive separators (paragraphs -> sentences -> words)."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        # Break text down into small atomic paragraphs / sentences
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        atomic_units = []
        for p in paragraphs:
            if len(p) <= self.chunk_size:
                atomic_units.append(p)
            else:
                # Split large paragraphs by sentence boundary
                sentences = [s.strip() + "." for s in p.replace("! ", ". ").replace("? ", ". ").split(". ") if s.strip()]
                for s in sentences:
                    if len(s) <= self.chunk_size:
                        atomic_units.append(s)
                    else:
                        # Fallback: split long sentences by words
                        words = s.split(" ")
                        cur_w = []
                        cur_w_len = 0
                        for w in words:
                            if cur_w_len + len(w) + 1 > self.chunk_size and cur_w:
                                atomic_units.append(" ".join(cur_w))
                                cur_w = [w]
                                cur_w_len = len(w)
                            else:
                                cur_w.append(w)
                                cur_w_len += len(w) + 1
                        if cur_w:
                            atomic_units.append(" ".join(cur_w))

        # Re-assemble atomic units into target-sized chunks with sliding overlap
        chunks: List[str] = []
        current_chunk_parts: List[str] = []
        current_length = 0

        for unit in atomic_units:
            unit_len = len(unit)
            if current_length + unit_len + 2 > self.chunk_size and current_chunk_parts:
                assembled_chunk = "\n\n".join(current_chunk_parts)
                chunks.append(assembled_chunk)

                # Overlap logic: retain the trailing units up to chunk_overlap
                overlap_parts: List[str] = []
                overlap_len = 0
                for part in reversed(current_chunk_parts):
                    if overlap_len + len(part) + 2 <= self.chunk_overlap:
                        overlap_parts.insert(0, part)
                        overlap_len += len(part) + 2
                    else:
                        break

                current_chunk_parts = overlap_parts + [unit]
                current_length = sum(len(p) for p in current_chunk_parts) + 2 * (len(current_chunk_parts) - 1)
            else:
                current_chunk_parts.append(unit)
                current_length += unit_len + 2

        if current_chunk_parts:
            final_chunk = "\n\n".join(current_chunk_parts)
            if not chunks or chunks[-1] != final_chunk:
                chunks.append(final_chunk)

        return chunks

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[DocumentChunk]:
        """Chunks an entire parsed document, preserving page numbers and assigning unique chunk IDs."""
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in parsed_doc.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            raw_chunks = self._split_text_into_chunks(page_text)

            for chunk_str in raw_chunks:
                # Filter out microscopic fragments (less than 25 chars)
                if len(chunk_str) < 25:
                    continue

                chunk_id = f"{parsed_doc.doc_id}_p{page.page_number}_c{global_chunk_idx}"
                doc_chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=parsed_doc.doc_id,
                    source_file=parsed_doc.filename,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    text=chunk_str,
                    char_count=len(chunk_str),
                )
                all_chunks.append(doc_chunk)
                global_chunk_idx += 1

        return all_chunks
