"""RAG retriever for resume content.

Chunks a resume by paragraphs, embeds each chunk with
``sentence-transformers/all-MiniLM-L6-v2``, and uses FAISS for cosine-
similarity retrieval. Each agent in the workflow retrieves the top-k chunks
most relevant to its role rather than receiving the full resume blob.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List
import re


@lru_cache(maxsize=1)
def _get_model():
    """Cached singleton: avoids reloading the 80MB embedding model per call."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def chunk_text(text: str, max_chars: int = 400) -> List[str]:
    """Split a resume into chunks of ~max_chars.

    Strategy: prefer paragraph boundaries (double newline). If the source
    text was extracted without paragraph breaks — common with pypdf on
    LaTeX-generated resumes — fall back to character-based chunking with
    sentence-boundary preference and short overlap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if len(paragraphs) > 1:
        chunks: List[str] = []
        current = ""
        for p in paragraphs:
            if not current:
                current = p
            elif len(current) + len(p) + 2 < max_chars:
                current = current + "\n\n" + p
            else:
                chunks.append(current)
                current = p
        if current:
            chunks.append(current)
        if len(chunks) > 1:
            return chunks

    return _chunk_by_chars(text, max_chars, overlap=80)


def _chunk_by_chars(text: str, chunk_size: int, overlap: int = 80) -> List[str]:
    """Character-based chunker with sentence-boundary preference and overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ("\n", ". ", "! ", "? ", "; "):
                idx = text.rfind(sep, max(start, end - 100), end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


class ResumeRetriever:
    """Embeds resume chunks and retrieves the top-k most relevant per query."""

    def __init__(self, text: str):
        import numpy as np
        import faiss

        self.chunks = chunk_text(text)
        model = _get_model()
        embeddings = model.encode(
            self.chunks, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.dim = embeddings.shape[1]

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        import faiss

        model = _get_model()
        q_emb = model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(q_emb)

        k = min(k, len(self.chunks))
        _scores, indices = self.index.search(q_emb, k)
        return [self.chunks[i] for i in indices[0]]

    def num_chunks(self) -> int:
        return len(self.chunks)
