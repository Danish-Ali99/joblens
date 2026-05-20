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


def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """Split a resume into paragraph chunks, merging short ones up to ~max_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
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
    return chunks or [text]


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
