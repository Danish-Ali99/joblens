"""RAG retriever for resume content (BM25-based).

Chunks a resume by paragraphs (with character-based fallback for tight
LaTeX-rendered PDFs) and indexes with BM25Okapi. Each agent in the workflow
retrieves the top-k chunks most relevant to its role rather than receiving
the full resume blob.

BM25 over sparse text is a standard RAG retriever — used in production
hybrid-retrieval systems alongside dense embeddings. We use it here instead
of dense vectors to keep the deployment lightweight (no torch / no
sentence-transformers / no FAISS) and the cold start fast.
"""
from __future__ import annotations

from typing import List
import re


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def chunk_text(text: str, max_chars: int = 300) -> List[str]:
    """Split a resume into chunks of ~max_chars.

    Strategy: prefer paragraph boundaries (double newline). Merge only very
    short adjacent paragraphs (< 100 chars). For tight LaTeX-rendered PDFs
    that come through pypdf without paragraph breaks, fall back to
    character-based chunking with sentence-boundary preference and overlap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if len(paragraphs) >= 3:
        chunks: List[str] = []
        current = ""
        for p in paragraphs:
            if not current:
                current = p
            elif len(current) < 100 and len(current) + len(p) + 2 <= max_chars:
                current = current + "\n\n" + p
            else:
                chunks.append(current)
                current = p
        if current:
            chunks.append(current)
        if len(chunks) >= 2:
            return chunks

    return _chunk_by_chars(text, max_chars, overlap=60)


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
    """BM25-based retriever over chunked resume text."""

    def __init__(self, text: str):
        from rank_bm25 import BM25Okapi

        self.chunks = chunk_text(text)
        self._tokenized = [_tokenize(c) for c in self.chunks]
        # BM25Okapi needs at least one non-empty doc list
        safe = [tokens or ["_"] for tokens in self._tokenized]
        self._bm25 = BM25Okapi(safe)

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        tokenized_q = _tokenize(query) or ["_"]
        scores = self._bm25.get_scores(tokenized_q)
        k = min(k, len(self.chunks))
        # argsort descending
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i] for i in top_indices]

    def num_chunks(self) -> int:
        return len(self.chunks)
