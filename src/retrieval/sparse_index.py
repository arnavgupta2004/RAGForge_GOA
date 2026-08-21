"""BM25 sparse index via bm25s.

bm25s is a pure-numpy BM25 implementation (no Java/Lucene dependency), fast
enough at this corpus size to run synchronously alongside dense search rather
than needing a separate retrieval service.
"""

from __future__ import annotations

from pathlib import Path

import bm25s


def build_sparse_index(corpus_texts: list[str], k1: float, b: float) -> bm25s.BM25:
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus_tokens, show_progress=False)
    return retriever


def save_sparse_index(retriever: bm25s.BM25, path: str | Path) -> None:
    retriever.save(str(path))


def load_sparse_index(path: str | Path) -> bm25s.BM25:
    return bm25s.BM25.load(str(path), load_corpus=False)


def search_sparse(retriever: bm25s.BM25, query_text: str, top_k: int) -> tuple[list[float], list[int]]:
    query_tokens = bm25s.tokenize([query_text], stopwords="en", show_progress=False)
    ids, scores = retriever.retrieve(query_tokens, k=top_k, show_progress=False)
    return scores[0].tolist(), ids[0].tolist()
