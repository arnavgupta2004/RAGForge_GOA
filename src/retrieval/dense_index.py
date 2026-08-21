"""FAISS flat inner-product index for dense retrieval.

Corpus size here is tens of thousands of chunks -- small enough that an exact
flat index is both fast (<5ms for a single query on CPU) and simpler/more
reliable than an approximate index (IVF/HNSW), which would trade accuracy for
speed we don't need at this scale. If the corpus grew by 100x this would be
the first thing to swap.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_dense_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_dense_index(index: faiss.Index, path: str | Path) -> None:
    faiss.write_index(index, str(path))


def load_dense_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def search_dense(index: faiss.Index, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores, ids) for a single query vector, shape (top_k,) each."""
    scores, ids = index.search(query_vec.reshape(1, -1), top_k)
    return scores[0], ids[0]
