"""Exact dense retrieval via a plain in-memory NumPy matrix.

This deliberately does NOT use FAISS. faiss-cpu bundles its own OpenMP
runtime, which collides with PyTorch's (sentence-transformers) OpenMP runtime
in the same process -- reproducibly segfaults the moment a real forward pass
runs after FAISS has been loaded (verified during development; see
docs/decisions.md). Rather than paper over that with an unsafe
KMP_DUPLICATE_LIB_OK env flag (which the OpenMP project itself documents as
"may cause crashes or silently produce incorrect results"), we avoid the
conflict entirely.

At this corpus size (~10^5 chunks x 384 dims) a brute-force matrix-vector
product is exact -- identical math to FAISS's IndexFlatIP -- and takes single-
digit milliseconds, so there is no quality or meaningful latency cost. If the
corpus grew by 100x, an approximate index would be the first thing to
reconsider, in a process that never also loads torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_dense_index(embeddings: np.ndarray) -> np.ndarray:
    return embeddings.astype("float32")


def save_dense_index(index: np.ndarray, path: str | Path) -> None:
    np.save(str(path), index)


def load_dense_index(path: str | Path) -> np.ndarray:
    return np.load(str(path))


def search_dense(index: np.ndarray, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores, ids) for a single query vector, shape (top_k,) each,
    sorted descending by score. index rows are assumed L2-normalized, so the
    dot product is cosine similarity."""
    scores = index @ query_vec
    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return scores[top_idx], top_idx
