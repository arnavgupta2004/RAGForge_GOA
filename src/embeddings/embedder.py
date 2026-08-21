"""Sentence embedding wrapper.

Uses sentence-transformers/msmarco-MiniLM-L6-cos-v5 -- a 6-layer model
distilled and fine-tuned specifically for MS MARCO passage-ranking cosine
similarity. It is a deliberate fit for this dataset (not a generic
off-the-shelf choice): small enough for CPU inference at demo latency, and
trained on exactly the query/passage relevance signal we evaluate against.

Vectors are L2-normalized so FAISS IndexFlatIP (inner product) computes
cosine similarity directly, avoiding a separate normalize step at query time.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 128) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return vecs.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=2)
def get_embedder(model_name: str, device: str = "cpu", batch_size: int = 128) -> Embedder:
    """Process-wide cached embedder so the runtime API loads the model once
    at startup, not per request."""
    return Embedder(model_name, device=device, batch_size=batch_size)
