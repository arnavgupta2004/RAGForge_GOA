"""Cross-encoder reranker, used only for the `hybrid_rerank` route.

cross-encoder/ms-marco-MiniLM-L-6-v2 -- again a deliberate MS MARCO-native
choice, not a generic cross-encoder. Only invoked on the router's small
top-k candidate pool (config: retrieval.rerank_top_k source, typically the
top ~10-20 fused candidates), since scoring the full corpus pairwise would
blow the latency budget; a cross-encoder is O(top_k) forward passes, not
O(corpus).
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(_MODEL_NAME)


def rerank(query: str, candidates: list[str]) -> list[float]:
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, c) for c in candidates]
    scores = model.predict(pairs)
    return [float(s) for s in scores]
