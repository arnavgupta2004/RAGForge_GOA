"""Dense/sparse score fusion.

Dense (cosine, roughly 0..1) and BM25 (unbounded, corpus-dependent) scores
live on different scales, so a raw weighted sum would let whichever score
has the larger numeric range dominate regardless of `hybrid_alpha`. We
min-max normalize each candidate set to [0, 1] over the pool of chunks
retrieved by either method, then combine with a single readable weight,
alpha, so a demo can point at one number and say "this is how much dense
counted vs. sparse."
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedCandidate:
    chunk_idx: int
    dense_score: float | None
    sparse_score: float | None
    fused_score: float


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse(
    dense: dict[int, float],
    sparse: dict[int, float],
    alpha: float,
) -> list[FusedCandidate]:
    """dense/sparse: {chunk_idx: raw_score}. alpha weights dense vs (1-alpha) sparse."""
    dense_norm = _min_max_normalize(dense)
    sparse_norm = _min_max_normalize(sparse)

    all_ids = set(dense_norm) | set(sparse_norm)
    candidates = []
    for idx in all_ids:
        d = dense_norm.get(idx, 0.0)
        s = sparse_norm.get(idx, 0.0)
        fused = alpha * d + (1 - alpha) * s
        candidates.append(
            FusedCandidate(
                chunk_idx=idx,
                dense_score=dense.get(idx),
                sparse_score=sparse.get(idx),
                fused_score=fused,
            )
        )
    candidates.sort(key=lambda c: c.fused_score, reverse=True)
    return candidates
