"""A single calibrated confidence score, comparable across every retrieval
strategy, used by the guardrail confidence gate and reported to the client.

This is NOT the fusion score used for ranking. `fuse()` min-max normalizes
scores within one query's own candidate pool, which makes the top result
~1.0 by construction regardless of whether it's actually a good match --
useless as an absolute confidence signal (verified empirically: off-topic
and genuinely-unanswerable queries scored just as "confident" as answerable
ones once fusion-normalized). Guardrail confidence instead reads the raw,
un-normalized score of the top candidate and maps it onto a common 0..1
scale per signal type:

  - dense cosine similarity is already 0..1, used as-is.
  - BM25 is unbounded; squashed via score / (score + k) with k picked from
    the observed corpus scale (see docs/decisions.md calibration run).
  - cross-encoder rerank logits are unbounded; squashed via sigmoid.

When both dense and sparse fired (hybrid strategies), we take the max of the
two calibrated confidences -- either signal independently finding a strong
match is evidence enough; requiring both would only punish sparse-poor
paraphrases or dense-poor exact-term queries.
"""

from __future__ import annotations

import math

from src.retrieval.models import RetrievedChunk

_BM25_SATURATION_K = 8.0


def _bm25_confidence(raw: float) -> float:
    return max(raw, 0.0) / (max(raw, 0.0) + _BM25_SATURATION_K)


def _rerank_confidence(raw: float) -> float:
    return 1 / (1 + math.exp(-raw))


def calibrated_confidence(top: RetrievedChunk | None) -> float:
    if top is None:
        return 0.0

    signals = []
    if top.dense_score is not None:
        signals.append(max(0.0, min(1.0, top.dense_score)))
    if top.sparse_score is not None:
        signals.append(_bm25_confidence(top.sparse_score))
    if top.rerank_score is not None:
        signals.append(_rerank_confidence(top.rerank_score))

    if not signals:
        return 0.0
    return max(signals)
