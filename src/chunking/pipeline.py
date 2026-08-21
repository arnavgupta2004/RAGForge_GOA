"""Offline chunking pipeline: passage record -> list[Chunk].

Two entry points are exposed:

- `chunk_passage_adaptive` (the RAGForge strategy): short passages stay
  atomic (parent == child, nothing to gain by slicing a 40-token passage).
  Passages longer than `atomic_max_tokens` get split THREE ways -- fixed,
  sentence-aware, and overlapping -- all pointing back at the same parent
  passage. All three strategies get embedded and indexed side by side, tagged
  by `chunking_strategy`, so evaluation can slice retrieval results by
  strategy and measure which one actually contributes hits per query type.

- `chunk_passage_baseline`: a single naive fixed-window split applied
  uniformly regardless of passage length, with no parent-child linkage. This
  is the "naive RAG" baseline the ablation study (docs/evaluation.md) compares
  against.

This module never touches the network or a model checkpoint -- it is pure
text processing so it can run once, offline, in scripts/build_index.py.
"""

from __future__ import annotations

from src.chunking.models import Chunk
from src.chunking.strategies import (
    count_tokens,
    fixed_token_chunks,
    sentence_aware_chunks,
)


class PassageRecord:
    def __init__(
        self,
        doc_id: str,
        text: str,
        query_id: int,
        query_type: str,
        is_selected: bool,
    ) -> None:
        self.doc_id = doc_id
        self.text = text
        self.query_id = query_id
        self.query_type = query_type
        self.is_selected = is_selected


def _make_chunks(
    passage: PassageRecord,
    texts: list[str],
    strategy: str,
) -> list[Chunk]:
    chunk_ids = [f"{passage.doc_id}::{strategy}::{i}" for i in range(len(texts))]
    chunks = []
    for i, text in enumerate(texts):
        neighbors = [cid for j, cid in enumerate(chunk_ids) if j != i and abs(j - i) == 1]
        chunks.append(
            Chunk(
                chunk_id=chunk_ids[i],
                doc_id=passage.doc_id,
                parent_id=passage.doc_id,
                source="msmarco-xi",
                text=text,
                original_text=passage.text,
                chunking_strategy=strategy,  # type: ignore[arg-type]
                position=i,
                token_count=count_tokens(text),
                neighbor_chunk_ids=neighbors,
                query_id=passage.query_id,
                query_type=passage.query_type,
                is_selected=passage.is_selected,
            )
        )
    return chunks


def chunk_passage_adaptive(
    passage: PassageRecord,
    atomic_max_tokens: int,
    fixed_window: int,
    sentence_per_chunk: int,
    overlap_window: int,
    overlap_tokens: int,
) -> list[Chunk]:
    total_tokens = count_tokens(passage.text)
    if total_tokens <= atomic_max_tokens:
        return _make_chunks(passage, [passage.text], "atomic")

    out: list[Chunk] = []
    out += _make_chunks(passage, fixed_token_chunks(passage.text, fixed_window, 0), "fixed")
    out += _make_chunks(passage, sentence_aware_chunks(passage.text, sentence_per_chunk), "sentence")
    out += _make_chunks(
        passage,
        fixed_token_chunks(passage.text, overlap_window, overlap_tokens),
        "overlapping",
    )
    return out


def chunk_passage_baseline(passage: PassageRecord, window_tokens: int) -> list[Chunk]:
    texts = fixed_token_chunks(passage.text, window_tokens, 0)
    return _make_chunks(passage, texts, "baseline_fixed")
