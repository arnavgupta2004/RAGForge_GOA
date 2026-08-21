"""Typed chunk record persisted to the index and used throughout retrieval.

Every chunk retains enough metadata to (a) expand back to its parent passage
for context-expansion retrieval, (b) walk to sibling chunks within the same
passage, and (c) be sliced by chunking strategy / query type for the
strategy-comparison and ablation experiments in src/evaluation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ChunkingStrategy = Literal["atomic", "fixed", "sentence", "overlapping", "baseline_fixed"]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str  # id of the parent passage this chunk was derived from
    parent_id: str  # == doc_id; kept as a distinct field for clarity at call sites
    source: str

    text: str
    original_text: str  # full parent passage text, for context expansion

    chunking_strategy: ChunkingStrategy
    position: int
    token_count: int
    neighbor_chunk_ids: list[str] = []

    # corpus-side provenance, needed for retrieval evaluation (recall@k/MRR)
    query_id: int
    query_type: str
    is_selected: bool
