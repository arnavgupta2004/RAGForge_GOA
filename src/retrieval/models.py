"""Result types shared between retrieval, guardrails, generation, and the API
response -- these are exactly what the frontend's retrieval-explainability
panel renders (dense/BM25/fused score per source, chunk id, strategy)."""

from __future__ import annotations

from pydantic import BaseModel

from src.chunking.models import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    final_score: float


class ContextItem(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    expanded: bool
    chunking_strategy: str
    final_score: float
    dense_score: float | None
    sparse_score: float | None
    fused_score: float | None
    rerank_score: float | None


class RetrievalResult(BaseModel):
    strategy: str
    strategy_reason: str
    query_variants: list[str]
    retrieved: list[RetrievedChunk]
    context: list[ContextItem]
    top_score: float
    embedding_ms: float
    retrieval_ms: float
    reranking_ms: float
