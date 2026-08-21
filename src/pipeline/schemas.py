"""Typed request/response contracts for the RAGForge pipeline and API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Status = Literal["answered", "refused", "error"]


class SourceOut(BaseModel):
    chunk_id: str
    doc_id: str
    snippet: str
    chunking_strategy: str
    expanded: bool
    dense_score: float | None
    sparse_score: float | None
    fused_score: float | None
    rerank_score: float | None
    final_score: float


class PipelineResponse(BaseModel):
    request_id: str
    status: Status
    answer: str
    sources: list[SourceOut] = []
    confidence: float
    retrieval_strategy: str | None = None
    retrieval_reason: str | None = None
    query_variants: list[str] = []
    transcript: str | None = None
    detected_language: str | None = None
    reason: str | None = None
    prompt_injection_detected: bool = False
    groundedness_overlap: float | None = None
    latency_ms: dict[str, float]
