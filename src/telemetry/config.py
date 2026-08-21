"""Loads a RAGForge config file (yaml) merged with environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class ChunkingConfig(BaseModel):
    atomic_max_tokens: int
    fixed: dict[str, int]
    overlapping: dict[str, int]
    sentence: dict[str, int]


class EmbeddingsConfig(BaseModel):
    model_name: str
    batch_size: int
    device: str


class SparseConfig(BaseModel):
    backend: str
    k1: float
    b: float


class RetrievalConfig(BaseModel):
    top_k_dense: int
    top_k_sparse: int
    top_k_fused: int
    rerank_top_k: int
    hybrid_alpha: float
    context_expansion_neighbors: int


class RouterConfig(BaseModel):
    min_query_tokens_for_multi_query: int
    lexical_specificity_threshold: float
    entity_query_types: list[str]


class GuardrailsConfig(BaseModel):
    min_retrieval_score: float
    min_context_chunks: int
    groundedness_min_overlap: float


class GenerationConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int
    temperature: float
    timeout_s: float
    max_retries: int


class ASRConfig(BaseModel):
    provider: str
    model: str
    mode: str
    timeout_s: float
    max_retries: int


class DataConfig(BaseModel):
    raw_path: str
    processed_dir: str


class AppConfig(BaseModel):
    env: str
    log_level: str


class TelemetryConfig(BaseModel):
    log_dir: str


class Config(BaseModel):
    app: AppConfig
    data: DataConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    sparse: SparseConfig
    retrieval: RetrievalConfig
    router: RouterConfig
    guardrails: GuardrailsConfig
    generation: GenerationConfig
    asr: ASRConfig
    telemetry: TelemetryConfig

    def path(self, relative: str) -> Path:
        return REPO_ROOT / relative


@lru_cache(maxsize=8)
def load_config(config_path: str | None = None) -> Config:
    path = Path(config_path or os.environ.get("RAGFORGE_CONFIG", "configs/development.yaml"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return Config.model_validate(raw)
