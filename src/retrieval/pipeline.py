"""Executes the retrieval strategy chosen by src/routing/router.py.

Concurrency: dense and sparse search both run for hybrid* strategies. They
are cheap (single-digit ms each at this corpus size) and independent, so they
run via asyncio.gather over thread-pool-wrapped calls rather than serially --
a real, if modest, latency optimization at this scale, and the right pattern
if the corpus grows.

Embedding, index search, and reranking are timed separately (not just lumped
into one "retrieval" number) so the P50/P70/... latency dashboard can show
where time actually goes.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from src.chunking.models import Chunk
from src.embeddings.embedder import Embedder
from src.retrieval.confidence import calibrated_confidence
from src.retrieval.dense_index import search_dense
from src.retrieval.fusion import fuse
from src.retrieval.index_store import IndexStore
from src.retrieval.models import ContextItem, RetrievalResult, RetrievedChunk
from src.retrieval.query_variants import generate_variants
from src.retrieval.reranker import rerank
from src.retrieval.sparse_index import search_sparse
from src.routing.router import RoutingDecision
from src.telemetry.config import RetrievalConfig


async def _embed_async(embedder: Embedder, texts: list[str]) -> list[np.ndarray]:
    def _run() -> list[np.ndarray]:
        return [embedder.encode_one(t) for t in texts]

    return await asyncio.to_thread(_run)


async def _dense_search_vec_async(store: IndexStore, vec: np.ndarray, top_k: int) -> dict[int, float]:
    def _run() -> dict[int, float]:
        scores, ids = search_dense(store.dense_index, vec, top_k)
        return {int(i): float(s) for i, s in zip(ids, scores) if i != -1}

    return await asyncio.to_thread(_run)


async def _sparse_search_async(store: IndexStore, query: str, top_k: int) -> dict[int, float]:
    def _run() -> dict[int, float]:
        if store.sparse_index is None:
            return {}
        scores, ids = search_sparse(store.sparse_index, query, top_k)
        return {int(i): float(s) for i, s in zip(ids, scores)}

    return await asyncio.to_thread(_run)


def _build_context(candidates: list[RetrievedChunk]) -> list[ContextItem]:
    """Dedupe by parent passage and expand non-atomic chunks back to their
    full parent text -- the parent-child context-expansion strategy. Highest
    scoring chunk per parent wins if multiple sibling chunks were retrieved."""
    best_by_doc: dict[str, RetrievedChunk] = {}
    for cand in candidates:
        doc_id = cand.chunk.doc_id
        if doc_id not in best_by_doc or cand.final_score > best_by_doc[doc_id].final_score:
            best_by_doc[doc_id] = cand

    items = []
    for cand in best_by_doc.values():
        c: Chunk = cand.chunk
        expanded = c.chunking_strategy != "atomic"
        items.append(
            ContextItem(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.original_text if expanded else c.text,
                expanded=expanded,
                chunking_strategy=c.chunking_strategy,
                final_score=cand.final_score,
                dense_score=cand.dense_score,
                sparse_score=cand.sparse_score,
                fused_score=cand.fused_score,
                rerank_score=cand.rerank_score,
            )
        )
    items.sort(key=lambda i: i.final_score, reverse=True)
    return items


async def retrieve(
    query_text: str,
    decision: RoutingDecision,
    store: IndexStore,
    embedder: Embedder,
    cfg: RetrievalConfig,
) -> RetrievalResult:
    strategy = decision.strategy
    query_variants = [query_text] if strategy != "multi_query" else generate_variants(query_text)

    embed_t0 = time.perf_counter()
    needs_embedding = strategy in ("dense", "multi_query", "hybrid", "hybrid_rerank")
    query_vecs = await _embed_async(embedder, query_variants) if needs_embedding else []
    embedding_ms = (time.perf_counter() - embed_t0) * 1000

    retrieval_t0 = time.perf_counter()

    if strategy == "sparse":
        sparse = await _sparse_search_async(store, query_text, cfg.top_k_sparse)
        dense: dict[int, float] = {}
        fused = fuse(dense, sparse, alpha=0.0)

    elif strategy == "dense":
        dense = await _dense_search_vec_async(store, query_vecs[0], cfg.top_k_dense)
        sparse = {}
        fused = fuse(dense, sparse, alpha=1.0)

    elif strategy == "multi_query":
        dense_results = await asyncio.gather(
            *[_dense_search_vec_async(store, vec, cfg.top_k_dense) for vec in query_vecs]
        )
        dense = {}
        for result in dense_results:
            for idx, score in result.items():
                dense[idx] = max(dense.get(idx, 0.0), score)
        sparse = {}
        fused = fuse(dense, sparse, alpha=1.0)

    else:  # hybrid, hybrid_rerank
        dense, sparse = await asyncio.gather(
            _dense_search_vec_async(store, query_vecs[0], cfg.top_k_dense),
            _sparse_search_async(store, query_text, cfg.top_k_sparse),
        )
        fused = fuse(dense, sparse, alpha=cfg.hybrid_alpha)

    pool = fused[: cfg.top_k_fused]
    retrieval_ms = (time.perf_counter() - retrieval_t0) * 1000

    rerank_t0 = time.perf_counter()
    if strategy == "hybrid_rerank" and pool:
        texts = [store.chunk(c.chunk_idx).text for c in pool]
        rerank_scores = await asyncio.to_thread(rerank, query_text, texts)
        candidates = [
            RetrievedChunk(
                chunk=store.chunk(c.chunk_idx),
                dense_score=c.dense_score,
                sparse_score=c.sparse_score,
                fused_score=c.fused_score,
                rerank_score=rerank_scores[i],
                final_score=rerank_scores[i],
            )
            for i, c in enumerate(pool)
        ]
        candidates.sort(key=lambda c: c.final_score, reverse=True)
    else:
        candidates = [
            RetrievedChunk(
                chunk=store.chunk(c.chunk_idx),
                dense_score=c.dense_score,
                sparse_score=c.sparse_score,
                fused_score=c.fused_score,
                rerank_score=None,
                final_score=c.fused_score,
            )
            for c in pool
        ]
    reranking_ms = (time.perf_counter() - rerank_t0) * 1000

    # Dedupe by parent document BEFORE truncating to the final k, not after --
    # sibling chunks from the same long passage (fixed/sentence/overlapping
    # variants) would otherwise cannibalize slots that should cover distinct
    # documents, undercounting document-level recall@k purely as an artifact
    # of chunking granularity rather than retrieval quality.
    seen_docs: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for c in candidates:
        if c.chunk.doc_id not in seen_docs:
            seen_docs.add(c.chunk.doc_id)
            deduped.append(c)
    candidates = deduped[: cfg.rerank_top_k]
    context = _build_context(candidates)
    top_score = calibrated_confidence(candidates[0] if candidates else None)

    return RetrievalResult(
        strategy=strategy,
        strategy_reason=decision.reason,
        query_variants=query_variants,
        retrieved=candidates,
        context=context,
        top_score=top_score,
        embedding_ms=round(embedding_ms, 3),
        retrieval_ms=round(retrieval_ms, 3),
        reranking_ms=round(reranking_ms, 3),
    )
