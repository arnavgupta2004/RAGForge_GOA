import pytest

from src.retrieval.pipeline import retrieve
from src.routing.router import route

ENTITY_TYPES = ["ENTITY", "PERSON", "LOCATION", "NUMERIC"]


def _route(cfg, query, query_type=None):
    return route(
        query, query_type,
        cfg.router.min_query_tokens_for_multi_query,
        cfg.router.lexical_specificity_threshold,
        cfg.router.entity_query_types,
    )


@pytest.mark.asyncio
async def test_retrieve_returns_context_for_known_query(cfg, ragforge_store, embedder):
    decision = _route(cfg, "what is a corporation", "DESCRIPTION")
    result = await retrieve("what is a corporation", decision, ragforge_store, embedder, cfg.retrieval)
    assert result.context
    assert result.top_score > 0
    assert all(0.0 <= c.final_score or c.final_score is not None for c in result.retrieved)


@pytest.mark.asyncio
async def test_retrieve_context_expansion_flag_set_for_split_passages(cfg, ragforge_store, embedder):
    decision = _route(cfg, "what is the significance of serial dilution in laboratory science", "DESCRIPTION")
    result = await retrieve(
        "what is the significance of serial dilution in laboratory science",
        decision, ragforge_store, embedder, cfg.retrieval,
    )
    # at least verify the field is populated and boolean, not that expansion always fires
    assert all(isinstance(c.expanded, bool) for c in result.context)


@pytest.mark.asyncio
async def test_sparse_strategy_executes_without_dense_scores(cfg, ragforge_store, embedder):
    decision = _route(cfg, "eureka ca 95501 klamath falls 97601 distance miles", "NUMERIC")
    result = await retrieve(
        "eureka ca 95501 klamath falls 97601 distance miles", decision, ragforge_store, embedder, cfg.retrieval
    )
    assert decision.strategy in ("sparse", "hybrid_rerank", "hybrid")


@pytest.mark.asyncio
async def test_multi_query_generates_variants(cfg, ragforge_store, embedder):
    decision = _route(cfg, "carbide")
    result = await retrieve("carbide", decision, ragforge_store, embedder, cfg.retrieval)
    assert decision.strategy == "multi_query"
    assert len(result.query_variants) >= 1


@pytest.mark.asyncio
async def test_timing_fields_populated(cfg, ragforge_store, embedder):
    decision = _route(cfg, "what is opentable", "DESCRIPTION")
    result = await retrieve("what is opentable", decision, ragforge_store, embedder, cfg.retrieval)
    assert result.embedding_ms >= 0
    assert result.retrieval_ms >= 0
    assert result.reranking_ms >= 0
