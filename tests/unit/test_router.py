from src.routing.router import route

ENTITY_TYPES = ["ENTITY", "PERSON", "LOCATION", "NUMERIC"]


def test_very_short_query_routes_multi_query():
    decision = route("carbide", None, min_tokens_for_multi_query=3, lexical_specificity_threshold=0.6, entity_query_types=ENTITY_TYPES)
    assert decision.strategy == "multi_query"


def test_numeric_entity_heavy_query_routes_sparse_or_hybrid_rerank():
    decision = route(
        "how far is eureka ca 95501 to klamath falls 97601",
        "NUMERIC",
        min_tokens_for_multi_query=3,
        lexical_specificity_threshold=0.6,
        entity_query_types=ENTITY_TYPES,
    )
    assert decision.strategy in ("sparse", "hybrid_rerank")
    assert decision.features.lexical_specificity >= 0.6


def test_descriptive_conceptual_query_routes_dense():
    decision = route(
        "what is the significance of serial dilution in laboratory science",
        "DESCRIPTION",
        min_tokens_for_multi_query=3,
        lexical_specificity_threshold=0.6,
        entity_query_types=ENTITY_TYPES,
    )
    assert decision.strategy == "dense"


def test_moderately_specific_query_routes_hybrid():
    decision = route(
        "how does the criminal justice system define actuarial risk",
        "DESCRIPTION",
        min_tokens_for_multi_query=3,
        lexical_specificity_threshold=0.6,
        entity_query_types=ENTITY_TYPES,
    )
    assert decision.strategy in ("hybrid", "dense")


def test_reason_is_populated():
    decision = route("what is opentable", "DESCRIPTION", 3, 0.6, ENTITY_TYPES)
    assert decision.reason
