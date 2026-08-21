"""Retrieval router: picks a retrieval strategy from query features.

A deterministic, inspectable heuristic rather than a learned classifier --
justified because the signal is cheap and strong: MSMARCO's own `query_type`
taxonomy (DESCRIPTION/NUMERIC/ENTITY/PERSON/LOCATION) already tells us how
lexical-specific a query is, and we corroborate it with cheap surface
features (token count, digit/proper-noun density) so the router also works
on live ASR transcripts that have no query_type label.

Strategies:
  - sparse            short, entity/number-heavy queries: exact term match wins.
  - dense              longer, descriptive/conceptual queries: semantic match wins.
  - hybrid             the common middle case: fuse both signals.
  - hybrid_rerank      hybrid retrieval is ambiguous (top score low / scores
                       close together) -- worth the extra cross-encoder pass.
  - multi_query        very short or vague queries, where a single embedding
                       may not capture intent; generate lightweight variants.

This module only classifies; src/retrieval/pipeline.py executes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RetrievalStrategy = Literal["sparse", "dense", "hybrid", "hybrid_rerank", "multi_query"]

_DIGIT_RE = re.compile(r"\d")
_CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-z]+\b")


@dataclass
class QueryFeatures:
    token_count: int
    has_digits: bool
    proper_noun_count: int
    query_type: str | None
    lexical_specificity: float  # 0..1 heuristic score


@dataclass
class RoutingDecision:
    strategy: RetrievalStrategy
    features: QueryFeatures
    reason: str


def _lexical_specificity(text: str, query_type: str | None, entity_types: set[str]) -> float:
    tokens = text.split()
    n = max(len(tokens), 1)
    digit_ratio = len(_DIGIT_RE.findall(text)) / n
    proper_nouns = len(_CAPITALIZED_WORD_RE.findall(text))
    proper_ratio = proper_nouns / n
    type_bonus = 0.35 if (query_type and query_type.upper() in entity_types) else 0.0
    score = min(1.0, digit_ratio * 1.5 + proper_ratio * 1.2 + type_bonus)
    return round(score, 3)


def route(
    query_text: str,
    query_type: str | None,
    min_tokens_for_multi_query: int,
    lexical_specificity_threshold: float,
    entity_query_types: list[str],
) -> RoutingDecision:
    text = query_text.strip()
    tokens = text.split()
    entity_types = {t.upper() for t in entity_query_types}

    features = QueryFeatures(
        token_count=len(tokens),
        has_digits=bool(_DIGIT_RE.search(text)),
        proper_noun_count=len(_CAPITALIZED_WORD_RE.findall(text)),
        query_type=query_type,
        lexical_specificity=_lexical_specificity(text, query_type, entity_types),
    )

    if features.token_count <= min_tokens_for_multi_query:
        return RoutingDecision(
            strategy="multi_query",
            features=features,
            reason=f"very short query ({features.token_count} tokens) -- generating variants to disambiguate intent",
        )

    if features.lexical_specificity >= lexical_specificity_threshold:
        if features.lexical_specificity >= 0.85:
            return RoutingDecision(
                strategy="sparse",
                features=features,
                reason=f"high lexical specificity ({features.lexical_specificity}) -- exact-term match should dominate",
            )
        return RoutingDecision(
            strategy="hybrid_rerank",
            features=features,
            reason=f"moderately lexical-specific ({features.lexical_specificity}), entity/number-bearing -- hybrid with reranking",
        )

    if features.lexical_specificity <= 0.15:
        return RoutingDecision(
            strategy="dense",
            features=features,
            reason=f"low lexical specificity ({features.lexical_specificity}) -- conceptual/descriptive query, semantic match wins",
        )

    return RoutingDecision(
        strategy="hybrid",
        features=features,
        reason=f"mixed lexical specificity ({features.lexical_specificity}) -- fuse dense and sparse signals",
    )
