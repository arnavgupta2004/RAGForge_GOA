"""Guardrail gates evaluated before and after generation.

Three gates run BEFORE calling the LLM (cheap, retrieval-signal-only checks
that save a generation call entirely when we already know the answer should
be a refusal -- a real latency optimization, not just a safety feature):

  1. retrieval confidence gate -- top fused/rerank score below threshold
  2. context sufficiency gate  -- fewer than min_context_chunks survived
  3. relevance gate            -- (folded into confidence gate here: a query
     genuinely off-topic for this corpus will not clear the score threshold
     either, since nothing in the corpus is close to it)

One gate runs AFTER generation:

  4. groundedness verification -- does the answer's content actually overlap
     the retrieved context, or did the model add unsupported claims?

Deliberately simple, explainable checks (score thresholds, token overlap)
over anything that needs its own model call -- every extra gate is latency
we can't get back, and an unexplainable one is worse for a judged demo than
no gate at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieval.models import ContextItem

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "but", "with", "as", "by",
    "at", "from", "that", "this", "it", "its", "not", "no", "do", "does",
    "did", "has", "have", "had", "can", "could", "will", "would", "should",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


@dataclass
class GateResult:
    passed: bool
    reason: str | None = None


def retrieval_confidence_gate(top_score: float, min_score: float) -> GateResult:
    if top_score < min_score:
        return GateResult(
            passed=False,
            reason=f"top retrieval score {top_score:.3f} below confidence threshold {min_score:.3f}",
        )
    return GateResult(passed=True)


def context_sufficiency_gate(context: list[ContextItem], min_chunks: int) -> GateResult:
    if len(context) < min_chunks:
        return GateResult(
            passed=False,
            reason=f"only {len(context)} supporting chunk(s) retrieved, need >= {min_chunks}",
        )
    return GateResult(passed=True)


_SELF_DECLINE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"does not (contain|provide|include)",
        r"doesn'?t (contain|provide|include|have)",
        r"(no|not) (enough|sufficient) (information|evidence|context)",
        r"insufficient (information|evidence|context)",
        r"context does not",
        r"context (doesn'?t|does not) (answer|address)",
        r"cannot (be )?answer(ed)?",
        r"unable to answer",
        r"i (don'?t|do not) have (enough|sufficient|specific)",
    ]
]
# Observed in production (Hindi query, model declined in Hindi -- see
# docs/decisions.md): lexical overlap alone can't catch a non-English
# decline, so a few concrete Hindi phrasings are matched directly rather
# than attempting a general multilingual solution.
_SELF_DECLINE_PATTERNS_HI = [
    re.compile(p) for p in [
        r"जानकारी उपलब्ध नहीं",
        r"विशिष्ट जानकारी नहीं",
        r"पर्याप्त जानकारी नहीं",
        r"संदर्भ में उपलब्ध नहीं",
    ]
]


def looks_like_self_decline(answer: str) -> bool:
    """Catches the case a pure lexical-overlap check misses: the model
    HONESTLY explains that the context doesn't answer the question, but in
    doing so it naturally repeats the context's own topic words (e.g. "the
    context only discusses X, Y, Z"), which can score as high overlap
    despite the answer being substantively a decline, not a grounded claim.
    Checked before the overlap heuristic so an honest decline is reported as
    a refusal, not a spuriously "grounded" answer."""
    return any(p.search(answer) for p in _SELF_DECLINE_PATTERNS) or any(
        p.search(answer) for p in _SELF_DECLINE_PATTERNS_HI
    )


def groundedness_check(answer: str, context: list[ContextItem], min_overlap: float) -> tuple[GateResult, float]:
    """Lexical-overlap groundedness proxy: what fraction of the answer's
    content words also appear somewhere in the retrieved context. Cheap,
    deterministic, and language-model-free -- appropriate as a fast guardrail
    layered underneath (not instead of) the generation prompt's own
    "answer only from context" instruction."""
    if looks_like_self_decline(answer):
        return GateResult(passed=False, reason="model declined to answer from the retrieved context"), 0.0

    answer_words = _content_words(answer)
    if not answer_words:
        return GateResult(passed=False, reason="empty or non-substantive answer"), 0.0

    context_words: set[str] = set()
    for item in context:
        context_words |= _content_words(item.text)

    overlap = len(answer_words & context_words) / len(answer_words)
    if overlap < min_overlap:
        return (
            GateResult(
                passed=False,
                reason=f"answer overlap with retrieved context {overlap:.2f} below threshold {min_overlap:.2f}",
            ),
            overlap,
        )
    return GateResult(passed=True), overlap


_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all|any|previous|prior|the) instructions",
        r"disregard (all|any|previous|prior|the) instructions",
        r"you are now",
        r"system prompt",
        r"reveal your (instructions|prompt|system)",
        r"act as (an?|the) (?!assistant\b)",
        r"new instructions?:",
        r"override (your|the) (rules|guidelines|instructions)",
    ]
]


def detect_prompt_injection(text: str) -> bool:
    """Flags likely injection attempts in either the user query or retrieved
    context. Retrieved documents are always treated as untrusted data in the
    generation prompt (see src/generation/prompts.py) regardless of this
    flag -- this check exists to log/demo the attempt, not as the sole
    defense."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


_UNSAFE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhow (to|do i|can i) (make|build|synthesize) (a )?(bomb|explosive|weapon)",
        r"\b(kill|harm|hurt) (myself|someone|others)\b",
    ]
]


def detect_unsafe_query(text: str) -> bool:
    return any(p.search(text) for p in _UNSAFE_PATTERNS)
