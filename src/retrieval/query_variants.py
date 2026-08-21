"""Lightweight, non-LLM multi-query expansion.

The spec calls for generating query variants for ambiguous/short queries.
An LLM call per variant would add a full model round-trip to the latency
budget for exactly the queries where we can least afford it (short, vague
ones already flagged by the router as needing help). Instead we generate
cheap lexical variants -- strip interrogative scaffolding, keep content
words -- and let dense retrieval do the semantic work across all variants.
This is a deliberate latency-over-cleverness tradeoff; see docs/decisions.md.
"""

from __future__ import annotations

import re

_LEAD_WORDS = re.compile(
    r"^\s*(what|who|whom|whose|where|when|why|how|which|is|are|was|were|do|does|did|can|could|should|would)\b\.?\s*",
    re.IGNORECASE,
)
_TRAILING_PUNCT = re.compile(r"[?.!]+\s*$")


def generate_variants(query: str, max_variants: int = 2) -> list[str]:
    variants = [query.strip()]

    stripped = _TRAILING_PUNCT.sub("", query).strip()
    without_lead = _LEAD_WORDS.sub("", stripped).strip()
    # strip a second leading interrogative scaffold, e.g. "what is a corporation" -> "a corporation"
    without_lead = _LEAD_WORDS.sub("", without_lead).strip()

    if without_lead and without_lead.lower() != query.strip().lower():
        variants.append(without_lead)

    seen = set()
    out = []
    for v in variants:
        key = v.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out[:max_variants]
