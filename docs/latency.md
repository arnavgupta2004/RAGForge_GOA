# Latency

## What's measured

`scripts/benchmark.py` runs a sample of real text queries through the full
live pipeline — router → retrieval → generation → guardrails — with real
network calls to the Gemini API (no mocking, no cached responses). ASR
latency is not included in this specific run (it benchmarks text queries;
Sarvam's own latency is a separate, additive network round-trip measured
the same way once voice input is exercised). Every stage is timed
independently via `time.perf_counter` in `src/telemetry/timing.py`.

**Status: pending `GEMINI_API_KEY`.** No numbers are reported below because
none have been measured yet — per the project's explicit instruction, this
file will only ever contain real benchmark output, never a hard-coded or
best-case-only number.

Reproduce once a key is available:

```bash
python scripts/benchmark.py --n 100
```

This writes `data/benchmarks/latency_report.json` and will be summarized
here: P50/P70/P90/P95/P99/P100/mean/min/max for total latency, cold (first 5
requests, includes lazy model warmup e.g. the cross-encoder reranker's first
load) reported separately from warm, plus the same breakdown per stage
(asr/query_processing/embedding/retrieval/reranking/generation/guardrail).

## Known-cheap paths (already measured, no API key needed)

Retrieval-only latency (embedding + dense/sparse search + fusion + optional
rerank), measured directly against the live index during
`scripts/evaluate.py`'s retrieval eval (150 queries, 2026-08-21): the full
150-query retrieval-only pass — no generation — completed in 11.2s wall
clock, i.e. ~75ms/query average across a mix of strategies including
cross-encoder reranking on the queries the router selected it for. This is
consistent with the design goal in `docs/decisions.md`: dense/sparse search
itself is single-digit milliseconds at this corpus size (~120k chunks); the
bulk of that per-query average is Python/asyncio overhead and the
occasional cross-encoder pass, not the index search itself.

The dominant cost in the full pipeline will be the generation network call
(a real LLM round-trip), not retrieval — which is why the architecture
optimizes for skipping generation entirely on low-confidence queries (the
pre-generation guardrail gates) rather than trying to make generation itself
faster than a network call to a hosted model can be.
