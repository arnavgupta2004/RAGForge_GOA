# Latency

## What's measured

`scripts/benchmark.py` runs a sample of real text queries through the full
live pipeline — router → retrieval → generation → guardrails — with real
network calls to the Gemini API (no mocking, no cached responses). ASR
latency is not included in this specific run (it benchmarks text queries;
Sarvam's own latency is a separate, additive network round-trip measured
the same way once voice input is exercised). Every stage is timed
independently via `time.perf_counter` in `src/telemetry/timing.py`.

## Full-pipeline results (real, run 2026-08-21, n=100)

`python scripts/benchmark.py --n 100 --rpm 12` (paced to stay under the
free-tier Gemini quota; pacing only affects how long the harness takes to
run, not any individual request's measured latency). 100 real text queries
sampled from the indexed corpus, real Gemini calls, no mocking.

**Total latency, warm (n=95, first 5 requests reported separately as cold):**

| | p50 | p70 | p90 | p95 | p99 | p100 | mean | min | max |
|---|---|---|---|---|---|---|---|---|---|
| warm | 936ms | 970ms | 1059ms | 1105ms | 1251ms | 1755ms | 884ms | 22ms | 1755ms |
| cold (n=5) | 920ms | 1016ms | 1206ms | 1206ms | 1206ms | 1206ms | 969ms | 847ms | 1206ms |

Cold and warm are close here — unlike a from-scratch process start, this
harness measures requests *after* the embedding model and index are already
loaded (that load happens once, outside the timed loop), so what "cold"
captures in this run is mainly first-request JIT/connection-pool effects,
not full model warmup. The min=22ms warm outlier is a query that hit the
retrieval confidence gate and refused before ever calling Gemini — the
pre-generation guardrail paying for itself exactly as designed.

**Per-stage breakdown, warm (ms):**

| stage | p50 | p95 | p99 | mean |
|---|---|---|---|---|
| asr | 0 (text queries; not exercised here) | — | — | 0 |
| query_processing | 0.09 | 0.16 | 0.30 | 0.10 |
| embedding | 25.9 | 32.5 | 38.7 | 25.5 |
| retrieval | 5.97 | 9.07 | 17.3 | 6.37 |
| reranking | 0.07 | 0.13 | 0.17 | 0.07 |
| **generation** | **897** | **1068** | **1240** | **851** |
| guardrail | 0.18 | 0.30 | 0.43 | 0.18 |
| **total** | **936** | **1105** | **1251** | **884** |

**What this says, honestly**: for TEXT queries specifically (no ASR — this
benchmark never touches Sarvam), generation is ~96% of total latency (851ms
of 884ms mean) — every other stage combined is under 35ms. This figure does
NOT generalize to voice queries; see "Voice input latency" below, where ASR
is a separate, real network cost that is frequently comparable to or larger
than generation, not a rounding error next to it. This is exactly
what the architecture was built to accept rather than fight: a real network
round-trip to a hosted LLM cannot be made sub-200ms, so the design puts the
optimization effort into *skipping that call entirely* when possible (the
pre-generation guardrail gates) rather than chasing an unrealistic
end-to-end target. The measurement boundary is explicit: this is the full
live path — router, retrieval, generation, and guardrails, real network
calls, no ASR in this specific run (see the retrieval-only and voice
sections below for those pieces measured separately).

One honest caveat on the `embedding` row: it's higher here (25.5ms mean)
than in the isolated micro-benchmark below (5.0ms mean) for the same model.
Two real, identified causes, not a mystery: (1) ~29% of queries in this
sample were routed to `multi_query`, which embeds two lexical variants
sequentially rather than one; (2) a second local dev server was briefly
running on this machine during part of this benchmark run for an unrelated
verification check, adding real CPU contention. The isolated number below is
the more representative "how fast is embedding alone" figure; this row is
the more representative "what a real request mix actually pays."

Reproduce: `python scripts/benchmark.py --n 100 --rpm 12`

## Voice input latency (real, 5 live requests, same audio clip via Sarvam)

The benchmark above uses text queries (ASR not exercised). Separately, 5
real voice requests (English speech, `saaras:v3`, `mode=translate`) through
the full pipeline:

| | asr | generation | total |
|---|---|---|---|
| run 1 | 583ms | 1182ms | 1934ms |
| run 2 | 557ms | 905ms | 1494ms |
| run 3 | 1377ms | 821ms | 2226ms |
| run 4 | 534ms | 953ms | 1505ms |
| run 5 | 1086ms | 888ms | 2005ms |
| **mean** | **828ms** | **950ms** | **1833ms** |

ASR adds a second real network round-trip (~0.5-1.4s here) on top of
generation, additive as expected — a full voice query is meaningfully
slower than a typed one, and that's reported plainly rather than only
benchmarking the cheaper text path and calling it "voice latency."

**Post-deployment production re-check** (Railway, real Hindi audio, 3 back-
to-back requests, after the thread-pinning fix in `docs/decisions.md`):

| | asr | generation | total |
|---|---|---|---|
| run 1 | 1207ms | 712ms | 1937ms |
| run 2 | 1193ms | 662ms | 1872ms |
| run 3 | 1189ms | 618ms | 1822ms |

Same pattern holds in production as it did locally: ASR is not a rounding
error next to generation, it's the larger of the two here. In one live
session during testing, a single request showed ASR at ~2.8-3.5s (still
under 5s, within normal network variance for a real third-party API call,
but a real reminder that Sarvam's round-trip time is not fixed) — that
request's total (~3.5s) was correctly ASR-dominated, not a telemetry bug.
The stage-to-field mapping in the frontend (`frontend/src/App.tsx`'s
`stages` array) reads `asr`/`retrieval`/`generation`/`guardrail`/`total`
directly off the same JSON keys the backend's `StageTimer` produces
(`src/telemetry/timing.py`, `src/pipeline/orchestrator.py`) — verified by
inspection, no relabeling or reassignment between backend and UI. When a
query is refused before generation ever runs (a pre-generation guardrail
gate), the `generation` key is simply absent from `latency_ms` and the UI
correctly omits that bar rather than showing a misleading zero.

## Known-cheap paths (already measured, no API key needed)

**Retrieval-only micro-benchmark** (`scripts/benchmark_retrieval.py`, embed +
dense search, no generation, no ASR, no router/fusion overhead), 50 queries
against the live 122,580-chunk RAGForge index, 2026-08-21:

| stage | avg | p50 | p95 | p99 |
|---|---|---|---|---|
| embed | 5.01ms | 4.87ms | 5.90ms | 7.15ms |
| search | 3.50ms | 3.45ms | 3.90ms | 3.95ms |
| **total** | **8.51ms** | **8.40ms** | **9.40ms** | **10.54ms** |

p95 total: **9.40ms** against a 50ms budget — pass, with a wide margin. This
confirms the NumPy dense-index decision in `docs/decisions.md` cost nothing
in practice: embedding the query (a real forward pass through
`msmarco-MiniLM-L6-cos-v5`) dominates over the actual matrix search.

Separately, `scripts/evaluate.py`'s full retrieval eval (150 queries,
including router decisions, hybrid fusion, and cross-encoder reranking on
the queries the router selected it for) completed in 11.2s wall clock, i.e.
~75ms/query average — still cheap, and consistent with the isolated number
above once router/fusion/occasional-rerank overhead is included.

The dominant cost in the full pipeline is the generation network call (a
real LLM round-trip, see the cold/warm table above), not retrieval — which
is why the architecture optimizes for skipping generation entirely on
low-confidence queries (the pre-generation guardrail gates) rather than
trying to make generation itself faster than a network call to a hosted
model can be.
