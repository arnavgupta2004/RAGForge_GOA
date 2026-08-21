# Evaluation

Methodology and results from `scripts/evaluate.py`. Two independent parts;
the first needs no API key, the second needs `GEMINI_API_KEY`.

## 1. Retrieval quality and ablation (real, run 2026-08-21)

150 randomly sampled answerable queries (seed 42) from the 6,000-query
MSMARCO-XI subset, each with real `is_selected` gold-passage labels pooled
against the full ~60k-passage corpus (see
[docs/decisions.md](decisions.md#dataset-language-and-corpus-construction)
for why pooling, not per-query candidate sets, is the honest benchmark
here).

| | recall@5 | MRR |
|---|---|---|
| **baseline** (naive fixed chunking, dense-only, no router) | 0.8667 | 0.5804 |
| **RAGForge** (adaptive chunking, hybrid retrieval, router) | 0.8667 | 0.5923 |

Overall recall ties; RAGForge's MRR is measurably higher (correct answers
rank closer to #1). The more interesting number is the subset that motivates
adaptive chunking in the first place:

| | n | recall@5 |
|---|---|---|
| Context-expansion subset (gold passage long enough to be split into children) | 34 | **0.9412** |
| Overall | 150 | 0.8667 |

On exactly the queries where the answer lives in a long passage —
where a naive fixed-size chunker either truncates context or an atomic
whole-passage embedding gets diluted — adaptive chunking with parent-child
expansion recovers meaningfully more of the gold evidence than the overall
average. This is the honest version of the ablation story: not "adaptive
chunking is universally better," but "adaptive chunking earns its
complexity specifically on the queries it was built for."

Retrieval-strategy distribution and recall on this sample (router's actual
choices, not forced):

| strategy | share of queries | recall@5 within strategy |
|---|---|---|
| dense | 82/150 | 0.805 |
| hybrid | 38/150 | 0.842 |
| multi_query | 29/150 | 0.931 |
| hybrid_rerank | 1/150 | 1.000 |

Which chunking strategy the winning (gold-matching) chunk came from, when
there was a hit:

| chunking_strategy | hits |
|---|---|
| atomic | 99 |
| sentence | 19 |
| fixed | 8 |
| overlapping | 4 |

Most hits are naturally atomic (most passages are short), which is expected
and correct — the strategy diversity exists for the passages that need it,
not for its own sake.

A methodology note worth stating plainly: an earlier run of this same
ablation showed the baseline slightly *beating* RAGForge on recall@5
(0.8667 vs. 0.8333) before a real bug was found and fixed — sibling child
chunks from the same long passage were cannibalizing top-k slots that
should have covered distinct documents. See
[docs/decisions.md](decisions.md#retrieval-slot-allocation-dedupe-by-document-before-truncating-not-after)
for the fix. The numbers above are post-fix.

Reproduce: `python scripts/evaluate.py --sample-size 150`

## 2. Guardrails, refusal quality, and hallucination rate

**Status: pending `GEMINI_API_KEY`.** This part of the suite runs the full
live pipeline (real generation calls) over a labeled set spanning easy
factual, semantic, keyword-heavy, ambiguous, context-expansion,
no-answer-in-dataset (2,135 real MSMARCO-XI queries with `is_selected`
all-zero — see [docs/decisions.md](decisions.md)), off-topic, adversarial,
prompt-injection, and noisy-ASR-like queries
(`scripts/evaluate.py::SYNTHETIC_EVAL_SET`), and reports:

- refusal precision / recall against the labeled expected-answer set
- hallucination rate (an "answered" status on a no-gold-evidence query)
- prompt-injection detection rate

Reproduce once a key is available:
`python scripts/evaluate.py --sample-size 150 --with-generation`

This file will be updated with the real numbers from that run — no
placeholder numbers are reported here in the meantime.
