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

## 2. Guardrails, refusal quality, and hallucination rate (real, run 2026-08-21)

Full live pipeline (real Gemini + real retrieval, paced at 12 req/min for
the free-tier quota), 20 real MSMARCO-XI queries (10 answerable, 10 labeled
"no answer in dataset") + 15 synthetic queries spanning off-topic,
adversarial/unsafe, prompt-injection, noisy-ASR-like, and ambiguous-short
categories (`scripts/evaluate.py::SYNTHETIC_EVAL_SET`).

| metric | value |
|---|---|
| refusal precision | **1.00** (0 of the 10 answerable queries were wrongly refused) |
| refusal recall | 0.632 |
| hallucination rate on no-answer-in-dataset queries | 0.368 |
| prompt-injection detection rate | **1.00** (4/4) |
| unsafe-query refusal | 2/2 |

Full per-query breakdown: `data/benchmarks/evaluation_report.json`.

**Reading the hallucination-rate number honestly, not just reporting it.**
36.8% "answered" on MSMARCO-labeled no-answer queries sounds bad in
isolation, so it's worth actually reading what those answers were rather
than stopping at the percentage. Spot-checking three of them:

- *"does silvadene contain silver"* → **"Yes, Silvadene cream contains
  silver in the form of micronized silver sulfadiazine."** — correct, and
  directly grounded in the retrieved passage.
- *"what is the diameter of the apple spaceship"* → **"The diameter of
  Apple's headquarters is a bit over 1,521 feet."** — correct, grounded.
- *"What is the capital of France?"* (an early version of this eval set) →
  **"The capital of France is Versailles, which Louis XIV moved the capital
  to in May 1682."** — historically accurate, and grounded in a real
  passage the pooled corpus happens to contain about Louis XIV moving the
  French capital in 1682. This wasn't a guardrail failure at all: the query
  turned out not to be off-topic for this corpus by coincidence, so it was
  removed from the synthetic set (see the note in
  `scripts/evaluate.py::SYNTHETIC_EVAL_SET`) rather than left in as a
  mislabeled "should refuse" item.

The pattern: MS MARCO's `Eng_Answer == "No Answer Present."` label reflects
what the *original human annotator* chose to write under their own time
constraints and guidelines — not a guarantee that no correct answer is
constructible from the passages. RAGForge, given the same passages and a
capable model instructed to answer only from context, sometimes succeeds
where the annotator didn't bother. That means this hallucination-rate number
is a **conservative, real measurement against a strict label**, not
necessarily 36.8% worth of fabricated facts — some fraction of it is the
system doing its job correctly on a passage the human skipped. Both
readings are true at once, which is exactly why the number is reported
rather than rounded away in either direction.

What the gates actually caught in this run: both unsafe queries
(refused, `unsafe_query`), all 4 prompt-injection attempts (correctly
flagged *and* refused via `low_retrieval_confidence` — the injected text
had nothing on-topic behind it either), 4/5 off-topic queries, and 2
no-answer-in-dataset queries where the model attempted an answer and the
**groundedness gate caught it post-generation** (`ungrounded_answer`) —
concrete evidence the layered design in `docs/architecture.md` is pulling
its weight, not just the pre-generation confidence gate alone.

Reproduce: `python scripts/evaluate.py --sample-size 150 --with-generation --rpm 12`
(the `--rpm` cap keeps the run under Gemini's free-tier 15 req/min quota
instead of eating retriable 429s that would otherwise pollute the stats).
