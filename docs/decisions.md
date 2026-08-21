# Engineering decisions

Real tradeoffs made while building RAGForge Goa, in the order they came up.
Where a decision was forced by something we measured rather than a
preference, the measurement is included.

## Dataset: language and corpus construction

`ai4bharat/MSMARCO-XI` is machine-translated MS MARCO passage-ranking data:
each row is a `(query, target_lang)` pair with `English_passages`,
`Translated_passages`, `is_selected` gold labels, and a `query_type`
(DESCRIPTION/NUMERIC/ENTITY/PERSON/LOCATION). The full dataset is
10M+/1.37M rows (~55GB) across ~23 languages -- far too large for a
demo-scale index, so we pulled a curated subset: streaming the validation
split (has ground-truth relevance labels, unlike train) and keeping rows
where `target_lang == "hin_Deva"`, which deduplicates to one row per unique
`query_id` (the English fields don't vary by language) while also giving us
a real Hindi translation of the query/answer at no extra cost.

We index and generate in **English** (`Eng_Query`/`English_passages`/
`Eng_Answer`), not the translated text. Embedding and generation quality for
English is materially better than for the translated Indic text, and the
dataset itself is originally English (translation is the round-trip, not the
source). Voice input still needs to handle Indic languages -- see the ASR
section below for how that's reconciled without a separate translation
stage or multilingual embeddings.

**Corpus scale**: 6,000 unique queries x ~10 passages each = ~60,000
passages, pooled into one shared corpus rather than kept as separate
per-query candidate sets. This matters for what "retrieval" actually means
here: MS MARCO's passages-per-query were originally the top-1000 BM25 hits
from the full MS MARCO corpus for that query. Pooling all 6,000 queries'
passages together and requiring the retriever to re-find each query's own
gold passage among ~60k pooled candidates (mostly other queries' passages,
acting as distractors) is a real retrieval benchmark, not a trivial 10-way
closed-book lookup.

**Free "no evidence" ground truth**: 2,135 of 6,000 queries (35.6%) have
`Eng_Answer` == "No Answer Present." -- and for every one of these, all of
that query's passages have `is_selected == 0`. This is a real,
dataset-native "insufficient evidence" label, used as the refusal-recall
test set in `scripts/evaluate.py` instead of only synthetic off-topic
queries.

## ASR: Sarvam, `mode="translate"`, single call

Verified via `docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe`
before implementing (per the instruction to check current APIs rather than
rely on training-time knowledge). `saaras:v3` with `mode="translate"`
transcribes AND translates non-English speech to English text in one call.
This collapses what would otherwise be a two-stage "transcribe, then
detect-language-and-translate" pipeline into one API call: Hindi (or any of
Sarvam's 22+ supported languages) speech in, English text out, with
`language_code` still returned so the UI can show what was actually spoken.
For English speech, translate mode passes the transcript through unchanged.
This is the entirety of the "query normalization" stage for language --
no separate translation call, no multilingual embedding model needed.

## Dense retrieval: NumPy, not FAISS

`faiss-cpu` and PyTorch (via sentence-transformers) each bundle their own
OpenMP runtime. Loading both in the same process and then doing real work
with each -- reproducibly, deterministically -- segfaults the moment
sentence-transformers runs its first real forward pass after FAISS has been
used:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
...
Fatal Python error: Aborted
```

Setting `KMP_DUPLICATE_LIB_OK=TRUE` (the commonly-suggested workaround)
converts the clean abort into a silent segfault later during actual
threaded computation instead of fixing anything -- consistent with OpenMP's
own documentation, which describes that flag as unsafe and says it "may
cause crashes or silently produce incorrect results." Given the instruction
to prioritize reliability over architectural fashion, and that a demo crash
is the single worst outcome for a judged live system, we removed FAISS from
the runtime path entirely rather than fight the conflict. Dense retrieval is
now a plain NumPy matrix (`src/retrieval/dense_index.py`): embeddings are
L2-normalized at index-build time, so `matrix @ query_vector` is an exact
cosine-similarity search, identical math to FAISS's `IndexFlatIP`. At this
corpus size (~120k chunks x 384 dims) it runs in single-digit milliseconds
-- no accuracy or meaningful latency cost, and one less fragile
cross-library dependency in the request path. If the corpus grew ~100x, an
approximate index would be the first thing to reconsider, in a process that
never also loads torch.

## Guardrail confidence: calibrated per-signal, not the fusion score

The initial design used the hybrid fusion score (min-max normalized dense +
sparse scores within one query's candidate pool) as the retrieval-confidence
gate's input. This is wrong, and the bug was caught empirically: min-max
normalization over a single query's own top-k pool makes the top result
~1.0 by construction, regardless of whether it's actually a good match.
Measured on 20 answerable, 20 dataset-labeled-unanswerable, and 4 genuinely
off-topic queries, the "confidence" scores were statistically indistinguishable
(~0.5-1.0 across all three groups) -- useless as an absolute signal.

Fixed by introducing `src/retrieval/confidence.py`: a confidence score
computed from the RAW (pre-fusion) score of the top candidate, mapped onto a
comparable 0..1 scale per signal type -- dense cosine used as-is, BM25
squashed via `score / (score + k)`, cross-encoder logits via sigmoid -- and
taking the max across whichever signals fired. Re-measured on the same
groups: answerable avg 0.735, MSMARCO-labeled-unanswerable avg 0.700,
off-topic avg ~0.51. The gate (`min_retrieval_score: 0.55`) is deliberately
calibrated to catch **off-topic / out-of-corpus** queries specifically --
see the next section for why it can't reliably distinguish
"in-corpus-but-MSMARCO-says-no-clean-answer" from "answerable," and why that
distinction is made downstream instead.

## Why the confidence gate can't (and shouldn't) catch MSMARCO's "no answer" cases

Both answerable and dataset-labeled-unanswerable queries have their own
topically-relevant passages in the pooled corpus (MS MARCO's own top-1000
BM25 pass already filtered for topical relevance before a human ever judged
answerability) -- so both groups retrieve similarly well. The distinction
MS MARCO is actually making ("these passages are on-topic but don't state
the specific fact asked") is a **groundedness** distinction, not a
**retrieval** distinction, and is correctly caught by the post-generation
groundedness gate instead: if Gemini, instructed to answer only from
context, declines because the retrieved passages don't actually contain the
answer, the lexical-overlap check on that decline will (correctly) fail to
find a substantive grounded claim and the pipeline refuses. Layering the
gates this way -- cheap retrieval-side checks for "wrong corpus entirely,"
model-side checks for "right corpus, insufficient specific evidence" --
matches what each signal actually knows.

## Retrieval slot allocation: dedupe by document before truncating, not after

First ablation run showed the naive baseline (uniform fixed-window chunking,
dense-only) slightly *beating* RAGForge on recall@5 (0.867 vs 0.833) --
surprising, and worth chasing down rather than reporting as-is. Cause: long
passages get split into up to ~8 sibling child chunks (fixed + sentence +
overlapping variants). When the top-k candidate list is truncated to 5
chunks *before* deduping by parent document, several of those 5 slots can be
consumed by siblings of the same (right or wrong) document, starving the
list of the document diversity a flat, one-chunk-per-passage baseline gets
for free. Fixed by deduping by `doc_id` over the full fused candidate pool
(config: `top_k_fused: 20`) before truncating to the final `rerank_top_k`,
so "top 5" means five distinct documents for both variants. After the fix:
RAGForge ties baseline overall (0.867 recall@5, better MRR: 0.592 vs 0.580)
and pulls ahead specifically on the queries that motivate adaptive chunking
in the first place -- see `docs/evaluation.md` for the context-expansion
subset numbers.

## Multi-query expansion: lexical variants, not an LLM call

The router flags very short/ambiguous queries for multi-query retrieval. An
LLM call to generate variants would add a full model round-trip to exactly
the queries the router has already flagged as needing extra help -- the
worst place to add latency. Instead, `src/retrieval/query_variants.py`
strips interrogative scaffolding ("what is" / "who is" / etc.) via regex to
produce a cheap second phrasing, and both variants get embedded and
dense-searched, merged by max score. This is a deliberate latency-over-cleverness
tradeoff, not an oversight.

## Generation: Gemini, not Claude

Initially built against the Anthropic Messages API (`claude-haiku-4-5`).
Switched to Google's Gemini API (`google-genai` SDK) per explicit direction.
The switch only touches `src/generation/llm_client.py` and config
(`generation.provider`/`model`) -- the prompt construction, guardrails, and
orchestrator are provider-agnostic by design (they depend only on
`LLMClient.generate(query, context) -> GenerationResult`).

**Model choice within Gemini**: `gemini-2.5-flash` and `gemini-2.0-flash`
both returned `404` ("no longer available to new users") when actually
called -- the API's own error message pointed at `gemini-3.6-flash` as the
current replacement, confirming the value of testing against the live API
rather than trusting a remembered model name. `gemini-3.6-flash` does work,
but is a "thinking" model by default: in testing it took **16.4s** for a
3-sentence grounded answer and the visible output was truncated mid-sentence
even with `max_output_tokens=300`, because thinking tokens are drawn from
the same budget as output tokens. That's disqualifying for a latency-critical
demo. `gemini-flash-lite-latest` answered the same grounded-QA prompt
correctly (including declining when context was insufficient, and ignoring
an injected instruction embedded in retrieved context) in under a second
(918ms / 1021ms across two runs), so it's the configured default
(`configs/*.yaml: generation.model`).

## Release-pass bugs found via live browser testing, not code review

Two genuine bugs surfaced only by actually clicking through the deployed
frontend against the real backend -- neither was visible from reading the
code in isolation.

**Groundedness check scored an honest refusal as "grounded."** Clicking the
frontend's "Hindi" demo (at the time, a Hindi *text* query -- see below)
retrieved several genuinely irrelevant passages (an Urdu dictionary entry,
Hindi phrase-meaning trivia, Japanese BSOD troubleshooting text). Gemini,
correctly following its instructions, declined in Hindi: *"the given context
doesn't contain specific information about the dataset, as it only contains
Urdu meanings of counterfeit, Hindi phrases, ... and details about
PayNetCafe."* The UI showed this as **✓ GROUNDED, overlap 1.000** -- because
an honest decline naturally repeats the same topic words as the (irrelevant)
context it's declining to use ("counterfeit," "Urdu," "Hindi phrases"),
which is exactly what the lexical-overlap heuristic measures. Fixed by
adding `looks_like_self_decline()` in `src/guardrails/gates.py`: a small set
of decline-phrase patterns (English, plus the specific Hindi phrasing
observed in production) checked *before* the overlap calculation, so an
honest decline is reported as `ungrounded_answer` regardless of how much
vocabulary it happens to share with the context. This is a real limitation
of a keyword-based approach -- it won't catch every language or phrasing --
but it's a concrete, evidence-driven improvement over having no check at
all, and it's cheap and explainable, consistent with why this guardrail
layer avoids a second model call in the first place.

**The Hindi demo button bypassed the actual Hindi pipeline.** The original
demo sent Hindi *text* directly to `/api/query`'s `text` field. Sarvam's
translate step only runs on *audio* input (see the ASR section above) --
a text query skips it entirely and hits the English-trained embedding model
with raw Hindi, which is exactly why the retrieval above was garbage.
The button was demonstrating a code path that has nothing to do with the
Hindi *voice* capability it claimed to show. Fixed by bundling a short
recorded Hindi audio clip (`frontend/public/demo-hindi.aiff`, the same
"कॉर्पोरेशन क्या है?" clip verified against the live Sarvam API during
development) and having the Hindi demo button submit it via `queryAudio()`
-- a one-click demo that now exercises the real ASR → translate → retrieve
→ generate path end-to-end, the same as actually speaking into the
microphone.

Also corrected in the same pass: the "Semantic" and "Keyword" demo buttons
originally asked meta-questions about "this dataset" itself (e.g. "what are
the key findings in this dataset"), which the corpus -- a pool of ~6,000
unrelated web-passage Q&A pairs -- has no passages about, so both correctly
refused every time, silently defeating the demo's purpose. Replaced with
verified-answerable real queries (serial dilution; Eureka–Klamath Falls
distance) that exercise dense and hybrid-rerank routing respectively, as
the buttons were always meant to.
