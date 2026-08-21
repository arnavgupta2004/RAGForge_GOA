"""RAGForge request orchestrator.

Request -> [ASR] -> normalize/safety -> route -> retrieve -> pre-generation
guardrails -> generate -> groundedness guardrail -> response.

Every external call (ASR, LLM) is wrapped so its failure produces a typed
"error" response rather than an unhandled exception reaching the API layer.
Guardrail failures produce a "refused" response with a reason, never a raw
error -- refusal is a normal, successful outcome of this system, not a
failure mode.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from src.asr.sarvam_client import ASRError, SarvamClient
from src.embeddings.embedder import Embedder
from src.generation.llm_client import GenerationError, LLMClient
from src.guardrails.gates import (
    context_sufficiency_gate,
    detect_prompt_injection,
    detect_unsafe_query,
    groundedness_check,
    retrieval_confidence_gate,
)
from src.pipeline.schemas import PipelineResponse, SourceOut
from src.retrieval.index_store import IndexStore
from src.retrieval.models import RetrievalResult
from src.retrieval.pipeline import retrieve
from src.routing.router import route
from src.telemetry.config import Config
from src.telemetry.logging_utils import log_request
from src.telemetry.timing import StageTimer

logger = logging.getLogger("ragforge.pipeline")

REFUSAL_MESSAGES = {
    "unsafe_query": "This request falls outside what I can help with.",
    "low_retrieval_confidence": "I couldn't find sufficient evidence in the knowledge base to answer this question reliably.",
    "insufficient_context": "I couldn't find sufficient evidence in the knowledge base to answer this question reliably.",
    "ungrounded_answer": "I don't have enough evidence in the knowledge base to answer that reliably.",
}


@dataclass
class PipelineDeps:
    cfg: Config
    store: IndexStore
    embedder: Embedder
    llm_client: LLMClient
    asr_client: SarvamClient | None


def _sources_from_context(retrieval_result: RetrievalResult) -> list[SourceOut]:
    sources = []
    for item in retrieval_result.context:
        snippet = item.text if len(item.text) <= 280 else item.text[:277] + "..."
        sources.append(
            SourceOut(
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                snippet=snippet,
                chunking_strategy=item.chunking_strategy,
                expanded=item.expanded,
                dense_score=item.dense_score,
                sparse_score=item.sparse_score,
                fused_score=item.fused_score,
                rerank_score=item.rerank_score,
                final_score=item.final_score,
            )
        )
    return sources


def _base_kwargs(request_id: str, timer: StageTimer, transcript: str | None, detected_language: str | None) -> dict:
    return {
        "request_id": request_id,
        "transcript": transcript,
        "detected_language": detected_language,
        "latency_ms": timer.as_dict(),
    }


def _log_and_return(deps: PipelineDeps, response: PipelineResponse) -> PipelineResponse:
    log_request(
        deps.cfg.telemetry.log_dir,
        {
            "request_id": response.request_id,
            "status": response.status,
            "retrieval_strategy": response.retrieval_strategy,
            "confidence": response.confidence,
            "reason": response.reason,
            "latency_ms": response.latency_ms,
        },
    )
    return response


async def run_pipeline(
    deps: PipelineDeps,
    query_text: str | None = None,
    audio_bytes: bytes | None = None,
    audio_filename: str = "audio.wav",
) -> PipelineResponse:
    request_id = uuid.uuid4().hex
    timer = StageTimer()
    transcript: str | None = None
    detected_language: str | None = None

    if audio_bytes is not None:
        if deps.asr_client is None:
            return _log_and_return(
                deps,
                PipelineResponse(
                    status="error",
                    answer="Voice input is not configured on this deployment.",
                    confidence=0.0,
                    **_base_kwargs(request_id, timer, transcript, detected_language),
                ),
            )
        asr_error: ASRError | None = None
        with timer.stage("asr"):
            try:
                asr_result = await deps.asr_client.transcribe(audio_bytes, audio_filename)
                transcript = asr_result.transcript
                detected_language = asr_result.language_code
                query_text = transcript
            except ASRError as e:
                asr_error = e

        if asr_error is not None:
            logger.warning("ASR failed: %s", asr_error)
            return _log_and_return(
                deps,
                PipelineResponse(
                    status="error",
                    answer="I couldn't process that audio. Please try again.",
                    confidence=0.0,
                    **_base_kwargs(request_id, timer, transcript, detected_language),
                ),
            )
    else:
        timer.record("asr", 0.0)

    if not query_text or not query_text.strip():
        return _log_and_return(
            deps,
            PipelineResponse(
                status="error",
                answer="No question was provided.",
                confidence=0.0,
                **_base_kwargs(request_id, timer, transcript, detected_language),
            ),
        )

    query_text = query_text.strip()

    with timer.stage("query_processing"):
        is_unsafe = detect_unsafe_query(query_text)
        injection_detected = detect_prompt_injection(query_text)
        decision = None
        if not is_unsafe:
            decision = route(
                query_text,
                query_type=None,
                min_tokens_for_multi_query=deps.cfg.router.min_query_tokens_for_multi_query,
                lexical_specificity_threshold=deps.cfg.router.lexical_specificity_threshold,
                entity_query_types=deps.cfg.router.entity_query_types,
            )

    # returned outside the `with` block so the stage's duration is already
    # recorded by the time timer.as_dict() is read (a `return` from inside
    # the block would build the response before the context manager's
    # `finally` records it)
    if is_unsafe:
        return _log_and_return(
            deps,
            PipelineResponse(
                status="refused",
                answer=REFUSAL_MESSAGES["unsafe_query"],
                reason="unsafe_query",
                confidence=0.0,
                **_base_kwargs(request_id, timer, transcript, detected_language),
            ),
        )

    retrieval_result = await retrieve(query_text, decision, deps.store, deps.embedder, deps.cfg.retrieval)
    timer.record("embedding", retrieval_result.embedding_ms)
    timer.record("retrieval", retrieval_result.retrieval_ms)
    timer.record("reranking", retrieval_result.reranking_ms)

    with timer.stage("guardrail"):
        conf_gate = retrieval_confidence_gate(retrieval_result.top_score, deps.cfg.guardrails.min_retrieval_score)
        suff_gate = context_sufficiency_gate(retrieval_result.context, deps.cfg.guardrails.min_context_chunks)

    if not conf_gate.passed:
        return _log_and_return(
            deps,
            PipelineResponse(
                status="refused",
                answer=REFUSAL_MESSAGES["low_retrieval_confidence"],
                reason="low_retrieval_confidence",
                confidence=round(retrieval_result.top_score, 3),
                sources=_sources_from_context(retrieval_result),
                retrieval_strategy=retrieval_result.strategy,
                retrieval_reason=retrieval_result.strategy_reason,
                query_variants=retrieval_result.query_variants,
                prompt_injection_detected=injection_detected,
                **_base_kwargs(request_id, timer, transcript, detected_language),
            ),
        )
    if not suff_gate.passed:
        return _log_and_return(
            deps,
            PipelineResponse(
                status="refused",
                answer=REFUSAL_MESSAGES["insufficient_context"],
                reason="insufficient_context",
                confidence=round(retrieval_result.top_score, 3),
                sources=_sources_from_context(retrieval_result),
                retrieval_strategy=retrieval_result.strategy,
                retrieval_reason=retrieval_result.strategy_reason,
                query_variants=retrieval_result.query_variants,
                prompt_injection_detected=injection_detected,
                **_base_kwargs(request_id, timer, transcript, detected_language),
            ),
        )

    try:
        with timer.stage("generation"):
            gen_result = await deps.llm_client.generate(query_text, retrieval_result.context)
    except GenerationError as e:
        logger.warning("Generation failed: %s", e)
        return _log_and_return(
            deps,
            PipelineResponse(
                status="error",
                answer="I hit an error generating a response. Please try again.",
                confidence=round(retrieval_result.top_score, 3),
                retrieval_strategy=retrieval_result.strategy,
                retrieval_reason=retrieval_result.strategy_reason,
                **_base_kwargs(request_id, timer, transcript, detected_language),
            ),
        )

    with timer.stage("guardrail"):
        ground_gate, overlap = groundedness_check(
            gen_result.answer, retrieval_result.context, deps.cfg.guardrails.groundedness_min_overlap
        )

    if not ground_gate.passed:
        return _log_and_return(
            deps,
            PipelineResponse(
                status="refused",
                answer=REFUSAL_MESSAGES["ungrounded_answer"],
                reason="ungrounded_answer",
                confidence=round(retrieval_result.top_score, 3),
                sources=_sources_from_context(retrieval_result),
                retrieval_strategy=retrieval_result.strategy,
                retrieval_reason=retrieval_result.strategy_reason,
                query_variants=retrieval_result.query_variants,
                prompt_injection_detected=injection_detected,
                groundedness_overlap=round(overlap, 3),
                **_base_kwargs(request_id, timer, transcript, detected_language),
            ),
        )

    return _log_and_return(
        deps,
        PipelineResponse(
            status="answered",
            answer=gen_result.answer,
            sources=_sources_from_context(retrieval_result),
            confidence=round(retrieval_result.top_score, 3),
            retrieval_strategy=retrieval_result.strategy,
            retrieval_reason=retrieval_result.strategy_reason,
            query_variants=retrieval_result.query_variants,
            prompt_injection_detected=injection_detected,
            groundedness_overlap=round(overlap, 3),
            **_base_kwargs(request_id, timer, transcript, detected_language),
        ),
    )
