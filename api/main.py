"""RAGForge Goa API.

Endpoints:
  POST /api/query      voice (multipart audio) or text query -> PipelineResponse
  GET  /health          liveness + what's loaded
  GET  /metrics         latency percentiles from the request log + corpus stats
  POST /benchmark       runs a small in-process benchmark sample, returns percentiles

All heavy artifacts (embedding model, dense/sparse indices) are loaded once
at startup via the lifespan handler -- no request recomputes them.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.asr.sarvam_client import ASRError, SarvamClient
from src.embeddings.embedder import Embedder
from src.generation.llm_client import GenerationError, LLMClient
from src.pipeline.orchestrator import PipelineDeps, run_pipeline
from src.pipeline.schemas import PipelineResponse
from src.retrieval.index_store import IndexStore
from src.telemetry.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragforge.api")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(os.environ.get("RAGFORGE_CONFIG", "configs/development.yaml"))
    logger.info("loading ragforge index...")
    store = IndexStore(cfg.path(cfg.data.processed_dir) / "ragforge")
    logger.info("loading embedder %s...", cfg.embeddings.model_name)
    embedder = Embedder(cfg.embeddings.model_name, device=cfg.embeddings.device, batch_size=cfg.embeddings.batch_size)

    llm_client = None
    try:
        llm_client = LLMClient(
            model=cfg.generation.model,
            max_tokens=cfg.generation.max_tokens,
            temperature=cfg.generation.temperature,
            timeout_s=cfg.generation.timeout_s,
            max_retries=cfg.generation.max_retries,
        )
    except GenerationError as e:
        logger.warning("LLM client unavailable: %s", e)

    asr_client = None
    try:
        asr_client = SarvamClient(
            model=cfg.asr.model, mode=cfg.asr.mode, timeout_s=cfg.asr.timeout_s, max_retries=cfg.asr.max_retries
        )
    except ASRError as e:
        logger.warning("ASR client unavailable: %s", e)

    STATE["cfg"] = cfg
    STATE["store"] = store
    STATE["embedder"] = embedder
    STATE["llm_client"] = llm_client
    STATE["asr_client"] = asr_client
    STATE["started_at"] = time.time()
    logger.info("ragforge ready: %d chunks indexed", len(store.chunks))
    yield
    STATE.clear()


app = FastAPI(title="RAGForge Goa", lifespan=lifespan)

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _deps() -> PipelineDeps:
    return PipelineDeps(
        cfg=STATE["cfg"],
        store=STATE["store"],
        embedder=STATE["embedder"],
        llm_client=STATE["llm_client"],
        asr_client=STATE["asr_client"],
    )


@app.post("/api/query", response_model=PipelineResponse)
async def query(
    text: str | None = Form(default=None),
    audio: UploadFile | None = File(default=None),
) -> PipelineResponse:
    if not text and not audio:
        raise HTTPException(status_code=400, detail="Provide either 'text' or an 'audio' file")

    if STATE.get("llm_client") is None:
        raise HTTPException(status_code=503, detail="Generation is not configured (GEMINI_API_KEY missing)")

    audio_bytes = None
    filename = "audio.wav"
    if audio is not None:
        if STATE.get("asr_client") is None:
            raise HTTPException(status_code=503, detail="Voice input is not configured (SARVAM_API_KEY missing)")
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        filename = audio.filename or filename

    return await run_pipeline(_deps(), query_text=text, audio_bytes=audio_bytes, audio_filename=filename)


@app.get("/health")
async def health() -> dict:
    store: IndexStore | None = STATE.get("store")
    return {
        "status": "ok" if store is not None else "starting",
        "uptime_s": round(time.time() - STATE.get("started_at", time.time()), 1),
        "chunks_indexed": len(store.chunks) if store else 0,
        "llm_configured": STATE.get("llm_client") is not None,
        "asr_configured": STATE.get("asr_client") is not None,
    }


@app.get("/metrics")
async def metrics() -> dict:
    cfg = STATE["cfg"]
    log_path = cfg.path(cfg.telemetry.log_dir) / "requests.jsonl"
    if not log_path.exists():
        return {"count": 0, "message": "no requests logged yet"}

    latencies = []
    statuses: dict[str, int] = {}
    strategies: dict[str, int] = {}
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total = rec.get("latency_ms", {}).get("total")
            if total is not None:
                latencies.append(total)
            statuses[rec.get("status", "unknown")] = statuses.get(rec.get("status", "unknown"), 0) + 1
            strat = rec.get("retrieval_strategy")
            if strat:
                strategies[strat] = strategies.get(strat, 0) + 1

    if not latencies:
        return {"count": 0, "message": "no completed requests with latency yet"}

    latencies.sort()

    def pct(p: float) -> float:
        idx = min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1))))
        return round(latencies[idx], 1)

    return {
        "count": len(latencies),
        "latency_ms": {
            "p50": pct(50),
            "p70": pct(70),
            "p90": pct(90),
            "p95": pct(95),
            "p99": pct(99),
            "p100": pct(100),
            "mean": round(sum(latencies) / len(latencies), 1),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
        },
        "status_counts": statuses,
        "strategy_counts": strategies,
    }


@app.post("/benchmark")
async def benchmark(n: int = 20) -> dict:
    """Runs a small live sample through the full pipeline (real ASR-free text
    queries, real generation calls) and returns latency percentiles. Capped
    to keep this endpoint safe to expose -- the authoritative 100+ query
    benchmark is scripts/benchmark.py, run offline and checked into
    docs/latency.md."""
    n = max(1, min(n, 50))
    if STATE.get("llm_client") is None:
        raise HTTPException(status_code=503, detail="Generation is not configured (GEMINI_API_KEY missing)")

    cfg = STATE["cfg"]
    queries_path = cfg.path(cfg.data.processed_dir) / "queries.jsonl"
    all_queries = [json.loads(l) for l in queries_path.open(encoding="utf-8")]
    sample = random.sample(all_queries, min(n, len(all_queries)))

    totals = []
    for q in sample:
        result = await run_pipeline(_deps(), query_text=q["eng_query"])
        totals.append(result.latency_ms["total"])

    totals.sort()

    def pct(p: float) -> float:
        idx = min(len(totals) - 1, int(round(p / 100 * (len(totals) - 1))))
        return round(totals[idx], 1)

    return {
        "n": len(totals),
        "latency_ms": {
            "p50": pct(50),
            "p70": pct(70),
            "p90": pct(90),
            "p100": pct(100),
            "mean": round(sum(totals) / len(totals), 1),
            "min": round(min(totals), 1),
            "max": round(max(totals), 1),
        },
    }
