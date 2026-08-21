"""Latency benchmark harness.

Runs a real sample of queries through the full live pipeline (real ASR-free
text queries -> router -> retrieval -> generation -> guardrails, hitting the
actual Gemini API) and reports P50/P70/P90/P95/P99/P100, mean, min, max --
both for total latency and each stage. No numbers are hard-coded; this is the
only source of truth for docs/latency.md.

Reports COLD (first N requests, model/index just loaded, includes first-call
warmup effects like lazy cross-encoder loading) separately from WARM (the
rest), since the brief requires distinguishing the two rather than only
reporting the best-case warm number.

Run: python scripts/benchmark.py --n 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.embeddings.embedder import Embedder
from src.generation.llm_client import GenerationError, LLMClient
from src.pipeline.orchestrator import PipelineDeps, run_pipeline
from src.retrieval.index_store import IndexStore
from src.telemetry.config import load_config

RANDOM_SEED = 123


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return round(values[idx], 2)


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "p50": percentile(values, 50),
        "p70": percentile(values, 70),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "p100": percentile(values, 100),
        "mean": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "n": len(values),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--cold-n", type=int, default=5, help="how many of the first requests count as 'cold'")
    parser.add_argument("--rpm", type=int, default=12, help="requests/minute cap (free-tier Gemini quota is 15); pacing doesn't affect per-request latency, only how long the whole run takes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    store = IndexStore(cfg.path(cfg.data.processed_dir) / "ragforge")
    embedder = Embedder(cfg.embeddings.model_name, device=cfg.embeddings.device, batch_size=cfg.embeddings.batch_size)

    try:
        llm_client = LLMClient(
            model=cfg.generation.model, max_tokens=cfg.generation.max_tokens,
            temperature=cfg.generation.temperature, timeout_s=cfg.generation.timeout_s,
            max_retries=cfg.generation.max_retries,
        )
    except GenerationError as e:
        print(f"ERROR: cannot benchmark generation without a working LLM client: {e}", file=sys.stderr)
        sys.exit(1)

    deps = PipelineDeps(cfg=cfg, store=store, embedder=embedder, llm_client=llm_client, asr_client=None)

    queries_path = cfg.path(cfg.data.processed_dir) / "queries.jsonl"
    all_queries = [json.loads(l) for l in queries_path.open(encoding="utf-8")]
    random.seed(RANDOM_SEED)
    sample = random.sample(all_queries, min(args.n, len(all_queries)))

    delay_s = 60.0 / args.rpm
    per_request: list[dict] = []
    print(f"Running {len(sample)} live requests against the full pipeline (paced at {args.rpm}/min)...", file=sys.stderr)
    for i, q in enumerate(sample):
        if i > 0:
            await asyncio.sleep(delay_s)
        t0 = time.perf_counter()
        response = await run_pipeline(deps, query_text=q["eng_query"])
        wall_ms = (time.perf_counter() - t0) * 1000
        per_request.append({"latency_ms": response.latency_ms, "status": response.status, "wall_ms": wall_ms})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample)}", file=sys.stderr)

    cold = per_request[: args.cold_n]
    warm = per_request[args.cold_n :]

    def stage_series(rows: list[dict], stage: str) -> list[float]:
        return [r["latency_ms"].get(stage, 0.0) for r in rows]

    stages = ["asr", "query_processing", "embedding", "retrieval", "reranking", "generation", "guardrail", "total"]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_total": len(per_request),
        "n_cold": len(cold),
        "n_warm": len(warm),
        "cold": {
            "total_latency_ms": summarize(stage_series(cold, "total")),
        },
        "warm": {
            "total_latency_ms": summarize(stage_series(warm, "total")),
            "by_stage_ms": {s: summarize(stage_series(warm, s)) for s in stages},
        },
        "status_counts": {
            s: sum(1 for r in per_request if r["status"] == s) for s in {r["status"] for r in per_request}
        },
    }

    out_dir = cfg.path(cfg.telemetry.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
