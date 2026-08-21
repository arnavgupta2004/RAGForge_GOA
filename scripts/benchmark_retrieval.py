"""Retrieval-only latency micro-benchmark: embed + dense search, isolated
from the network-bound generation call.

scripts/benchmark.py measures the full live pipeline (ASR-free text query
through router -> retrieval -> generation -> guardrails, real Gemini calls)
and is the authoritative end-to-end number in docs/latency.md. This script
answers a narrower, complementary question: how fast is OUR retrieval path
by itself, against the real RAGForge index, with no LLM in the loop at all?
That isolation matters because the dominant cost in the full pipeline is a
generation network round-trip (hundreds of ms to a hosted model, outside our
control), not retrieval -- this script is the evidence for that claim rather
than an assertion of it.

Usage:
    python scripts/benchmark_retrieval.py [--n 50] [--budget-ms 50]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import Embedder
from src.retrieval.dense_index import search_dense
from src.retrieval.index_store import IndexStore
from src.telemetry.config import load_config

QUERIES = [
    "what is a corporation",
    "how far is eureka ca to klamath falls",
    "what is the significance of serial dilution",
    "define militia",
    "garnet price per carat",
    "what is opentable",
    "criminal justice actuarial definition",
    "what is good feng shui",
]


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--budget-ms", type=float, default=50.0, help="p95 latency budget for embed+search combined")
    args = parser.parse_args()

    cfg = load_config(args.config)
    store = IndexStore(cfg.path(cfg.data.processed_dir) / "ragforge")
    embedder = Embedder(cfg.embeddings.model_name, device=cfg.embeddings.device, batch_size=cfg.embeddings.batch_size)

    print("Warming up (model load + first inference)...", file=sys.stderr)
    embedder.encode_one(QUERIES[0])
    search_dense(store.dense_index, embedder.encode_one(QUERIES[0]), top_k=cfg.retrieval.top_k_dense)

    embed_ms, search_ms, total_ms = [], [], []
    for i in range(args.n):
        query = QUERIES[i % len(QUERIES)]

        t0 = time.perf_counter()
        vec = embedder.encode_one(query)
        t1 = time.perf_counter()
        search_dense(store.dense_index, vec, top_k=cfg.retrieval.top_k_dense)
        t2 = time.perf_counter()

        embed_ms.append((t1 - t0) * 1000)
        search_ms.append((t2 - t1) * 1000)
        total_ms.append((t2 - t0) * 1000)

    print(f"\nRan {args.n} queries against {len(store.chunks)} indexed chunks (no generation, no ASR)\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [("embed", embed_ms), ("search", search_ms), ("total", total_ms)]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nRetrieval-only latency budget: {args.budget_ms}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= args.budget_ms:
        print("PASS: retrieval itself is within budget -- see docs/latency.md for why full-pipeline latency is dominated by the generation call instead")
    else:
        print("FAIL: retrieval alone exceeds budget -- see docs/decisions.md (dense index) before assuming this is a generation-latency issue")
        sys.exit(1)


if __name__ == "__main__":
    main()
