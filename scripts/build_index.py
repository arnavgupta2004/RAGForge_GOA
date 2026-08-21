"""Offline preprocessing: chunk the MSMARCO-XI subset, embed every chunk, and
persist FAISS + BM25 indices to data/processed/.

This is the ONLY place chunking and embedding happen. The runtime API loads
the artifacts this script produces and never recomputes them -- see
src/pipeline/orchestrator.py.

Builds two parallel index variants so the ablation study
(docs/evaluation.md) has something concrete to compare:

  - baseline/  naive fixed-window chunking (uniform, no parent-child links),
               dense-only retrieval. The "generic RAG" strawman.
  - ragforge/  adaptive chunking (atomic short passages; fixed + sentence +
               overlapping children for long ones, all parent-linked) with
               a BM25 index alongside dense, for hybrid retrieval.

Chunk insertion order is preserved 1:1 with FAISS's positional ids: row i of
chunks.jsonl is always vector i in dense.faiss.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.pipeline import PassageRecord, chunk_passage_adaptive, chunk_passage_baseline
from src.embeddings.embedder import Embedder
from src.retrieval.dense_index import build_dense_index, save_dense_index
from src.retrieval.sparse_index import build_sparse_index, save_sparse_index
from src.telemetry.config import load_config


def load_raw(raw_path: Path) -> list[dict]:
    rows = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_queries_file(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            qid = row["query_id"]
            is_selected = row["is_selected"]
            gold_doc_ids = [f"q{qid}_p{i}" for i, sel in enumerate(is_selected) if sel == 1]
            has_no_answer = "no answer present" in row["eng_answer"].lower()
            record = {
                "query_id": qid,
                "query_type": row["query_type"],
                "eng_query": row["eng_query"],
                "hin_query": row["hin_query"],
                "eng_answer": row["eng_answer"],
                "hin_answer": row["hin_answer"],
                "gold_doc_ids": gold_doc_ids,
                "is_answerable": len(gold_doc_ids) > 0 and not has_no_answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_variant(
    rows: list[dict],
    variant: str,
    cfg,
    embedder: Embedder,
    out_dir: Path,
) -> None:
    print(f"[{variant}] chunking...", file=sys.stderr)
    t0 = time.perf_counter()
    chunks = []
    for row in rows:
        qid = row["query_id"]
        query_type = row["query_type"]
        is_selected = row["is_selected"]
        for i, passage_text in enumerate(row["passages"]):
            passage = PassageRecord(
                doc_id=f"q{qid}_p{i}",
                text=passage_text,
                query_id=qid,
                query_type=query_type,
                is_selected=bool(is_selected[i]),
            )
            if variant == "ragforge":
                chunks.extend(
                    chunk_passage_adaptive(
                        passage,
                        atomic_max_tokens=cfg.chunking.atomic_max_tokens,
                        fixed_window=cfg.chunking.fixed["window_tokens"],
                        sentence_per_chunk=cfg.chunking.sentence["sentences_per_chunk"],
                        overlap_window=cfg.chunking.overlapping["window_tokens"],
                        overlap_tokens=cfg.chunking.overlapping["overlap_tokens"],
                    )
                )
            else:
                chunks.extend(chunk_passage_baseline(passage, window_tokens=cfg.chunking.fixed["window_tokens"]))
    chunk_ms = (time.perf_counter() - t0) * 1000
    print(f"[{variant}] {len(chunks)} chunks from {len(rows)} queries in {chunk_ms:.0f}ms", file=sys.stderr)

    print(f"[{variant}] embedding...", file=sys.stderr)
    t0 = time.perf_counter()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embed_ms = (time.perf_counter() - t0) * 1000
    print(f"[{variant}] embedded {len(texts)} chunks in {embed_ms:.0f}ms", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{variant}] building dense index...", file=sys.stderr)
    dense_index = build_dense_index(embeddings)
    save_dense_index(dense_index, out_dir / "dense.faiss")

    sparse_built = False
    if variant == "ragforge":
        print(f"[{variant}] building sparse (BM25) index...", file=sys.stderr)
        sparse_index = build_sparse_index(texts, k1=cfg.sparse.k1, b=cfg.sparse.b)
        save_sparse_index(sparse_index, out_dir / "bm25")
        sparse_built = True

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    meta = {
        "variant": variant,
        "num_chunks": len(chunks),
        "num_queries": len(rows),
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.dim,
        "sparse_index": sparse_built,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_ms": round(chunk_ms, 1),
        "embed_ms": round(embed_ms, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{variant}] done -> {out_dir}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/development.yaml")
    parser.add_argument("--limit", type=int, default=None, help="cap number of queries, for quick dev iteration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = cfg.path(cfg.data.raw_path)
    processed_dir = cfg.path(cfg.data.processed_dir)

    rows = load_raw(raw_path)
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} queries from {raw_path}", file=sys.stderr)

    build_queries_file(rows, processed_dir / "queries.jsonl")

    embedder = Embedder(
        cfg.embeddings.model_name,
        device=cfg.embeddings.device,
        batch_size=cfg.embeddings.batch_size,
    )

    build_variant(rows, "baseline", cfg, embedder, processed_dir / "baseline")
    build_variant(rows, "ragforge", cfg, embedder, processed_dir / "ragforge")


if __name__ == "__main__":
    main()
