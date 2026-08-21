"""Loads a built index variant (baseline/ or ragforge/) into memory once.

This is the load side of scripts/build_index.py's persistence contract: chunk
row i in chunks.jsonl is vector i in dense.faiss. Loaded once at process
startup (see src/pipeline/orchestrator.py) and reused for every request --
the runtime path never touches the raw dataset or re-embeds the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.chunking.models import Chunk
from src.retrieval.dense_index import load_dense_index
from src.retrieval.sparse_index import load_sparse_index


class IndexStore:
    def __init__(self, variant_dir: Path) -> None:
        self.variant_dir = variant_dir
        self.meta = json.loads((variant_dir / "meta.json").read_text())

        self.chunks: list[Chunk] = []
        with (variant_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                self.chunks.append(Chunk.model_validate_json(line))

        self.dense_index = load_dense_index(variant_dir / "dense.faiss")
        self.sparse_index = load_sparse_index(variant_dir / "bm25") if self.meta.get("sparse_index") else None

    def chunk(self, idx: int) -> Chunk:
        return self.chunks[idx]

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        # small enough corpus that a lazy dict build on first miss is fine
        if not hasattr(self, "_by_id"):
            self._by_id = {c.chunk_id: c for c in self.chunks}
        return self._by_id.get(chunk_id)
