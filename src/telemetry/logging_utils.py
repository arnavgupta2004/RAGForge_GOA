"""Structured, append-only JSONL request logging.

Each pipeline run appends one JSON object with request_id, timestamp,
retrieval strategy, guardrail decision, latency breakdown, and final status.
Never logs raw audio bytes or full transcripts by default -- only what is
needed to reconstruct latency/quality metrics offline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

_write_lock = Lock()


def log_request(log_dir: str | Path, record: dict) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {"logged_at": time.time(), **record}
    line = json.dumps(record, ensure_ascii=False)
    path = log_dir / "requests.jsonl"
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
