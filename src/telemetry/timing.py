"""High-resolution stage timing for request telemetry.

Every pipeline stage records its own wall-clock duration into a shared
`StageTimer`, which produces the `latency_ms` block returned to clients and
written to the benchmark log. Uses `time.perf_counter` (monotonic, sub-ms
resolution) rather than `time.time()`, which is not guaranteed monotonic and
is coarser on some platforms.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self._durations_ms: dict[str, float] = {}
        self._total_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._durations_ms[name] = self._durations_ms.get(name, 0.0) + elapsed_ms

    def record(self, name: str, elapsed_ms: float) -> None:
        self._durations_ms[name] = self._durations_ms.get(name, 0.0) + elapsed_ms

    def total_ms(self) -> float:
        return (time.perf_counter() - self._total_start) * 1000

    def as_dict(self) -> dict[str, float]:
        out = {k: round(v, 3) for k, v in self._durations_ms.items()}
        out["total"] = round(self.total_ms(), 3)
        return out
