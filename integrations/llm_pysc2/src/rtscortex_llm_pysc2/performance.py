"""Cumulative phase profiler for the managed LLM-PySC2 worker."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _PhaseStats:
    count: int = 0
    total_ns: int = 0
    max_ns: int = 0


class PhaseProfiler:
    """Collect cumulative phase timings without changing simulation cadence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phases: dict[str, _PhaseStats] = {}

    def observe_milliseconds(self, phase: str, elapsed_ms: float) -> None:
        elapsed_ns = max(0, int(float(elapsed_ms) * 1_000_000))
        with self._lock:
            stats = self._phases.setdefault(phase, _PhaseStats())
            stats.count += 1
            stats.total_ns += elapsed_ns
            stats.max_ns = max(stats.max_ns, elapsed_ns)

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - started
            self.observe_milliseconds(phase, elapsed / 1_000_000)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                phase: {
                    "count": stats.count,
                    "total_ms": stats.total_ns / 1_000_000,
                    "mean_ms": (stats.total_ns / stats.count / 1_000_000 if stats.count else 0.0),
                    "max_ms": stats.max_ns / 1_000_000,
                }
                for phase, stats in sorted(self._phases.items())
            }
