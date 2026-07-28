"""Run a bounded EventStore throughput smoke with acceptance-facing metrics."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from rtscortex.memory import EventStore


def profile(*, loops: int, events_per_loop: int) -> dict[str, float | int]:
    if loops < 1 or events_per_loop < 1:
        raise ValueError("loops and events_per_loop must be positive")
    with tempfile.TemporaryDirectory(prefix="rtscortex-event-smoke-") as directory:
        root = Path(directory)
        store = EventStore(
            root / "events.sqlite3",
            root / "events.jsonl",
        )
        started = time.perf_counter()
        for game_loop in range(loops):
            for event_index in range(events_per_loop):
                store.append_event(
                    run_id="event-store-smoke",
                    episode_id="episode-1",
                    step_id=game_loop,
                    event_type="performance_smoke",
                    payload={
                        "game_loop": game_loop,
                        "event_index": event_index,
                    },
                )
        store.flush()
        elapsed_seconds = time.perf_counter() - started
        performance = store.performance_snapshot()
        store.close()
        total_events = loops * events_per_loop
        return {
            "loops": loops,
            "events": total_events,
            "elapsed_seconds": elapsed_seconds,
            "loops_per_second": loops / elapsed_seconds,
            "events_per_loop": total_events / loops,
            "bytes_per_loop": performance.journal_bytes / loops,
            "queue_peak": performance.max_queue_depth,
            "queue_capacity": performance.queue_capacity,
            "writer_lag_ms_p95": performance.writer_lag_ms_p95,
            "writer_lag_ms_max": performance.writer_lag_ms_max,
            "blocked_append_count": performance.blocked_append_count,
            "dropped_sampled_event_count": performance.dropped_sampled_event_count,
            "append_latency_ms_mean": performance.append_latency_ms_mean,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, default=2_000)
    parser.add_argument("--events-per-loop", type=int, default=6)
    arguments = parser.parse_args()
    print(
        json.dumps(
            profile(
                loops=arguments.loops,
                events_per_loop=arguments.events_per_loop,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
