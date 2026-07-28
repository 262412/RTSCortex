from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from rtscortex.contracts import EpisodeOutcome, EpisodeResult, EpisodeSummary
from rtscortex.memory import DisabledMemoryRetriever, EventStore, read_event_log


def test_event_store_persists_events_lessons_and_episode(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    journal = tmp_path / "events.jsonl"
    store = EventStore(database, journal)
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="observation",
        payload={"value": 1},
    )
    store.add_lesson(
        run_id="run",
        episode_id="episode",
        source_step_id=0,
        content="Keep units together.",
    )
    store.record_episode(
        EpisodeResult(
            run_id="run",
            episode_id="episode",
            scenario="test",
            seed=0,
            outcome=EpisodeOutcome.VICTORY,
            steps=1,
        )
    )
    store.close()

    reopened = EventStore(database, journal)
    assert reopened.lessons("run", "episode") == ["Keep units together."]
    assert reopened.lesson_records("run", "episode")[0].source_step_id == 0
    assert reopened.last_event("run", "episode", "episode_result") is not None
    reopened.close()
    assert [event.event_type for event in read_event_log(journal)] == [
        "observation",
        "episode_result",
    ]


def test_semantic_memory_is_explicitly_disabled() -> None:
    hits = asyncio.run(DisabledMemoryRetriever().search("enemy strategy"))
    assert hits == []


def test_episode_summaries_persist_with_run_isolation(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    journal = tmp_path / "events.jsonl"
    store = EventStore(database, journal)
    summary = EpisodeSummary(
        run_id="run-a",
        episode_id="episode-1",
        scenario="pvz_task1_level1",
        outcome=EpisodeOutcome.VICTORY,
        summary="Won after holding the first attack.",
        lessons=["Keep the army together."],
        source_step_id=4,
    )
    store.record_episode_summary(summary)
    store.record_episode_summary(
        summary.model_copy(update={"run_id": "run-b", "episode_id": "episode-2"})
    )
    store.close()

    reopened = EventStore(database, journal)
    assert reopened.episode_summary("run-a", "episode-1") == summary
    assert reopened.recent_episode_summaries("run-a") == [summary]
    reopened.close()


def test_event_store_pages_events_and_notifies_best_effort_subscribers(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    seen: list[int] = []
    unsubscribe = store.subscribe(lambda event: seen.append(event.event_id))
    store.subscribe(lambda _: (_ for _ in ()).throw(RuntimeError("observer failed")))

    for step_id in range(3):
        store.append_event(
            run_id="run-a",
            episode_id="episode",
            step_id=step_id,
            event_type="observation",
            payload={"step": step_id},
        )
    store.append_event(
        run_id="run-b",
        episode_id="episode",
        step_id=0,
        event_type="observation",
        payload={},
    )

    first_page = store.events_after("run-a", 0, 2)
    second_page = store.events_after("run-a", first_page[-1].event_id, 2)
    assert [event.step_id for event in first_page] == [0, 1]
    assert [event.step_id for event in second_page] == [2]
    assert store.latest_event_id("run-a") == second_page[-1].event_id
    assert len(seen) == 4

    unsubscribe()
    store.append_event(
        run_id="run-a",
        episode_id="episode",
        step_id=4,
        event_type="observation",
        payload={},
    )
    assert len(seen) == 4
    store.close()


def test_event_subscriber_can_cross_a_durability_barrier_without_locking_append(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    completed = threading.Event()

    def subscriber(event: object) -> None:
        del event
        store.flush()
        completed.set()

    store.subscribe(subscriber)
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="semantic",
        payload={"value": 1},
    )

    assert completed.wait(timeout=1)
    metrics = store.performance_snapshot()
    assert metrics.enqueued_events == 1
    assert metrics.written_events == 1
    store.close()


def test_event_writer_queue_is_bounded(tmp_path: Path) -> None:
    store = EventStore(
        tmp_path / "events.sqlite3",
        tmp_path / "events.jsonl",
        writer_queue_size=3,
    )

    snapshot = store.performance_snapshot()

    assert store._write_queue.maxsize == 3
    assert snapshot.queue_capacity == 3
    assert snapshot.current_queue_depth <= snapshot.queue_capacity
    store.close()


def test_event_store_reconciles_malformed_jsonl_tail_from_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    journal = tmp_path / "events.jsonl"
    store = EventStore(database, journal)
    for step_id in range(3):
        store.append_event(
            run_id="run",
            episode_id="episode",
            step_id=step_id,
            event_type="semantic",
            payload={"step": step_id},
        )
    store.close()
    with journal.open("a", encoding="utf-8") as stream:
        stream.write("{malformed-tail")

    reopened = EventStore(database, journal)
    reopened.close()

    assert [event.event_id for event in read_event_log(journal)] == [1, 2, 3]


def test_runtime_snapshot_is_ordered_after_events_and_replaced_monotonically(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    first = store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=1,
        event_type="command_lifecycle",
        payload={"status": "pending"},
    )
    snapshot = store.record_snapshot(
        run_id="run",
        episode_id="episode",
        snapshot_type="runtime-test-v1",
        step_id=1,
        payload={"state": "pending"},
    )
    tail = store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=2,
        event_type="command_lifecycle",
        payload={"status": "dispatched"},
    )
    store.flush()

    recovered = store.latest_snapshot("run", "episode", "runtime-test-v1")
    assert recovered == snapshot
    assert recovered is not None
    assert recovered.through_event_id == first.event_id
    assert [
        event.event_id
        for event in store.events_of_type(
            "run",
            "episode",
            "command_lifecycle",
            after_event_id=recovered.through_event_id,
        )
    ] == [tail.event_id]
    store.close()
