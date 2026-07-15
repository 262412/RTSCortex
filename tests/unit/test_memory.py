from __future__ import annotations

import asyncio
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
