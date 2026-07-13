from __future__ import annotations

import asyncio
from pathlib import Path

from rtscortex.contracts import EpisodeOutcome, EpisodeResult
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
    assert reopened.last_event("run", "episode", "episode_result") is not None
    reopened.close()
    assert [event.event_type for event in read_event_log(journal)] == [
        "observation",
        "episode_result",
    ]


def test_semantic_memory_is_explicitly_disabled() -> None:
    hits = asyncio.run(DisabledMemoryRetriever().search("enemy strategy"))
    assert hits == []
