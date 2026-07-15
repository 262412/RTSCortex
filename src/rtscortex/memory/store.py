"""SQLite query store with an append-only JSONL event journal."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rtscortex.contracts import EpisodeResult, EpisodeSummary


@dataclass(frozen=True)
class StoredEvent:
    event_id: int
    run_id: str
    episode_id: str
    step_id: int
    event_type: str
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class StoredLesson:
    lesson_id: int
    run_id: str
    episode_id: str
    source_step_id: int
    content: str
    created_at: str


def _json_payload(payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


class EventStore:
    """Store runtime events in SQLite and mirror each event to JSONL."""

    def __init__(self, database_path: Path, journal_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.journal_path = journal_path
        self._lock = threading.Lock()
        self._reader_lock = threading.Lock()
        self._subscriber_lock = threading.Lock()
        self._subscribers: dict[int, Callable[[StoredEvent], None]] = {}
        self._next_subscriber_id = 0
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()
        self._reader_connection = sqlite3.connect(database_path, check_same_thread=False)
        self._reader_connection.row_factory = sqlite3.Row

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                step_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_episode
                ON events (run_id, episode_id, event_id);
            CREATE TABLE IF NOT EXISTS lessons (
                lesson_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                source_step_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episode_results (
                run_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, episode_id)
            );
            CREATE TABLE IF NOT EXISTS episode_summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE (run_id, episode_id)
            );
            """
        )
        self._connection.commit()

    def append_event(
        self,
        *,
        run_id: str,
        episode_id: str,
        step_id: int,
        event_type: str,
        payload: BaseModel | dict[str, Any],
    ) -> StoredEvent:
        created_at = datetime.now(UTC).isoformat()
        normalized = _json_payload(payload)
        encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO events (
                    run_id, episode_id, step_id, event_type, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, episode_id, step_id, event_type, created_at, encoded),
            )
            self._connection.commit()
            assert cursor.lastrowid is not None
            event_id = cursor.lastrowid
            record = StoredEvent(
                event_id=event_id,
                run_id=run_id,
                episode_id=episode_id,
                step_id=step_id,
                event_type=event_type,
                created_at=created_at,
                payload=normalized,
            )
            with self.journal_path.open("a", encoding="utf-8") as journal:
                journal.write(
                    json.dumps(record.__dict__, ensure_ascii=False, sort_keys=True) + "\n"
                )
            # Sinks enqueue only; publishing under the write lock preserves event-id order.
            self._publish(record)
        return record

    def subscribe(self, subscriber: Callable[[StoredEvent], None]) -> Callable[[], None]:
        """Subscribe a non-blocking event sink and return its unsubscribe function.

        Subscribers run after the event is durable. Their failures never affect the
        runtime write path. Subscribers should only enqueue work and return immediately.
        """

        with self._subscriber_lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[subscriber_id] = subscriber

        def unsubscribe() -> None:
            with self._subscriber_lock:
                self._subscribers.pop(subscriber_id, None)

        return unsubscribe

    def _publish(self, event: StoredEvent) -> None:
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers.values())
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:
                # Console and other observers are best-effort and must never stop a run.
                continue

    def events_after(
        self,
        run_id: str,
        after_event_id: int,
        limit: int,
        *,
        episode_id: str | None = None,
    ) -> list[StoredEvent]:
        """Return persisted events after an event id in ascending order."""

        if after_event_id < 0:
            raise ValueError("after_event_id must be non-negative")
        if limit < 1:
            raise ValueError("limit must be positive")
        if episode_id is None:
            query = """
                SELECT * FROM events
                WHERE run_id = ? AND event_id > ?
                ORDER BY event_id LIMIT ?
            """
            parameters: tuple[object, ...] = (run_id, after_event_id, limit)
        else:
            query = """
                SELECT * FROM events
                WHERE run_id = ? AND episode_id = ? AND event_id > ?
                ORDER BY event_id LIMIT ?
            """
            parameters = (run_id, episode_id, after_event_id, limit)
        with self._reader_lock:
            rows = self._reader_connection.execute(query, parameters).fetchall()
        return [self._row_to_event(row) for row in rows]

    def latest_event_id(self, run_id: str, *, episode_id: str | None = None) -> int:
        if episode_id is None:
            query = "SELECT MAX(event_id) AS event_id FROM events WHERE run_id = ?"
            parameters: tuple[object, ...] = (run_id,)
        else:
            query = """
                SELECT MAX(event_id) AS event_id FROM events
                WHERE run_id = ? AND episode_id = ?
            """
            parameters = (run_id, episode_id)
        with self._reader_lock:
            row = self._reader_connection.execute(query, parameters).fetchone()
        return 0 if row is None or row["event_id"] is None else int(row["event_id"])

    def recent_events(self, run_id: str, episode_id: str, limit: int) -> list[StoredEvent]:
        rows = self._connection.execute(
            """
            SELECT * FROM events
            WHERE run_id = ? AND episode_id = ?
            ORDER BY event_id DESC LIMIT ?
            """,
            (run_id, episode_id, limit),
        ).fetchall()
        return [self._row_to_event(row) for row in reversed(rows)]

    def last_event(self, run_id: str, episode_id: str, event_type: str) -> StoredEvent | None:
        row = self._connection.execute(
            """
            SELECT * FROM events
            WHERE run_id = ? AND episode_id = ? AND event_type = ?
            ORDER BY event_id DESC LIMIT 1
            """,
            (run_id, episode_id, event_type),
        ).fetchone()
        return None if row is None else self._row_to_event(row)

    def events_of_type(
        self,
        run_id: str,
        episode_id: str,
        event_type: str,
    ) -> list[StoredEvent]:
        rows = self._connection.execute(
            """
            SELECT * FROM events
            WHERE run_id = ? AND episode_id = ? AND event_type = ?
            ORDER BY event_id
            """,
            (run_id, episode_id, event_type),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            event_id=int(row["event_id"]),
            run_id=str(row["run_id"]),
            episode_id=str(row["episode_id"]),
            step_id=int(row["step_id"]),
            event_type=str(row["event_type"]),
            created_at=str(row["created_at"]),
            payload=json.loads(str(row["payload_json"])),
        )

    def add_lesson(
        self,
        *,
        run_id: str,
        episode_id: str,
        source_step_id: int,
        content: str,
    ) -> None:
        if not content.strip():
            return
        self._connection.execute(
            """
            INSERT INTO lessons (
                run_id, episode_id, source_step_id, content, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, episode_id, source_step_id, content, datetime.now(UTC).isoformat()),
        )
        self._connection.commit()

    def lessons(self, run_id: str, episode_id: str, limit: int = 10) -> list[str]:
        return [lesson.content for lesson in self.lesson_records(run_id, episode_id, limit)]

    def lesson_records(
        self,
        run_id: str,
        episode_id: str,
        limit: int = 10,
    ) -> list[StoredLesson]:
        rows = self._connection.execute(
            """
            SELECT * FROM lessons
            WHERE run_id = ? AND episode_id = ?
            ORDER BY lesson_id DESC LIMIT ?
            """,
            (run_id, episode_id, limit),
        ).fetchall()
        return [
            StoredLesson(
                lesson_id=int(row["lesson_id"]),
                run_id=str(row["run_id"]),
                episode_id=str(row["episode_id"]),
                source_step_id=int(row["source_step_id"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
            )
            for row in reversed(rows)
        ]

    def record_episode(self, result: EpisodeResult) -> None:
        encoded = result.model_dump_json()
        self._connection.execute(
            """
            INSERT INTO episode_results (run_id, episode_id, payload_json)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id, episode_id) DO UPDATE SET payload_json = excluded.payload_json
            """,
            (result.run_id, result.episode_id, encoded),
        )
        self._connection.commit()
        self.append_event(
            run_id=result.run_id,
            episode_id=result.episode_id,
            step_id=result.steps,
            event_type="episode_result",
            payload=result,
        )

    def record_episode_summary(self, summary: EpisodeSummary) -> None:
        self._connection.execute(
            """
            INSERT INTO episode_summaries (
                run_id, episode_id, created_at, payload_json
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, episode_id) DO UPDATE SET
                created_at = excluded.created_at,
                payload_json = excluded.payload_json
            """,
            (
                summary.run_id,
                summary.episode_id,
                summary.created_at.isoformat(),
                summary.model_dump_json(),
            ),
        )
        self._connection.commit()
        self.append_event(
            run_id=summary.run_id,
            episode_id=summary.episode_id,
            step_id=summary.source_step_id,
            event_type="episode_summary",
            payload=summary,
        )

    def episode_summary(self, run_id: str, episode_id: str) -> EpisodeSummary | None:
        row = self._connection.execute(
            """
            SELECT payload_json FROM episode_summaries
            WHERE run_id = ? AND episode_id = ?
            """,
            (run_id, episode_id),
        ).fetchone()
        if row is None:
            return None
        return EpisodeSummary.model_validate_json(str(row["payload_json"]))

    def recent_episode_summaries(self, run_id: str, limit: int = 5) -> list[EpisodeSummary]:
        rows = self._connection.execute(
            """
            SELECT payload_json FROM episode_summaries
            WHERE run_id = ?
            ORDER BY summary_id DESC LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [
            EpisodeSummary.model_validate_json(str(row["payload_json"])) for row in reversed(rows)
        ]

    def close(self) -> None:
        with self._subscriber_lock:
            self._subscribers.clear()
        self._reader_connection.close()
        self._connection.close()


def read_event_log(path: Path) -> Iterable[StoredEvent]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            yield StoredEvent(**payload)
