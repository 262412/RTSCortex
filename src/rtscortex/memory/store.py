"""SQLite query store with an append-only JSONL event journal."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rtscortex.contracts import EpisodeResult


@dataclass(frozen=True)
class StoredEvent:
    event_id: int
    run_id: str
    episode_id: str
    step_id: int
    event_type: str
    created_at: str
    payload: dict[str, Any]


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
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

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
        return record

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
        rows = self._connection.execute(
            """
            SELECT content FROM lessons
            WHERE run_id = ? AND episode_id = ?
            ORDER BY lesson_id DESC LIMIT ?
            """,
            (run_id, episode_id, limit),
        ).fetchall()
        return [str(row["content"]) for row in reversed(rows)]

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

    def close(self) -> None:
        self._connection.close()


def read_event_log(path: Path) -> Iterable[StoredEvent]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            yield StoredEvent(**payload)
