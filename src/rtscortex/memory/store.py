"""SQLite query store with an append-only JSONL event journal."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from collections import deque
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


@dataclass(frozen=True)
class StoredSnapshot:
    run_id: str
    episode_id: str
    snapshot_type: str
    through_event_id: int
    step_id: int
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class EventStorePerformance:
    enqueued_events: int
    written_events: int
    max_queue_depth: int
    current_queue_depth: int
    queue_capacity: int
    append_latency_ms_mean: float
    writer_lag_ms_p95: float
    writer_lag_ms_max: float
    blocked_append_count: int
    sampled_drop_supported: bool
    subscriber_dropped_events: int
    journal_bytes: int


@dataclass(frozen=True)
class _FlushBarrier:
    completed: threading.Event


@dataclass(frozen=True)
class _StopWriter:
    pass


@dataclass(frozen=True)
class _SnapshotRecord:
    snapshot: StoredSnapshot


@dataclass
class _SubscriberWorker:
    callback: Callable[[StoredEvent], None]
    queue: queue.Queue[StoredEvent | None]
    thread: threading.Thread
    dropped_events: int = 0


def _json_payload(payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


class EventStore:
    """Store runtime events in SQLite and mirror each event to JSONL."""

    def __init__(
        self,
        database_path: Path,
        journal_path: Path,
        *,
        flush_event_limit: int = 64,
        flush_interval_seconds: float = 0.25,
        writer_queue_size: int = 8192,
        subscriber_queue_size: int = 256,
    ) -> None:
        if flush_event_limit < 1:
            raise ValueError("flush_event_limit must be positive")
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        if writer_queue_size < 1:
            raise ValueError("writer_queue_size must be positive")
        if subscriber_queue_size < 1:
            raise ValueError("subscriber_queue_size must be positive")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.journal_path = journal_path
        self._id_lock = threading.Lock()
        self._metadata_lock = threading.Lock()
        self._reader_lock = threading.Lock()
        self._subscriber_lock = threading.Lock()
        self._subscribers: dict[int, _SubscriberWorker] = {}
        self._next_subscriber_id = 0
        self._flush_event_limit = flush_event_limit
        self._flush_interval_seconds = flush_interval_seconds
        self._writer_queue_size = writer_queue_size
        self._subscriber_queue_size = subscriber_queue_size
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()
        self._reader_connection = sqlite3.connect(database_path, check_same_thread=False)
        self._reader_connection.row_factory = sqlite3.Row
        self._reader_connection.execute("PRAGMA busy_timeout=5000")
        row = self._connection.execute("SELECT MAX(event_id) AS event_id FROM events").fetchone()
        self._next_event_id = (
            1 if row is None or row["event_id"] is None else int(row["event_id"]) + 1
        )
        self._reconcile_journal()
        self._write_queue: queue.Queue[
            StoredEvent | _SnapshotRecord | _FlushBarrier | _StopWriter
        ] = queue.Queue(maxsize=writer_queue_size)
        self._writer_error: BaseException | None = None
        self._closed = False
        self._enqueued_events = 0
        self._written_events = 0
        self._max_queue_depth = 0
        self._append_latency_ns = 0
        self._event_enqueued_ns: dict[int, int] = {}
        self._writer_lag_ns_max = 0
        self._writer_lag_samples_ns: deque[int] = deque(maxlen=8192)
        self._blocked_append_count = 0
        self._writer = threading.Thread(
            target=self._writer_main,
            name=f"rtscortex-event-writer-{id(self):x}",
            daemon=True,
        )
        self._writer.start()

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
            CREATE INDEX IF NOT EXISTS idx_events_episode_type
                ON events (run_id, episode_id, event_type, event_id);
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
            CREATE TABLE IF NOT EXISTS runtime_snapshots (
                run_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                snapshot_type TEXT NOT NULL,
                through_event_id INTEGER NOT NULL,
                step_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, episode_id, snapshot_type)
            );
            """
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
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
        append_started = time.perf_counter_ns()
        created_at = datetime.now(UTC).isoformat()
        normalized = _json_payload(payload)
        self._raise_writer_error()
        if self._closed:
            raise RuntimeError("event store is closed")
        with self._id_lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            record = StoredEvent(
                event_id=event_id,
                run_id=run_id,
                episode_id=episode_id,
                step_id=step_id,
                event_type=event_type,
                created_at=created_at,
                payload=normalized,
            )
            self._event_enqueued_ns[event_id] = time.perf_counter_ns()
            self._enqueue(record)
        self._enqueued_events += 1
        self._max_queue_depth = max(self._max_queue_depth, self._write_queue.qsize())
        # Console and other best-effort observers must never share the durable
        # writer lock or delay the SC2 tick path.
        self._publish(record)
        self._append_latency_ns += time.perf_counter_ns() - append_started
        return record

    def performance_snapshot(self) -> EventStorePerformance:
        mean_ms = (
            0.0
            if self._enqueued_events == 0
            else self._append_latency_ns / self._enqueued_events / 1_000_000
        )
        lag_samples = sorted(self._writer_lag_samples_ns)
        lag_p95_ns = (
            0
            if not lag_samples
            else lag_samples[
                max(0, min(len(lag_samples) - 1, (len(lag_samples) * 95 + 99) // 100 - 1))
            ]
        )
        return EventStorePerformance(
            enqueued_events=self._enqueued_events,
            written_events=self._written_events,
            max_queue_depth=self._max_queue_depth,
            current_queue_depth=self._write_queue.qsize(),
            queue_capacity=self._writer_queue_size,
            append_latency_ms_mean=mean_ms,
            writer_lag_ms_p95=lag_p95_ns / 1_000_000,
            writer_lag_ms_max=self._writer_lag_ns_max / 1_000_000,
            blocked_append_count=self._blocked_append_count,
            sampled_drop_supported=False,
            subscriber_dropped_events=sum(
                subscriber.dropped_events for subscriber in self._subscribers.values()
            ),
            journal_bytes=(self.journal_path.stat().st_size if self.journal_path.exists() else 0),
        )

    def record_snapshot(
        self,
        *,
        run_id: str,
        episode_id: str,
        snapshot_type: str,
        step_id: int,
        payload: BaseModel | dict[str, Any],
    ) -> StoredSnapshot:
        """Queue a compact recovery snapshot after all currently assigned events.

        The writer processes snapshots in queue order, so ``through_event_id`` is
        a durable replay boundary: recovery only needs the snapshot plus events
        with a greater id.
        """

        if not snapshot_type:
            raise ValueError("snapshot_type must not be empty")
        self._raise_writer_error()
        if self._closed:
            raise RuntimeError("event store is closed")
        with self._id_lock:
            through_event_id = self._next_event_id - 1
            snapshot = StoredSnapshot(
                run_id=run_id,
                episode_id=episode_id,
                snapshot_type=snapshot_type,
                through_event_id=through_event_id,
                step_id=step_id,
                created_at=datetime.now(UTC).isoformat(),
                payload=_json_payload(payload),
            )
            self._enqueue(_SnapshotRecord(snapshot))
        self._max_queue_depth = max(self._max_queue_depth, self._write_queue.qsize())
        return snapshot

    def flush(self) -> None:
        """Wait until every event queued before this call is durable."""

        self._raise_writer_error()
        if self._closed:
            return
        barrier = _FlushBarrier(threading.Event())
        self._enqueue(barrier)
        if not barrier.completed.wait(timeout=30):
            raise TimeoutError("event writer did not acknowledge the durability barrier")
        self._raise_writer_error()

    def _enqueue(
        self,
        item: StoredEvent | _SnapshotRecord | _FlushBarrier | _StopWriter,
    ) -> None:
        """Bound memory while applying explicit backpressure to durable events."""

        try:
            self._write_queue.put_nowait(item)
            return
        except queue.Full:
            self._blocked_append_count += 1
        try:
            self._write_queue.put(item, timeout=30)
        except queue.Full as error:
            raise TimeoutError("event writer queue remained saturated for 30 seconds") from error

    def _writer_main(self) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        journal = self.journal_path.open("a", encoding="utf-8", buffering=1024 * 1024)
        batch: list[StoredEvent] = []
        try:
            while True:
                try:
                    item = self._write_queue.get(
                        timeout=self._flush_interval_seconds if batch else None
                    )
                except queue.Empty:
                    self._write_batch(connection, journal, batch)
                    batch.clear()
                    continue
                if isinstance(item, StoredEvent):
                    batch.append(item)
                    if len(batch) >= self._flush_event_limit:
                        self._write_batch(connection, journal, batch)
                        batch.clear()
                    continue
                self._write_batch(connection, journal, batch)
                batch.clear()
                if isinstance(item, _SnapshotRecord):
                    self._write_snapshot(connection, item.snapshot)
                    continue
                if isinstance(item, _FlushBarrier):
                    item.completed.set()
                    continue
                if isinstance(item, _StopWriter):
                    return
        except BaseException as error:
            self._writer_error = error
            while True:
                try:
                    pending = self._write_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending, _FlushBarrier):
                    pending.completed.set()
        finally:
            journal.close()
            connection.close()

    def _write_batch(
        self,
        connection: sqlite3.Connection,
        journal: Any,
        records: list[StoredEvent],
    ) -> None:
        if not records:
            return
        encoded = [
            (
                record.event_id,
                record.run_id,
                record.episode_id,
                record.step_id,
                record.event_type,
                record.created_at,
                json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
            )
            for record in records
        ]
        connection.executemany(
            """
            INSERT INTO events (
                event_id, run_id, episode_id, step_id, event_type, created_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            encoded,
        )
        # SQLite is the canonical durable authority. The JSONL stream is its
        # ordered compatibility mirror and may only lag, never lead, a commit.
        connection.commit()
        now = time.perf_counter_ns()
        for record in records:
            enqueued = self._event_enqueued_ns.pop(record.event_id, now)
            lag = now - enqueued
            self._writer_lag_ns_max = max(self._writer_lag_ns_max, lag)
            self._writer_lag_samples_ns.append(lag)
        for record in records:
            journal.write(json.dumps(record.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
        journal.flush()
        self._written_events += len(records)

    @staticmethod
    def _write_snapshot(
        connection: sqlite3.Connection,
        snapshot: StoredSnapshot,
    ) -> None:
        connection.execute(
            """
            INSERT INTO runtime_snapshots (
                run_id, episode_id, snapshot_type, through_event_id,
                step_id, created_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, episode_id, snapshot_type) DO UPDATE SET
                through_event_id = excluded.through_event_id,
                step_id = excluded.step_id,
                created_at = excluded.created_at,
                payload_json = excluded.payload_json
            WHERE excluded.through_event_id >= runtime_snapshots.through_event_id
            """,
            (
                snapshot.run_id,
                snapshot.episode_id,
                snapshot.snapshot_type,
                snapshot.through_event_id,
                snapshot.step_id,
                snapshot.created_at,
                json.dumps(snapshot.payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        connection.commit()

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("event writer failed") from self._writer_error

    def subscribe(self, subscriber: Callable[[StoredEvent], None]) -> Callable[[], None]:
        """Subscribe a non-blocking event sink and return its unsubscribe function.

        Subscribers receive events in event-id order before the next bounded durable
        flush. Their failures never affect the runtime write path. Subscribers should
        only enqueue work and return immediately.
        """

        with self._subscriber_lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            subscriber_queue: queue.Queue[StoredEvent | None] = queue.Queue(
                maxsize=self._subscriber_queue_size
            )
            worker = _SubscriberWorker(
                callback=subscriber,
                queue=subscriber_queue,
                thread=threading.Thread(),
            )
            worker.thread = threading.Thread(
                target=self._subscriber_main,
                args=(worker,),
                name=f"rtscortex-event-subscriber-{subscriber_id}",
                daemon=True,
            )
            self._subscribers[subscriber_id] = worker
            worker.thread.start()

        def unsubscribe() -> None:
            with self._subscriber_lock:
                worker = self._subscribers.pop(subscriber_id, None)
            if worker is not None:
                self._stop_subscriber(worker)

        return unsubscribe

    def _publish(self, event: StoredEvent) -> None:
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers.values())
        for subscriber in subscribers:
            try:
                subscriber.queue.put_nowait(event)
            except queue.Full:
                subscriber.dropped_events += 1

    @staticmethod
    def _subscriber_main(worker: _SubscriberWorker) -> None:
        while True:
            event = worker.queue.get()
            if event is None:
                return
            try:
                worker.callback(event)
            except Exception:
                continue

    @staticmethod
    def _stop_subscriber(worker: _SubscriberWorker) -> None:
        try:
            worker.queue.put_nowait(None)
        except queue.Full:
            with worker.queue.mutex:
                worker.queue.queue.clear()
            worker.queue.put_nowait(None)
        worker.thread.join(timeout=5)

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
        self.flush()
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
        self.flush()
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
        self.flush()
        with self._reader_lock:
            rows = self._reader_connection.execute(
                """
                SELECT * FROM events
                WHERE run_id = ? AND episode_id = ?
                ORDER BY event_id DESC LIMIT ?
                """,
                (run_id, episode_id, limit),
            ).fetchall()
        return [self._row_to_event(row) for row in reversed(rows)]

    def last_event(
        self,
        run_id: str,
        episode_id: str,
        event_type: str,
        *,
        after_event_id: int = 0,
    ) -> StoredEvent | None:
        if after_event_id < 0:
            raise ValueError("after_event_id must be non-negative")
        self.flush()
        with self._reader_lock:
            row = self._reader_connection.execute(
                """
                SELECT * FROM events
                WHERE run_id = ? AND episode_id = ? AND event_type = ?
                    AND event_id > ?
                ORDER BY event_id DESC LIMIT 1
                """,
                (run_id, episode_id, event_type, after_event_id),
            ).fetchone()
        return None if row is None else self._row_to_event(row)

    def events_of_type(
        self,
        run_id: str,
        episode_id: str,
        event_type: str,
        *,
        after_event_id: int = 0,
    ) -> list[StoredEvent]:
        if after_event_id < 0:
            raise ValueError("after_event_id must be non-negative")
        self.flush()
        with self._reader_lock:
            rows = self._reader_connection.execute(
                """
                SELECT * FROM events
                WHERE run_id = ? AND episode_id = ? AND event_type = ?
                    AND event_id > ?
                ORDER BY event_id
                """,
                (run_id, episode_id, event_type, after_event_id),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def latest_snapshot(
        self,
        run_id: str,
        episode_id: str,
        snapshot_type: str,
    ) -> StoredSnapshot | None:
        self.flush()
        with self._reader_lock:
            row = self._reader_connection.execute(
                """
                SELECT * FROM runtime_snapshots
                WHERE run_id = ? AND episode_id = ? AND snapshot_type = ?
                """,
                (run_id, episode_id, snapshot_type),
            ).fetchone()
        if row is None:
            return None
        return StoredSnapshot(
            run_id=str(row["run_id"]),
            episode_id=str(row["episode_id"]),
            snapshot_type=str(row["snapshot_type"]),
            through_event_id=int(row["through_event_id"]),
            step_id=int(row["step_id"]),
            created_at=str(row["created_at"]),
            payload=json.loads(str(row["payload_json"])),
        )

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
        with self._metadata_lock:
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
        with self._metadata_lock:
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
        with self._metadata_lock:
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
        with self._metadata_lock:
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
        with self._metadata_lock:
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
        with self._metadata_lock:
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
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._enqueue(_StopWriter())
        self._writer.join(timeout=30)
        if self._writer.is_alive():
            raise TimeoutError("event writer did not stop")
        self._raise_writer_error()
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers.values())
            self._subscribers.clear()
        for subscriber in subscribers:
            self._stop_subscriber(subscriber)
        self._reader_connection.close()
        self._connection.close()

    def _reconcile_journal(self) -> None:
        """Repair the compatibility JSONL mirror from canonical SQLite rows."""

        last_event_id = 0
        valid_prefix = True
        if self.journal_path.exists():
            try:
                with self.journal_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        event_id = int(payload["event_id"])
                        if event_id != last_event_id + 1:
                            valid_prefix = False
                            break
                        last_event_id = event_id
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                valid_prefix = False
        database_max = self._next_event_id - 1
        if not valid_prefix or last_event_id > database_max:
            last_event_id = 0
            self.journal_path.write_text("", encoding="utf-8")
        rows = self._connection.execute(
            "SELECT * FROM events WHERE event_id > ? ORDER BY event_id",
            (last_event_id,),
        )
        mode = "a" if last_event_id else "w"
        with self.journal_path.open(mode, encoding="utf-8") as journal:
            for row in rows:
                event = self._row_to_event(row)
                journal.write(json.dumps(event.__dict__, ensure_ascii=False, sort_keys=True) + "\n")


def read_event_log(path: Path) -> Iterable[StoredEvent]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            yield StoredEvent(**payload)
