"""SQLite persistence and deterministic retrieval for CortexPlaybook."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from rtscortex.playbook.models import (
    DecisionCase,
    LessonStatus,
    PlaybookHit,
    PlaybookLesson,
    PlaybookQuery,
    PlaybookRuleKind,
    PlaybookSelection,
)


class PlaybookStore:
    """Persist reusable experience outside any individual run directory."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS decision_cases (
                case_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_playbook_cases_episode
                ON decision_cases (run_id, episode_id);
            CREATE TABLE IF NOT EXISTS playbook_lessons (
                lesson_id TEXT PRIMARY KEY,
                signature TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def add_case(self, case: DecisionCase) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO decision_cases (
                    case_id, run_id, episode_id, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (case.case_id, case.run_id, case.episode_id, case.model_dump_json()),
            )
            self._connection.commit()
        return cursor.rowcount == 1

    def cases(self, *, run_id: str | None = None) -> list[DecisionCase]:
        if run_id is None:
            rows = self._connection.execute(
                "SELECT payload_json FROM decision_cases ORDER BY rowid"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT payload_json FROM decision_cases WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [DecisionCase.model_validate_json(str(row["payload_json"])) for row in rows]

    def lesson_by_signature(self, signature: str) -> PlaybookLesson | None:
        row = self._connection.execute(
            "SELECT payload_json FROM playbook_lessons WHERE signature = ?",
            (signature,),
        ).fetchone()
        if row is None:
            return None
        return PlaybookLesson.model_validate_json(str(row["payload_json"]))

    def upsert_lesson(self, lesson: PlaybookLesson) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO playbook_lessons (lesson_id, signature, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    lesson_id = excluded.lesson_id,
                    payload_json = excluded.payload_json
                """,
                (lesson.lesson_id, lesson.signature, lesson.model_dump_json()),
            )
            self._connection.commit()

    def lessons(self) -> list[PlaybookLesson]:
        rows = self._connection.execute(
            "SELECT payload_json FROM playbook_lessons ORDER BY rowid"
        ).fetchall()
        return [PlaybookLesson.model_validate_json(str(row["payload_json"])) for row in rows]

    def retrieve(self, query: PlaybookQuery) -> PlaybookSelection:
        hits: list[PlaybookHit] = []
        for lesson in self.lessons():
            if lesson.confidence < query.min_confidence:
                continue
            if lesson.status is not LessonStatus.PROMOTED and not query.include_candidates:
                continue
            score, reasons = _match_score(query, lesson)
            required_reasons = (
                {"race"}
                if lesson.rule_kind is PlaybookRuleKind.EXECUTION_GUARD
                else {"race", "opponent"}
            )
            if not required_reasons.issubset(reasons):
                continue
            hits.append(PlaybookHit(lesson=lesson, score=score, match_reasons=tuple(reasons)))
        hits.sort(key=lambda hit: (-hit.score, hit.lesson.lesson_id))
        return PlaybookSelection(query=query, hits=tuple(hits[: query.top_k]))

    def close(self) -> None:
        self._connection.close()


def _match_score(query: PlaybookQuery, lesson: PlaybookLesson) -> tuple[float, list[str]]:
    expected = query.context
    actual = lesson.context
    score = lesson.confidence + min(lesson.support_count, 5) * 0.1
    reasons: list[str] = []
    if expected.agent_race == actual.agent_race:
        score += 4.0
        reasons.append("race")
    if expected.opponent_race == actual.opponent_race:
        score += 3.0
        reasons.append("opponent")
    if expected.phase is actual.phase:
        score += 2.0
        reasons.append("phase")
    if expected.map_name == actual.map_name:
        score += 1.0
        reasons.append("map")
    overlap = set(expected.tags).intersection(actual.tags)
    if overlap:
        score += min(len(overlap), 3) * 0.25
        reasons.append("tags")
    return score, reasons
