"""SQLite persistence and deterministic retrieval for CortexPlaybook."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from rtscortex.playbook.models import (
    DecisionCase,
    LessonStatus,
    PlaybookCondition,
    PlaybookConditionOperator,
    PlaybookContext,
    PlaybookHit,
    PlaybookLesson,
    PlaybookQuery,
    PlaybookRule,
    PlaybookRuleApplication,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleKind,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    PlaybookSelection,
    StrategicConsequenceType,
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
            CREATE TABLE IF NOT EXISTS playbook_rules_v2 (
                rule_id TEXT PRIMARY KEY,
                canonical_key TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS playbook_rule_applications_v2 (
                application_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_playbook_rule_applications_episode
                ON playbook_rule_applications_v2 (run_id, episode_id);
            CREATE TABLE IF NOT EXISTS playbook_migrations (
                migration_id TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._connection.commit()
        self.migrate_legacy_lessons()

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

    def upsert_rule(self, rule: PlaybookRule) -> PlaybookRule:
        with self._lock:
            existing_row = self._connection.execute(
                "SELECT payload_json FROM playbook_rules_v2 WHERE canonical_key = ?",
                (rule.canonical_key,),
            ).fetchone()
            if existing_row is not None:
                existing = PlaybookRule.model_validate_json(str(existing_row["payload_json"]))
                rule = _merge_rule_evidence(existing, rule)
            self._connection.execute(
                """
                INSERT INTO playbook_rules_v2 (rule_id, canonical_key, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(canonical_key) DO UPDATE SET
                    rule_id = excluded.rule_id,
                    payload_json = excluded.payload_json
                """,
                (rule.rule_id, rule.canonical_key, rule.model_dump_json()),
            )
            self._connection.commit()
        return rule

    def upsert_legacy_lesson_rule(self, lesson: PlaybookLesson) -> PlaybookRule:
        rule = _legacy_rule(lesson)
        return self.upsert_rule(rule)

    def upsert_lesson_rule_candidate(
        self,
        lesson: PlaybookLesson,
        source_case: DecisionCase,
    ) -> PlaybookRule:
        return self.upsert_rule(_candidate_rule(lesson, source_case))

    def rules(self) -> list[PlaybookRule]:
        rows = self._connection.execute(
            "SELECT payload_json FROM playbook_rules_v2 ORDER BY rowid"
        ).fetchall()
        return [PlaybookRule.model_validate_json(str(row["payload_json"])) for row in rows]

    def rules_for_guard(
        self,
        *,
        context: PlaybookContext | None = None,
        max_hard: int = 8,
        max_soft: int = 8,
    ) -> tuple[PlaybookRule, ...]:
        rules = self.rules()
        if context is not None:
            rules = [rule for rule in rules if _rule_matches_context(rule, context)]
        active = [rule for rule in rules if rule.status is PlaybookRuleStatus.ACTIVE]
        now = datetime.now(UTC)
        active = [rule for rule in active if _rule_is_unexpired(rule, now)]
        hard = sorted(
            (rule for rule in active if rule.strength is PlaybookRuleStrength.HARD),
            key=lambda rule: (-rule.confidence, rule.rule_id),
        )[:max_hard]
        soft = sorted(
            (rule for rule in active if rule.strength is PlaybookRuleStrength.SOFT),
            key=lambda rule: (-rule.confidence, rule.rule_id),
        )[:max_soft]
        advisory = sorted(
            (
                rule
                for rule in rules
                if rule.status is PlaybookRuleStatus.LEGACY
                and rule.strength is PlaybookRuleStrength.ADVISORY
            ),
            key=lambda rule: (-rule.confidence, rule.rule_id),
        )[:8]
        candidates = sorted(
            (
                rule
                for rule in rules
                if rule.status is PlaybookRuleStatus.CANDIDATE
                and bool(rule.action_names or rule.role_ids)
            ),
            key=lambda rule: (-rule.confidence, rule.rule_id),
        )[:16]
        return tuple([*hard, *soft, *candidates, *advisory])

    def record_rule_application(self, application: PlaybookRuleApplication) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO playbook_rule_applications_v2 (
                    application_id, rule_id, run_id, episode_id, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    application.application_id,
                    application.rule_id,
                    application.run_id,
                    application.episode_id,
                    application.model_dump_json(),
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted and application.matched:
                row = self._connection.execute(
                    "SELECT payload_json FROM playbook_rules_v2 WHERE rule_id = ?",
                    (application.rule_id,),
                ).fetchone()
                if row is not None:
                    rule = PlaybookRule.model_validate_json(str(row["payload_json"]))
                    if rule.status is PlaybookRuleStatus.CANDIDATE:
                        updated = rule.model_copy(
                            update={"shadow_state_count": rule.shadow_state_count + 1}
                        )
                        self._connection.execute(
                            "UPDATE playbook_rules_v2 SET payload_json = ? WHERE rule_id = ?",
                            (updated.model_dump_json(), updated.rule_id),
                        )
            self._connection.commit()
        return inserted

    def rule_applications(
        self,
        *,
        run_id: str | None = None,
    ) -> list[PlaybookRuleApplication]:
        if run_id is None:
            rows = self._connection.execute(
                "SELECT payload_json FROM playbook_rule_applications_v2 ORDER BY rowid"
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT payload_json FROM playbook_rule_applications_v2
                WHERE run_id = ? ORDER BY rowid
                """,
                (run_id,),
            ).fetchall()
        return [
            PlaybookRuleApplication.model_validate_json(str(row["payload_json"])) for row in rows
        ]

    def migrate_legacy_lessons(self) -> int:
        migration_id = "legacy-lessons-to-v2-advisory"
        applied = self._connection.execute(
            "SELECT 1 FROM playbook_migrations WHERE migration_id = ?",
            (migration_id,),
        ).fetchone()
        if applied is not None:
            return 0
        lessons = self.lessons()
        if lessons:
            backup_path = self.database_path.with_name(
                f"{self.database_path.stem}.pre-v2{self.database_path.suffix}"
            )
            if not backup_path.exists():
                backup = sqlite3.connect(backup_path)
                try:
                    self._connection.backup(backup)
                finally:
                    backup.close()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for lesson in lessons:
                    rule = _legacy_rule(lesson)
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO playbook_rules_v2 (
                            rule_id, canonical_key, payload_json
                        ) VALUES (?, ?, ?)
                        """,
                        (rule.rule_id, rule.canonical_key, rule.model_dump_json()),
                    )
                self._connection.execute(
                    "INSERT INTO playbook_migrations (migration_id) VALUES (?)",
                    (migration_id,),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        return len(lessons)

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


def _rule_matches_context(rule: PlaybookRule, context: PlaybookContext) -> bool:
    values: dict[str, object] = {
        "agent_race": context.agent_race,
        "opponent_race": context.opponent_race,
        "phase": context.phase.value,
        "map_name": context.map_name,
        "alert": context.tags,
    }
    for condition in rule.conditions:
        if condition.field not in values:
            continue
        actual = values[condition.field]
        expected = condition.value
        if condition.operator is PlaybookConditionOperator.EQ and actual != expected:
            return False
        if condition.operator is PlaybookConditionOperator.IN:
            options = expected if isinstance(expected, tuple) else (expected,)
            if actual not in options:
                return False
        if condition.operator is PlaybookConditionOperator.CONTAINS:
            if not isinstance(actual, tuple) or expected not in actual:
                return False
    return True


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


def _legacy_rule(lesson: PlaybookLesson) -> PlaybookRule:
    conditions = (
        PlaybookCondition(field="agent_race", value=lesson.context.agent_race),
        PlaybookCondition(field="opponent_race", value=lesson.context.opponent_race),
        PlaybookCondition(field="phase", value=lesson.context.phase.value),
        PlaybookCondition(field="map_name", value=lesson.context.map_name),
    )
    effect = (
        PlaybookRuleEffect.PREFER
        if lesson.recommended_action is not None or lesson.recommended_role is not None
        else PlaybookRuleEffect.AVOID
    )
    actions = tuple(
        action for action in (lesson.recommended_action, lesson.avoid_action) if action is not None
    )
    roles = tuple(
        role for role in (lesson.recommended_role, lesson.avoid_role) if role is not None
    )
    canonical = hashlib.sha256(
        f"legacy|{lesson.signature}|{effect.value}|{'|'.join(actions)}".encode()
    ).hexdigest()
    return PlaybookRule(
        rule_id=f"playbook-rule:{canonical}",
        canonical_key=canonical,
        category=(
            PlaybookRuleCategory.EXECUTION_GUARD
            if lesson.rule_kind is PlaybookRuleKind.EXECUTION_GUARD
            else PlaybookRuleCategory.MATCHUP_STRATEGY
        ),
        conditions=conditions,
        effect=effect,
        strength=PlaybookRuleStrength.ADVISORY,
        status=PlaybookRuleStatus.LEGACY,
        action_names=actions,
        role_ids=roles,
        confidence=lesson.confidence,
        support_count=lesson.support_count,
        contradiction_count=lesson.contradiction_count,
        source_case_ids=lesson.source_case_ids,
        source_run_ids=tuple(
            dict.fromkeys(source.split("/", 1)[0] for source in lesson.source_episode_ids)
        ),
        evidence={"legacy_lesson_id": lesson.lesson_id, "statement": lesson.statement},
    )


def _candidate_rule(lesson: PlaybookLesson, source_case: DecisionCase) -> PlaybookRule:
    conditions: tuple[PlaybookCondition, ...] = (
        PlaybookCondition(field="agent_race", value=lesson.context.agent_race),
        PlaybookCondition(field="opponent_race", value=lesson.context.opponent_race),
        PlaybookCondition(field="phase", value=lesson.context.phase.value),
        PlaybookCondition(field="map_name", value=lesson.context.map_name),
    )
    condition_values = source_case.evidence.get("condition_values")
    if isinstance(condition_values, dict) and source_case.consequence_type is not None:
        conditions = (
            *conditions,
            *tuple(
                PlaybookCondition(field=field, value=str(condition_values[field]))  # type: ignore[arg-type]
                for field in ("threat_level", "economy_status", "army_readiness")
                if isinstance(condition_values.get(field), str)
            ),
        )
    effect = (
        PlaybookRuleEffect.PREFER
        if lesson.recommended_action is not None or lesson.recommended_role is not None
        else PlaybookRuleEffect.AVOID
    )
    actions = tuple(
        action for action in (lesson.recommended_action, lesson.avoid_action) if action is not None
    )
    roles = tuple(
        role for role in (lesson.recommended_role, lesson.avoid_role) if role is not None
    )
    canonical_payload = "|".join(
        (
            lesson.rule_kind.value,
            *(
                f"{condition.field}:{condition.operator.value}:{condition.value}"
                for condition in conditions
            ),
            effect.value,
            *actions,
            *roles,
        )
    )
    canonical = hashlib.sha256(canonical_payload.encode()).hexdigest()
    seed = source_case.evidence.get("seed")
    return PlaybookRule(
        rule_id=f"playbook-rule:{canonical}",
        canonical_key=canonical,
        category=_rule_category(lesson),
        conditions=conditions,
        effect=effect,
        strength=PlaybookRuleStrength.ADVISORY,
        status=PlaybookRuleStatus.CANDIDATE,
        action_names=actions,
        role_ids=roles,
        confidence=lesson.confidence,
        support_count=1,
        source_case_ids=(source_case.case_id,),
        source_run_ids=(source_case.run_id,),
        source_seeds=((int(seed),) if isinstance(seed, int) else ()),
        censored_source_run_ids=(
            (source_case.run_id,) if source_case.evidence.get("censored") is True else ()
        ),
        censored_source_seeds=(
            (int(seed),)
            if source_case.evidence.get("censored") is True and isinstance(seed, int)
            else ()
        ),
        evidence={
            "lesson_id": lesson.lesson_id,
            "statement": lesson.statement,
            "consequence_type": (
                None if lesson.consequence_type is None else lesson.consequence_type.value
            ),
        },
    )


def _rule_category(lesson: PlaybookLesson) -> PlaybookRuleCategory:
    if lesson.rule_kind is PlaybookRuleKind.EXECUTION_GUARD:
        return PlaybookRuleCategory.EXECUTION_GUARD
    if lesson.consequence_type in {
        StrategicConsequenceType.EXPANSION_DELAYED,
        StrategicConsequenceType.PRODUCTION_IMBALANCE,
    }:
        return PlaybookRuleCategory.RACE_MACRO
    if lesson.consequence_type in {
        StrategicConsequenceType.THREAT_UNANSWERED,
        StrategicConsequenceType.TIMING_ATTACK_FAILED,
        StrategicConsequenceType.UNNECESSARY_RETREAT,
        StrategicConsequenceType.ADVANTAGE_NOT_CONVERTED,
        StrategicConsequenceType.SUCCESSFUL_KEY_DECISION,
    }:
        return PlaybookRuleCategory.TACTICAL_RESPONSE
    return PlaybookRuleCategory.MATCHUP_STRATEGY


def _merge_rule_evidence(existing: PlaybookRule, incoming: PlaybookRule) -> PlaybookRule:
    """Keep one canonical rule while preserving independent evidence lineage."""

    contradiction_seeds = tuple(
        dict.fromkeys((*existing.contradiction_seeds, *incoming.contradiction_seeds))
    )
    source_run_ids = tuple(dict.fromkeys((*existing.source_run_ids, *incoming.source_run_ids)))
    preserve_active = (
        incoming.status is PlaybookRuleStatus.CANDIDATE
        and existing.status is PlaybookRuleStatus.ACTIVE
    )
    return incoming.model_copy(
        update={
            "status": existing.status if preserve_active else incoming.status,
            "strength": existing.strength if preserve_active else incoming.strength,
            "confidence": (
                max(existing.confidence, incoming.confidence)
                if preserve_active
                else incoming.confidence
            ),
            "support_count": max(
                existing.support_count,
                incoming.support_count,
                len(source_run_ids),
            ),
            "contradiction_count": len(contradiction_seeds),
            "source_case_ids": tuple(
                dict.fromkeys((*existing.source_case_ids, *incoming.source_case_ids))
            ),
            "source_run_ids": source_run_ids,
            "source_seeds": tuple(dict.fromkeys((*existing.source_seeds, *incoming.source_seeds))),
            "censored_source_run_ids": tuple(
                dict.fromkeys(
                    (
                        *existing.censored_source_run_ids,
                        *incoming.censored_source_run_ids,
                    )
                )
            ),
            "censored_source_seeds": tuple(
                dict.fromkeys(
                    (
                        *existing.censored_source_seeds,
                        *incoming.censored_source_seeds,
                    )
                )
            ),
            "contradiction_seeds": contradiction_seeds,
            "shadow_state_count": max(existing.shadow_state_count, incoming.shadow_state_count),
            "false_block_count": max(existing.false_block_count, incoming.false_block_count),
            "evidence": {**existing.evidence, **incoming.evidence},
        }
    )


def _rule_is_unexpired(rule: PlaybookRule, now: datetime) -> bool:
    expires_at = rule.expires_at
    if expires_at is None:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at.astimezone(UTC) > now
