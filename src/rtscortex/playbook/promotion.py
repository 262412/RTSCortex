"""Replay historical situation states through conservative Playbook promotion gates."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rtscortex.playbook.lifecycle import PlaybookRuleLifecycle
from rtscortex.playbook.models import (
    PlaybookCondition,
    PlaybookConditionOperator,
    PlaybookRule,
    PlaybookRuleStatus,
)
from rtscortex.playbook.store import PlaybookStore

_SITUATION_FIELDS = {
    "phase",
    "threat_level",
    "economy_status",
    "army_readiness",
}
_CONTEXTUAL_FIELDS = {
    "threat_level",
    "economy_status",
    "army_readiness",
    "alert",
}


@dataclass(frozen=True, slots=True)
class PromotionSweepResult:
    inspected_rule_count: int
    coverage_updated_rule_ids: tuple[str, ...]
    promoted_rule_ids: tuple[str, ...]
    unavailable_run_ids: tuple[str, ...]
    matched_state_count_by_rule: dict[str, int]
    rejected_reason_by_rule: dict[str, str]


class PlaybookPromotionSweep:
    """Reconstruct exact shadow matches from each rule's own source runs."""

    def __init__(self, store: PlaybookStore, *, run_root: Path | None = None) -> None:
        self.store = store
        self.run_root = (
            store.database_path.parent if run_root is None else run_root.expanduser()
        )
        self.lifecycle = PlaybookRuleLifecycle()

    def run(self) -> PromotionSweepResult:
        candidates = [
            rule
            for rule in self.store.rules()
            if rule.status is PlaybookRuleStatus.CANDIDATE
        ]
        coverage_updated: list[str] = []
        promoted: list[str] = []
        unavailable_runs: set[str] = set()
        matched_counts: dict[str, int] = {}
        rejected: dict[str, str] = {}
        situation_cache: dict[str, tuple[dict[str, object], ...] | None] = {}

        for rule in candidates:
            preliminary_error = _preliminary_rejection(rule)
            if preliminary_error is not None:
                rejected[rule.rule_id] = preliminary_error
                continue
            states: list[dict[str, object]] = []
            for run_id in dict.fromkeys(rule.source_run_ids):
                if run_id not in situation_cache:
                    situation_cache[run_id] = self._load_situations(run_id)
                run_states = situation_cache[run_id]
                if run_states is None:
                    unavailable_runs.add(run_id)
                    continue
                states.extend(run_states)
            matched_count = sum(_matches_rule_situation(rule, state) for state in states)
            matched_counts[rule.rule_id] = matched_count
            updated = rule
            if matched_count > rule.shadow_state_count:
                evidence = {
                    **rule.evidence,
                    "promotion_sweep": {
                        "source": "historical_situation_shadow_replay",
                        "matched_state_count": matched_count,
                        "source_run_count": len(set(rule.source_run_ids)),
                    },
                }
                updated = self.store.upsert_rule(
                    rule.model_copy(
                        update={
                            "shadow_state_count": matched_count,
                            "evidence": evidence,
                        }
                    )
                )
                coverage_updated.append(rule.rule_id)
            try:
                promoted_rule = self.lifecycle.promote_to_soft(updated)
            except ValueError as error:
                rejected[rule.rule_id] = str(error)
                continue
            self.store.upsert_rule(promoted_rule)
            promoted.append(rule.rule_id)

        return PromotionSweepResult(
            inspected_rule_count=len(candidates),
            coverage_updated_rule_ids=tuple(coverage_updated),
            promoted_rule_ids=tuple(promoted),
            unavailable_run_ids=tuple(sorted(unavailable_runs)),
            matched_state_count_by_rule=matched_counts,
            rejected_reason_by_rule=rejected,
        )

    def _load_situations(self, run_id: str) -> tuple[dict[str, object], ...] | None:
        database_path = self.run_root / run_id / "events.sqlite3"
        if not database_path.is_file():
            sibling_path = self.run_root / f"{run_id}.sqlite3"
            if not sibling_path.is_file():
                return None
            database_path = sibling_path
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM events
                WHERE event_type = 'situation_assessed'
                ORDER BY event_id
                """
            ).fetchall()
        finally:
            connection.close()
        situations: list[dict[str, object]] = []
        for row in rows:
            try:
                payload = json.loads(str(row[0]))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                situations.append(payload)
        return tuple(situations)


def _preliminary_rejection(rule: PlaybookRule) -> str | None:
    if len(set(rule.source_run_ids)) < 2 or len(set(rule.source_seeds)) < 2:
        return "soft promotion requires evidence from two runs and seeds"
    if rule.confidence < 0.75 or rule.contradiction_count:
        return "soft promotion confidence or contradiction gate failed"
    fields = {condition.field for condition in rule.conditions}
    if not fields.intersection(_CONTEXTUAL_FIELDS):
        return "soft promotion requires contextual shadow conditions"
    if not rule.action_names and not rule.role_ids:
        return "soft promotion requires an action or role target"
    return None


def _matches_rule_situation(rule: PlaybookRule, state: dict[str, object]) -> bool:
    conditions = [
        condition for condition in rule.conditions if condition.field in _SITUATION_FIELDS
    ]
    return bool(conditions) and all(_matches(condition, state) for condition in conditions)


def _matches(condition: PlaybookCondition, state: dict[str, object]) -> bool:
    if condition.field not in state:
        return False
    actual = state[condition.field]
    expected = condition.value
    if condition.operator is PlaybookConditionOperator.EQ:
        return actual == expected
    if condition.operator is PlaybookConditionOperator.IN:
        return isinstance(expected, tuple) and actual in expected
    if condition.operator is PlaybookConditionOperator.CONTAINS:
        return isinstance(actual, (tuple, list, set)) and expected in actual
    if condition.operator is PlaybookConditionOperator.GTE:
        return (
            isinstance(actual, (int, float))
            and isinstance(expected, (int, float))
            and actual >= expected
        )
    if condition.operator is PlaybookConditionOperator.LTE:
        return (
            isinstance(actual, (int, float))
            and isinstance(expected, (int, float))
            and actual <= expected
        )
    return False
