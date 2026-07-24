"""Replay historical situation states through conservative Playbook promotion gates."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from rtscortex.playbook.lifecycle import PlaybookRuleLifecycle
from rtscortex.playbook.models import (
    PlaybookCondition,
    PlaybookConditionOperator,
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
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
_CONSOLIDATION_CONTEXT_FIELDS: tuple[
    Literal["threat_level", "economy_status", "army_readiness"], ...
] = ("threat_level", "economy_status", "army_readiness")
_ConsolidationKey = tuple[
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    str,
    str,
]


@dataclass(frozen=True, slots=True)
class PromotionSweepResult:
    inspected_rule_count: int
    coverage_updated_rule_ids: tuple[str, ...]
    promoted_rule_ids: tuple[str, ...]
    unavailable_run_ids: tuple[str, ...]
    matched_state_count_by_rule: dict[str, int]
    rejected_reason_by_rule: dict[str, str]
    consolidated_rule_ids: tuple[str, ...] = ()


class PlaybookPromotionSweep:
    """Reconstruct exact shadow matches from each rule's own source runs."""

    def __init__(
        self,
        store: PlaybookStore,
        *,
        run_root: Path | None = None,
        run_directories: Mapping[str, Path] | None = None,
    ) -> None:
        self.store = store
        self.run_root = (
            store.database_path.parent if run_root is None else run_root.expanduser()
        )
        self.run_directories = {
            run_id: path.expanduser()
            for run_id, path in ({} if run_directories is None else run_directories).items()
        }
        self.lifecycle = PlaybookRuleLifecycle()

    def run(self) -> PromotionSweepResult:
        consolidated = self._consolidate_compatible_candidates()
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
            consolidated_rule_ids=consolidated,
        )

    def _consolidate_compatible_candidates(self) -> tuple[str, ...]:
        """Merge fragmented strategic evidence without broadening execution guards."""

        groups: dict[_ConsolidationKey, list[PlaybookRule]] = defaultdict(list)
        for rule in self.store.rules():
            if (
                rule.status is not PlaybookRuleStatus.CANDIDATE
                or rule.category
                in {
                    PlaybookRuleCategory.ENGINE_INVARIANT,
                    PlaybookRuleCategory.EXECUTION_GUARD,
                }
                or rule.contradiction_count
            ):
                continue
            actions = tuple(
                action
                for action in rule.action_names
                if action.strip().casefold() not in {"", "unknown"}
            )
            if not actions and not rule.role_ids:
                continue
            core = _core_condition_values(rule)
            if core is None:
                continue
            groups[
                (
                    rule.category,
                    rule.effect,
                    actions,
                    rule.role_ids,
                    *core,
                )
            ].append(rule)

        consolidated: list[str] = []
        for key, rules in groups.items():
            if len(rules) < 2:
                continue
            run_ids = tuple(
                dict.fromkeys(
                    run_id
                    for rule in rules
                    for run_id in rule.source_run_ids
                )
            )
            seeds = tuple(
                dict.fromkeys(seed for rule in rules for seed in rule.source_seeds)
            )
            if len(set(run_ids)) < 2 or len(set(seeds)) < 2:
                continue
            contextual_conditions = _consolidated_contextual_conditions(rules)
            if not contextual_conditions:
                continue
            category, effect, actions, roles, agent_race, opponent_race, phase, map_name = key
            conditions = (
                PlaybookCondition(field="agent_race", value=str(agent_race)),
                PlaybookCondition(field="opponent_race", value=str(opponent_race)),
                PlaybookCondition(field="phase", value=str(phase)),
                PlaybookCondition(field="map_name", value=str(map_name)),
                *contextual_conditions,
            )
            canonical_payload = "|".join(
                (
                    str(category),
                    str(effect),
                    *(
                        f"{condition.field}:{condition.operator.value}:{condition.value}"
                        for condition in conditions
                    ),
                    *(str(action) for action in actions),
                    *(str(role) for role in roles),
                )
            )
            canonical = hashlib.sha256(canonical_payload.encode()).hexdigest()
            rule = self.store.upsert_rule(
                PlaybookRule(
                    rule_id=f"playbook-rule:{canonical}",
                    canonical_key=canonical,
                    category=category,
                    conditions=conditions,
                    effect=effect,
                    strength=PlaybookRuleStrength.ADVISORY,
                    status=PlaybookRuleStatus.CANDIDATE,
                    action_names=actions,
                    role_ids=roles,
                    confidence=min(rule.confidence for rule in rules),
                    support_count=len(run_ids),
                    source_case_ids=tuple(
                        dict.fromkeys(
                            case_id
                            for rule in rules
                            for case_id in rule.source_case_ids
                        )
                    ),
                    source_run_ids=run_ids,
                    source_seeds=seeds,
                    censored_source_run_ids=tuple(
                        dict.fromkeys(
                            run_id
                            for rule in rules
                            for run_id in rule.censored_source_run_ids
                        )
                    ),
                    censored_source_seeds=tuple(
                        dict.fromkeys(
                            seed
                            for rule in rules
                            for seed in rule.censored_source_seeds
                        )
                    ),
                    evidence={
                        "consolidation": "typed_multi_run_strategy",
                        "source_rule_ids": [rule.rule_id for rule in rules],
                    },
                )
            )
            consolidated.append(rule.rule_id)
        return tuple(dict.fromkeys(consolidated))

    def _load_situations(self, run_id: str) -> tuple[dict[str, object], ...] | None:
        run_directory = self.run_directories.get(run_id, self.run_root / run_id)
        database_path = run_directory / "events.sqlite3"
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


def _core_condition_values(
    rule: PlaybookRule,
) -> tuple[str, str, str, str] | None:
    values: dict[str, object] = {
        condition.field: condition.value
        for condition in rule.conditions
        if condition.operator is PlaybookConditionOperator.EQ
    }
    core = tuple(
        values.get(field)
        for field in ("agent_race", "opponent_race", "phase", "map_name")
    )
    if not all(isinstance(value, str) for value in core):
        return None
    return cast(tuple[str, str, str, str], core)


def _consolidated_contextual_conditions(
    rules: list[PlaybookRule],
) -> tuple[PlaybookCondition, ...]:
    conditions: list[PlaybookCondition] = []
    for field in _CONSOLIDATION_CONTEXT_FIELDS:
        values = tuple(
            dict.fromkeys(
                str(condition.value)
                for rule in rules
                for condition in rule.conditions
                if condition.field == field
                and condition.operator is PlaybookConditionOperator.EQ
                and isinstance(condition.value, str)
            )
        )
        if not values:
            continue
        conditions.append(
            PlaybookCondition(
                field=field,
                operator=(
                    PlaybookConditionOperator.EQ
                    if len(values) == 1
                    else PlaybookConditionOperator.IN
                ),
                value=values[0] if len(values) == 1 else values,
            )
        )
    return tuple(conditions)
