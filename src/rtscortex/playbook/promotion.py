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

import yaml

from rtscortex.memory import StoredEvent, read_event_log
from rtscortex.playbook.conditions import condition_matches
from rtscortex.playbook.lifecycle import PlaybookRuleLifecycle
from rtscortex.playbook.models import (
    PlaybookCondition,
    PlaybookConditionOperator,
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleKind,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
)
from rtscortex.playbook.retry_coverage import (
    TypedRetryCoverage,
    TypedRetryCoverageBySeed,
    analyze_typed_retry_coverage,
)
from rtscortex.playbook.semantics import evaluation_kind
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
        self.run_root = store.database_path.parent if run_root is None else run_root.expanduser()
        self.run_directories = {
            run_id: path.expanduser()
            for run_id, path in ({} if run_directories is None else run_directories).items()
        }
        self.lifecycle = PlaybookRuleLifecycle()

    def run(self) -> PromotionSweepResult:
        consolidated = self._consolidate_compatible_candidates()
        candidates = [
            rule for rule in self.store.rules() if rule.status is PlaybookRuleStatus.CANDIDATE
        ]
        coverage_updated: list[str] = []
        promoted: list[str] = []
        unavailable_runs: set[str] = set()
        matched_counts: dict[str, int] = {}
        rejected: dict[str, str] = {}
        situation_cache: dict[str, tuple[tuple[dict[str, object], int], ...] | None] = {}
        event_cache: dict[str, tuple[StoredEvent, ...] | None] = {}

        for rule in candidates:
            preliminary_error = _preliminary_rejection(rule)
            if preliminary_error is not None:
                rejected[rule.rule_id] = preliminary_error
                continue
            typed_coverage: dict[str, object] | None = None
            typed_opportunity_count: int | None = None
            if (
                evaluation_kind(rule.category) is PlaybookRuleKind.EXECUTION_GUARD
                and rule.retry_guard is not None
            ):
                typed_by_seed: dict[int, TypedRetryCoverage] = {}
                typed_reasons: dict[str, tuple[str, ...]] = {}
                missing_typed_run: str | None = None
                source_run_ids = tuple(dict.fromkeys(rule.source_run_ids))
                for run_id in source_run_ids:
                    if run_id not in event_cache:
                        event_cache[run_id] = self._load_events(run_id)
                    events = event_cache[run_id]
                    if events is None:
                        missing_typed_run = run_id
                        unavailable_runs.add(run_id)
                        break
                    seed = self._run_seed(run_id, events)
                    if seed is None or seed in typed_by_seed:
                        missing_typed_run = run_id
                        unavailable_runs.add(run_id)
                        break
                    coverage = analyze_typed_retry_coverage(
                        events,
                        rule=rule,
                        context_values=self._load_runtime_context(run_id),
                    )
                    typed_by_seed[seed] = coverage
                    if coverage.unavailable_reasons:
                        typed_reasons[str(seed)] = coverage.unavailable_reasons
                if missing_typed_run is not None:
                    rejected[rule.rule_id] = "typed_retry_coverage_unavailable:" + missing_typed_run
                    continue
                typed_coverage_by_seed = TypedRetryCoverageBySeed(by_seed=typed_by_seed)
                if (
                    typed_reasons
                    or set(typed_by_seed)
                    != set(rule.source_seeds) - set(rule.censored_source_seeds)
                    or any(
                        coverage.typed_retry_opportunity_count <= 0
                        for coverage in typed_by_seed.values()
                    )
                ):
                    rejected[rule.rule_id] = (
                        "typed_retry_coverage_unavailable:"
                        if typed_reasons
                        else "typed_retry_opportunity_coverage:"
                    ) + json.dumps(
                        {
                            "reasons_by_seed": {
                                seed: list(reasons) for seed, reasons in typed_reasons.items()
                            },
                            "opportunity_count_by_seed": {
                                str(seed): coverage.typed_retry_opportunity_count
                                for seed, coverage in sorted(typed_by_seed.items())
                            },
                        },
                        sort_keys=True,
                    )
                    continue
                typed_opportunity_count = sum(
                    coverage.typed_retry_opportunity_count for coverage in typed_by_seed.values()
                )
                typed_coverage = typed_coverage_by_seed.as_dict()
            states: list[tuple[dict[str, object], int]] = []
            for run_id in dict.fromkeys(rule.source_run_ids):
                if run_id not in situation_cache:
                    situation_cache[run_id] = self._load_situations(run_id)
                run_states = situation_cache[run_id]
                if run_states is None:
                    unavailable_runs.add(run_id)
                    continue
                states.extend(run_states)
            broad_matched_count = sum(
                count for state, count in states if _matches_rule_situation(rule, state)
            )
            matched_counts[rule.rule_id] = broad_matched_count
            updated = rule
            if typed_opportunity_count is not None or broad_matched_count > rule.shadow_state_count:
                promotion_sweep_evidence: dict[str, object] = {
                    "source": "historical_situation_shadow_replay",
                    "matched_state_count": broad_matched_count,
                    "source_run_count": len(set(rule.source_run_ids)),
                }
                if typed_opportunity_count is not None:
                    promotion_sweep_evidence["typed_retry_opportunity_count"] = (
                        typed_opportunity_count
                    )
                    if typed_coverage is not None:
                        promotion_sweep_evidence["typed_retry_opportunity_count_by_seed"] = (
                            typed_coverage.get("typed_retry_opportunity_count_by_seed", {})
                        )
                evidence = {
                    **rule.evidence,
                    "promotion_sweep": promotion_sweep_evidence,
                }
                if typed_coverage is not None:
                    evidence["typed_retry_coverage"] = typed_coverage
                updated = self.store.upsert_rule(
                    rule.model_copy(
                        update={
                            "shadow_state_count": max(rule.shadow_state_count, broad_matched_count),
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

    def _load_events(self, run_id: str) -> tuple[StoredEvent, ...] | None:
        run_directory = self.run_directories.get(run_id, self.run_root / run_id)
        events_path = run_directory / "events.jsonl"
        if not events_path.is_file():
            return None
        try:
            return tuple(read_event_log(events_path))
        except (OSError, ValueError):
            return None

    def _load_runtime_context(self, run_id: str) -> dict[str, object]:
        run_directory = self.run_directories.get(run_id, self.run_root / run_id)
        config_path = run_directory / "config.yaml"
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            environment = payload["environment"]
            if not isinstance(environment, dict):
                return {}
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
            return {}
        context: dict[str, object] = {}
        for field in ("agent_race", "opponent_race", "scenario"):
            value = environment.get(field)
            if isinstance(value, str) and value:
                context["map_name" if field == "scenario" else field] = value
        return context

    def _run_seed(self, run_id: str, events: tuple[StoredEvent, ...]) -> int | None:
        run_directory = self.run_directories.get(run_id, self.run_root / run_id)
        config_path = run_directory / "config.yaml"
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            run_config = payload.get("run") if isinstance(payload, dict) else None
            seed = run_config.get("seed") if isinstance(run_config, dict) else None
            if isinstance(seed, int) and not isinstance(seed, bool):
                return seed
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            pass
        for event in events:
            if event.event_type != "episode_result":
                continue
            seed = event.payload.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                return seed
        return None

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
                dict.fromkeys(run_id for rule in rules for run_id in rule.source_run_ids)
            )
            seeds = tuple(dict.fromkeys(seed for rule in rules for seed in rule.source_seeds))
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
                        dict.fromkeys(case_id for rule in rules for case_id in rule.source_case_ids)
                    ),
                    source_run_ids=run_ids,
                    source_seeds=seeds,
                    censored_source_run_ids=tuple(
                        dict.fromkeys(
                            run_id for rule in rules for run_id in rule.censored_source_run_ids
                        )
                    ),
                    censored_source_seeds=tuple(
                        dict.fromkeys(seed for rule in rules for seed in rule.censored_source_seeds)
                    ),
                    evidence={
                        "consolidation": "typed_multi_run_strategy",
                        "source_rule_ids": [rule.rule_id for rule in rules],
                    },
                )
            )
            consolidated.append(rule.rule_id)
        return tuple(dict.fromkeys(consolidated))

    def _load_situations(self, run_id: str) -> tuple[tuple[dict[str, object], int], ...] | None:
        run_directory = self.run_directories.get(run_id, self.run_root / run_id)
        database_path = run_directory / "events.sqlite3"
        if not database_path.is_file():
            sibling_path = self.run_root / f"{run_id}.sqlite3"
            if not sibling_path.is_file():
                return None
            database_path = sibling_path
        connection = sqlite3.connect(f"file:{database_path}?mode=ro&immutable=1", uri=True)
        try:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM events
                WHERE event_type = 'situation_assessed'
                ORDER BY event_id
                """
            ).fetchall()
            retention_row = connection.execute(
                """
                SELECT payload_json
                FROM events
                WHERE event_type = 'event_retention_summary'
                ORDER BY event_id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            connection.close()
        retained_states = _retained_situation_states(retention_row)
        if retained_states is not None:
            return retained_states
        situations: list[tuple[dict[str, object], int]] = []
        for row in rows:
            try:
                payload = json.loads(str(row[0]))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                situations.append((payload, 1))
        return tuple(situations)


def _retained_situation_states(
    row: tuple[object, ...] | None,
) -> tuple[tuple[dict[str, object], int], ...] | None:
    if row is None:
        return None
    try:
        summary = json.loads(str(row[0]))
    except (TypeError, ValueError):
        return None
    aggregates = summary.get("aggregates") if isinstance(summary, dict) else None
    situation = aggregates.get("situation_assessed") if isinstance(aggregates, dict) else None
    state_counts = situation.get("state_counts") if isinstance(situation, dict) else None
    if not isinstance(state_counts, dict):
        return None
    states: list[tuple[dict[str, object], int]] = []
    for encoded_state, raw_count in state_counts.items():
        if not isinstance(encoded_state, str) or not isinstance(raw_count, int) or raw_count < 1:
            continue
        try:
            state = json.loads(encoded_state)
        except (TypeError, ValueError):
            continue
        if isinstance(state, dict):
            states.append((state, raw_count))
    return tuple(states)


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
    return bool(conditions) and all(condition_matches(condition, state) for condition in conditions)


def _core_condition_values(
    rule: PlaybookRule,
) -> tuple[str, str, str, str] | None:
    values: dict[str, object] = {
        condition.field: condition.value
        for condition in rule.conditions
        if condition.operator is PlaybookConditionOperator.EQ
    }
    core = tuple(
        values.get(field) for field in ("agent_race", "opponent_race", "phase", "map_name")
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
