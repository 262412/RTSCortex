"""Shared selection and identity for executable hard Playbook rules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime

from rtscortex.playbook.models import (
    PlaybookCondition,
    PlaybookConditionOperator,
    PlaybookRule,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
)

_STATIC_CONTEXT_FIELDS = frozenset({"agent_race", "opponent_race", "map_name"})


def runtime_hard_rule_candidates(
    rules: Sequence[PlaybookRule],
    *,
    agent_race: str | None,
    opponent_race: str | None,
    map_name: str | None,
    now: datetime | None = None,
) -> tuple[PlaybookRule, ...]:
    """Return every hard rule the Runtime may execute during this match."""

    current_time = now or datetime.now(UTC)
    context = {
        key: value
        for key, value in {
            "agent_race": agent_race,
            "opponent_race": opponent_race,
            "map_name": map_name,
        }.items()
        if value is not None
    }
    candidates = (
        rule
        for rule in rules
        if rule.status is PlaybookRuleStatus.ACTIVE
        and rule.strength is PlaybookRuleStrength.HARD
        and rule_is_unexpired(rule, current_time)
        and rule_matches_static_context(rule, context)
    )
    return tuple(sorted(candidates, key=lambda rule: (-rule.confidence, rule.rule_id)))


def select_runtime_hard_rules(
    rules: Sequence[PlaybookRule],
    *,
    agent_race: str | None,
    opponent_race: str | None,
    map_name: str | None,
    max_hard: int,
    now: datetime | None = None,
) -> tuple[PlaybookRule, ...]:
    """Apply the Runtime hard-rule ordering and cap."""

    return runtime_hard_rule_candidates(
        rules,
        agent_race=agent_race,
        opponent_race=opponent_race,
        map_name=map_name,
        now=now,
    )[:max_hard]


def hard_rule_set_sha256(
    *,
    baseline_sha256: str,
    max_hard_rules: int,
    rules: Sequence[PlaybookRule],
) -> str:
    """Bind the approved IDs, order, and complete rule semantics."""

    payload = {
        "baseline_sha256": baseline_sha256,
        "max_hard_rules": max_hard_rules,
        "rules": [rule.model_dump(mode="json") for rule in rules],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rule_matches_static_context(
    rule: PlaybookRule,
    context: dict[str, str],
) -> bool:
    """Match only fields fixed for the complete game."""

    return all(
        condition.field not in _STATIC_CONTEXT_FIELDS
        or condition.field not in context
        or _condition_matches_static(condition, context[condition.field])
        for condition in rule.conditions
    )


def rule_is_unexpired(rule: PlaybookRule, now: datetime) -> bool:
    expires_at = rule.expires_at
    if expires_at is None:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at.astimezone(UTC) > now


def _condition_matches_static(condition: PlaybookCondition, actual: str) -> bool:
    expected = condition.value
    if condition.operator is PlaybookConditionOperator.EQ:
        return str(expected).casefold() == actual.casefold()
    if condition.operator is PlaybookConditionOperator.IN:
        values = expected if isinstance(expected, tuple) else (str(expected),)
        return actual.casefold() in {str(value).casefold() for value in values}
    if condition.operator is PlaybookConditionOperator.CONTAINS:
        return str(expected).casefold() in actual.casefold()
    # Invalid static operators remain selectable so readiness must reject them.
    return True
