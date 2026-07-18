"""Deterministic, typed Playbook guards for intents and executable candidates."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from rtscortex.cortex.models import ExecutableCandidate, SituationAssessment
from rtscortex.cortex.strategic import StrategicIntent
from rtscortex.playbook.models import (
    PlaybookCondition,
    PlaybookConditionOperator,
    PlaybookContext,
    PlaybookRule,
    PlaybookRuleApplication,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
)


@dataclass(frozen=True, slots=True)
class GuardResult:
    blocked: bool
    score_delta: float
    rule_ids: tuple[str, ...]
    applications: tuple[PlaybookRuleApplication, ...]


class PlaybookIntentGuard:
    """Apply validated Playbook rules before strategic intent arbitration."""

    def evaluate(
        self,
        intent: StrategicIntent,
        *,
        context: PlaybookContext,
        situation: SituationAssessment,
        rules: Sequence[PlaybookRule],
        game_loop: int,
        mode: Literal["shadow", "active"] = "shadow",
    ) -> GuardResult:
        values = _values(
            context,
            situation,
            action_name=intent.action_names[0],
            role=intent.role.value,
        )
        return _evaluate(
            rules,
            values,
            run_id=intent.run_id,
            episode_id=intent.episode_id,
            step_id=intent.step_id,
            game_loop=game_loop,
            target_kind="intent",
            target_id=intent.intent_id,
            action_name=intent.action_names[0],
            role=intent.role.value,
            mode=mode,
        )


class PlaybookCandidateGuard:
    """Filter exact candidates immediately before the Fast Executor."""

    def evaluate(
        self,
        candidate: ExecutableCandidate,
        *,
        role: str,
        context: PlaybookContext,
        situation: SituationAssessment,
        rules: Sequence[PlaybookRule],
        run_id: str,
        episode_id: str,
        step_id: int,
        game_loop: int,
        mode: Literal["shadow", "active"] = "shadow",
    ) -> GuardResult:
        values = _values(
            context,
            situation,
            action_name=candidate.action_name,
            role=role,
        )
        return _evaluate(
            rules,
            values,
            run_id=run_id,
            episode_id=episode_id,
            step_id=step_id,
            game_loop=game_loop,
            target_kind="candidate",
            target_id=candidate.candidate_id,
            action_name=candidate.action_name,
            role=role,
            mode=mode,
        )


def _values(
    context: PlaybookContext,
    situation: SituationAssessment,
    *,
    action_name: str,
    role: str,
) -> dict[str, object]:
    return {
        "agent_race": context.agent_race,
        "opponent_race": context.opponent_race,
        "phase": context.phase.value,
        "map_name": context.map_name,
        "action_name": action_name,
        "role": role,
        "threat_level": situation.threat_level.value,
        "economy_status": situation.economy_status.value,
        "army_readiness": situation.army_readiness.value,
        "alert": tuple(context.tags),
    }


def _evaluate(
    rules: Sequence[PlaybookRule],
    values: Mapping[str, object],
    *,
    run_id: str,
    episode_id: str,
    step_id: int,
    game_loop: int,
    target_kind: Literal["intent", "candidate"],
    target_id: str,
    action_name: str,
    role: str,
    mode: Literal["shadow", "active"],
) -> GuardResult:
    applicable = [
        rule
        for rule in rules
        if rule.status in {PlaybookRuleStatus.LEGACY, PlaybookRuleStatus.ACTIVE}
        and all(_matches(condition, values) for condition in rule.conditions)
    ]
    required = {
        action
        for rule in applicable
        if rule.effect is PlaybookRuleEffect.REQUIRE
        and rule.strength is PlaybookRuleStrength.HARD
        for action in rule.action_names
    }
    blocked = False
    delta = 0.0
    applications: list[PlaybookRuleApplication] = []
    applied_ids: list[str] = []
    for rule in applicable:
        targets_action = not rule.action_names or action_name in rule.action_names
        targets_role = not rule.role_ids or role in rule.role_ids
        matched = targets_action and targets_role
        rule_blocked = False
        rule_delta = 0.0
        if rule.strength is not PlaybookRuleStrength.ADVISORY:
            if rule.effect is PlaybookRuleEffect.PREFER and matched:
                rule_delta = 1.0 if rule.strength is PlaybookRuleStrength.HARD else 0.5
            elif rule.effect is PlaybookRuleEffect.AVOID and matched:
                rule_delta = -1.0 if rule.strength is PlaybookRuleStrength.HARD else -0.5
            elif rule.effect is PlaybookRuleEffect.FORBID and matched:
                rule_blocked = rule.strength is PlaybookRuleStrength.HARD
                rule_delta = -1.0 if rule.strength is PlaybookRuleStrength.SOFT else 0.0
            elif rule.effect is PlaybookRuleEffect.REQUIRE:
                rule_blocked = (
                    rule.strength is PlaybookRuleStrength.HARD
                    and bool(required)
                    and action_name not in required
                )
                rule_delta = 0.5 if matched and not rule_blocked else 0.0
        effective_block = rule_blocked and mode == "active"
        blocked = blocked or effective_block
        delta += rule_delta
        if matched or rule_blocked:
            applied_ids.append(rule.rule_id)
        if matched or rule_blocked or rule_delta:
            identity = hashlib.sha256(
                f"{rule.rule_id}|{target_kind}|{target_id}|{game_loop}".encode()
            ).hexdigest()
            applications.append(
                PlaybookRuleApplication(
                    application_id=f"rule-application:{identity}",
                    rule_id=rule.rule_id,
                    run_id=run_id,
                    episode_id=episode_id,
                    step_id=step_id,
                    game_loop=game_loop,
                    target_kind=target_kind,
                    target_id=target_id,
                    matched=matched,
                    blocked=effective_block,
                    score_delta=rule_delta,
                    reason=(
                        "shadow_would_block"
                        if rule_blocked and mode == "shadow"
                        else "rule_blocked"
                        if effective_block
                        else "rule_scored"
                        if rule_delta
                        else "advisory_match"
                    ),
                )
            )
    return GuardResult(
        blocked=blocked,
        score_delta=delta,
        rule_ids=tuple(dict.fromkeys(applied_ids)),
        applications=tuple(applications),
    )


def _matches(condition: PlaybookCondition, values: Mapping[str, object]) -> bool:
    actual = values[condition.field]
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
