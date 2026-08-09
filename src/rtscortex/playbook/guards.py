"""Deterministic, typed Playbook guards for intents and executable candidates."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from rtscortex.cortex.models import ExecutableCandidate, SituationAssessment
from rtscortex.cortex.strategic import StrategicIntent
from rtscortex.playbook.conditions import condition_matches
from rtscortex.playbook.models import (
    PlaybookContext,
    PlaybookRetryGuardBinding,
    PlaybookRoleId,
    PlaybookRule,
    PlaybookRuleApplication,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    playbook_predicate_fingerprint,
    playbook_rule_fingerprint,
)
from rtscortex.playbook.semantics import evaluation_kind


@dataclass(frozen=True, slots=True)
class GuardResult:
    blocked: bool
    score_delta: float
    rule_ids: tuple[str, ...]
    applications: tuple[PlaybookRuleApplication, ...]


@dataclass(frozen=True, slots=True)
class RecentTerminalFeedback:
    signature: str
    action_name: str
    actor: str
    failure_code: str
    command_id: str
    operation_id: str
    attempt_ordinal: int
    terminal_game_loop: int
    expires_game_loop: int
    hard_suppression: bool


def typed_retry_binding_matches(
    binding: PlaybookRetryGuardBinding,
    *,
    target_kind: Literal["intent", "candidate"],
    counterfactual_signature: str,
    recent_feedback: Sequence[RecentTerminalFeedback],
    operation_id: str | None,
    attempt_ordinal: int | None,
    game_loop: int,
) -> bool:
    """Apply the exact runtime retry predicate to one candidate.

    Historical replay and the live candidate guard both call this helper.  It
    deliberately keeps the runtime's five identity checks together: failure
    code, candidate signature, operation lineage, next attempt ordinal, and
    bounded freshness.
    """

    if target_kind != "candidate":
        return False
    return any(
        feedback.failure_code == binding.failure_code
        and _terminal_feedback_matches(
            feedback,
            signature=counterfactual_signature,
            operation_id=operation_id,
            attempt_ordinal=attempt_ordinal,
            game_loop=game_loop,
            max_age_game_loops=binding.max_age_game_loops,
        )
        for feedback in recent_feedback
    )


def typed_candidate_predicate_matches(
    rule: PlaybookRule,
    *,
    action_name: str,
    role: str | None,
    values: Mapping[str, object],
) -> bool:
    """Evaluate the candidate rule predicate shared by live and replay paths.

    ``values`` must contain the same context, situation, action, and role
    fields that :class:`PlaybookCandidateGuard` passes to ``_evaluate``.
    Keeping condition and target matching here prevents historical replay from
    drifting from the runtime's action/role semantics.
    """

    if not all(condition_matches(condition, values) for condition in rule.conditions):
        return False
    action_key = _action_key(action_name)
    targets_action = not rule.action_names or action_key in {
        _action_key(action) for action in rule.action_names
    }
    targets_role = not rule.role_ids or (role is not None and role in rule.role_ids)
    return targets_action and targets_role


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
        behavior_before_hash: str | None = None,
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
            counterfactual_signature=_intent_signature(intent),
            behavior_before_hash=behavior_before_hash,
            decision_epoch=game_loop,
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
        behavior_before_hash: str | None = None,
        mode: Literal["shadow", "active"] = "shadow",
        recent_feedback: Sequence[RecentTerminalFeedback] = (),
        operation_id: str | None = None,
        attempt_ordinal: int | None = None,
    ) -> GuardResult:
        values = _values(
            context,
            situation,
            action_name=candidate.action_name,
            role=role,
        )
        rules_result = _evaluate(
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
            counterfactual_signature=candidate_signature(
                candidate.action_name,
                candidate.actor,
                candidate.arguments,
            ),
            behavior_before_hash=behavior_before_hash,
            decision_epoch=game_loop,
            mode=mode,
            recent_feedback=recent_feedback,
            operation_id=operation_id,
            attempt_ordinal=attempt_ordinal,
        )
        signature = candidate_signature(
            candidate.action_name,
            candidate.actor,
            candidate.arguments,
        )
        feedback = next(
            (
                item
                for item in recent_feedback
                if _terminal_feedback_matches(
                    item,
                    signature=signature,
                    operation_id=operation_id,
                    attempt_ordinal=attempt_ordinal,
                    game_loop=game_loop,
                )
            ),
            None,
        )
        if feedback is None:
            return rules_result
        blocked = mode == "active"
        score_delta = -2.0
        application = PlaybookRuleApplication(
            application_id=(
                "rule-application:"
                + hashlib.sha256(
                    f"{feedback.command_id}|{candidate.candidate_id}|{game_loop}".encode()
                ).hexdigest()
            ),
            rule_id=f"terminal-feedback:{feedback.command_id}",
            run_id=run_id,
            episode_id=episode_id,
            step_id=step_id,
            game_loop=game_loop,
            target_kind="candidate",
            target_id=candidate.candidate_id,
            matched=True,
            blocked=blocked,
            score_delta=score_delta,
            reason=(
                "recent_terminal_failure_block"
                if feedback.hard_suppression
                else "recent_terminal_failure_cooldown"
            ),
        )
        return GuardResult(
            blocked=rules_result.blocked or blocked,
            score_delta=(rules_result.score_delta + score_delta if mode == "active" else 0.0),
            rule_ids=(*rules_result.rule_ids, application.rule_id),
            applications=(*rules_result.applications, application),
        )


def candidate_signature(
    action_name: str,
    actor: str,
    arguments: Sequence[object],
) -> str:
    payload = json.dumps(
        {
            "action_name": action_name,
            "actor": actor,
            "arguments": list(arguments),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _intent_signature(intent: StrategicIntent) -> str:
    """Return one run-neutral identity for the exact semantic operation."""

    payload = json.dumps(
        {
            "role": intent.role.value,
            "action_names": list(intent.action_names),
            "actor_scopes": sorted(scope.casefold() for scope in intent.actor_scopes),
            "semantic_target_key": intent.semantic_target_key.casefold(),
            "desired_effect": intent.desired_effect,
            "producer_types": sorted(item.casefold() for item in intent.producer_types),
            "resource_claim": intent.resource_claim.model_dump(mode="json"),
            "dependency_semantic_keys": sorted(intent.dependency_semantic_keys),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


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
    counterfactual_signature: str,
    behavior_before_hash: str | None,
    decision_epoch: int,
    mode: Literal["shadow", "active"],
    recent_feedback: Sequence[RecentTerminalFeedback] = (),
    operation_id: str | None = None,
    attempt_ordinal: int | None = None,
) -> GuardResult:
    applicable = [
        rule
        for rule in rules
        if (
            rule.status in {PlaybookRuleStatus.LEGACY, PlaybookRuleStatus.ACTIVE}
            or rule.status is PlaybookRuleStatus.CANDIDATE
        )
        and all(condition_matches(condition, values) for condition in rule.conditions)
        and _rule_retry_binding_matches(
            rule,
            target_kind=target_kind,
            counterfactual_signature=counterfactual_signature,
            recent_feedback=recent_feedback,
            operation_id=operation_id,
            attempt_ordinal=attempt_ordinal,
            game_loop=game_loop,
        )
    ]
    required = {
        _action_key(action)
        for rule in applicable
        if rule.effect is PlaybookRuleEffect.REQUIRE and rule.strength is PlaybookRuleStrength.HARD
        for action in rule.action_names
    }
    blocked = False
    delta = 0.0
    applications: list[PlaybookRuleApplication] = []
    applied_ids: list[str] = []
    for rule in applicable:
        shadow_candidate = rule.status is PlaybookRuleStatus.CANDIDATE
        matched = typed_candidate_predicate_matches(
            rule,
            action_name=action_name,
            role=role,
            values=values,
        )
        action_key = _action_key(action_name)
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
                    and action_key not in required
                )
                rule_delta = 0.5 if matched and not rule_blocked else 0.0
        effective_block = rule_blocked and mode == "active" and not shadow_candidate
        blocked = blocked or effective_block
        if not shadow_candidate:
            delta += rule_delta
        if matched or rule_blocked:
            applied_ids.append(rule.rule_id)
        if matched or rule_blocked or rule_delta:
            counterfactual_key = _counterfactual_key(
                rule.rule_id,
                target_kind=target_kind,
                counterfactual_signature=counterfactual_signature,
                behavior_before_hash=behavior_before_hash,
                decision_epoch=decision_epoch,
            )
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
                    rule_kind=evaluation_kind(rule.category),
                    action_name=action_name,
                    role=cast(PlaybookRoleId, role),
                    counterfactual_key=counterfactual_key,
                    counterfactual_signature=counterfactual_signature,
                    behavior_before_hash=behavior_before_hash,
                    decision_epoch=decision_epoch,
                    rule_fingerprint=playbook_rule_fingerprint(rule),
                    predicate_fingerprint=playbook_predicate_fingerprint(rule),
                    matched=matched,
                    blocked=effective_block,
                    score_delta=rule_delta,
                    reason=(
                        "candidate_shadow_match"
                        if shadow_candidate
                        else "shadow_would_block"
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
        score_delta=delta if mode == "active" else 0.0,
        rule_ids=tuple(dict.fromkeys(applied_ids)),
        applications=tuple(applications),
    )


def _rule_retry_binding_matches(
    rule: PlaybookRule,
    *,
    target_kind: Literal["intent", "candidate"],
    counterfactual_signature: str,
    recent_feedback: Sequence[RecentTerminalFeedback],
    operation_id: str | None,
    attempt_ordinal: int | None,
    game_loop: int,
) -> bool:
    binding = rule.retry_guard
    if binding is None:
        return not (
            rule.category is PlaybookRuleCategory.EXECUTION_GUARD
            and rule.effect in {PlaybookRuleEffect.AVOID, PlaybookRuleEffect.FORBID}
            and rule.evidence.get("canary_fixture") is not True
        )
    return typed_retry_binding_matches(
        binding,
        target_kind=target_kind,
        counterfactual_signature=counterfactual_signature,
        recent_feedback=recent_feedback,
        operation_id=operation_id,
        attempt_ordinal=attempt_ordinal,
        game_loop=game_loop,
    )


def _terminal_feedback_matches(
    feedback: RecentTerminalFeedback,
    *,
    signature: str,
    operation_id: str | None,
    attempt_ordinal: int | None,
    game_loop: int,
    max_age_game_loops: int | None = None,
) -> bool:
    if operation_id is None or attempt_ordinal is None:
        return False
    age = game_loop - feedback.terminal_game_loop
    return (
        feedback.signature == signature
        and feedback.operation_id == operation_id
        and attempt_ordinal == feedback.attempt_ordinal + 1
        and age >= 0
        and feedback.expires_game_loop >= game_loop
        and (max_age_game_loops is None or age <= max_age_game_loops)
    )


def _counterfactual_key(
    rule_id: str,
    *,
    target_kind: Literal["intent", "candidate"],
    counterfactual_signature: str,
    behavior_before_hash: str | None,
    decision_epoch: int,
) -> str:
    """Identify one exact rule decision across matched active/shadow runs."""

    payload = json.dumps(
        {
            "rule_id": rule_id,
            "target_kind": target_kind,
            "counterfactual_signature": counterfactual_signature,
            "behavior_before_hash": behavior_before_hash,
            "decision_epoch": decision_epoch,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"counterfactual:{hashlib.sha256(payload.encode()).hexdigest()}"


def _action_key(action_name: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]", "", action_name).upper()
    if key.startswith("BUILD"):
        for suffix in ("SCREEN", "NEAR"):
            if key.endswith(suffix):
                return key[: -len(suffix)]
    return key
