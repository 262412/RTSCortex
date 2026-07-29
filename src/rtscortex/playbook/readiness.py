"""Fail-closed readiness and qualification for executable hard Playbook rules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from rtscortex.contracts.models import ContractModel
from rtscortex.playbook.lifecycle import PlaybookRuleLifecycle, StrategicABEvidence
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
from rtscortex.policy.capabilities import DEFAULT_RUNTIME_CAPABILITIES
from rtscortex.races import race_profile

_STATIC_CONTEXT_FIELDS = frozenset({"agent_race", "opponent_race", "map_name"})
_VALID_ROLES = frozenset(
    {
        "economy",
        "technology",
        "production",
        "defense",
        "offense",
        "focus_fire",
        "retreat",
    }
)


class PlaybookRuleReadiness(ContractModel):
    """One rule's complete, read-only hard-promotion audit."""

    rule_id: str
    parent_rule_id: str | None = None
    category: PlaybookRuleCategory
    status: PlaybookRuleStatus
    strength: PlaybookRuleStrength
    effect: PlaybookRuleEffect
    action_names: tuple[str, ...]
    role_ids: tuple[str, ...]
    conditions: tuple[PlaybookCondition, ...]
    context_applicable: bool
    target_reachable: bool
    source_run_ids: tuple[str, ...]
    source_seed_ids: tuple[int, ...]
    censored_source_run_ids: tuple[str, ...]
    censored_source_seed_ids: tuple[int, ...]
    confidence: float
    contradiction_count: int
    shadow_state_count: int
    execution_false_block_count: int
    execution_false_block_rate: float
    strategic_regret_count: int | None = None
    code_revision: str | None = None
    sc2_patch: str | None = None
    qualified_at_git_sha: str | None = None
    qualified_at_sc2_patch: str | None = None
    qualification_seed_ids: tuple[int, ...]
    evaluation_seed_ids: tuple[int, ...]
    evidence_hashes: tuple[str, ...]
    canary_fixture: bool
    rejection_reasons: tuple[str, ...]


class PlaybookHardReadinessReport(ContractModel):
    """Machine-readable preflight artifact emitted before any GPU or SC2 work."""

    schema_version: str = "1.0"
    baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    sc2_patch: str = Field(min_length=1)
    agent_race: str = Field(min_length=1)
    opponent_race: str = Field(min_length=1)
    map_name: str = Field(min_length=1)
    evaluation_seed_ids: tuple[int, ...] = ()
    active_soft_count: int = Field(ge=0)
    active_hard_count: int = Field(ge=0)
    active_blocking_hard_count: int = Field(ge=0)
    context_applicable_blocking_hard_count: int = Field(ge=0)
    reachable_blocking_hard_count: int = Field(ge=0)
    canary_fixture_rule_ids: tuple[str, ...] = ()
    canary_runnable: bool
    rejection_reasons: dict[str, tuple[str, ...]]
    rules: tuple[PlaybookRuleReadiness, ...]


def analyze_hard_readiness(
    rules: Sequence[PlaybookRule],
    *,
    baseline_sha256: str,
    expected_git_sha: str,
    sc2_patch: str,
    agent_race: str,
    opponent_race: str,
    map_name: str,
    evaluation_seed_ids: Sequence[int] = (),
    allow_canary_fixture: bool = False,
) -> PlaybookHardReadinessReport:
    """Classify every rule and require one context-applicable blocking hard rule."""

    normalized_race = agent_race.casefold()
    profile = race_profile(normalized_race).data
    reachable_actions = {
        *DEFAULT_RUNTIME_CAPABILITIES.supported_actions,
        *(spec.name for spec in profile.progress_action_specs),
        *profile.controller_capabilities,
        *profile.controller_managed_actions,
    }
    evaluation_seeds = tuple(dict.fromkeys(int(seed) for seed in evaluation_seed_ids))
    active = [rule for rule in rules if rule.status is PlaybookRuleStatus.ACTIVE]
    active_hard = [rule for rule in active if rule.strength is PlaybookRuleStrength.HARD]
    blocking = [rule for rule in active_hard if rule.effect is PlaybookRuleEffect.FORBID]
    context = {
        "agent_race": normalized_race,
        "opponent_race": opponent_race.casefold(),
        "map_name": map_name,
    }
    context_applicable = [rule for rule in blocking if _context_is_applicable(rule, context)]
    rejection_reasons: dict[str, tuple[str, ...]] = {}
    reachable: list[PlaybookRule] = []
    fixture_rule_ids: list[str] = []
    rule_audits: list[PlaybookRuleReadiness] = []
    rules_by_id = {rule.rule_id: rule for rule in rules}
    for rule in rules:
        fixture = rule.evidence.get("canary_fixture") is True
        if fixture:
            fixture_rule_ids.append(rule.rule_id)
        reasons = _hard_rejection_reasons(
            rule,
            expected_git_sha=expected_git_sha,
            sc2_patch=sc2_patch,
            context=context,
            reachable_actions=reachable_actions,
            evaluation_seed_ids=evaluation_seeds,
            allow_canary_fixture=allow_canary_fixture,
            rules_by_id=rules_by_id,
        )
        if reasons:
            rejection_reasons[rule.rule_id] = reasons
        elif rule in context_applicable:
            reachable.append(rule)
        strategic_regret = rule.evidence.get("strategic_regret_count")
        rule_audits.append(
            PlaybookRuleReadiness(
                rule_id=rule.rule_id,
                parent_rule_id=rule.parent_rule_id,
                category=rule.category,
                status=rule.status,
                strength=rule.strength,
                effect=rule.effect,
                action_names=rule.action_names,
                role_ids=rule.role_ids,
                conditions=rule.conditions,
                context_applicable=_context_is_applicable(rule, context),
                target_reachable=_rule_target_is_reachable(rule, reachable_actions),
                source_run_ids=rule.source_run_ids,
                source_seed_ids=rule.source_seeds,
                censored_source_run_ids=rule.censored_source_run_ids,
                censored_source_seed_ids=rule.censored_source_seeds,
                confidence=rule.confidence,
                contradiction_count=rule.contradiction_count,
                shadow_state_count=rule.shadow_state_count,
                execution_false_block_count=rule.false_block_count,
                execution_false_block_rate=rule.false_block_rate,
                strategic_regret_count=(
                    int(strategic_regret) if isinstance(strategic_regret, int | float) else None
                ),
                code_revision=rule.code_revision,
                sc2_patch=rule.sc2_patch,
                qualified_at_git_sha=rule.qualified_at_git_sha,
                qualified_at_sc2_patch=rule.qualified_at_sc2_patch,
                qualification_seed_ids=rule.qualification_seed_ids,
                evaluation_seed_ids=rule.evaluation_seed_ids,
                evidence_hashes=rule.evidence_hashes,
                canary_fixture=fixture,
                rejection_reasons=reasons,
            )
        )
    return PlaybookHardReadinessReport(
        baseline_sha256=baseline_sha256,
        expected_git_sha=expected_git_sha,
        sc2_patch=sc2_patch,
        agent_race=normalized_race,
        opponent_race=opponent_race.casefold(),
        map_name=map_name,
        evaluation_seed_ids=evaluation_seeds,
        active_soft_count=sum(rule.strength is PlaybookRuleStrength.SOFT for rule in active),
        active_hard_count=len(active_hard),
        active_blocking_hard_count=len(blocking),
        context_applicable_blocking_hard_count=len(context_applicable),
        reachable_blocking_hard_count=len(reachable),
        canary_fixture_rule_ids=tuple(sorted(fixture_rule_ids)),
        canary_runnable=bool(reachable),
        rejection_reasons=rejection_reasons,
        rules=tuple(rule_audits),
    )


def analyze_hard_readiness_database(
    database_path: Path,
    *,
    expected_git_sha: str,
    sc2_patch: str,
    agent_race: str,
    opponent_race: str,
    map_name: str,
    evaluation_seed_ids: Sequence[int] = (),
    allow_canary_fixture: bool = False,
) -> PlaybookHardReadinessReport:
    """Read one immutable baseline and return its readiness report."""

    resolved = database_path.expanduser().resolve()
    store = PlaybookStore(resolved, read_only=True)
    try:
        rules = store.rules()
    finally:
        store.close()
    return analyze_hard_readiness(
        rules,
        baseline_sha256=_sha256_file(resolved),
        expected_git_sha=expected_git_sha,
        sc2_patch=sc2_patch,
        agent_race=agent_race,
        opponent_race=opponent_race,
        map_name=map_name,
        evaluation_seed_ids=evaluation_seed_ids,
        allow_canary_fixture=allow_canary_fixture,
    )


def qualify_hard_rule(
    store: PlaybookStore,
    *,
    parent_rule_id: str,
    expected_git_sha: str,
    sc2_patch: str,
    evidence_paths: Sequence[Path],
    evaluation_seed_ids: Sequence[int],
    strategic_ab: StrategicABEvidence | None = None,
) -> PlaybookRule:
    """Create a derived hard rule without mutating its soft parent."""

    parent = next((rule for rule in store.rules() if rule.rule_id == parent_rule_id), None)
    if parent is None:
        raise ValueError(f"unknown parent Playbook rule {parent_rule_id!r}")
    if parent.status is not PlaybookRuleStatus.ACTIVE:
        raise ValueError("hard qualification requires an active parent rule")
    if parent.strength is not PlaybookRuleStrength.SOFT:
        raise ValueError("hard qualification requires a soft parent rule")
    effect = _qualified_effect(parent.effect)
    qualification_seeds = tuple(
        sorted(set(parent.source_seeds) - set(parent.censored_source_seeds))
    )
    held_out_seeds = tuple(sorted(set(int(seed) for seed in evaluation_seed_ids)))
    if not held_out_seeds:
        raise ValueError("hard qualification requires held-out evaluation seeds")
    overlap = set(qualification_seeds) & set(held_out_seeds)
    if overlap:
        raise ValueError(
            "qualification and held-out evaluation seeds overlap: "
            + ", ".join(str(seed) for seed in sorted(overlap))
        )
    evidence_hashes = tuple(
        sorted({_sha256_file(path.expanduser().resolve()) for path in evidence_paths})
    )
    if not evidence_hashes:
        raise ValueError("hard qualification requires hashed evidence artifacts")
    qualification_kind: Literal["execution", "strategic"] = (
        "execution"
        if parent.category
        in {
            PlaybookRuleCategory.ENGINE_INVARIANT,
            PlaybookRuleCategory.EXECUTION_GUARD,
            PlaybookRuleCategory.TACTICAL_RESPONSE,
        }
        else "strategic"
    )
    if qualification_kind == "strategic" and strategic_ab is None:
        raise ValueError("strategic hard qualification requires paired outcome evidence")
    canonical_payload = {
        "parent_rule_id": parent.rule_id,
        "parent_canonical_key": parent.canonical_key,
        "effect": effect.value,
        "expected_git_sha": expected_git_sha,
        "sc2_patch": sc2_patch,
        "evidence_hashes": evidence_hashes,
        "qualification_seed_ids": qualification_seeds,
        "evaluation_seed_ids": held_out_seeds,
    }
    canonical_key = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    candidate = parent.model_copy(
        update={
            "schema_version": "2.1",
            "rule_id": f"playbook-rule:{canonical_key}",
            "canonical_key": canonical_key,
            "effect": effect,
            "strength": PlaybookRuleStrength.SOFT,
            "code_revision": expected_git_sha,
            "sc2_patch": sc2_patch,
            "parent_rule_id": parent.rule_id,
            "evidence_hashes": evidence_hashes,
            "qualified_at_git_sha": expected_git_sha,
            "qualified_at_sc2_patch": sc2_patch,
            "qualification_seed_ids": qualification_seeds,
            "evaluation_seed_ids": held_out_seeds,
            "qualification_kind": qualification_kind,
            "evidence": {
                **parent.evidence,
                "qualification": canonical_payload,
            },
        }
    )
    qualified = PlaybookRuleLifecycle().promote_to_hard(
        candidate,
        current_code_revision=expected_git_sha,
        current_sc2_patch=sc2_patch,
        strategic_ab=strategic_ab,
    )
    return store.upsert_rule(qualified)


def create_canary_fixture(
    database_path: Path,
    *,
    expected_git_sha: str,
    sc2_patch: str,
) -> PlaybookRule:
    """Create an isolated one-rule baseline for bounded canary infrastructure tests."""

    resolved = database_path.expanduser().resolve()
    if resolved.exists():
        raise ValueError(f"canary fixture database already exists: {resolved}")
    canonical_payload = {
        "fixture_version": "counterfactual-canary-v1",
        "action_name": "Build_Pylon_Screen",
        "context": {
            "agent_race": "protoss",
            "opponent_race": "zerg",
            "phase": "early",
            "map_name": "Simple64",
        },
    }
    canonical_key = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rule = PlaybookRule(
        rule_id=f"playbook-rule:{canonical_key}",
        canonical_key=canonical_key,
        category=PlaybookRuleCategory.EXECUTION_GUARD,
        conditions=(
            PlaybookCondition(field="agent_race", value="protoss"),
            PlaybookCondition(field="opponent_race", value="zerg"),
            PlaybookCondition(field="phase", value="early"),
            PlaybookCondition(field="map_name", value="Simple64"),
        ),
        effect=PlaybookRuleEffect.FORBID,
        strength=PlaybookRuleStrength.HARD,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("Build_Pylon_Screen",),
        role_ids=("economy",),
        confidence=1.0,
        code_revision=expected_git_sha,
        sc2_patch=sc2_patch,
        qualified_at_git_sha=expected_git_sha,
        qualified_at_sc2_patch=sc2_patch,
        qualification_kind="execution",
        evidence={
            "canary_fixture": True,
            "fixture_version": "counterfactual-canary-v1",
            "one_shot": True,
        },
    )
    store = PlaybookStore(resolved)
    try:
        return store.upsert_rule(rule)
    finally:
        store.close()


def contains_canary_fixture(rules: Iterable[PlaybookRule]) -> bool:
    return any(rule.evidence.get("canary_fixture") is True for rule in rules)


def _hard_rejection_reasons(
    rule: PlaybookRule,
    *,
    expected_git_sha: str,
    sc2_patch: str,
    context: dict[str, str],
    reachable_actions: set[str],
    evaluation_seed_ids: tuple[int, ...],
    allow_canary_fixture: bool,
    rules_by_id: dict[str, PlaybookRule],
) -> tuple[str, ...]:
    fixture = rule.evidence.get("canary_fixture") is True
    reasons: list[str] = []
    if rule.status is not PlaybookRuleStatus.ACTIVE:
        reasons.append(f"status_is_{rule.status.value}")
    if rule.strength is not PlaybookRuleStrength.HARD:
        reasons.append(f"strength_is_{rule.strength.value}")
    if rule.effect is not PlaybookRuleEffect.FORBID:
        reasons.append(f"effect_is_{rule.effect.value}_not_forbid")
    if not _context_is_applicable(rule, context):
        reasons.append("context_not_applicable")
    if not _rule_target_is_reachable(rule, reachable_actions):
        reasons.append("action_or_role_unreachable")
    if fixture and not allow_canary_fixture:
        reasons.append("canary_fixture_not_allowed")
    if not (fixture and allow_canary_fixture):
        if rule.code_revision is None:
            reasons.append("missing_code_revision")
        elif rule.code_revision != expected_git_sha:
            reasons.append("code_revision_mismatch")
        if rule.sc2_patch is None:
            reasons.append("missing_sc2_patch")
        elif rule.sc2_patch != sc2_patch:
            reasons.append("sc2_patch_mismatch")
        if rule.qualified_at_git_sha is None:
            reasons.append("missing_qualified_git_sha")
        elif rule.qualified_at_git_sha != expected_git_sha:
            reasons.append("qualified_git_sha_mismatch")
        if rule.qualified_at_sc2_patch is None:
            reasons.append("missing_qualified_sc2_patch")
        elif rule.qualified_at_sc2_patch != sc2_patch:
            reasons.append("qualified_sc2_patch_mismatch")
        parent = None if rule.parent_rule_id is None else rules_by_id.get(rule.parent_rule_id)
        if rule.parent_rule_id is None:
            reasons.append("missing_parent_rule")
        elif parent is None:
            reasons.append("parent_rule_not_found")
        elif not (
            parent.status is PlaybookRuleStatus.ACTIVE
            and parent.strength is PlaybookRuleStrength.SOFT
            and parent.effect is PlaybookRuleEffect.AVOID
        ):
            reasons.append("parent_is_not_active_soft_avoid")
        qualification_seeds = set(rule.qualification_seed_ids or rule.source_seeds)
        uncensored_seeds = qualification_seeds - set(rule.censored_source_seeds)
        uncensored_runs = set(rule.source_run_ids) - set(rule.censored_source_run_ids)
        if len(uncensored_seeds) < 3:
            reasons.append("insufficient_source_seeds")
        if len(uncensored_runs) < 3:
            reasons.append("insufficient_source_runs")
        if rule.confidence < 0.9:
            reasons.append("confidence_below_0_9")
        if rule.contradiction_count:
            reasons.append("has_contradictions")
        if rule.shadow_state_count < 48:
            reasons.append("insufficient_shadow_coverage")
        if rule.false_block_rate > 0.01:
            reasons.append("false_block_rate_above_1_percent")
        if not rule.evidence_hashes:
            reasons.append("missing_evidence_hashes")
        elif any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in rule.evidence_hashes
        ):
            reasons.append("invalid_evidence_hash")
        if not evaluation_seed_ids:
            reasons.append("missing_evaluation_seed_set")
        elif set(rule.evaluation_seed_ids) != set(evaluation_seed_ids):
            reasons.append("evaluation_seed_set_mismatch")
        overlap = qualification_seeds & set(evaluation_seed_ids)
        if overlap:
            reasons.append("qualification_evaluation_seed_overlap")
    return tuple(reasons)


def _context_is_applicable(rule: PlaybookRule, context: dict[str, str]) -> bool:
    return all(
        condition.field not in _STATIC_CONTEXT_FIELDS
        or _condition_matches_static(condition, context[condition.field])
        for condition in rule.conditions
    )


def _condition_matches_static(condition: PlaybookCondition, actual: str) -> bool:
    expected = condition.value
    if condition.operator is PlaybookConditionOperator.EQ:
        return str(expected).casefold() == actual.casefold()
    if condition.operator is PlaybookConditionOperator.IN:
        values = expected if isinstance(expected, tuple) else (str(expected),)
        return actual.casefold() in {str(value).casefold() for value in values}
    if condition.operator is PlaybookConditionOperator.CONTAINS:
        return str(expected).casefold() in actual.casefold()
    return True


def _rule_target_is_reachable(rule: PlaybookRule, reachable_actions: set[str]) -> bool:
    if not rule.action_names and not rule.role_ids:
        return False
    action_reachable = not rule.action_names or any(
        action in reachable_actions for action in rule.action_names
    )
    role_reachable = not rule.role_ids or any(role in _VALID_ROLES for role in rule.role_ids)
    return action_reachable and role_reachable


def _qualified_effect(effect: PlaybookRuleEffect) -> PlaybookRuleEffect:
    if effect is PlaybookRuleEffect.AVOID:
        return PlaybookRuleEffect.FORBID
    raise ValueError(
        f"hard blocking qualification requires an active soft avoid rule; got {effect.value!r}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
