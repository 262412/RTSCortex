"""Role intents and deterministic strategic arbitration."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable, Sequence
from enum import StrEnum

from pydantic import Field, model_validator

from rtscortex.contracts import ObservationEnvelope
from rtscortex.contracts.models import ContractModel
from rtscortex.cortex.models import CortexIntent, MacroIntent, ReflexIntent, TacticalIntent
from rtscortex.races import ActionDomain, RaceProfile


class RoleId(StrEnum):
    ECONOMY = "economy"
    TECHNOLOGY = "technology"
    PRODUCTION = "production"
    DEFENSE = "defense"
    OFFENSE = "offense"
    FOCUS_FIRE = "focus_fire"
    RETREAT = "retreat"


class IntentDecisionStatus(StrEnum):
    SELECTED = "selected"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    PREEMPTED = "preempted"


class IntentConflictKind(StrEnum):
    ROLE = "role"
    ACTOR = "actor"
    PRODUCER = "producer"
    RESOURCE = "resource"
    OBJECTIVE = "objective"
    DEPENDENCY = "dependency"
    COMMITMENT = "commitment"


class ResourceClaim(ContractModel):
    minerals: int = Field(default=0, ge=0)
    vespene: int = Field(default=0, ge=0)
    supply: int = Field(default=0, ge=0)
    reservation_game_loops: int = Field(default=1, ge=1)


class StrategicIntent(ContractModel):
    schema_version: str = "2.0"
    intent_id: str = Field(min_length=1)
    continuity_key: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_id: int = Field(ge=0)
    created_game_loop: int = Field(ge=0)
    role: RoleId
    objective: str = Field(min_length=1)
    desired_effect: str = Field(min_length=1)
    action_names: tuple[str, ...] = Field(min_length=1)
    actor_scopes: tuple[str, ...] = ()
    producer_types: tuple[str, ...] = ()
    resource_claim: ResourceClaim = Field(default_factory=ResourceClaim)
    dependency_intent_ids: tuple[str, ...] = ()
    mutually_exclusive_groups: tuple[str, ...] = ()
    hard_blockers: tuple[str, ...] = ()
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    strategic_alignment: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    expected_progress: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: int = Field(default=50, ge=0, le=100)
    emergency: bool = False
    horizon_game_loops: int = Field(default=112, ge=1)
    ttl_game_loops: int = Field(default=112, ge=1)
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_intent_id: str | None = None
    situation_assessment_id: str | None = None
    race_brain_plan_id: str | None = None
    playbook_rule_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_unique_fields(self) -> StrategicIntent:
        for name, values in (
            ("action_names", self.action_names),
            ("actor_scopes", self.actor_scopes),
            ("producer_types", self.producer_types),
            ("dependency_intent_ids", self.dependency_intent_ids),
            ("mutually_exclusive_groups", self.mutually_exclusive_groups),
            ("playbook_rule_ids", self.playbook_rule_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


class IntentConflict(ContractModel):
    conflict_id: str = Field(min_length=1)
    kind: IntentConflictKind
    intent_ids: tuple[str, ...] = Field(min_length=1)
    detail: str = Field(min_length=1)


class IntentScore(ContractModel):
    expected_progress: float
    urgency: float
    strategic_alignment: float
    confidence: float
    playbook_delta: float = 0.0
    risk_penalty: float
    switch_penalty: float
    resource_pressure_penalty: float
    total: float


class IntentDecision(ContractModel):
    intent_id: str = Field(min_length=1)
    status: IntentDecisionStatus
    reason_code: str = Field(min_length=1)
    score: IntentScore
    conflict_ids: tuple[str, ...] = ()
    playbook_rule_ids: tuple[str, ...] = ()
    reserved_resources: ResourceClaim = Field(default_factory=ResourceClaim)
    commitment_until_game_loop: int | None = Field(default=None, ge=0)


class StrategicAgenda(ContractModel):
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    game_loop: int = Field(ge=0)
    active_intent_ids: tuple[str, ...] = ()
    active_continuity_keys: tuple[str, ...] = ()
    reserved_resources: ResourceClaim = Field(default_factory=ResourceClaim)
    commitment_until_game_loop: int = Field(default=0, ge=0)
    total_score: float = 0.0


class StrategicArbitration(ContractModel):
    decisions: tuple[IntentDecision, ...]
    selected_intent_ids: tuple[str, ...]
    conflicts: tuple[IntentConflict, ...]
    agenda: StrategicAgenda

    @model_validator(mode="after")
    def validate_conservation(self) -> StrategicArbitration:
        decision_ids = [decision.intent_id for decision in self.decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("each strategic intent requires exactly one decision")
        selected = {
            decision.intent_id
            for decision in self.decisions
            if decision.status is IntentDecisionStatus.SELECTED
        }
        if selected != set(self.selected_intent_ids):
            raise ValueError("selected intent IDs must match selected decisions")
        return self


class StrategicIntentAdapter:
    """Project the current Cortex roles into the v2 intent contract."""

    adapter_id = "cortex-intent-adapter"
    adapter_version = "2.0.0"

    def __init__(self, profile: RaceProfile) -> None:
        self.profile = profile
        self._specs = {spec.name: spec for spec in profile.data.progress_action_specs}

    def adapt(self, intent: CortexIntent) -> StrategicIntent:
        role = self._role(intent)
        first_action = intent.action_names[0]
        spec = self._specs.get(first_action)
        commitment = _default_commitment(role)
        identity = hashlib.sha256(
            f"{intent.intent_id}|{role.value}|strategic-v2".encode()
        ).hexdigest()
        return StrategicIntent(
            intent_id=f"strategic:{identity}",
            continuity_key="|".join(
                (role.value, first_action, intent.target.kind.value, intent.target.region or "")
            ),
            run_id=intent.run_id,
            episode_id=intent.episode_id,
            step_id=intent.step_id,
            created_game_loop=intent.created_game_loop,
            role=role,
            objective=intent.objective,
            desired_effect=intent.objective,
            action_names=tuple(intent.action_names),
            actor_scopes=tuple(intent.actor_scopes),
            producer_types=self.profile.data.producers_for_action(first_action),
            resource_claim=ResourceClaim(
                minerals=0 if spec is None else spec.minerals,
                vespene=0 if spec is None else spec.vespene,
                supply=0 if spec is None else spec.supply,
                reservation_game_loops=commitment,
            ),
            urgency=_urgency(role),
            strategic_alignment=1.0 if isinstance(intent, MacroIntent) else 0.8,
            confidence=1.0,
            expected_progress=1.0 if isinstance(intent, MacroIntent) else 0.8,
            risk=0.25 if role is RoleId.OFFENSE else 0.05,
            priority=intent.priority,
            # A planned static-defense building belongs to Defense, but is not an
            # emergency by itself.  Only an actual reflex/retreat signal may bypass
            # commitments and the normal switch margin.
            emergency=(
                role is RoleId.RETREAT
                or isinstance(intent, ReflexIntent)
                and first_action
                not in {
                    "Effect_InjectLarva",
                    "Build_CreepTumor_Queen_Screen",
                    "Build_CreepTumor_Tumor_Screen",
                    "Train_SCV",
                    "Morph_OrbitalCommand",
                    "Effect_CalldownMULE_Screen",
                }
            ),
            horizon_game_loops=commitment,
            ttl_game_loops=intent.ttl_game_loops,
            source_id=intent.source_id,
            source_version=intent.source_version,
            source_intent_id=intent.intent_id,
            situation_assessment_id=intent.situation_assessment_id,
            race_brain_plan_id=(intent.macro_plan_id if isinstance(intent, MacroIntent) else None),
        )

    def _role(self, intent: CortexIntent) -> RoleId:
        if isinstance(intent, MacroIntent):
            domain = self.profile.domain_for_action(intent.action_names[0])
            return _role_for_domain(domain)
        if isinstance(intent, ReflexIntent):
            return RoleId.RETREAT if "retreat" in intent.objective.casefold() else RoleId.DEFENSE
        if isinstance(intent, TacticalIntent):
            if "retreat" in intent.objective.casefold():
                return RoleId.RETREAT
            if intent.action_names[0] == "Attack_Unit":
                return RoleId.FOCUS_FIRE
            return RoleId.OFFENSE
        return RoleId.PRODUCTION


class IntentArbiter:
    """Select a deterministic feasible set before commands are materialized."""

    arbiter_id = "deterministic-strategic-intent-arbiter"
    arbiter_version = "2.0.0"

    def __init__(self, *, switch_margin: float = 1.0, max_intents: int = 7) -> None:
        if switch_margin < 0:
            raise ValueError("switch_margin cannot be negative")
        if max_intents < 1:
            raise ValueError("max_intents must be positive")
        self.switch_margin = switch_margin
        self.max_intents = max_intents

    def arbitrate(
        self,
        intents: Sequence[StrategicIntent],
        observation: ObservationEnvelope,
        *,
        previous_agenda: StrategicAgenda | None = None,
        playbook_deltas: dict[str, float] | None = None,
        playbook_rule_ids: dict[str, tuple[str, ...]] | None = None,
    ) -> StrategicArbitration:
        _validate_intent_batch(intents, observation)
        deltas = playbook_deltas or {}
        rules = playbook_rule_ids or {}
        conflicts = _build_conflicts(intents)
        rejected: dict[str, tuple[IntentDecisionStatus, str]] = {}
        eligible: list[StrategicIntent] = []
        role_winners: dict[RoleId, StrategicIntent] = {}
        for intent in sorted(intents, key=lambda item: (-item.priority, item.intent_id)):
            if observation.game_loop - intent.created_game_loop >= intent.ttl_game_loops:
                rejected[intent.intent_id] = (IntentDecisionStatus.REJECTED, "intent_expired")
                continue
            if intent.hard_blockers:
                rejected[intent.intent_id] = (
                    IntentDecisionStatus.REJECTED,
                    "hard_precondition_failed",
                )
                continue
            if intent.role in role_winners:
                rejected[intent.intent_id] = (
                    IntentDecisionStatus.DEFERRED,
                    "lower_priority_same_role",
                )
                continue
            role_winners[intent.role] = intent
            eligible.append(intent)
        eligible = eligible[: self.max_intents]
        scores = {
            intent.intent_id: _score(
                intent,
                observation,
                previous_agenda=previous_agenda,
                playbook_delta=deltas.get(intent.intent_id, 0.0),
            )
            for intent in intents
        }
        selected = _best_feasible_subset(eligible, observation, scores, conflicts)
        selected_ids = {intent.intent_id for intent in selected}
        selected_score = sum(scores[intent.intent_id].total for intent in selected)

        if (
            previous_agenda is not None
            and observation.game_loop < previous_agenda.commitment_until_game_loop
        ):
            retained = [
                intent
                for intent in eligible
                if intent.continuity_key in previous_agenda.active_continuity_keys
            ]
            if retained and _subset_is_feasible(retained, observation, conflicts):
                retained_score = sum(scores[intent.intent_id].total for intent in retained)
                if selected_score < retained_score + self.switch_margin and not any(
                    intent.emergency for intent in selected if intent not in retained
                ):
                    selected = retained
                    selected_ids = {intent.intent_id for intent in selected}
                    selected_score = retained_score

        total_claim = _sum_claims(intent.resource_claim for intent in selected)
        commitment_until = max(
            (
                observation.game_loop + intent.resource_claim.reservation_game_loops
                for intent in selected
            ),
            default=observation.game_loop,
        )
        agenda = StrategicAgenda(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            game_loop=observation.game_loop,
            active_intent_ids=tuple(sorted(selected_ids)),
            active_continuity_keys=tuple(sorted(intent.continuity_key for intent in selected)),
            reserved_resources=total_claim,
            commitment_until_game_loop=commitment_until,
            total_score=selected_score,
        )
        conflicts_by_intent = _conflicts_by_intent(conflicts)
        decisions: list[IntentDecision] = []
        for intent in intents:
            explicit = rejected.get(intent.intent_id)
            if intent.intent_id in selected_ids:
                status = IntentDecisionStatus.SELECTED
                reason = "selected_feasible_set"
                reserved = intent.resource_claim
                committed = observation.game_loop + intent.resource_claim.reservation_game_loops
            elif explicit is not None:
                status, reason = explicit
                reserved = ResourceClaim()
                committed = None
            else:
                status = (
                    IntentDecisionStatus.PREEMPTED
                    if previous_agenda is not None
                    and intent.continuity_key in previous_agenda.active_continuity_keys
                    and any(candidate.emergency for candidate in selected)
                    else IntentDecisionStatus.DEFERRED
                )
                reason = (
                    "emergency_preemption"
                    if status is IntentDecisionStatus.PREEMPTED
                    else "not_in_best_feasible_set"
                )
                reserved = ResourceClaim()
                committed = None
            decisions.append(
                IntentDecision(
                    intent_id=intent.intent_id,
                    status=status,
                    reason_code=reason,
                    score=scores[intent.intent_id],
                    conflict_ids=tuple(
                        conflict.conflict_id
                        for conflict in conflicts_by_intent.get(intent.intent_id, ())
                    ),
                    playbook_rule_ids=rules.get(intent.intent_id, ()),
                    reserved_resources=reserved,
                    commitment_until_game_loop=committed,
                )
            )
        return StrategicArbitration(
            decisions=tuple(decisions),
            selected_intent_ids=tuple(sorted(selected_ids)),
            conflicts=tuple(conflicts),
            agenda=agenda,
        )


def _role_for_domain(domain: ActionDomain | None) -> RoleId:
    if domain is None:
        return RoleId.PRODUCTION
    return RoleId(domain.value)


def _default_commitment(role: RoleId) -> int:
    if role in {RoleId.RETREAT, RoleId.FOCUS_FIRE}:
        return 8
    if role in {RoleId.DEFENSE, RoleId.OFFENSE}:
        return 16
    return 112


def _urgency(role: RoleId) -> float:
    if role is RoleId.RETREAT:
        return 1.0
    if role is RoleId.DEFENSE:
        return 0.9
    if role is RoleId.FOCUS_FIRE:
        return 0.8
    return 0.5


def _validate_intent_batch(
    intents: Sequence[StrategicIntent],
    observation: ObservationEnvelope,
) -> None:
    identities = [intent.intent_id for intent in intents]
    if len(identities) != len(set(identities)):
        raise ValueError("strategic intent IDs must be unique")
    expected = (observation.run_id, observation.episode_id, observation.step_id)
    for intent in intents:
        if (intent.run_id, intent.episode_id, intent.step_id) != expected:
            raise ValueError("strategic intent does not match the observation")


def _build_conflicts(intents: Sequence[StrategicIntent]) -> list[IntentConflict]:
    conflicts: list[IntentConflict] = []
    for left, right in itertools.combinations(intents, 2):
        kind: IntentConflictKind | None = None
        detail = ""
        if left.role is right.role:
            kind = IntentConflictKind.ROLE
            detail = f"both intents own role {left.role.value}"
        elif set(left.actor_scopes).intersection(right.actor_scopes):
            kind = IntentConflictKind.ACTOR
            detail = "actor scopes overlap"
        elif set(left.producer_types).intersection(right.producer_types):
            kind = IntentConflictKind.PRODUCER
            detail = "producer claims overlap"
        elif set(left.mutually_exclusive_groups).intersection(right.mutually_exclusive_groups):
            kind = IntentConflictKind.OBJECTIVE
            detail = "strategic objective groups are mutually exclusive"
        if kind is None:
            continue
        pair = tuple(sorted((left.intent_id, right.intent_id)))
        digest = hashlib.sha256(f"{kind.value}|{'|'.join(pair)}".encode()).hexdigest()
        conflicts.append(
            IntentConflict(
                conflict_id=f"conflict:{digest}",
                kind=kind,
                intent_ids=pair,
                detail=detail,
            )
        )
    return conflicts


def _score(
    intent: StrategicIntent,
    observation: ObservationEnvelope,
    *,
    previous_agenda: StrategicAgenda | None,
    playbook_delta: float,
) -> IntentScore:
    economy = observation.state.economy
    resource_pressure = max(
        intent.resource_claim.minerals / max(economy.minerals, 1),
        intent.resource_claim.vespene / max(economy.vespene, 1),
        intent.resource_claim.supply / max(economy.supply_cap - economy.supply_used, 1),
    )
    resource_pressure = min(resource_pressure, 1.0)
    switched = (
        previous_agenda is not None
        and bool(previous_agenda.active_continuity_keys)
        and intent.continuity_key not in previous_agenda.active_continuity_keys
    )
    switch_penalty = 1.0 if switched and not intent.emergency else 0.0
    progress = 4.0 * intent.expected_progress
    urgency = 3.0 * intent.urgency
    alignment = 2.0 * intent.strategic_alignment
    confidence = intent.confidence
    risk = 3.0 * intent.risk
    switch = 2.0 * switch_penalty
    resource = resource_pressure
    total = progress + urgency + alignment + confidence + playbook_delta - risk - switch - resource
    return IntentScore(
        expected_progress=progress,
        urgency=urgency,
        strategic_alignment=alignment,
        confidence=confidence,
        playbook_delta=playbook_delta,
        risk_penalty=risk,
        switch_penalty=switch,
        resource_pressure_penalty=resource,
        total=total,
    )


def _best_feasible_subset(
    intents: Sequence[StrategicIntent],
    observation: ObservationEnvelope,
    scores: dict[str, IntentScore],
    conflicts: Sequence[IntentConflict],
) -> list[StrategicIntent]:
    candidates: list[tuple[tuple[int, float, int, tuple[str, ...]], list[StrategicIntent]]] = []
    for count in range(len(intents) + 1):
        for subset_tuple in itertools.combinations(intents, count):
            subset = list(subset_tuple)
            if not _subset_is_feasible(subset, observation, conflicts):
                continue
            key = (
                sum(intent.emergency for intent in subset),
                sum(scores[intent.intent_id].total for intent in subset),
                len(subset),
                tuple(sorted(intent.intent_id for intent in subset)),
            )
            candidates.append((key, subset))
    return max(candidates, key=lambda item: item[0], default=((0, 0.0, 0, ()), []))[1]


def _subset_is_feasible(
    intents: Sequence[StrategicIntent],
    observation: ObservationEnvelope,
    conflicts: Sequence[IntentConflict],
) -> bool:
    selected_ids = {intent.intent_id for intent in intents}
    if any(set(conflict.intent_ids).issubset(selected_ids) for conflict in conflicts):
        return False
    if any(not set(intent.dependency_intent_ids).issubset(selected_ids) for intent in intents):
        return False
    claim = _sum_claims(intent.resource_claim for intent in intents)
    economy = observation.state.economy
    return (
        claim.minerals <= economy.minerals
        and claim.vespene <= economy.vespene
        and claim.supply <= max(0, economy.supply_cap - economy.supply_used)
    )


def _sum_claims(claims: Iterable[ResourceClaim]) -> ResourceClaim:
    values = list(claims)
    return ResourceClaim(
        minerals=sum(claim.minerals for claim in values),
        vespene=sum(claim.vespene for claim in values),
        supply=sum(claim.supply for claim in values),
        reservation_game_loops=max(
            (claim.reservation_game_loops for claim in values),
            default=1,
        ),
    )


def _conflicts_by_intent(
    conflicts: Sequence[IntentConflict],
) -> dict[str, tuple[IntentConflict, ...]]:
    mapping: dict[str, list[IntentConflict]] = {}
    for conflict in conflicts:
        for intent_id in conflict.intent_ids:
            mapping.setdefault(intent_id, []).append(conflict)
    return {intent_id: tuple(values) for intent_id, values in mapping.items()}
