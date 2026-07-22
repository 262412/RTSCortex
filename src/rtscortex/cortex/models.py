"""Typed internal contracts for the SC2-native Cortex runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from rtscortex.contracts import CommandLifecycleSnapshot, ObservationEnvelope
from rtscortex.contracts.models import ContractModel
from rtscortex.game_phase import GamePhase as GamePhase
from rtscortex.progress import GoalProgressReport


class CortexRole(StrEnum):
    """A single-owner role in the Cortex decision hierarchy."""

    SITUATION = "situation"
    MACRO = "macro"
    TACTICAL = "tactical"
    REFLEX = "reflex"
    EXECUTOR = "executor"


class ThreatLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"
    CRITICAL = "critical"


class EconomyStatus(StrEnum):
    CONSTRAINED = "constrained"
    STABLE = "stable"
    FLOATING = "floating"


class ArmyReadiness(StrEnum):
    EMPTY = "empty"
    FORMING = "forming"
    READY = "ready"
    ENGAGED = "engaged"


class KnowledgeStatus(StrEnum):
    """How strongly one Situation v2 fact is supported by observations."""

    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class SituationFact(ContractModel):
    """One auditable fact or inference used by the Cortex decision hierarchy."""

    name: str = Field(min_length=1)
    status: KnowledgeStatus
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()


class ForceComposition(ContractModel):
    counts: dict[str, int] = Field(default_factory=dict)
    total_units: int = Field(default=0, ge=0)
    ground_units: int = Field(default=0, ge=0)
    air_units: int = Field(default=0, ge=0)
    estimated_resource_value: int = Field(default=0, ge=0)
    unknown_unit_types: tuple[str, ...] = ()


class BaseAssessment(ContractModel):
    own_base_count: int = Field(default=0, ge=0)
    visible_enemy_base_count: int = Field(default=0, ge=0)
    own_production_capacity: int = Field(default=0, ge=0)
    visible_enemy_production_capacity: int = Field(default=0, ge=0)


class SpatialAssessment(ContractModel):
    nearest_threat_distance: float | None = Field(default=None, ge=0.0)
    threat_eta_seconds: float | None = Field(default=None, ge=0.0)
    visible_enemy_regions: tuple[str, ...] = ()
    map_control_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


class ScoutingAssessment(ContractModel):
    enemy_visible: bool = False
    last_enemy_seen_game_loop: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confirmed_enemy_tech: tuple[str, ...] = ()
    inferred_enemy_tech: tuple[str, ...] = ()
    possible_transitions: tuple[str, ...] = ()


class SituationAssessment(ContractModel):
    """Compact game-state interpretation with explicit source provenance."""

    assessment_id: str = Field(min_length=1)
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    valid_until_game_loop: int = Field(ge=0)
    phase: GamePhase
    threat_level: ThreatLevel
    economy_status: EconomyStatus
    army_readiness: ArmyReadiness
    threats: list[str] = Field(default_factory=list)
    information_gaps: list[str] = Field(default_factory=list)
    own_force: ForceComposition = Field(default_factory=ForceComposition)
    visible_enemy_force: ForceComposition = Field(default_factory=ForceComposition)
    bases: BaseAssessment = Field(default_factory=BaseAssessment)
    spatial: SpatialAssessment = Field(default_factory=SpatialAssessment)
    scouting: ScoutingAssessment = Field(default_factory=ScoutingAssessment)
    facts: list[SituationFact] = Field(default_factory=list)
    source_kind: Literal["deterministic", "model"]
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lifetime(self) -> SituationAssessment:
        if self.valid_until_game_loop < self.game_loop:
            raise ValueError("assessment cannot expire before its source observation")
        return self


class MacroStepStatus(StrEnum):
    PENDING = "pending"
    DEFERRED = "deferred"
    DISPATCHED = "dispatched"
    CONFIRMED = "confirmed"
    BLOCKED = "blocked"
    OBSOLETE = "obsolete"


class MacroStep(ContractModel):
    """One ordered semantic step retained from a specialist macro policy."""

    ordinal: int = Field(ge=0)
    semantic_action: str = Field(min_length=1)
    runtime_actions: list[str] = Field(default_factory=list)
    repeat: int = Field(default=1, ge=1)
    completed_repeats: int = Field(default=0, ge=0)
    status: MacroStepStatus = MacroStepStatus.PENDING
    reason: str | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> MacroStep:
        if self.completed_repeats > self.repeat:
            raise ValueError("completed_repeats cannot exceed repeat")
        if len(self.runtime_actions) != len(set(self.runtime_actions)):
            raise ValueError("runtime_actions must be unique")
        return self


class MacroPlan(ContractModel):
    """Versioned, ordered output from an SC2-specialized macro policy."""

    plan_id: str = Field(min_length=1)
    run_id: str
    episode_id: str
    source_step_id: int = Field(ge=0)
    created_game_loop: int = Field(ge=0)
    expires_game_loop: int = Field(ge=0)
    strategic_objective: str = Field(min_length=1)
    steps: list[MacroStep] = Field(default_factory=list)
    source_model_id: str = Field(min_length=1)
    source_model_revision: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    vocabulary_version: str = Field(min_length=1)
    raw_proposal: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_plan(self) -> MacroPlan:
        if self.expires_game_loop <= self.created_game_loop:
            raise ValueError("macro plan must expire after it is created")
        ordinals = [step.ordinal for step in self.steps]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("macro plan step ordinals must be unique")
        return self


class IntentTargetKind(StrEnum):
    """Semantic targets only; concrete tags and coordinates live in candidates."""

    NONE = "none"
    ENEMY = "enemy"
    DEFENSIVE_REGION = "defensive_region"
    RETREAT_REGION = "retreat_region"
    EXPANSION = "expansion"
    GEYSER = "geyser"
    PRODUCTION = "production"


class IntentTarget(ContractModel):
    kind: IntentTargetKind = IntentTargetKind.NONE
    unit_tag: str | None = None
    unit_type: str | None = None
    structure_type: str | None = None
    region: str | None = None
    position: tuple[int, int] | None = None


class _IntentBase(ContractModel):
    intent_id: str = Field(min_length=1)
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    created_game_loop: int = Field(ge=0)
    objective: str = Field(min_length=1)
    action_names: list[str] = Field(min_length=1)
    actor_scopes: list[str] = Field(default_factory=list)
    target: IntentTarget = Field(default_factory=IntentTarget)
    priority: int = Field(default=50, ge=0, le=100)
    ttl_game_loops: int = Field(default=1, ge=1)
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    situation_assessment_id: str | None = None

    @model_validator(mode="after")
    def validate_unique_constraints(self) -> _IntentBase:
        if len(self.action_names) != len(set(self.action_names)):
            raise ValueError("intent action_names must be unique")
        if len(self.actor_scopes) != len(set(self.actor_scopes)):
            raise ValueError("intent actor_scopes must be unique")
        return self


class MacroIntent(_IntentBase):
    intent_kind: Literal["macro"] = "macro"
    source_role: Literal[CortexRole.MACRO] = CortexRole.MACRO
    macro_plan_id: str = Field(min_length=1)


class TacticalIntent(_IntentBase):
    intent_kind: Literal["tactical"] = "tactical"
    source_role: Literal[CortexRole.TACTICAL] = CortexRole.TACTICAL


class ReflexIntent(_IntentBase):
    intent_kind: Literal["reflex"] = "reflex"
    source_role: Literal[CortexRole.REFLEX] = CortexRole.REFLEX


CortexIntent: TypeAlias = Annotated[
    MacroIntent | TacticalIntent | ReflexIntent,
    Field(discriminator="intent_kind"),
]


class CandidateFeatures(ContractModel):
    """Stable, model-ready attributes without any hidden executable values."""

    action_rank: int = Field(ge=0)
    actor_rank: int = Field(ge=0)
    argument_rank: int = Field(ge=0)
    compile_ordinal: int = Field(ge=0)
    advances_goal: bool = False
    playbook_score: float = 0.0


class ExecutableCandidate(ContractModel):
    """One observation-bound action already present in the legal candidate domain."""

    candidate_id: str = Field(pattern=r"^candidate:[0-9a-f]{64}$")
    observation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent_id: str = Field(min_length=1)
    action_name: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    arguments: list[Any] = Field(default_factory=list)
    features: CandidateFeatures


class FastExecutorContext(ContractModel):
    """All information available to the fast, candidate-selecting motor policy."""

    observation: ObservationEnvelope
    intent: CortexIntent
    goal_progress: GoalProgressReport | None = None
    busy_actors: list[str] = Field(default_factory=list)
    recent_commands: list[CommandLifecycleSnapshot] = Field(default_factory=list)
    candidates: list[ExecutableCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_context_identity(self) -> FastExecutorContext:
        observation = self.observation
        intent = self.intent
        if (intent.run_id, intent.episode_id, intent.step_id, intent.created_game_loop) != (
            observation.run_id,
            observation.episode_id,
            observation.step_id,
            observation.game_loop,
        ):
            raise ValueError("intent must be bound to the context observation")
        if len(self.busy_actors) != len(set(self.busy_actors)):
            raise ValueError("busy_actors must be unique")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        if any(candidate.intent_id != intent.intent_id for candidate in self.candidates):
            raise ValueError("every candidate must belong to the context intent")
        report = self.goal_progress
        if report is not None and (
            report.run_id,
            report.episode_id,
            report.step_id,
            report.game_loop,
        ) != (
            observation.run_id,
            observation.episode_id,
            observation.step_id,
            observation.game_loop,
        ):
            raise ValueError("goal progress must describe the context observation")
        return self


class CandidateSelectionStatus(StrEnum):
    SELECTED = "selected"
    ABSTAINED = "abstained"


class CandidateSelection(ContractModel):
    selection_id: str = Field(pattern=r"^selection:[0-9a-f]{64}$")
    intent_id: str = Field(min_length=1)
    status: CandidateSelectionStatus
    candidate_id: str | None = Field(default=None, pattern=r"^candidate:[0-9a-f]{64}$")
    executor_id: str = Field(min_length=1)
    executor_version: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> CandidateSelection:
        if self.status is CandidateSelectionStatus.SELECTED and self.candidate_id is None:
            raise ValueError("selected outcomes require candidate_id")
        if self.status is CandidateSelectionStatus.ABSTAINED:
            if self.candidate_id is not None:
                raise ValueError("abstained outcomes cannot identify a candidate")
            if not self.fallback_reason:
                raise ValueError("abstained outcomes require fallback_reason")
        return self


class CommandLineage(ContractModel):
    """Trace one wire command back to its specialist intent and motor selection."""

    command_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    candidate_id: str = Field(pattern=r"^candidate:[0-9a-f]{64}$")
    selection_id: str = Field(pattern=r"^selection:[0-9a-f]{64}$")
    source_role: CortexRole
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    executor_id: str = Field(min_length=1)
    executor_version: str = Field(min_length=1)
    situation_assessment_id: str | None = None
    macro_plan_id: str | None = None
    strategic_intent_id: str | None = None
    responsibility: (
        Literal[
            "economy",
            "technology",
            "production",
            "defense",
            "offense",
            "focus_fire",
            "retreat",
        ]
        | None
    ) = None
    arbiter_mode: Literal["disabled", "shadow", "active"] = "disabled"
    intent_decision: Literal["selected", "deferred", "rejected", "preempted"] | None = None
    playbook_rule_ids: tuple[str, ...] = ()
    selected_game_loop: int = Field(ge=0)
