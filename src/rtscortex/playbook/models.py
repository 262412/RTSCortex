"""Typed records for the cross-episode Cortex tactical playbook."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from rtscortex.contracts.models import ContractModel
from rtscortex.game_phase import GamePhase


class DecisionQuality(StrEnum):
    ADVANTAGE_GAINED = "advantage_gained"
    CORRECT_EXECUTION = "correct_execution"
    STRATEGIC_ERROR = "strategic_error"
    EXECUTION_ERROR = "execution_error"
    INCONCLUSIVE = "inconclusive"


class StrategicConsequenceType(StrEnum):
    THREAT_UNANSWERED = "threat_unanswered"
    EXPANSION_DELAYED = "expansion_delayed"
    PRODUCTION_IMBALANCE = "production_imbalance"
    TIMING_ATTACK_FAILED = "timing_attack_failed"
    UNNECESSARY_RETREAT = "unnecessary_retreat"
    ADVANTAGE_NOT_CONVERTED = "advantage_not_converted"
    SUCCESSFUL_KEY_DECISION = "successful_key_decision"


class FailureOwner(StrEnum):
    CORTEX = "cortex"
    EXECUTOR = "executor"
    BRIDGE = "bridge"
    ENVIRONMENT = "environment"
    NONE = "none"
    UNKNOWN = "unknown"


class LessonStatus(StrEnum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    CONTRADICTED = "contradicted"
    RETIRED = "retired"


class PlaybookRuleKind(StrEnum):
    """Minimal rule split; finer tactical taxonomy is intentionally deferred."""

    STRATEGY = "strategy"
    EXECUTION_GUARD = "execution_guard"


class PlaybookRuleCategory(StrEnum):
    ENGINE_INVARIANT = "engine_invariant"
    EXECUTION_GUARD = "execution_guard"
    RACE_MACRO = "race_macro"
    MATCHUP_STRATEGY = "matchup_strategy"
    TACTICAL_RESPONSE = "tactical_response"
    MAP_SPECIFIC = "map_specific"


class PlaybookRuleEffect(StrEnum):
    REQUIRE = "require"
    FORBID = "forbid"
    PREFER = "prefer"
    AVOID = "avoid"


class PlaybookRuleStrength(StrEnum):
    ADVISORY = "advisory"
    SOFT = "soft"
    HARD = "hard"


class PlaybookRuleStatus(StrEnum):
    LEGACY = "legacy"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class PlaybookConditionOperator(StrEnum):
    EQ = "eq"
    IN = "in"
    CONTAINS = "contains"
    GTE = "gte"
    LTE = "lte"


PlaybookConditionValue: TypeAlias = str | int | float | bool | tuple[str, ...]
PlaybookRoleId: TypeAlias = Literal[
    "economy",
    "technology",
    "production",
    "defense",
    "offense",
    "focus_fire",
    "retreat",
]


class PlaybookCondition(ContractModel):
    field: Literal[
        "agent_race",
        "opponent_race",
        "phase",
        "map_name",
        "action_name",
        "role",
        "threat_level",
        "economy_status",
        "army_readiness",
        "alert",
    ]
    operator: PlaybookConditionOperator = PlaybookConditionOperator.EQ
    value: PlaybookConditionValue


class PlaybookRule(ContractModel):
    schema_version: str = "2.0"
    rule_id: str = Field(min_length=1)
    canonical_key: str = Field(min_length=1)
    category: PlaybookRuleCategory
    conditions: tuple[PlaybookCondition, ...]
    effect: PlaybookRuleEffect
    strength: PlaybookRuleStrength
    status: PlaybookRuleStatus
    action_names: tuple[str, ...] = ()
    role_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    support_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    source_case_ids: tuple[str, ...] = ()
    source_run_ids: tuple[str, ...] = ()
    source_seeds: tuple[int, ...] = ()
    censored_source_run_ids: tuple[str, ...] = ()
    censored_source_seeds: tuple[int, ...] = ()
    contradiction_seeds: tuple[int, ...] = ()
    code_revision: str | None = None
    sc2_patch: str | None = None
    expires_at: datetime | None = None
    shadow_state_count: int = Field(default=0, ge=0)
    false_block_count: int = Field(default=0, ge=0)
    evidence: dict[str, object] = Field(default_factory=dict)

    @property
    def false_block_rate(self) -> float:
        if self.shadow_state_count == 0:
            return 0.0
        return self.false_block_count / self.shadow_state_count


class PlaybookRuleApplication(ContractModel):
    application_id: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    target_kind: Literal["intent", "candidate"]
    target_id: str = Field(min_length=1)
    matched: bool
    blocked: bool = False
    score_delta: float = 0.0
    reason: str = Field(min_length=1)


class PlaybookContext(ContractModel):
    agent_race: str = Field(min_length=1)
    opponent_race: str = Field(min_length=1)
    phase: GamePhase
    map_name: str = Field(min_length=1)
    patch: str | None = None
    tags: tuple[str, ...] = ()


class StrategicConditionSnapshot(ContractModel):
    phase: GamePhase
    threat_level: Literal["none", "low", "high", "critical"]
    economy_status: Literal["constrained", "stable", "floating"]
    army_readiness: Literal["empty", "forming", "ready", "engaged"]


class StrategicConsequence(ContractModel):
    """One bounded, evidence-backed strategic outcome extracted after a full match."""

    schema_version: str = "1.0"
    consequence_id: str = Field(pattern=r"^consequence:[0-9a-f]{64}$")
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    consequence_type: StrategicConsequenceType
    quality: DecisionQuality
    effect: PlaybookRuleEffect
    role: PlaybookRoleId | None = None
    semantic_action: str | None = None
    objective: str = Field(min_length=1)
    start_game_loop: int = Field(ge=0)
    end_game_loop: int = Field(ge=0)
    source_event_ids: tuple[int, ...] = Field(min_length=1)
    condition: StrategicConditionSnapshot
    explanation: str = Field(min_length=1, max_length=500)
    evidence: dict[str, object] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    censored: bool = False

    @model_validator(mode="after")
    def validate_consequence(self) -> StrategicConsequence:
        if self.end_game_loop < self.start_game_loop:
            raise ValueError("strategic consequence cannot end before it starts")
        if self.role is None and self.semantic_action is None:
            raise ValueError("strategic consequence requires a role or semantic action")
        if self.quality not in {
            DecisionQuality.STRATEGIC_ERROR,
            DecisionQuality.ADVANTAGE_GAINED,
        }:
            raise ValueError("strategic consequence must identify an error or advantage")
        return self


class DecisionCase(ContractModel):
    case_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    source_event_id: int = Field(ge=1)
    source_step_id: int = Field(ge=0)
    command_id: str = Field(min_length=1)
    macro_plan_id: str | None = None
    semantic_action: str = Field(min_length=1)
    objective: str | None = None
    context: PlaybookContext
    quality: DecisionQuality
    failure_owner: FailureOwner
    consequence: str = Field(min_length=1)
    evidence: dict[str, object] = Field(default_factory=dict)
    episode_outcome: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    consequence_id: str | None = None
    consequence_type: StrategicConsequenceType | None = None


class PlaybookLesson(ContractModel):
    lesson_id: str = Field(min_length=1)
    signature: str = Field(min_length=1)
    context: PlaybookContext
    rule_kind: PlaybookRuleKind = PlaybookRuleKind.STRATEGY
    statement: str = Field(min_length=1, max_length=360)
    recommended_action: str | None = None
    avoid_action: str | None = None
    recommended_role: PlaybookRoleId | None = None
    avoid_role: PlaybookRoleId | None = None
    consequence_type: StrategicConsequenceType | None = None
    status: LessonStatus
    confidence: float = Field(ge=0.0, le=1.0)
    support_count: int = Field(ge=0)
    contradiction_count: int = Field(ge=0)
    source_case_ids: tuple[str, ...] = ()
    source_episode_ids: tuple[str, ...] = ()


class PlaybookQuery(ContractModel):
    context: PlaybookContext
    top_k: int = Field(default=6, ge=1, le=20)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    include_candidates: bool = False


class PlaybookHit(ContractModel):
    lesson: PlaybookLesson
    score: float
    match_reasons: tuple[str, ...]


class PlaybookSelection(ContractModel):
    query: PlaybookQuery
    hits: tuple[PlaybookHit, ...] = ()

    @property
    def lesson_ids(self) -> tuple[str, ...]:
        return tuple(hit.lesson.lesson_id for hit in self.hits)
