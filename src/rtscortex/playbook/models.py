"""Typed records for the cross-episode Cortex tactical playbook."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from rtscortex.contracts.models import ContractModel
from rtscortex.game_phase import GamePhase


class DecisionQuality(StrEnum):
    ADVANTAGE_GAINED = "advantage_gained"
    CORRECT_EXECUTION = "correct_execution"
    STRATEGIC_ERROR = "strategic_error"
    EXECUTION_ERROR = "execution_error"
    INCONCLUSIVE = "inconclusive"


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


class PlaybookContext(ContractModel):
    agent_race: str = Field(min_length=1)
    opponent_race: str = Field(min_length=1)
    phase: GamePhase
    map_name: str = Field(min_length=1)
    patch: str | None = None
    tags: tuple[str, ...] = ()


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


class PlaybookLesson(ContractModel):
    lesson_id: str = Field(min_length=1)
    signature: str = Field(min_length=1)
    context: PlaybookContext
    rule_kind: PlaybookRuleKind = PlaybookRuleKind.STRATEGY
    statement: str = Field(min_length=1, max_length=360)
    recommended_action: str | None = None
    avoid_action: str | None = None
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
