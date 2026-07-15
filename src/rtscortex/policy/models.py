"""Typed, shadow-only policy comparison models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rtscortex.agents.models import ActionProposal
from rtscortex.contracts import ObservationEnvelope
from rtscortex.progress import GoalProgressReport, GoalSpec


class PolicyModel(BaseModel):
    """Base model for immutable policy-observation records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicyProviderKind(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    HUGGING_FACE_TRANSFORMERS = "hugging_face_transformers"
    TENSORFLOW_CHECKPOINT = "tensorflow_checkpoint"


class PolicyAvailabilityStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class PolicyShadowStatus(StrEnum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"
    FAILED = "failed"


class PolicySubagentSpec(PolicyModel):
    """Static identity and integration requirements for one policy candidate."""

    subagent_id: str
    display_name: str
    provider_kind: PolicyProviderKind
    model_id: str
    role: str
    race: Literal["Protoss", "Terran", "Zerg", "any"] = "any"
    action_interface: str
    requires_external_weights: bool
    license_id: str | None = None
    shadow_only: Literal[True] = True


class PolicyAvailability(PolicyModel):
    """Runtime availability of one candidate without probing or downloading it."""

    status: PolicyAvailabilityStatus
    reason: str | None = None

    @model_validator(mode="after")
    def require_reason_when_not_available(self) -> PolicyAvailability:
        if self.status is not PolicyAvailabilityStatus.AVAILABLE and not self.reason:
            raise ValueError("unavailable and skipped policies require a reason")
        return self


class PolicyObservationFixture(PolicyModel):
    """One historical observation shared unchanged by every shadow policy."""

    fixture_id: str
    observation: ObservationEnvelope
    goal_spec: GoalSpec | None = None
    goal_progress: GoalProgressReport | None = None

    @model_validator(mode="after")
    def validate_goal_context(self) -> PolicyObservationFixture:
        report = self.goal_progress
        if report is None:
            return self
        observation = self.observation
        if (
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
            raise ValueError("goal progress must describe the fixture observation")
        if self.goal_spec is not None and report.goal_id != self.goal_spec.goal_id:
            raise ValueError("goal progress must match the fixture goal spec")
        return self


class PolicyProposal(PolicyModel):
    """A non-executable policy suggestion normalized for offline comparison."""

    strategic_goal: str = Field(max_length=200)
    steps: list[str] = Field(default_factory=list, max_length=3)
    proposed_actions: list[ActionProposal] = Field(default_factory=list, max_length=3)


class PolicyShadowRecord(PolicyModel):
    """Result of evaluating one policy against one immutable fixture."""

    fixture_id: str
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    spec: PolicySubagentSpec
    availability: PolicyAvailability
    status: PolicyShadowStatus
    proposal: PolicyProposal | None = None
    goal_id: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    proposed_action_count: int = Field(default=0, ge=0)
    legal_action_count: int = Field(default=0, ge=0)
    goal_advancing_action_count: int = Field(default=0, ge=0)
    control_action_violation_count: int = Field(default=0, ge=0)
    legal_action_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    goal_advancing_action_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    shadow_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_outcome(self) -> PolicyShadowRecord:
        if self.status is PolicyShadowStatus.COMPLETED and self.proposal is None:
            raise ValueError("completed shadow records require a proposal")
        if self.status is not PolicyShadowStatus.COMPLETED and self.proposal is not None:
            raise ValueError("non-completed shadow records cannot contain a proposal")
        if self.status is PolicyShadowStatus.FAILED and not self.error:
            raise ValueError("failed shadow records require an error")
        return self


class PolicyShadowSummary(PolicyModel):
    subagent_id: str
    fixtures: int = Field(ge=0)
    completed: int = Field(ge=0)
    unavailable: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)
    proposals: int = Field(ge=0)
    legal_actions: int = Field(ge=0)
    goal_advancing_actions: int = Field(ge=0)
    goal_opportunity_fixtures: int = Field(ge=0)
    goal_opportunity_proposals: int = Field(ge=0)
    control_action_violation_count: int = Field(ge=0)
    legal_action_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    goal_advancing_action_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class PolicyShadowComparison(PolicyModel):
    """Serializable result set for a fair, same-fixture comparison."""

    fixture_ids: list[str]
    fixtures: list[PolicyObservationFixture]
    candidate_ids: list[str]
    records: list[PolicyShadowRecord]
    summaries: list[PolicyShadowSummary]
