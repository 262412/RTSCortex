"""Typed, shadow-only policy comparison models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rtscortex.agents.models import ActionProposal
from rtscortex.contracts import ObservationEnvelope, ProtocolVersion
from rtscortex.progress import GoalProgressReport, GoalSpec


class PolicyModel(BaseModel):
    """Base model for immutable policy-observation records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicyProviderKind(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    HUGGING_FACE_TRANSFORMERS = "hugging_face_transformers"
    TENSORFLOW_CHECKPOINT = "tensorflow_checkpoint"


class PolicyGenerationMetadata(PolicyModel):
    """Auditable settings and outcome for one local policy generation."""

    provider_kind: PolicyProviderKind
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)
    checkpoint_verified: bool
    license_acknowledged: bool
    deterministic: bool
    max_new_tokens: int = Field(ge=1)
    prompt_token_count: int = Field(ge=0)
    completion_token_count: int = Field(ge=0)
    eos_reached: bool
    truncated: bool


class PolicyAvailabilityStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class PolicyShadowStatus(StrEnum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"
    FAILED = "failed"


class PolicyFixtureStratum(StrEnum):
    """Primary, mutually exclusive corpus bucket for one policy fixture."""

    EARLY = "early"
    TECHNOLOGY = "technology"
    PRODUCTION = "production"
    COMBAT = "combat"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"


class PolicyActionClassification(StrEnum):
    """One terminal shadow assessment for every discovered macro step."""

    PARSE_ERROR = "parse_error"
    UNSUPPORTED_BY_RUNTIME = "unsupported_by_runtime"
    MAPPED_FUTURE = "mapped_future"
    MAPPED_LEGAL_NOW = "mapped_legal_now"
    MAPPED_DEFERRED = "mapped_deferred"
    ILLEGAL_ACTION = "illegal_action"
    OBSOLETE = "obsolete"


class PolicyActionClassificationCounts(PolicyModel):
    """One conserved classification vector for logical or effective actions."""

    parse_error: int = Field(default=0, ge=0)
    unsupported_by_runtime: int = Field(default=0, ge=0)
    mapped_future: int = Field(default=0, ge=0)
    mapped_legal_now: int = Field(default=0, ge=0)
    mapped_deferred: int = Field(default=0, ge=0)
    illegal_action: int = Field(default=0, ge=0)
    obsolete: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return (
            self.parse_error
            + self.unsupported_by_runtime
            + self.mapped_future
            + self.mapped_legal_now
            + self.mapped_deferred
            + self.illegal_action
            + self.obsolete
        )


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


class PolicyFixtureSource(PolicyModel):
    """Immutable provenance for one observation selected into a corpus."""

    run_id: str
    episode_id: str
    event_id: int = Field(ge=0)
    seed: int | None = None
    map_name: str | None = None
    game_loop: int = Field(ge=0)
    protocol_version: ProtocolVersion
    journal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PolicyObservationFixture(PolicyModel):
    """One historical observation shared unchanged by every shadow policy."""

    fixture_id: str
    observation: ObservationEnvelope
    previous_actions: list[str] = Field(default_factory=list)
    goal_spec: GoalSpec | None = None
    goal_progress: GoalProgressReport | None = None
    primary_stratum: PolicyFixtureStratum | None = None
    phase_tags: list[str] = Field(default_factory=list)
    condition_tags: list[str] = Field(default_factory=list)
    blocker_tags: list[str] = Field(default_factory=list)
    selection_evidence: list[str] = Field(default_factory=list)
    source: PolicyFixtureSource | None = None
    state_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_goal_context(self) -> PolicyObservationFixture:
        observation = self.observation
        source = self.source
        if source is not None and (
            source.run_id,
            source.episode_id,
            source.game_loop,
            source.protocol_version,
        ) != (
            observation.run_id,
            observation.episode_id,
            observation.game_loop,
            observation.protocol_version,
        ):
            raise ValueError("fixture source must describe the fixture observation")
        report = self.goal_progress
        if report is None:
            return self
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

    proposal_kind: Literal["native"] = "native"
    strategic_goal: str = Field(max_length=200)
    steps: list[str] = Field(default_factory=list, max_length=3)
    proposed_actions: list[ActionProposal] = Field(default_factory=list, max_length=3)


class TacticalRationale(PolicyModel):
    """Explicit HIMA rationale sections, excluding hidden model reasoning."""

    immediate: str = ""
    short_term: str = ""
    long_term: str = ""


class MacroActionStep(PolicyModel):
    """One compact, ordered macro action emitted by a policy model."""

    ordinal: int = Field(ge=0)
    canonical_action: str = Field(min_length=1)
    category: Literal["train", "build", "research"]
    repeat: int = Field(default=1, ge=1)
    raw_token: str = Field(min_length=1)


class ParseDiagnostic(PolicyModel):
    """Deterministic parser feedback retained with the raw model output."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    raw_token: str | None = None
    ordinal: int | None = Field(default=None, ge=0)
    repeat: int = Field(default=1, ge=1)


class MacroPolicyProposal(PolicyModel):
    """A shadow-only ordered macro policy that is not directly executable."""

    proposal_kind: Literal["macro"] = "macro"
    strategic_objective: str = Field(max_length=500)
    tactical_rationale: TacticalRationale = Field(default_factory=TacticalRationale)
    horizon_seconds: int = Field(default=180, ge=1)
    steps: list[MacroActionStep] = Field(default_factory=list)
    raw_output: str = ""
    adapter_version: str = Field(default="not_recorded", min_length=1)
    vocabulary_version: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    generation_metadata: PolicyGenerationMetadata | None = None
    diagnostics: list[ParseDiagnostic] = Field(default_factory=list)


class PolicyActionAssessment(PolicyModel):
    """Terminal classification of one discovered native or macro action."""

    ordinal: int = Field(ge=0)
    repeat: int = Field(default=1, ge=1)
    source_action: str = Field(min_length=1)
    runtime_action: str | None = None
    classification: PolicyActionClassification
    reason_code: str | None = None
    is_logical_frontier: bool = False
    is_runtime_frontier: bool = False
    is_frontier: bool = False

    @model_validator(mode="before")
    @classmethod
    def synchronize_frontier_aliases(cls, value: object) -> object:
        """Keep the deprecated frontier flag aligned to the runtime frontier."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy = data.get("is_frontier")
        runtime = data.get("is_runtime_frontier")
        if runtime is None and legacy is not None:
            data["is_runtime_frontier"] = legacy
        elif legacy is None and runtime is not None:
            data["is_frontier"] = runtime
        return data

    @model_validator(mode="after")
    def require_runtime_action_for_mapped_classes(self) -> PolicyActionAssessment:
        mapped = {
            PolicyActionClassification.MAPPED_FUTURE,
            PolicyActionClassification.MAPPED_LEGAL_NOW,
            PolicyActionClassification.MAPPED_DEFERRED,
            PolicyActionClassification.ILLEGAL_ACTION,
            PolicyActionClassification.OBSOLETE,
        }
        if self.classification in mapped and not self.runtime_action:
            raise ValueError("mapped action assessments require runtime_action")
        if self.classification not in mapped and self.runtime_action is not None:
            raise ValueError("unmapped action assessments cannot contain runtime_action")
        if self.is_frontier != self.is_runtime_frontier:
            raise ValueError("is_frontier must match is_runtime_frontier")
        return self


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
    proposal: PolicyProposal | MacroPolicyProposal | None = None
    goal_id: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    proposed_action_count: int = Field(default=0, ge=0)
    legal_action_count: int = Field(default=0, ge=0)
    goal_advancing_action_count: int = Field(default=0, ge=0)
    control_action_violation_count: int = Field(default=0, ge=0)
    legal_action_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    goal_advancing_action_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    action_assessments: list[PolicyActionAssessment] = Field(default_factory=list)
    logical_classification_counts: PolicyActionClassificationCounts = Field(
        default_factory=PolicyActionClassificationCounts
    )
    effective_classification_counts: PolicyActionClassificationCounts = Field(
        default_factory=PolicyActionClassificationCounts
    )
    discovered_macro_step_count: int = Field(default=0, ge=0)
    parsed_known_action_count: int = Field(default=0, ge=0)
    effective_action_count: int = Field(default=0, ge=0)
    parse_error_count: int = Field(default=0, ge=0)
    unsupported_by_runtime_count: int = Field(default=0, ge=0)
    mapped_future_count: int = Field(default=0, ge=0)
    mapped_legal_now_count: int = Field(default=0, ge=0)
    mapped_deferred_count: int = Field(default=0, ge=0)
    illegal_action_count: int = Field(default=0, ge=0)
    obsolete_count: int = Field(default=0, ge=0)
    parse_validity: float | None = Field(default=None, ge=0.0, le=1.0)
    mapping_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    frontier_illegal_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    shadow_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_outcome(self) -> PolicyShadowRecord:
        if self.status is PolicyShadowStatus.COMPLETED and self.proposal is None:
            raise ValueError("completed shadow records require a proposal")
        if self.status is not PolicyShadowStatus.COMPLETED and self.proposal is not None:
            raise ValueError("non-completed shadow records cannot contain a proposal")
        if self.status is PolicyShadowStatus.FAILED and not self.error:
            raise ValueError("failed shadow records require an error")
        if isinstance(self.proposal, MacroPolicyProposal):
            _validate_macro_classification_conservation(
                logical=self.logical_classification_counts,
                effective=self.effective_classification_counts,
                discovered=self.discovered_macro_step_count,
                parsed=self.parsed_known_action_count,
                effective_total=self.effective_action_count,
                scalar_counts=(
                    self.parse_error_count,
                    self.unsupported_by_runtime_count,
                    self.mapped_future_count,
                    self.mapped_legal_now_count,
                    self.mapped_deferred_count,
                    self.illegal_action_count,
                    self.obsolete_count,
                ),
            )
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
    logical_classification_counts: PolicyActionClassificationCounts = Field(
        default_factory=PolicyActionClassificationCounts
    )
    effective_classification_counts: PolicyActionClassificationCounts = Field(
        default_factory=PolicyActionClassificationCounts
    )
    discovered_macro_step_count: int = Field(default=0, ge=0)
    parsed_known_action_count: int = Field(default=0, ge=0)
    effective_action_count: int = Field(default=0, ge=0)
    parse_error_count: int = Field(default=0, ge=0)
    unsupported_by_runtime_count: int = Field(default=0, ge=0)
    mapped_future_count: int = Field(default=0, ge=0)
    mapped_legal_now_count: int = Field(default=0, ge=0)
    mapped_deferred_count: int = Field(default=0, ge=0)
    illegal_action_count: int = Field(default=0, ge=0)
    obsolete_count: int = Field(default=0, ge=0)
    parse_validity: float | None = Field(default=None, ge=0.0, le=1.0)
    mapping_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    frontier_illegal_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_macro_classification_counts(self) -> PolicyShadowSummary:
        if self.discovered_macro_step_count:
            _validate_macro_classification_conservation(
                logical=self.logical_classification_counts,
                effective=self.effective_classification_counts,
                discovered=self.discovered_macro_step_count,
                parsed=self.parsed_known_action_count,
                effective_total=self.effective_action_count,
                scalar_counts=(
                    self.parse_error_count,
                    self.unsupported_by_runtime_count,
                    self.mapped_future_count,
                    self.mapped_legal_now_count,
                    self.mapped_deferred_count,
                    self.illegal_action_count,
                    self.obsolete_count,
                ),
            )
        return self


class PolicyShadowComparison(PolicyModel):
    """Serializable result set for a fair, same-fixture comparison."""

    comparison_version: Literal["0.2"] = "0.2"
    fixture_ids: list[str]
    fixtures: list[PolicyObservationFixture]
    candidate_ids: list[str]
    records: list[PolicyShadowRecord]
    summaries: list[PolicyShadowSummary]


def _validate_macro_classification_conservation(
    *,
    logical: PolicyActionClassificationCounts,
    effective: PolicyActionClassificationCounts,
    discovered: int,
    parsed: int,
    effective_total: int,
    scalar_counts: tuple[int, int, int, int, int, int, int],
) -> None:
    if logical.total != discovered:
        raise ValueError("logical classification counts must equal discovered macro steps")
    if effective.total != effective_total:
        raise ValueError("effective classification counts must equal effective actions")
    if parsed != logical.total - logical.parse_error:
        raise ValueError("parsed known actions must exclude logical parse errors")
    if scalar_counts != (
        logical.parse_error,
        logical.unsupported_by_runtime,
        logical.mapped_future,
        logical.mapped_legal_now,
        logical.mapped_deferred,
        logical.illegal_action,
        logical.obsolete,
    ):
        raise ValueError("classification scalar counts must match the logical vector")
