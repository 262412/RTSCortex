"""Versioned wire models for RTSCortex v1."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

ProtocolVersion = Literal["1.0", "1.1"]
CURRENT_PROTOCOL_VERSION: Literal["1.1"] = "1.1"


class ContractModel(BaseModel):
    """Base model shared by all immutable public contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionSource(StrEnum):
    PLANNER = "planner"
    REFLEX = "reflex"
    FALLBACK = "fallback"


class ActionArgumentType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    POSITION = "position"
    TAG = "tag"
    ANY = "any"


class IdleReason(StrEnum):
    WAITING_FOR_PLANNER = "waiting_for_planner"
    PLAN_COMMANDS_DEFERRED = "plan_commands_deferred"
    PLAN_EXHAUSTED = "plan_exhausted"
    NO_LEGAL_ACTION = "no_legal_action"
    PLANNER_TIMEOUT = "planner_timeout"
    NOOP_BASELINE = "noop_baseline"


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCONFIRMED = "unconfirmed"


class ExecutionStage(StrEnum):
    PRE_DISPATCH = "pre_dispatch"
    TRANSLATION = "translation"
    PYSC2_ACCEPTANCE = "pysc2_acceptance"
    EFFECT_VERIFICATION = "effect_verification"
    EPISODE_END = "episode_end"


class PrimitiveOrigin(StrEnum):
    TRANSLATOR = "translator"
    ORCHESTRATION = "orchestration"


class EpisodeOutcome(StrEnum):
    VICTORY = "victory"
    DEFEAT = "defeat"
    DRAW = "draw"
    TRUNCATED = "truncated"
    ERROR = "error"


class EconomyState(ContractModel):
    minerals: int = Field(default=0, ge=0)
    vespene: int = Field(default=0, ge=0)
    supply_used: int = Field(default=0, ge=0)
    supply_cap: int = Field(default=0, ge=0)
    workers: int = Field(default=0, ge=0)
    army_supply: int = Field(default=0, ge=0)


class ProductionItem(ContractModel):
    name: str
    producer_id: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)


class UnitState(ContractModel):
    unit_id: str
    unit_type: str
    alliance: Literal["self", "ally", "enemy", "neutral"]
    position: tuple[float, float] | None = None
    health_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    energy: float | None = Field(default=None, ge=0.0)
    status: str | None = None


class SC2State(ContractModel):
    """Canonical state prepared for planning and future StarWM integration."""

    economy: EconomyState = Field(default_factory=EconomyState)
    production_queue: list[ProductionItem] = Field(default_factory=list)
    own_units: list[UnitState] = Field(default_factory=list)
    own_structures: list[UnitState] = Field(default_factory=list)
    visible_enemies: list[UnitState] = Field(default_factory=list)
    upgrades: list[str] = Field(default_factory=list)


class AvailableAction(ContractModel):
    name: str
    argument_names: list[str] = Field(default_factory=list)
    argument_types: list[ActionArgumentType] = Field(default_factory=list)
    actor_scopes: list[str] = Field(default_factory=list)
    argument_candidates: list[list[Any]] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_argument_candidates(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("argument_candidates") is None:
            return value
        normalized = dict(value)
        names = normalized.get("argument_names", [])
        types = normalized.get("argument_types", [])
        candidates = normalized["argument_candidates"]
        if not isinstance(candidates, list):
            raise ValueError("argument_candidates must be a list of complete argument lists")
        normalized_candidates: list[list[Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, list):
                raise ValueError("each argument candidate must be a list")
            if len(candidate) != len(names):
                raise ValueError("each argument candidate must match argument_names")
            normalized_candidates.append(
                [
                    _normalize_candidate_argument(
                        argument, types[index] if index < len(types) else None
                    )
                    for index, argument in enumerate(candidate)
                ]
            )
        normalized["argument_candidates"] = normalized_candidates
        return normalized

    @model_validator(mode="after")
    def validate_argument_schema(self) -> AvailableAction:
        if len(self.argument_types) != len(self.argument_names):
            raise ValueError("argument_types must match argument_names")
        return self


class ObservationEnvelope(ContractModel):
    protocol_version: ProtocolVersion = CURRENT_PROTOCOL_VERSION
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state: SC2State
    text_observation: str = ""
    available_actions: list[AvailableAction] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    image_uri: str | None = None

    @model_validator(mode="after")
    def validate_v11_semantic_candidates(self) -> ObservationEnvelope:
        if self.protocol_version != CURRENT_PROTOCOL_VERSION:
            return self
        missing = [
            action.name
            for action in self.available_actions
            if any(
                argument_type in {ActionArgumentType.TAG, ActionArgumentType.POSITION}
                for argument_type in action.argument_types
            )
            and not action.argument_candidates
        ]
        if missing:
            rendered = ", ".join(sorted(set(missing)))
            raise ValueError(
                "protocol 1.1 tag and position actions require non-empty "
                f"argument_candidates: {rendered}"
            )
        return self


class ActionCommand(ContractModel):
    command_id: str
    actor: str
    name: str
    arguments: list[Any] = Field(default_factory=list)
    priority: int = Field(default=50, ge=0, le=100)
    ttl_game_loops: int = Field(default=1, ge=1)
    created_game_loop: int = Field(ge=0)
    source: ActionSource
    preconditions: dict[str, Any] = Field(default_factory=dict)


class ActionBatch(ContractModel):
    protocol_version: ProtocolVersion = CURRENT_PROTOCOL_VERSION
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    decision_id: str
    strategic_goal: str = ""
    summary: str = ""
    planner_pending: bool = False
    idle_reason: IdleReason | None = None
    commands: list[ActionCommand] = Field(default_factory=list)
    rejected_commands: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_v11_idle_reason(self) -> ActionBatch:
        if self.protocol_version == CURRENT_PROTOCOL_VERSION:
            if not self.commands and self.idle_reason is None:
                raise ValueError("protocol 1.1 empty action batches require idle_reason")
            if self.commands and self.idle_reason is not None:
                raise ValueError("protocol 1.1 non-empty action batches cannot be idle")
        return self


class PrimitiveTraceEntry(ContractModel):
    function: str | None = Field(
        default=None,
        validation_alias=AliasChoices("function", "function_name"),
    )
    requested_function_id: int | None = Field(default=None, ge=0)
    emitted_function_id: int | None = Field(default=None, ge=0)
    origin: PrimitiveOrigin
    ordinal: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=1)
    game_loop: int | None = Field(default=None, ge=0)
    accepted: bool
    failure_code: str | None = None
    raw_reason: str | None = Field(
        default=None,
        validation_alias=AliasChoices("raw_reason", "detail"),
    )


class EffectEvidence(ContractModel):
    effect_kind: (
        Literal[
            "build",
            "move",
            "production",
            "addon",
            "morph",
            "inject",
            "research",
            "ability",
        ]
        | None
    ) = None
    target_type: str | None = None
    target_position: tuple[float, float] | None = None
    target_tag: str | None = None
    builder_tag: str | None = None
    requested_producer_tag: str | None = None
    producer_tag: str | None = None
    producer_type: str | None = None
    producer_observed_type: str | None = None
    producer_consumed: bool = False
    expected_unit_type: str | None = None
    expected_order_id: int | None = Field(default=None, ge=0)
    baseline_structure_tags: list[str] = Field(default_factory=list)
    baseline_unit_tags: list[str] = Field(default_factory=list)
    new_structure_tag: str | None = Field(
        default=None,
        validation_alias=AliasChoices("new_structure_tag", "observed_structure_tag"),
    )
    new_unit_tag: str | None = None
    dispatch_game_loop: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("dispatch_game_loop", "dispatched_loop"),
    )
    accepted_game_loop: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("accepted_game_loop", "accepted_loop"),
    )
    confirmed_game_loop: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("confirmed_game_loop", "confirmed_loop"),
    )
    worker_orders: list[str] = Field(default_factory=list)
    baseline_producer_orders: list[int] = Field(default_factory=list)
    producer_orders: list[int] = Field(default_factory=list)
    resource_delta: dict[str, int] = Field(default_factory=dict)
    order_seen: bool = False
    production_order_seen: bool = False
    confirmation_kind: (
        Literal[
            "producer_order",
            "producer_morph",
            "source_morph",
            "target_buff",
            "new_unit",
            "new_structure",
            "upgrade_observed",
        ]
        | None
    ) = None
    order_last_seen_game_loop: int | None = Field(default=None, ge=0)
    post_order_grace_game_loops: int | None = Field(default=None, ge=1)
    mineral_delta: int | None = None
    elapsed_game_loops: int | None = Field(default=None, ge=0)
    base_timeout_game_loops: int | None = Field(default=None, ge=1)
    effective_timeout_game_loops: int | None = Field(default=None, ge=1)
    active_order_extension: bool = False
    source_build_progress: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_target_buff_ids: list[int] = Field(default_factory=list)
    target_buff_ids: list[int] = Field(default_factory=list)
    expected_upgrade: str | None = None
    expected_upgrade_id: int | None = Field(default=None, ge=0)
    baseline_upgrade_ids: list[int] = Field(default_factory=list)
    upgrade_ids: list[int] = Field(default_factory=list)
    baseline_builder_position: tuple[float, float] | None = None
    observed_builder_position: tuple[float, float] | None = None
    builder_displacement: float | None = Field(default=None, ge=0)
    move_order_seen: bool = False


class ExecutionReport(ContractModel):
    protocol_version: ProtocolVersion = CURRENT_PROTOCOL_VERSION
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    command_id: str
    success: bool
    action_name: str | None = None
    actor: str | None = None
    source: ActionSource | None = None
    requested_arguments: list[Any] = Field(default_factory=list)
    resolved_arguments: list[Any] = Field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.FAILED
    execution_stage: ExecutionStage | None = None
    failure_code: str | None = None
    primitive_trace: list[PrimitiveTraceEntry] = Field(default_factory=list)
    effect_evidence: EffectEvidence | None = None
    failure_reason: str | None = None
    pysc2_function: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    game_result: str | None = None

    @model_validator(mode="before")
    @classmethod
    def derive_legacy_status(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "status" in value:
            return value
        if value.get("protocol_version", CURRENT_PROTOCOL_VERSION) == CURRENT_PROTOCOL_VERSION:
            raise ValueError("protocol 1.1 execution reports require explicit status")
        normalized = dict(value)
        failure_reason = str(normalized.get("failure_reason") or "").lower()
        if normalized.get("success") is True:
            normalized["status"] = "succeeded"
        elif "episode ended before command completion" in failure_reason:
            normalized["status"] = "cancelled"
        else:
            normalized["status"] = "failed"
        return normalized

    @model_validator(mode="after")
    def validate_status_matches_success(self) -> ExecutionReport:
        if self.status is ExecutionStatus.SUCCEEDED and not self.success:
            raise ValueError("succeeded execution status requires success=true")
        if self.status is not ExecutionStatus.SUCCEEDED and self.success:
            raise ValueError("only succeeded execution status permits success=true")
        if self.protocol_version == CURRENT_PROTOCOL_VERSION:
            missing = [
                name
                for name, value in (
                    ("action_name", self.action_name),
                    ("actor", self.actor),
                    ("source", self.source),
                    ("execution_stage", self.execution_stage),
                )
                if value is None
            ]
            if missing:
                raise ValueError("protocol 1.1 execution report is missing: " + ", ".join(missing))
            if self.status is not ExecutionStatus.SUCCEEDED and self.failure_code is None:
                raise ValueError("protocol 1.1 non-success execution reports require failure_code")
        return self


class EpisodeResult(ContractModel):
    protocol_version: ProtocolVersion = CURRENT_PROTOCOL_VERSION
    run_id: str
    episode_id: str
    scenario: str
    seed: int
    outcome: EpisodeOutcome
    score: float = 0.0
    steps: int = Field(default=0, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    failure_reason: str | None = None


class EpisodeSummary(ContractModel):
    protocol_version: ProtocolVersion = CURRENT_PROTOCOL_VERSION
    run_id: str
    episode_id: str
    scenario: str
    seed: int | None = None
    outcome: EpisodeOutcome
    summary: str
    lessons: list[str] = Field(default_factory=list)
    source_step_id: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _normalize_candidate_argument(value: Any, argument_type: object) -> Any:
    if argument_type in {ActionArgumentType.TAG, ActionArgumentType.TAG.value}:
        if isinstance(value, bool):
            raise ValueError("tag candidates must be non-negative integers or hexadecimal strings")
        if isinstance(value, int):
            if value < 0:
                raise ValueError("tag candidates must be non-negative")
            return hex(value)
        if isinstance(value, str):
            try:
                parsed = int(value, 0)
            except ValueError as error:
                raise ValueError("tag candidates must be hexadecimal or integer strings") from error
            if parsed < 0:
                raise ValueError("tag candidates must be non-negative")
            return hex(parsed)
        raise ValueError("tag candidates must be non-negative integers or hexadecimal strings")
    if argument_type in {ActionArgumentType.POSITION, ActionArgumentType.POSITION.value}:
        if (
            not isinstance(value, list | tuple)
            or len(value) != 2
            or any(
                not isinstance(coordinate, int) or isinstance(coordinate, bool)
                for coordinate in value
            )
        ):
            raise ValueError("position candidates must contain exactly two integers")
        return [value[0], value[1]]
    return value
