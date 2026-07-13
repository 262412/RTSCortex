"""Versioned wire models for RTSCortex v1."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="after")
    def validate_argument_schema(self) -> AvailableAction:
        if self.argument_types and len(self.argument_types) != len(self.argument_names):
            raise ValueError("argument_types must match argument_names")
        return self


class ObservationEnvelope(ContractModel):
    protocol_version: Literal["1.0"] = "1.0"
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
    protocol_version: Literal["1.0"] = "1.0"
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    decision_id: str
    strategic_goal: str = ""
    summary: str = ""
    planner_pending: bool = False
    commands: list[ActionCommand] = Field(default_factory=list)
    rejected_commands: list[str] = Field(default_factory=list)


class ExecutionReport(ContractModel):
    protocol_version: Literal["1.0"] = "1.0"
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    command_id: str
    success: bool
    failure_reason: str | None = None
    pysc2_function: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    game_result: str | None = None


class EpisodeResult(ContractModel):
    protocol_version: Literal["1.0"] = "1.0"
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
    protocol_version: Literal["1.0"] = "1.0"
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
