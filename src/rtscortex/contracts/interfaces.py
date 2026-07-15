"""Extension interfaces owned by RTSCortex."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from rtscortex.contracts.models import (
    ActionBatch,
    ActionCommand,
    EpisodeResult,
    ExecutionReport,
    ObservationEnvelope,
    SC2State,
)


@dataclass(frozen=True)
class CommandLifecycleSnapshot:
    """Compact non-terminal command state exposed to deliberative modules."""

    command_id: str
    actor: str
    name: str
    arguments: tuple[Any, ...]
    source: str
    status: str
    reason: str | None
    created_game_loop: int
    ttl_game_loops: int

    @property
    def expires_at_game_loop(self) -> int:
        return self.created_game_loop + self.ttl_game_loops


@dataclass(frozen=True)
class ActivePlanSnapshot:
    """Current strategy plus every command that can still affect the game."""

    strategic_goal: str
    summary: str
    commands: tuple[CommandLifecycleSnapshot, ...]


@dataclass(frozen=True)
class AgentContext:
    observation: ObservationEnvelope
    memory: dict[str, Any] = field(default_factory=dict)
    last_execution: ExecutionReport | None = None
    last_decision: ActionBatch | None = None
    active_plan: ActivePlanSnapshot | None = None


@dataclass(frozen=True)
class ModuleResult:
    module: str
    updates: dict[str, Any] = field(default_factory=dict)
    commands: list[ActionCommand] = field(default_factory=list)
    model_call: bool = False


class AgentModule(Protocol):
    name: str

    async def run(self, context: AgentContext) -> ModuleResult: ...


class ReflexPolicy(Protocol):
    def evaluate(self, observation: ObservationEnvelope) -> list[ActionCommand]: ...


class WorldModel(Protocol):
    async def predict(
        self,
        state: SC2State,
        candidate_actions: list[ActionCommand],
        horizon_seconds: float = 5.0,
    ) -> SC2State | None: ...


class PerceptionProvider(Protocol):
    async def enrich(self, observation: ObservationEnvelope) -> ObservationEnvelope: ...


class EnvironmentAdapter(Protocol):
    async def reset(self, *, run_id: str, episode_id: str, seed: int) -> ObservationEnvelope: ...

    async def step(
        self, actions: ActionBatch
    ) -> tuple[ObservationEnvelope, list[ExecutionReport]]: ...

    async def close(self) -> None: ...


ResponseT = TypeVar("ResponseT", bound=BaseModel)


class LLMProvider(Protocol):
    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT: ...


class EpisodeSink(Protocol):
    def record_episode(self, result: EpisodeResult) -> None: ...


class NullWorldModel:
    async def predict(
        self,
        state: SC2State,
        candidate_actions: list[ActionCommand],
        horizon_seconds: float = 5.0,
    ) -> SC2State | None:
        return None


class NullPerceptionProvider:
    async def enrich(self, observation: ObservationEnvelope) -> ObservationEnvelope:
        return observation
