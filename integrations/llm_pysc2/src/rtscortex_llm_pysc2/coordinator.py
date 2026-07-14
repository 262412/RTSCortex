"""Worker-side orchestration seam for the RTSCortex runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from rtscortex_llm_pysc2.execution import ExecutionTracker
from rtscortex_llm_pysc2.observation import ObservationMapper
from rtscortex_llm_pysc2.routing import ActionRouter, RoutedActionBatch


class RuntimeAPI(Protocol):
    """Transport-independent runtime operations used by the worker."""

    def health(self) -> dict[str, Any]: ...

    def tick(self, observation: dict[str, Any]) -> dict[str, Any]: ...

    def execution(self, report: dict[str, Any]) -> None: ...

    def end_episode(self, result: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class BridgeDecision:
    observation: dict[str, Any]
    action_batch: dict[str, Any]
    routes: dict[str, RoutedActionBatch]

    def action_text(self, agent_name: str) -> str:
        return self.routes[agent_name].action_text


class BridgeCoordinator:
    """Perform one runtime tick and prepare each enabled upstream agent slot."""

    def __init__(
        self,
        runtime: RuntimeAPI,
        *,
        mapper: Optional[ObservationMapper] = None,
        router: Optional[ActionRouter] = None,
        tracker: Optional[ExecutionTracker] = None,
    ) -> None:
        self.runtime = runtime
        self.mapper = mapper or ObservationMapper()
        self.router = router or ActionRouter()
        self.tracker = tracker or ExecutionTracker()

    def decide(
        self,
        snapshot: Mapping[str, Any],
        agent_team_order: Mapping[str, Sequence[str]],
    ) -> BridgeDecision:
        observation = self.mapper.map(snapshot)
        batch = self.runtime.tick(observation)
        dispatch_batch = {
            **batch,
            "commands": [
                command
                for command in batch["commands"]
                if not self.tracker.has_seen(str(command["command_id"]))
            ],
        }
        available_actions = observation["available_actions"]
        routes = {
            agent_name: self.router.route(
                dispatch_batch,
                agent_name=agent_name,
                team_order=team_order,
                available_actions=available_actions,
            )
            for agent_name, team_order in agent_team_order.items()
        }

        expected = {str(command["command_id"]) for command in dispatch_batch["commands"]}
        routed = {command.command_id for route in routes.values() for command in route.commands}
        if routed != expected:
            missing = sorted(expected - routed)
            raise ValueError(f"no enabled LLM-PySC2 actor route for commands: {missing}")

        for route in routes.values():
            self.tracker.register(route)
        return BridgeDecision(observation=observation, action_batch=batch, routes=routes)

    def record_primitive(
        self,
        command_id: str,
        function_name: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        failure_reason: Optional[str] = None,
    ) -> None:
        self.tracker.record_primitive(
            command_id,
            function_name,
            success=success,
            latency_ms=latency_ms,
            failure_reason=failure_reason,
        )

    def complete_command(
        self, command_id: str, *, game_result: Optional[str] = None
    ) -> dict[str, Any]:
        report = self.tracker.complete(command_id, game_result=game_result)
        self.runtime.execution(report)
        return report

    def end_episode(self, result: dict[str, Any]) -> None:
        game_result = result.get("outcome")
        for report in self.tracker.drain_pending(
            failure_reason="episode ended before command completion",
            game_result=None if game_result is None else str(game_result),
        ):
            self.runtime.execution(report)
        self.runtime.end_episode(result)
