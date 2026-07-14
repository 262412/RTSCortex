"""Worker-side orchestration seam for the RTSCortex runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from rtscortex_llm_pysc2.effect_verifier import ActionEffectVerifier, EffectVerdict
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
        effect_verifier: Optional[ActionEffectVerifier] = None,
    ) -> None:
        self.runtime = runtime
        self.mapper = mapper or ObservationMapper()
        self.router = router or ActionRouter()
        self.tracker = tracker or ExecutionTracker()
        self.effect_verifier = effect_verifier or ActionEffectVerifier()

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
            for command in route.commands:
                self.effect_verifier.track(command)
        return BridgeDecision(observation=observation, action_batch=batch, routes=routes)

    def prepare_effect(
        self,
        command_id: str,
        observation: Any,
        *,
        builder_tag: Optional[int],
    ) -> None:
        if self.effect_verifier.is_tracked(command_id):
            self.effect_verifier.prepare(command_id, observation, builder_tag)

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
        self,
        command_id: str,
        *,
        game_result: Optional[str] = None,
        game_loop: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        if self.effect_verifier.is_tracked(command_id):
            if self.tracker.primitives_succeeded(command_id):
                if game_loop is None:
                    raise ValueError("game_loop is required to verify a build effect")
                self.effect_verifier.accept_primitive(command_id, game_loop=game_loop)
                return None
            self.effect_verifier.cancel(command_id)
        report = self.tracker.complete(command_id, game_result=game_result)
        self.runtime.execution(report)
        return report

    def observe_effects(self, observation: Any) -> list[dict[str, Any]]:
        return self._publish_effect_verdicts(self.effect_verifier.observe(observation))

    def end_episode(self, result: dict[str, Any]) -> None:
        game_result = result.get("outcome")
        normalized_result = None if game_result is None else str(game_result)
        self._publish_effect_verdicts(
            self.effect_verifier.fail_pending("episode ended before gameplay effect was confirmed"),
            game_result=normalized_result,
        )
        for report in self.tracker.drain_pending(
            failure_reason="episode ended before command completion",
            game_result=normalized_result,
        ):
            self.runtime.execution(report)
        self.runtime.end_episode(result)

    def _publish_effect_verdicts(
        self,
        verdicts: Sequence[EffectVerdict],
        *,
        game_result: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        reports = [
            self.tracker.complete(
                verdict.command_id,
                game_result=game_result,
                failure_reason=verdict.failure_reason if not verdict.success else None,
            )
            for verdict in verdicts
        ]
        for report in reports:
            self.runtime.execution(report)
        return reports
