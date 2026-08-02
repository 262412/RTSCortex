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

    def placement_transition(self, event: dict[str, Any]) -> None: ...

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
        profiler = getattr(self.runtime, "profiler", None)
        if profiler is None:
            observation = self.mapper.map(snapshot)
        else:
            with profiler.measure("observation_extraction"):
                observation = self.mapper.map(snapshot)
        service = self.effect_verifier.placement_service
        if service is not None:
            service.set_runtime_context(
                run_id=str(observation["run_id"]),
                episode_id=str(observation["episode_id"]),
                step_id=int(observation["step_id"]),
                game_loop=int(observation["game_loop"]),
            )
        batch = self.runtime.tick(observation)
        duplicate_ids = [
            str(command["command_id"])
            for command in batch["commands"]
            if self.tracker.has_seen(str(command["command_id"]))
        ]
        if duplicate_ids:
            duplicates = ", ".join(sorted(duplicate_ids))
            reason = f"duplicate command dispatch invariant violated: {duplicates}"
            for command_id in dict.fromkeys(duplicate_ids):
                self.fail_bridge_integrity(
                    command_id,
                    reason,
                    function_name="runtime_dispatch",
                    execution_stage="pre_dispatch",
                )
            raise RuntimeError(f"bridge_integrity_error: {reason}")
        dispatch_batch = batch
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
        producer_tag: Optional[int] = None,
        actor_tags: tuple[int, ...] = (),
        minimap_transform: Optional[tuple[float, float, float, float, float]] = None,
    ) -> None:
        if self.effect_verifier.is_tracked(command_id):
            self.effect_verifier.prepare(
                command_id,
                observation,
                builder_tag,
                producer_tag=producer_tag,
                actor_tags=actor_tags,
                minimap_transform=minimap_transform,
            )

    def record_primitive(
        self,
        command_id: str,
        function_name: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        failure_reason: Optional[str] = None,
        origin: str = "translator",
        ordinal: Optional[int] = None,
        total: Optional[int] = None,
        game_loop: Optional[int] = None,
        failure_code: Optional[str] = None,
        requested_function_id: Optional[int] = None,
        emitted_function_id: Optional[int] = None,
    ) -> None:
        self.tracker.record_primitive(
            command_id,
            function_name,
            success=success,
            latency_ms=latency_ms,
            failure_reason=failure_reason,
            origin=origin,
            ordinal=ordinal,
            total=total,
            game_loop=game_loop,
            failure_code=failure_code,
            requested_function_id=requested_function_id,
            emitted_function_id=emitted_function_id,
        )

    def resolve_arguments(self, command_id: str, arguments: list[Any]) -> None:
        self.tracker.resolve_arguments(command_id, arguments)
        if self.effect_verifier.is_tracked(command_id):
            self.effect_verifier.resolve_arguments(command_id, arguments)

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
                    raise ValueError("game_loop is required to verify a gameplay effect")
                self.effect_verifier.accept_primitive(command_id, game_loop=game_loop)
                return None
            self.effect_verifier.cancel(command_id)
        report = self.tracker.complete(command_id, game_result=game_result)
        report = self._attach_placement_transitions(report)
        self.runtime.execution(report)
        return report

    def fail_bridge_integrity(
        self,
        command_id: str,
        reason: str,
        *,
        function_name: str,
        execution_stage: str = "translation",
        origin: str = "translator",
        ordinal: Optional[int] = None,
        total: Optional[int] = None,
        game_loop: Optional[int] = None,
        requested_function_id: Optional[int] = None,
        emitted_function_id: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Publish one terminal report before aborting on a Bridge invariant."""

        if not self.tracker.is_pending(command_id):
            return None
        self.tracker.record_primitive(
            command_id,
            function_name,
            success=False,
            failure_reason=reason,
            origin=origin,
            ordinal=ordinal,
            total=total,
            game_loop=game_loop,
            failure_code="bridge_integrity_error",
            requested_function_id=requested_function_id,
            emitted_function_id=emitted_function_id,
        )
        self.effect_verifier.cancel(command_id)
        report = self.tracker.complete(
            command_id,
            status="failed",
            execution_stage=execution_stage,
            failure_code="bridge_integrity_error",
        )
        report = self._attach_placement_transitions(report)
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
            report = self._attach_placement_transitions(report)
            self.runtime.execution(report)
        self.runtime.end_episode(result)

    def _publish_effect_verdicts(
        self,
        verdicts: Sequence[EffectVerdict],
        *,
        game_result: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for verdict in verdicts:
            report = self.tracker.complete(
                verdict.command_id,
                game_result=game_result,
                failure_reason=verdict.failure_reason if not verdict.success else None,
                status=verdict.status,
                execution_stage=(
                    "effect_verification" if verdict.status != "unconfirmed" else "episode_end"
                ),
                failure_code=verdict.failure_code,
                effect_evidence=verdict.evidence,
            )
            service = self.effect_verifier.placement_service
            if (
                verdict.success
                and service is not None
                and service.command_target(verdict.command_id) is not None
            ):
                evidence = verdict.evidence or {}
                confirmed_loop = evidence.get("confirmed_game_loop")
                service.release_command(
                    verdict.command_id,
                    game_loop=(int(confirmed_loop) if isinstance(confirmed_loop, int) else 0),
                    reason="effect_terminal_persisted",
                )
            reports.append(self._attach_placement_transitions(report))
        for report in reports:
            self.runtime.execution(report)
        return reports

    def _attach_placement_transitions(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        service = self.effect_verifier.placement_service
        if service is None:
            return report
        if service.transitions_are_durable:
            return report
        transitions = service.drain_transition_history(str(report.get("command_id", "")))
        if not transitions:
            return report
        evidence = report.get("effect_evidence")
        merged = dict(evidence) if isinstance(evidence, dict) else {"effect_kind": "build"}
        merged["placement_ledger_transitions"] = transitions
        return {**report, "effect_evidence": merged}
