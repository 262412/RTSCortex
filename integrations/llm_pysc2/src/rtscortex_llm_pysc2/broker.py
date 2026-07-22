"""A shared barrier that turns concurrent upstream queries into one runtime tick."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Condition
from time import monotonic
from typing import Any, NoReturn, Optional

from rtscortex_llm_pysc2.coordinator import BridgeCoordinator, BridgeDecision
from rtscortex_llm_pysc2.extractor import TimeStepExtractor, current_team_order


class BridgeIntegrityError(RuntimeError):
    """Raised after a Bridge invariant violation has been accounted for."""


@dataclass
class _Submission:
    agent: Any
    timestep: Any
    text_observation: str


@dataclass
class _DecisionState:
    submissions: dict[str, _Submission] = field(default_factory=dict)
    decision: Optional[BridgeDecision] = None
    error: Optional[Exception] = None
    in_flight: bool = False
    consumed: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PrimitiveDispatch:
    command_id: str
    function_name: str
    final_primitive: bool
    origin: str = "translator"
    ordinal: Optional[int] = None
    total: Optional[int] = None
    failure_code: Optional[str] = None
    requested_function_id: Optional[int] = None
    emitted_function_id: Optional[int] = None


@dataclass
class _ActiveCommand:
    action_name: str
    command_id: str
    total: int
    next_ordinal: int = 0


@dataclass(frozen=True)
class ScreenRouteProvenance:
    world_target: tuple[float, float]
    anchor_tag: int


class SharedDecisionBroker:
    """Synchronize enabled LLMAgents without ticking once per subagent."""

    def __init__(
        self,
        coordinator: BridgeCoordinator,
        extractor: TimeStepExtractor,
        *,
        decision_timeout_seconds: float = 60.0,
        metrics_path: Optional[str] = None,
    ) -> None:
        if decision_timeout_seconds <= 0:
            raise ValueError("decision_timeout_seconds must be positive")
        self.coordinator = coordinator
        self.extractor = extractor
        self.decision_timeout_seconds = decision_timeout_seconds
        self._condition = Condition()
        self._agents: dict[str, Any] = {}
        self._states: dict[int, _DecisionState] = {}
        self._command_queues: dict[tuple[str, str, str], deque[str]] = defaultdict(deque)
        self._active_commands: dict[tuple[str, str], _ActiveCommand] = {}
        self._screen_route_provenance: dict[str, ScreenRouteProvenance] = {}
        self._planner_pending = False
        self._last_decision_game_loop: Optional[int] = None
        self._initial_decision_started = False
        self._initial_decision_complete = False
        self._initial_decision_error: Optional[Exception] = None
        self.unattributed_primitives = 0
        self.candidate_outside_pysc2_dispatches = 0
        self.observation_gap_watchdog_triggers = 0
        self.orchestration_recoveries = 0
        self.expansion_scout_camera_moves = 0
        self._metrics_path = None if metrics_path is None else Path(metrics_path)
        with self._condition:
            self._persist_metrics_locked()

    def metrics(self) -> dict[str, int]:
        """Return the live Worker counters included in every episode result."""

        with self._condition:
            return {
                "unattributed_primitives": self.unattributed_primitives,
                "candidate_outside_pysc2_dispatches": (self.candidate_outside_pysc2_dispatches),
                "observation_gap_watchdog_triggers": self.observation_gap_watchdog_triggers,
                "orchestration_recoveries": self.orchestration_recoveries,
                "expansion_scout_camera_moves": self.expansion_scout_camera_moves,
            }

    @property
    def last_decision_game_loop(self) -> Optional[int]:
        """Return the game loop represented by the latest Runtime decision."""

        with self._condition:
            return self._last_decision_game_loop

    def record_observation_gap_watchdog_trigger(self) -> None:
        with self._condition:
            self.observation_gap_watchdog_triggers += 1
            self._persist_metrics_locked()

    def record_orchestration_recovery(self) -> None:
        with self._condition:
            self.orchestration_recoveries += 1
            self._persist_metrics_locked()

    def record_expansion_scout_move(self) -> None:
        with self._condition:
            self.expansion_scout_camera_moves += 1
            self._persist_metrics_locked()

    def record_unattributed_primitive(self) -> None:
        with self._condition:
            self.unattributed_primitives += 1
            self._persist_metrics_locked()

    def reject_candidate_outside_dispatch(
        self,
        dispatch: PrimitiveDispatch,
        reason: str,
        *,
        game_loop: Optional[int],
    ) -> NoReturn:
        """Account for and abort a primitive that escaped its candidate domain."""

        with self._condition:
            self.candidate_outside_pysc2_dispatches += 1
            self._persist_metrics_locked()
        self._raise_integrity(
            reason,
            command_id=dispatch.command_id,
            function_name=dispatch.function_name,
            origin=dispatch.origin,
            ordinal=dispatch.ordinal,
            total=dispatch.total,
            game_loop=game_loop,
            requested_function_id=dispatch.requested_function_id,
            emitted_function_id=dispatch.emitted_function_id,
        )

    def settle_candidate_invalidation(
        self,
        dispatch: PrimitiveDispatch,
        *,
        failure_code: str,
        failure_reason: str,
        game_loop: Optional[int],
    ) -> None:
        """Fail one command whose previously valid semantic target became stale."""

        self.settle_primitive(
            replace(
                dispatch,
                failure_code=failure_code,
                emitted_function_id=0,
            ),
            success=False,
            failure_reason=failure_reason,
            game_loop=game_loop,
        )

    def raise_unattributed_integrity(self, reason: str) -> NoReturn:
        self._raise_integrity(reason, command_id=None)

    def fail_dispatch_integrity(
        self,
        dispatch: PrimitiveDispatch,
        reason: str,
        *,
        game_loop: Optional[int],
    ) -> NoReturn:
        """Fail one known command when MainAgent changes its translator output."""

        self._raise_integrity(
            reason,
            command_id=dispatch.command_id,
            function_name=dispatch.function_name,
            origin=dispatch.origin,
            ordinal=dispatch.ordinal,
            total=dispatch.total,
            game_loop=game_loop,
            requested_function_id=dispatch.requested_function_id,
            emitted_function_id=dispatch.emitted_function_id,
        )

    def fail_command_integrity(
        self,
        agent_name: str,
        team_name: Optional[str],
        action_name: str,
        reason: str,
        *,
        function_name: str = "bridge_integrity",
        game_loop: Optional[int] = None,
    ) -> NoReturn:
        """Fail the uniquely owned command when no dispatch object could be formed."""

        actor = self._resolve_dispatch_actor(
            agent_name,
            team_name,
            action_name,
            active_only=False,
        )
        command_id = None if actor is None else self._command_id_for_actor(actor, action_name)
        self._raise_integrity(
            reason,
            command_id=command_id,
            function_name=function_name,
            game_loop=game_loop,
        )

    @property
    def planner_pending(self) -> bool:
        with self._condition:
            return self._planner_pending

    @property
    def initial_decision_started(self) -> bool:
        with self._condition:
            return self._initial_decision_started

    def wait_for_initial_decision(self, timeout_seconds: Optional[float] = None) -> None:
        """Block the environment thread until the first Runtime decision exists."""

        timeout = self.decision_timeout_seconds if timeout_seconds is None else timeout_seconds
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._initial_decision_complete or self._initial_decision_error is not None,
                timeout=timeout,
            )
            if not ready:
                raise TimeoutError("initial runtime decision did not complete before timeout")
            if self._initial_decision_error is not None:
                raise RuntimeError("initial runtime decision failed") from (
                    self._initial_decision_error
                )

    def register(self, agent: Any) -> None:
        with self._condition:
            if agent.name in self._agents:
                raise ValueError(f"agent {agent.name!r} is already registered")
            self._agents[agent.name] = agent

    def submit(self, agent: Any, timestep: Any, text_observation: str) -> str:
        step_id = int(agent.main_loop_step)
        with self._condition:
            state = self._states.setdefault(step_id, _DecisionState())
            if state.error is not None:
                raise RuntimeError(
                    f"shared runtime decision failed at step {step_id}"
                ) from state.error
            if agent.name in state.submissions:
                raise ValueError(f"agent {agent.name!r} submitted step {step_id} twice")
            state.submissions[agent.name] = _Submission(agent, timestep, text_observation)
            self._condition.notify_all()

        deadline = monotonic() + self.decision_timeout_seconds
        while state.decision is None and state.error is None:
            leader = False
            with self._condition:
                expected = {name for name, value in self._agents.items() if value.enable}
                # Combat agents can be disabled after losing their final unit while
                # Builder/Developer threads are already waiting at this barrier.
                # Re-evaluate the live participant set instead of retaining a stale
                # name until the full decision timeout expires. Submitted agents are
                # still included in the snapshot even if they become disabled later.
                if not state.in_flight and expected.issubset(state.submissions):
                    state.in_flight = True
                    leader = True
                    if not self._initial_decision_started:
                        self._initial_decision_started = True
                        self._condition.notify_all()
                else:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        missing = sorted(expected.difference(state.submissions))
                        state.error = TimeoutError(
                            "not all enabled agents submitted runtime step "
                            f"{step_id}; missing={missing}"
                        )
                        if not self._initial_decision_complete:
                            self._initial_decision_error = state.error
                        self._states.pop(step_id, None)
                        self._condition.notify_all()
                    else:
                        self._condition.wait(timeout=min(0.05, remaining))
            if leader:
                self._decide(step_id)

        with self._condition:
            if state.error is not None:
                raise RuntimeError(
                    f"shared runtime decision failed at step {step_id}"
                ) from state.error
            assert state.decision is not None
            route = state.decision.routes[agent.name]
            state.consumed.add(agent.name)
            if state.consumed == set(state.submissions):
                self._states.pop(step_id, None)
            return route.action_text

    def claim_primitive(
        self,
        agent_name: str,
        team_name: Optional[str],
        action_name: str,
        function_name: str,
        *,
        final_primitive: bool,
        origin: str = "translator",
        ordinal: Optional[int] = None,
        total: Optional[int] = None,
        failure_code: Optional[str] = None,
        requested_function_id: Optional[int] = None,
        emitted_function_id: Optional[int] = None,
    ) -> Optional[PrimitiveDispatch]:
        if origin == "translator" and team_name is not None:
            explicit_active = self._active_commands.get((agent_name, team_name))
            if explicit_active is not None and explicit_active.action_name != action_name:
                self._raise_integrity(
                    "action changed before active command completed: "
                    f"{explicit_active.action_name!r} -> {action_name!r}",
                    command_id=explicit_active.command_id,
                    function_name=function_name,
                    ordinal=ordinal,
                    total=total,
                    requested_function_id=requested_function_id,
                    emitted_function_id=emitted_function_id,
                )
        actor = (
            self._resolve_orchestration_actor(agent_name, team_name, action_name)
            if origin == "orchestration"
            else self._resolve_dispatch_actor(
                agent_name,
                team_name,
                action_name,
                active_only=False,
            )
        )
        if actor is None:
            return None
        active = self._active_commands.get(actor)
        command_id = self._command_id_for_actor(actor, action_name)
        if origin == "orchestration":
            command_id = (
                active.command_id
                if active is not None and active.action_name == action_name
                else self._command_queues[(actor[0], actor[1], action_name)][0]
            )
            return PrimitiveDispatch(
                command_id,
                function_name,
                False,
                origin,
                requested_function_id=requested_function_id,
                emitted_function_id=emitted_function_id,
            )
        if origin != "translator":
            self._raise_integrity(
                f"unsupported primitive origin {origin!r}",
                command_id=command_id,
                function_name=function_name,
                origin=origin,
                requested_function_id=requested_function_id,
                emitted_function_id=emitted_function_id,
            )
        if ordinal is None or total is None or total <= 0:
            self._raise_integrity(
                "translator primitive lacks sequence data",
                command_id=command_id,
                function_name=function_name,
                ordinal=ordinal,
                total=total,
                requested_function_id=requested_function_id,
                emitted_function_id=emitted_function_id,
            )
        if active is not None and active.action_name != action_name:
            self._raise_integrity(
                "action changed before active command completed: "
                f"{active.action_name!r} -> {action_name!r}",
                command_id=active.command_id,
                function_name=function_name,
                ordinal=ordinal,
                total=total,
                requested_function_id=requested_function_id,
                emitted_function_id=emitted_function_id,
            )
        if active is None:
            if ordinal != 0:
                self._raise_integrity(
                    "translator sequence must begin at ordinal 0",
                    command_id=command_id,
                    function_name=function_name,
                    ordinal=ordinal,
                    total=total,
                    requested_function_id=requested_function_id,
                    emitted_function_id=emitted_function_id,
                )
            queue = self._command_queues[(actor[0], actor[1], action_name)]
            if not queue:
                return None
            active = _ActiveCommand(action_name, queue.popleft(), total)
            self._active_commands[actor] = active
        if total != active.total or ordinal != active.next_ordinal:
            self._raise_integrity(
                "invalid translator sequence for command "
                f"{active.command_id!r}; expected {active.next_ordinal}/{active.total}, "
                f"received {ordinal}/{total}",
                command_id=active.command_id,
                function_name=function_name,
                ordinal=ordinal,
                total=total,
                requested_function_id=requested_function_id,
                emitted_function_id=emitted_function_id,
            )
        expected_final = ordinal == total - 1
        if final_primitive != expected_final and not (final_primitive and failure_code is not None):
            self._raise_integrity(
                "translator final flag disagrees with sequence",
                command_id=active.command_id,
                function_name=function_name,
                ordinal=ordinal,
                total=total,
                requested_function_id=requested_function_id,
                emitted_function_id=emitted_function_id,
            )
        active.next_ordinal += 1
        return PrimitiveDispatch(
            active.command_id,
            function_name,
            final_primitive,
            origin,
            ordinal,
            total,
            failure_code,
            requested_function_id,
            emitted_function_id,
        )

    def command_id_for(
        self,
        agent_name: str,
        team_name: Optional[str],
        action_name: str,
    ) -> Optional[str]:
        actor = self._resolve_dispatch_actor(
            agent_name,
            team_name,
            action_name,
            active_only=False,
        )
        if actor is None:
            return None
        active = self._active_commands.get(actor)
        if active is not None and active.action_name == action_name:
            return active.command_id
        queue = self._command_queues[(actor[0], actor[1], action_name)]
        return queue[0] if queue else None

    def reject_command(
        self,
        agent_name: str,
        team_name: Optional[str],
        action_name: str,
        *,
        failure_code: str,
    ) -> Optional[PrimitiveDispatch]:
        """Claim a terminal pre-translation failure for one uniquely owned command."""

        actor = self._resolve_dispatch_actor(
            agent_name,
            team_name,
            action_name,
            active_only=False,
        )
        if actor is None:
            return None
        active = self._active_commands.get(actor)
        if active is None:
            ordinal, total = 0, 1
        else:
            if active.next_ordinal >= active.total:
                self._raise_integrity(
                    "cannot abort a completed translator sequence",
                    command_id=active.command_id,
                    function_name="pre_dispatch",
                    ordinal=active.next_ordinal,
                    total=active.total,
                )
            ordinal, total = active.next_ordinal, active.total
        return self.claim_primitive(
            agent_name,
            team_name,
            action_name,
            "pre_dispatch",
            final_primitive=True,
            origin="translator",
            ordinal=ordinal,
            total=total,
            failure_code=failure_code,
        )

    def resolve_arguments(self, command_id: str, arguments: list[Any]) -> None:
        self.coordinator.resolve_arguments(command_id, arguments)

    def screen_route_provenance(
        self,
        command_id: str,
    ) -> Optional[ScreenRouteProvenance]:
        return self._screen_route_provenance.get(command_id)

    def _resolve_dispatch_actor(
        self,
        agent_name: str,
        team_name: Optional[str],
        action_name: str,
        *,
        active_only: bool,
    ) -> Optional[tuple[str, str]]:
        if team_name is not None:
            preferred = (agent_name, team_name)
            active = self._active_commands.get(preferred)
            queued = self._command_queues.get((agent_name, team_name, action_name))
            if active is not None and active.action_name == action_name:
                return preferred
            if not active_only and queued:
                return preferred
            return None

        candidates = (
            set()
            if active_only
            else {
                (queued_agent, queued_team)
                for (
                    queued_agent,
                    queued_team,
                    queued_action,
                ), queue in self._command_queues.items()
                if queued_agent == agent_name and queued_action == action_name and queue
            }
        )
        candidates.update(
            actor
            for actor, active in self._active_commands.items()
            if actor[0] == agent_name and active.action_name == action_name
        )
        if len(candidates) > 1:
            rendered = ", ".join(f"{agent}/{team}" for agent, team in sorted(candidates))
            self._raise_integrity(
                f"primitive ownership is ambiguous for {agent_name}/{action_name}: {rendered}",
                command_id=None,
            )
        if len(candidates) == 1:
            return candidates.pop()
        return None

    def _resolve_orchestration_actor(
        self,
        agent_name: str,
        team_name: Optional[str],
        action_name: str,
    ) -> Optional[tuple[str, str]]:
        candidates = {
            (queued_agent, queued_team)
            for (queued_agent, queued_team, queued_action), queue in self._command_queues.items()
            if queued_agent == agent_name and queued_action == action_name and queue
        }
        candidates.update(
            actor
            for actor, active in self._active_commands.items()
            if actor[0] == agent_name and active.action_name == action_name
        )
        preferred = None if team_name is None else (agent_name, team_name)
        if preferred is not None and preferred in candidates:
            return preferred
        if len(candidates) > 1:
            rendered = ", ".join(f"{agent}/{team}" for agent, team in sorted(candidates))
            self._raise_integrity(
                "orchestration primitive ownership is ambiguous for "
                f"{agent_name}/{action_name}: {rendered}",
                command_id=None,
                origin="orchestration",
            )
        if len(candidates) == 1:
            return candidates.pop()
        return None

    def settle_primitive(
        self,
        dispatch: PrimitiveDispatch,
        *,
        success: bool,
        failure_reason: Optional[str] = None,
        game_loop: Optional[int] = None,
    ) -> None:
        self.coordinator.record_primitive(
            dispatch.command_id,
            dispatch.function_name,
            success=success,
            failure_reason=failure_reason,
            origin=dispatch.origin,
            ordinal=dispatch.ordinal,
            total=dispatch.total,
            game_loop=game_loop,
            failure_code=dispatch.failure_code,
            requested_function_id=dispatch.requested_function_id,
            emitted_function_id=dispatch.emitted_function_id,
        )
        if dispatch.final_primitive:
            self.coordinator.complete_command(dispatch.command_id, game_loop=game_loop)
            self._screen_route_provenance.pop(dispatch.command_id, None)
            for actor, value in list(self._active_commands.items()):
                if value.command_id == dispatch.command_id:
                    del self._active_commands[actor]

    def prepare_effect(
        self,
        dispatch: PrimitiveDispatch,
        observation: Any,
        *,
        builder_tag: Optional[int],
        producer_tag: Optional[int] = None,
    ) -> None:
        self.coordinator.prepare_effect(
            dispatch.command_id,
            observation,
            builder_tag=builder_tag,
            producer_tag=producer_tag,
        )

    def observe_effects(self, observation: Any) -> None:
        self.coordinator.observe_effects(observation)

    def end_episode(self, result: dict[str, Any]) -> None:
        self.coordinator.end_episode(result)
        self._screen_route_provenance.clear()

    def _command_id_for_actor(
        self,
        actor: tuple[str, str],
        action_name: str,
    ) -> Optional[str]:
        active = self._active_commands.get(actor)
        if active is not None:
            return active.command_id
        queue = self._command_queues.get((actor[0], actor[1], action_name))
        return queue[0] if queue else None

    def _raise_integrity(
        self,
        reason: str,
        *,
        command_id: Optional[str],
        function_name: str = "bridge_integrity",
        execution_stage: str = "translation",
        origin: str = "translator",
        ordinal: Optional[int] = None,
        total: Optional[int] = None,
        game_loop: Optional[int] = None,
        requested_function_id: Optional[int] = None,
        emitted_function_id: Optional[int] = None,
    ) -> NoReturn:
        rendered = reason.removeprefix("bridge_integrity_error: ")
        if command_id is None:
            self.record_unattributed_primitive()
        else:
            self.coordinator.fail_bridge_integrity(
                command_id,
                rendered,
                function_name=function_name,
                execution_stage=execution_stage,
                origin=origin,
                ordinal=ordinal,
                total=total,
                game_loop=game_loop,
                requested_function_id=requested_function_id,
                emitted_function_id=emitted_function_id,
            )
            self._forget_command(command_id)
        raise BridgeIntegrityError(f"bridge_integrity_error: {rendered}")

    def _forget_command(self, command_id: str) -> None:
        self._screen_route_provenance.pop(command_id, None)
        for actor, active in list(self._active_commands.items()):
            if active.command_id == command_id:
                del self._active_commands[actor]
        for key, queue in list(self._command_queues.items()):
            retained = deque(value for value in queue if value != command_id)
            if retained:
                self._command_queues[key] = retained
            else:
                self._command_queues.pop(key, None)

    def _persist_metrics_locked(self) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._metrics_path.with_name(f"{self._metrics_path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "unattributed_primitives": self.unattributed_primitives,
                    "candidate_outside_pysc2_dispatches": (self.candidate_outside_pysc2_dispatches),
                    "observation_gap_watchdog_triggers": (self.observation_gap_watchdog_triggers),
                    "orchestration_recoveries": self.orchestration_recoveries,
                    "expansion_scout_camera_moves": self.expansion_scout_camera_moves,
                }
            ),
            encoding="utf-8",
        )
        temporary.replace(self._metrics_path)

    def _decide(self, step_id: int) -> None:
        with self._condition:
            state = self._states[step_id]
            submissions = dict(state.submissions)
        try:
            first = next(iter(submissions.values()))
            agents = {name: item.agent for name, item in submissions.items()}
            snapshot = self.extractor.extract(
                first.timestep,
                agents,
                {name: item.text_observation for name, item in submissions.items()},
                step_id=step_id,
            )
            team_order = {
                name: current_team_order(item.agent) for name, item in submissions.items()
            }
            decision = self.coordinator.decide(snapshot, team_order)
        except Exception as error:
            with self._condition:
                state.error = error
                if not self._initial_decision_complete:
                    self._initial_decision_error = error
                self._states.pop(step_id, None)
                self._condition.notify_all()
            return

        with self._condition:
            state.decision = decision
            self._last_decision_game_loop = int(snapshot["game_loop"])
            self._initial_decision_complete = True
            self._planner_pending = bool(decision.action_batch.get("planner_pending", False))
            for route in decision.routes.values():
                for command in route.commands:
                    key = (route.agent_name, command.team_name, command.name)
                    self._command_queues[key].append(command.command_id)
                    if (
                        command.screen_world_target is not None
                        and command.screen_anchor_tag is not None
                    ):
                        self._screen_route_provenance[command.command_id] = ScreenRouteProvenance(
                            world_target=command.screen_world_target,
                            anchor_tag=command.screen_anchor_tag,
                        )
            self._condition.notify_all()
