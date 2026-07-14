"""A shared barrier that turns concurrent upstream queries into one runtime tick."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Condition
from typing import Any, Optional

from rtscortex_llm_pysc2.coordinator import BridgeCoordinator, BridgeDecision
from rtscortex_llm_pysc2.extractor import TimeStepExtractor, current_team_order


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


class SharedDecisionBroker:
    """Synchronize enabled LLMAgents without ticking once per subagent."""

    def __init__(
        self,
        coordinator: BridgeCoordinator,
        extractor: TimeStepExtractor,
        *,
        decision_timeout_seconds: float = 60.0,
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
        self._active_commands: dict[tuple[str, str], tuple[str, str]] = {}
        self._planner_pending = False
        self._initial_decision_started = False
        self._initial_decision_complete = False
        self._initial_decision_error: Optional[Exception] = None

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
                lambda: self._initial_decision_complete
                or self._initial_decision_error is not None,
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
        leader = False
        with self._condition:
            state = self._states.setdefault(step_id, _DecisionState())
            if state.error is not None:
                raise RuntimeError(
                    f"shared runtime decision failed at step {step_id}"
                ) from state.error
            if agent.name in state.submissions:
                raise ValueError(f"agent {agent.name!r} submitted step {step_id} twice")
            state.submissions[agent.name] = _Submission(agent, timestep, text_observation)
            expected = {name for name, value in self._agents.items() if value.enable}
            if set(state.submissions) == expected and not state.in_flight:
                state.in_flight = True
                leader = True
                if not self._initial_decision_started:
                    self._initial_decision_started = True
                    self._condition.notify_all()

        if leader:
            self._decide(step_id)

        with self._condition:
            ready = self._condition.wait_for(
                lambda: state.decision is not None or state.error is not None,
                timeout=self.decision_timeout_seconds,
            )
            if not ready:
                state.error = TimeoutError(
                    f"not all enabled agents submitted runtime step {step_id}"
                )
                if not self._initial_decision_complete:
                    self._initial_decision_error = state.error
                self._states.pop(step_id, None)
                self._condition.notify_all()
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
    ) -> Optional[PrimitiveDispatch]:
        actor = self._resolve_dispatch_actor(agent_name, team_name, action_name)
        if actor is None:
            return None
        active = self._active_commands.get(actor)
        if active is None or active[0] != action_name:
            queue = self._command_queues[(actor[0], actor[1], action_name)]
            if not queue:
                return None
            active = (action_name, queue.popleft())
            self._active_commands[actor] = active
        return PrimitiveDispatch(active[1], function_name, final_primitive)

    def _resolve_dispatch_actor(
        self,
        agent_name: str,
        team_name: Optional[str],
        action_name: str,
    ) -> Optional[tuple[str, str]]:
        if team_name is not None:
            preferred = (agent_name, team_name)
            active = self._active_commands.get(preferred)
            queued = self._command_queues.get((agent_name, team_name, action_name))
            if (active is not None and active[0] == action_name) or queued:
                return preferred

        candidates = {
            (queued_agent, queued_team)
            for (queued_agent, queued_team, queued_action), queue in self._command_queues.items()
            if queued_agent == agent_name and queued_action == action_name and queue
        }
        candidates.update(
            actor
            for actor, active in self._active_commands.items()
            if actor[0] == agent_name and active[0] == action_name
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
    ) -> None:
        self.coordinator.record_primitive(
            dispatch.command_id,
            dispatch.function_name,
            success=success,
            failure_reason=failure_reason,
        )
        if dispatch.final_primitive:
            self.coordinator.complete_command(dispatch.command_id)
            for actor, value in list(self._active_commands.items()):
                if value[1] == dispatch.command_id:
                    del self._active_commands[actor]

    def end_episode(self, result: dict[str, Any]) -> None:
        self.coordinator.end_episode(result)

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
            self._initial_decision_complete = True
            self._planner_pending = bool(decision.action_batch.get("planner_pending", False))
            for route in decision.routes.values():
                for command in route.commands:
                    key = (route.agent_name, command.team_name, command.name)
                    self._command_queues[key].append(command.command_id)
            self._condition.notify_all()
