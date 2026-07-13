"""Importable LLM-PySC2 worker classes for the pinned upstream checkout."""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

from rtscortex_llm_pysc2.broker import PrimitiveDispatch, SharedDecisionBroker
from rtscortex_llm_pysc2.coordinator import BridgeCoordinator
from rtscortex_llm_pysc2.extractor import TimeStepExtractor
from rtscortex_llm_pysc2.hook import RuntimeQueryMixin
from rtscortex_llm_pysc2.protocol import RuntimeClient

try:
    _upstream_agents = importlib.import_module("llm_pysc2.agents")
except ModuleNotFoundError as error:
    _UPSTREAM_IMPORT_ERROR: Optional[ModuleNotFoundError] = error
    _MainAgentBase: Any = object
    _LLMAgentBase: Any = object
else:
    _UPSTREAM_IMPORT_ERROR = None
    _MainAgentBase = _upstream_agents.MainAgent
    _LLMAgentBase = _upstream_agents.LLMAgent


@dataclass(frozen=True)
class WorkerSettings:
    run_id: str
    episode_id: str
    socket_path: Optional[str]
    runtime_url: str
    seed: int
    scenario: str = "pvz_task1_level1"
    pending_plan_step_delay_seconds: float = 0.0

    @classmethod
    def from_environment(cls) -> WorkerSettings:
        run_id = os.environ.get("RTSCORTEX_RUN_ID")
        episode_id = os.environ.get("RTSCORTEX_EPISODE_ID")
        if not run_id or not episode_id:
            raise RuntimeError("RTSCORTEX_RUN_ID and RTSCORTEX_EPISODE_ID must be set")
        pending_plan_step_delay_seconds = float(
            os.environ.get("RTSCORTEX_PENDING_PLAN_STEP_DELAY_SECONDS", "0")
        )
        if pending_plan_step_delay_seconds < 0:
            raise RuntimeError("RTSCORTEX_PENDING_PLAN_STEP_DELAY_SECONDS cannot be negative")
        return cls(
            run_id=run_id,
            episode_id=episode_id,
            socket_path=os.environ.get("RTSCORTEX_RUNTIME_SOCKET")
            or os.environ.get("RTSCORTEX_SOCKET"),
            runtime_url=os.environ.get("RTSCORTEX_RUNTIME_URL", "http://127.0.0.1:8765"),
            seed=int(os.environ.get("RTSCORTEX_SEED", "0")),
            scenario=os.environ.get("RTSCORTEX_SCENARIO", "pvz_task1_level1"),
            pending_plan_step_delay_seconds=pending_plan_step_delay_seconds,
        )


class RTSCortexLLMAgent(RuntimeQueryMixin, _LLMAgentBase):  # type: ignore[misc]
    """Real upstream LLMAgent subclass with its model call replaced by a broker."""

    def __init__(self, *args: Any, broker: SharedDecisionBroker, **kwargs: Any) -> None:
        _require_upstream()
        super().__init__(*args, **kwargs)
        self.broker = broker
        broker.register(self)


class RTSCortexMainAgent(_MainAgentBase):  # type: ignore[misc]
    """Keep the upstream environment loop and attach RTSCortex at its query seam."""

    def __init__(self) -> None:
        _require_upstream()
        self.worker_settings = WorkerSettings.from_environment()
        self.runtime_client = RuntimeClient(
            base_url=self.worker_settings.runtime_url,
            unix_socket=self.worker_settings.socket_path,
        )
        self.runtime_client.health()
        coordinator = BridgeCoordinator(self.runtime_client)
        unit_names, building_types = _unit_metadata()
        extractor = TimeStepExtractor(
            self.worker_settings.run_id,
            self.worker_settings.episode_id,
            unit_names=unit_names,
            building_types=building_types,
        )
        self.decision_broker = SharedDecisionBroker(coordinator, extractor)
        self._pending_primitive: Optional[PrimitiveDispatch] = None
        self._episode_reported = False

        config = _scenario_config(self.worker_settings.scenario)
        subagent = partial(RTSCortexLLMAgent, broker=self.decision_broker)
        super().__init__(config, subagent)
        _apply_scenario_bootstrap(self, self.worker_settings.scenario)

    def step(self, obs: Any) -> Any:
        self._settle_previous_primitive(obs)
        if _is_terminal(obs):
            return _finish_terminal(self, obs, _base_agent_step, _no_op)
        action = super().step(obs)
        self._capture_primitive(action)
        delay = _pending_plan_idle_delay(
            action,
            planner_pending=self.decision_broker.planner_pending,
            configured_delay_seconds=self.worker_settings.pending_plan_step_delay_seconds,
        )
        if delay:
            time.sleep(delay)
        return action

    def _settle_previous_primitive(self, obs: Any) -> None:
        dispatch = self._pending_primitive
        if dispatch is None:
            return
        action_results = list(getattr(obs.observation, "action_result", ()))
        failure_reason = None
        if action_results:
            failure_reason = ", ".join(
                f"PySC2 action result {int(value)}" for value in action_results
            )
        self.decision_broker.settle_primitive(
            dispatch,
            success=not action_results,
            failure_reason=failure_reason,
        )
        self._pending_primitive = None

    def _capture_primitive(self, action: Any) -> None:
        if self._pending_primitive is not None or not self.AGENT_NAMES:
            return
        agent = self.agents[self.AGENT_NAMES[self.agent_id]]
        team_name = getattr(agent, "team_unit_team_curr", None)
        action_name = getattr(agent, "curr_action_name", "")
        function_id = getattr(action, "function", None)
        if not team_name or not action_name or function_id is None:
            return
        action_specification = getattr(agent.translator_a, "ACTION_SPACE_DICT", {}).get(action_name)
        if action_specification is None:
            return
        expected_ids = {int(triple[0]) for triple in action_specification.get("func", ())}
        function_name = _function_name(int(function_id))
        final_primitive = int(function_id) in expected_ids and len(agent.func_list) == 0
        dispatch = self.decision_broker.claim_primitive(
            agent.name,
            str(team_name),
            str(action_name),
            function_name,
            final_primitive=final_primitive,
        )
        if dispatch is None:
            return
        if int(function_id) == 0 and 0 not in expected_ids:
            self.decision_broker.settle_primitive(
                PrimitiveDispatch(dispatch.command_id, function_name, True),
                success=False,
                failure_reason="upstream action translation or argument validation failed",
            )
            return
        self._pending_primitive = dispatch

    def _report_episode(self, obs: Any) -> None:
        reward = float(getattr(obs, "reward", 0.0) or 0.0)
        outcome = "victory" if reward > 0 else "defeat" if reward < 0 else "draw"
        self.decision_broker.end_episode(
            {
                "protocol_version": "1.0",
                "run_id": self.worker_settings.run_id,
                "episode_id": self.worker_settings.episode_id,
                "scenario": self.worker_settings.scenario,
                "seed": self.worker_settings.seed,
                "outcome": outcome,
                "score": reward,
                "steps": int(self.steps),
                "metrics": {},
                "failure_reason": None,
            }
        )
        self._episode_reported = True


def _require_upstream() -> None:
    if _UPSTREAM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LLM-PySC2 is unavailable; install the pinned submodule in the worker environment"
        ) from _UPSTREAM_IMPORT_ERROR


def _scenario_config(scenario: str) -> Any:
    definitions = {
        "pvz_task1_level1": (
            "llm_pysc2.agents.configs.llm_pysc2",
            "ConfigPysc2_Harass",
        ),
        "2s3z": ("llm_pysc2.agents.configs.llm_smac", "ConfigSmac_2s3z"),
    }
    try:
        module_name, class_name = definitions[scenario]
    except KeyError as error:
        supported = ", ".join(sorted(definitions))
        raise ValueError(
            f"unsupported worker scenario {scenario!r}; supported scenarios: {supported}"
        ) from error
    module = importlib.import_module(module_name)
    config = getattr(module, class_name)()
    config.reset_llm(
        model_name="gpt-3.5-turbo",
        api_base="http://127.0.0.1",
        api_key="rtscortex-unused",
    )
    if scenario == "pvz_task1_level1":
        for team in config.AGENTS["CombatGroup7"]["team"]:
            team["task"] = [
                {
                    "time": None,
                    "pos": [52, 32],
                    "info": "Reach minimap [52, 32] while avoiding detection and attacks.",
                },
                {
                    "time": None,
                    "pos": None,
                    "info": "Destroy as many enemy workers as possible.",
                },
            ]
    actions = importlib.import_module("pysc2.lib.actions")
    _ensure_no_operation(config, actions.FUNCTIONS.no_op)
    return config


def _ensure_no_operation(config: Any, no_op_function: Any) -> None:
    """Make the bridge's declared fallback a real upstream action."""

    for agent in config.AGENTS.values():
        action_space = agent["action"]
        for unit_type, unit_actions in list(action_space.items()):
            if any(action["name"] == "No_Operation" for action in unit_actions):
                continue
            action_space[unit_type] = [
                {
                    "name": "No_Operation",
                    "arg": [],
                    "func": [(0, no_op_function, ())],
                },
                *list(unit_actions),
            ]


def _apply_scenario_bootstrap(agent: Any, scenario: str) -> None:
    """Skip strict centering that the small SMAC arena cannot satisfy."""

    if scenario != "2s3z":
        return
    agent.world_xy_calibration = True


def _unit_metadata() -> tuple[dict[int, str], tuple[int, ...]]:
    units = importlib.import_module("pysc2.lib.units")
    names = {}
    for race in (units.Neutral, units.Protoss, units.Terran, units.Zerg):
        names.update({int(value): value.name for value in race})
    utils = importlib.import_module("llm_pysc2.lib.utils")
    return names, tuple(int(value) for value in utils.BUILDING_TYPE)


def _function_name(function_id: int) -> str:
    actions = importlib.import_module("pysc2.lib.actions")
    return str(actions.FUNCTIONS[function_id].name)


def _pending_plan_idle_delay(
    action: Any,
    *,
    planner_pending: bool,
    configured_delay_seconds: float,
) -> float:
    function_id = getattr(action, "function", None)
    if planner_pending and function_id == 0:
        return configured_delay_seconds
    return 0.0


def _finish_terminal(
    agent: Any,
    obs: Any,
    base_step: Callable[[Any, Any], None],
    no_op: Callable[[], Any],
) -> Any:
    """Finalize an episode without re-entering the upstream decision loop."""

    base_step(agent, obs)
    if not agent._episode_reported:
        try:
            agent._report_episode(obs)
        finally:
            agent.runtime_client.close()
    return no_op()


def _base_agent_step(agent: Any, obs: Any) -> None:
    module = importlib.import_module("pysc2.agents.base_agent")
    module.BaseAgent.step(agent, obs)


def _no_op() -> Any:
    actions = importlib.import_module("pysc2.lib.actions")
    return actions.FUNCTIONS.no_op()


def _is_terminal(obs: Any) -> bool:
    last = getattr(obs, "last", None)
    return bool(last()) if callable(last) else False
