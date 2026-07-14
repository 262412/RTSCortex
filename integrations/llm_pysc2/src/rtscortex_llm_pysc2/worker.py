"""Importable LLM-PySC2 worker classes for the pinned upstream checkout."""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

from rtscortex_llm_pysc2.broker import PrimitiveDispatch, SharedDecisionBroker
from rtscortex_llm_pysc2.clock import FixedRateGameClock, InitialPlanningBarrier
from rtscortex_llm_pysc2.coordinator import BridgeCoordinator
from rtscortex_llm_pysc2.effect_verifier import (
    DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS,
    ActionEffectVerifier,
)
from rtscortex_llm_pysc2.extractor import TimeStepExtractor, build_screen_candidates
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
    simulation_speed_multiplier: Optional[float] = None
    pause_until_first_plan: bool = False
    runtime_request_timeout_seconds: float = 60.0
    action_effect_timeout_game_loops: int = DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS

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
        simulation_speed = os.environ.get("RTSCORTEX_SIMULATION_SPEED_MULTIPLIER")
        runtime_request_timeout_seconds = float(
            os.environ.get("RTSCORTEX_RUNTIME_REQUEST_TIMEOUT_SECONDS", "60")
        )
        if runtime_request_timeout_seconds <= 0:
            raise RuntimeError("RTSCORTEX_RUNTIME_REQUEST_TIMEOUT_SECONDS must be positive")
        action_effect_timeout_game_loops = int(
            os.environ.get(
                "RTSCORTEX_ACTION_EFFECT_TIMEOUT_GAME_LOOPS",
                str(DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS),
            )
        )
        if action_effect_timeout_game_loops <= 0:
            raise RuntimeError("RTSCORTEX_ACTION_EFFECT_TIMEOUT_GAME_LOOPS must be positive")
        pause_until_first_plan = _environment_bool("RTSCORTEX_PAUSE_UNTIL_FIRST_PLAN", False)
        return cls(
            run_id=run_id,
            episode_id=episode_id,
            socket_path=os.environ.get("RTSCORTEX_RUNTIME_SOCKET")
            or os.environ.get("RTSCORTEX_SOCKET"),
            runtime_url=os.environ.get("RTSCORTEX_RUNTIME_URL", "http://127.0.0.1:8765"),
            seed=int(os.environ.get("RTSCORTEX_SEED", "0")),
            scenario=os.environ.get("RTSCORTEX_SCENARIO", "pvz_task1_level1"),
            pending_plan_step_delay_seconds=pending_plan_step_delay_seconds,
            simulation_speed_multiplier=(
                float(simulation_speed) if simulation_speed is not None else None
            ),
            pause_until_first_plan=pause_until_first_plan,
            runtime_request_timeout_seconds=runtime_request_timeout_seconds,
            action_effect_timeout_game_loops=action_effect_timeout_game_loops,
        )


class RTSCortexLLMAgent(RuntimeQueryMixin, _LLMAgentBase):  # type: ignore[misc]
    """Real upstream LLMAgent subclass with its model call replaced by a broker."""

    def __init__(self, *args: Any, broker: SharedDecisionBroker, **kwargs: Any) -> None:
        _require_upstream()
        super().__init__(*args, **kwargs)
        self.broker = broker
        broker.register(self)

    def get_func(self, obs: Any) -> Any:
        if not self.func_list and self.action_list:
            _refresh_build_action_position(self.action_list[0], obs.observation)
        return super().get_func(obs)


class RTSCortexMainAgent(_MainAgentBase):  # type: ignore[misc]
    """Keep the upstream environment loop and attach RTSCortex at its query seam."""

    def __init__(self) -> None:
        _require_upstream()
        self.worker_settings = WorkerSettings.from_environment()
        self.runtime_client = RuntimeClient(
            base_url=self.worker_settings.runtime_url,
            unix_socket=self.worker_settings.socket_path,
            timeout_seconds=self.worker_settings.runtime_request_timeout_seconds,
        )
        self.runtime_client.health()
        unit_names, building_types = _unit_metadata()
        coordinator = BridgeCoordinator(
            self.runtime_client,
            effect_verifier=ActionEffectVerifier(
                timeout_game_loops=self.worker_settings.action_effect_timeout_game_loops,
                unit_names=unit_names,
            ),
        )
        extractor = TimeStepExtractor(
            self.worker_settings.run_id,
            self.worker_settings.episode_id,
            unit_names=unit_names,
            building_types=building_types,
            action_source_types=_production_action_source_types(),
        )
        self.decision_broker = SharedDecisionBroker(coordinator, extractor)
        self._pending_primitive: Optional[PrimitiveDispatch] = None
        self._episode_reported = False
        self.initial_planning_barrier = InitialPlanningBarrier()
        self.game_clock = (
            FixedRateGameClock(self.worker_settings.simulation_speed_multiplier)
            if self.worker_settings.simulation_speed_multiplier is not None
            else None
        )

        config = _scenario_config(self.worker_settings.scenario)
        subagent = partial(RTSCortexLLMAgent, broker=self.decision_broker)
        super().__init__(config, subagent)
        _apply_scenario_bootstrap(self, self.worker_settings.scenario)

    def step(self, obs: Any) -> Any:
        self._settle_previous_primitive(obs)
        self.decision_broker.observe_effects(obs.observation)
        if _is_terminal(obs):
            return _finish_terminal(self, obs, _base_agent_step, _no_op)
        action = super().step(obs)
        if (
            self.worker_settings.pause_until_first_plan
            and self.initial_planning_barrier.blocks_steps
            and self.decision_broker.initial_decision_started
        ):
            self.decision_broker.wait_for_initial_decision(
                self.worker_settings.runtime_request_timeout_seconds + 5.0
            )
            self.initial_planning_barrier.release()
            if self.game_clock is not None:
                self.game_clock.reset()
        self._capture_primitive(action, obs)
        delay = _pending_plan_idle_delay(
            action,
            planner_pending=self.decision_broker.planner_pending,
            configured_delay_seconds=self.worker_settings.pending_plan_step_delay_seconds,
        )
        if self.game_clock is not None:
            self.game_clock.wait_for_step()
        elif delay:
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
            game_loop=_observation_game_loop(obs.observation),
        )
        self._pending_primitive = None

    def _capture_primitive(self, action: Any, obs: Any) -> None:
        if self._pending_primitive is not None or not self.AGENT_NAMES:
            return
        agent = self.agents[self.AGENT_NAMES[self.agent_id]]
        team_name = _execution_team_name(agent)
        action_name = getattr(agent, "curr_action_name", "")
        function_id = getattr(action, "function", None)
        if not action_name or function_id is None:
            return
        action_specification = getattr(agent.translator_a, "ACTION_SPACE_DICT", {}).get(action_name)
        if action_specification is None:
            return
        expected_ids = {int(triple[0]) for triple in action_specification.get("func", ())}
        function_name = _function_name(int(function_id))
        final_primitive = int(function_id) in expected_ids and len(agent.func_list) == 0
        dispatch = self.decision_broker.claim_primitive(
            agent.name,
            team_name,
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
                game_loop=_observation_game_loop(obs.observation),
            )
            return
        if dispatch.final_primitive:
            self.decision_broker.prepare_effect(
                dispatch,
                obs.observation,
                builder_tag=_execution_unit_tag(agent),
            )
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
        "Simple64": ("rtscortex_llm_pysc2.melee", "RTSCortexMeleeConfig"),
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


def _production_action_source_types() -> dict[int, int]:
    actions = importlib.import_module("pysc2.lib.actions")
    llm_action = importlib.import_module("llm_pysc2.lib.llm_action")
    result: dict[int, int] = {}
    for function_id in range(len(actions.FUNCTIONS)):
        source_type = llm_action.find_unit_type_the_func_belongs_to(function_id, "protoss")
        if source_type is not None:
            result[function_id] = int(source_type)
    return result


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


def _execution_team_name(agent: Any) -> Optional[str]:
    """Recover the implicit Empty actor omitted by the upstream execution loop."""

    team_name = getattr(agent, "team_unit_team_curr", None)
    if team_name is not None:
        return str(team_name)
    if getattr(agent, "flag_enable_empty_unit_group", False):
        return "Empty"
    return None


def _execution_unit_tag(agent: Any) -> Optional[int]:
    value = getattr(agent, "team_unit_tag_curr", None)
    return None if value is None else int(value)


def _observation_game_loop(observation: Any) -> int:
    value: Any = (
        observation.get("game_loop", 0)
        if isinstance(observation, Mapping)
        else getattr(observation, "game_loop", 0)
    )
    if isinstance(value, (str, bytes)):
        return int(value)
    try:
        if len(value) == 1:
            return int(value[0])
    except (TypeError, IndexError):
        pass
    return int(value)


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be 'true' or 'false'")


def _refresh_build_action_position(action: dict[str, Any], observation: Any) -> bool:
    candidates = build_screen_candidates(observation, str(action.get("name", "")))
    if not candidates:
        return False
    position = candidates[0]
    refreshed_functions = []
    for function_id, function, arguments in action.get("func", ()):
        refreshed_arguments = tuple(
            position
            if isinstance(argument, list)
            and len(argument) == 2
            and all(isinstance(value, (int, float)) for value in argument)
            else argument
            for argument in arguments
        )
        refreshed_functions.append((function_id, function, refreshed_arguments))
    action["func"] = refreshed_functions
    return True


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
