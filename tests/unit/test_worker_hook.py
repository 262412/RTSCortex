from __future__ import annotations

import importlib
import json
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rtscortex_llm_pysc2.broker import SharedDecisionBroker
from rtscortex_llm_pysc2.coordinator import BridgeCoordinator
from rtscortex_llm_pysc2.extractor import TimeStepExtractor
from rtscortex_llm_pysc2.hook import RuntimeDecisionBroker, RuntimeQueryMixin
from rtscortex_llm_pysc2.observation import ObservationMapper
from rtscortex_llm_pysc2.worker import (
    WorkerSettings,
    _apply_scenario_bootstrap,
    _finish_terminal,
    _pending_plan_idle_delay,
    _scenario_config,
)

from rtscortex.contracts import ObservationEnvelope


def test_timestep_extractor_produces_json_safe_five_part_snapshot() -> None:
    agents = {"CombatGroup7": FakeAgent("CombatGroup7", "Adept-1", _fake_timestep(), StubBroker())}
    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={311: "Adept", 59: "Nexus", 104: "Drone"},
        building_types=(59,),
    ).extract(
        _fake_timestep(),
        agents,
        {"CombatGroup7": "Adept sees a Drone."},
        step_id=3,
    )

    json.dumps(snapshot)
    envelope = ObservationEnvelope.model_validate(ObservationMapper().map(snapshot))

    assert envelope.state.economy.minerals == 375
    assert envelope.state.production_queue[0].name == "ability:141"
    assert envelope.state.own_units[0].unit_type == "Adept"
    assert envelope.state.own_structures[0].unit_type == "Nexus"
    assert envelope.state.visible_enemies[0].unit_type == "Drone"
    assert envelope.available_actions[1].argument_names == ["tag"]
    assert envelope.available_actions[1].argument_types == ["tag"]
    assert "Unsupported" not in {action.name for action in envelope.available_actions}


def test_shared_broker_calls_runtime_once_and_distributes_to_all_agents() -> None:
    runtime = FakeRuntime()
    broker = SharedDecisionBroker(
        BridgeCoordinator(runtime),
        TimeStepExtractor(
            "run-worker",
            "episode-worker",
            unit_names={311: "Adept", 59: "Nexus", 104: "Drone"},
            building_types=(59,),
        ),
    )
    timestep = _fake_timestep()
    first = FakeAgent("AgentA", "A", timestep, broker)
    second = FakeAgent("AgentB", "B", timestep, broker)
    broker.register(first)
    broker.register(second)

    threads = [threading.Thread(target=agent.query, args=(timestep,)) for agent in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert runtime.tick_calls == 1
    assert broker.planner_pending is False
    assert first.text_observation_calls == 1
    assert second.text_observation_calls == 1
    assert first.action_translation_calls == 1
    assert second.action_translation_calls == 1
    assert "<Attack_Unit(0x137)>" in first.action_text
    assert "Team B:\n        <No_Operation()>" in second.action_text

    dispatch = broker.claim_primitive(
        "AgentA",
        "A",
        "Attack_Unit",
        "Attack_screen",
        final_primitive=True,
    )
    assert dispatch is not None
    broker.settle_primitive(dispatch, success=True)
    broker.end_episode(_episode_result())

    assert runtime.execution_reports[0]["command_id"] == "command-attack"
    assert runtime.execution_reports[0]["pysc2_function"] == "Attack_screen"
    assert runtime.episode_results == [_episode_result()]


def test_query_mixin_delegates_to_upstream_base_methods() -> None:
    agent = MROAgent()

    agent.query(object())

    assert agent.calls == ["communication", "observation", "translation"]
    assert cast(StubBroker, agent.broker).submissions == [(agent, "upstream observation")]
    assert agent.action_lists == [[{"name": "translated"}]]


def test_broker_times_out_if_an_enabled_agent_never_submits() -> None:
    runtime = FakeRuntime()
    broker = SharedDecisionBroker(
        BridgeCoordinator(runtime),
        TimeStepExtractor("run-worker", "episode-worker"),
        decision_timeout_seconds=0.01,
    )
    timestep = _fake_timestep()
    first = FakeAgent("AgentA", "A", timestep, broker)
    second = FakeAgent("AgentB", "B", timestep, broker)
    broker.register(first)
    broker.register(second)

    with pytest.raises(RuntimeError, match="shared runtime decision failed"):
        broker.submit(first, timestep, "only one submission")

    assert runtime.tick_calls == 0


def test_terminal_seam_skips_main_loop_reports_and_closes() -> None:
    events: list[str] = []
    agent = TerminalAgent(events)

    def base_step(_agent: Any, _obs: Any) -> None:
        events.append("base-step")

    def no_op() -> str:
        events.append("no-op")
        return "no-op-action"

    result = _finish_terminal(agent, object(), base_step, no_op)

    assert result == "no-op-action"
    assert events == ["base-step", "episode-end", "close", "no-op"]


def test_worker_settings_prefer_canonical_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RTSCORTEX_RUN_ID", "run-env")
    monkeypatch.setenv("RTSCORTEX_EPISODE_ID", "episode-env")
    monkeypatch.setenv("RTSCORTEX_RUNTIME_SOCKET", "/tmp/canonical.sock")
    monkeypatch.setenv("RTSCORTEX_SOCKET", "/tmp/legacy.sock")
    monkeypatch.setenv("RTSCORTEX_SCENARIO", "pvz_task1_level1")
    monkeypatch.setenv("RTSCORTEX_SEED", "17")
    monkeypatch.setenv("RTSCORTEX_PENDING_PLAN_STEP_DELAY_SECONDS", "0.75")

    settings = WorkerSettings.from_environment()

    assert settings.socket_path == "/tmp/canonical.sock"
    assert settings.scenario == "pvz_task1_level1"
    assert settings.seed == 17
    assert settings.pending_plan_step_delay_seconds == 0.75


def test_shared_broker_exposes_pending_planner_state() -> None:
    runtime = FakeRuntime(planner_pending=True)
    broker = SharedDecisionBroker(
        BridgeCoordinator(runtime),
        TimeStepExtractor(
            "run-worker",
            "episode-worker",
            unit_names={311: "Adept", 59: "Nexus", 104: "Drone"},
            building_types=(59,),
        ),
    )
    timestep = _fake_timestep()
    agent = FakeAgent("AgentA", "A", timestep, broker)
    broker.register(agent)

    agent.query(timestep)

    assert broker.planner_pending is True


def test_pending_plan_pacing_only_delays_no_op() -> None:
    assert (
        _pending_plan_idle_delay(
            SimpleNamespace(function=0),
            planner_pending=True,
            configured_delay_seconds=0.75,
        )
        == 0.75
    )
    assert (
        _pending_plan_idle_delay(
            SimpleNamespace(function=12),
            planner_pending=True,
            configured_delay_seconds=0.75,
        )
        == 0.0
    )
    assert (
        _pending_plan_idle_delay(
            SimpleNamespace(function=0),
            planner_pending=False,
            configured_delay_seconds=0.75,
        )
        == 0.0
    )


def test_timestep_extractor_maps_sc2_attack_alerts() -> None:
    agent = FakeAgent("CombatGroupSmac", "Stalker-1", _fake_timestep(), StubBroker())
    snapshot = TimeStepExtractor("run-worker", "episode-worker").extract(
        _fake_timestep(alerts=[6, 19, 3]),
        {"CombatGroupSmac": agent},
        {"CombatGroupSmac": "under attack"},
        step_id=1,
    )

    assert snapshot["alerts"] == ["building_under_attack", "unit_under_attack", "alert:3"]


def test_worker_selects_2s3z_config_and_adds_no_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_op_function = object()

    class FakeSmacConfig:
        def __init__(self) -> None:
            self.AGENTS = {
                "CombatGroupSmac": {
                    "team": [
                        {"name": "Zealot-1"},
                        {"name": "Zealot-2"},
                        {"name": "Stalker-1"},
                    ],
                    "action": {
                        "Zealot": [{"name": "Attack_Unit", "arg": ["tag"], "func": []}],
                        "Stalker": [{"name": "Attack_Unit", "arg": ["tag"], "func": []}],
                    },
                }
            }
            self.reset_args: tuple[str, str, str] | None = None

        def reset_llm(self, model_name: str, api_base: str, api_key: str) -> None:
            self.reset_args = (model_name, api_base, api_key)

    def fake_import(name: str) -> Any:
        if name == "llm_pysc2.agents.configs.llm_smac":
            return SimpleNamespace(ConfigSmac_2s3z=FakeSmacConfig)
        if name == "pysc2.lib.actions":
            return SimpleNamespace(FUNCTIONS=SimpleNamespace(no_op=no_op_function))
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    config = _scenario_config("2s3z")

    assert config.reset_args == ("gpt-3.5-turbo", "http://127.0.0.1", "rtscortex-unused")
    assert [team["name"] for team in config.AGENTS["CombatGroupSmac"]["team"]] == [
        "Zealot-1",
        "Zealot-2",
        "Stalker-1",
    ]
    for actions in config.AGENTS["CombatGroupSmac"]["action"].values():
        assert actions[0] == {
            "name": "No_Operation",
            "arg": [],
            "func": [(0, no_op_function, ())],
        }


def test_worker_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="unsupported worker scenario"):
        _scenario_config("unknown")


def test_2s3z_bootstrap_skips_unreachable_camera_calibration() -> None:
    agent = SimpleNamespace(
        world_range=0,
        world_x_offset=7,
        world_y_offset=9,
        world_xy_calibration=False,
    )

    _apply_scenario_bootstrap(agent, "2s3z")

    assert agent.world_range == 0
    assert agent.world_x_offset == 7
    assert agent.world_y_offset == 9
    assert agent.world_xy_calibration is True


def test_standard_scenario_keeps_upstream_camera_calibration() -> None:
    agent = SimpleNamespace(
        world_range=0,
        world_x_offset=0,
        world_y_offset=0,
        world_xy_calibration=False,
    )

    _apply_scenario_bootstrap(agent, "pvz_task1_level1")

    assert agent.world_range == 0
    assert agent.world_xy_calibration is False


class FakeAgent(RuntimeQueryMixin):
    def __init__(
        self,
        name: str,
        team_name: str,
        timestep: Any,
        broker: RuntimeDecisionBroker,
    ) -> None:
        self.name = name
        self.main_loop_step = 3
        self.enable = True
        self.lock = threading.Lock()
        self.is_waiting = False
        self.first_action = False
        self.action_lists: list[Any] = []
        self.team_unit_team_list = [team_name]
        self.team_unit_obs_list = [timestep]
        self.text_observation_calls = 0
        self.action_translation_calls = 0
        self.action_text = ""
        self.broker = broker
        no_op = {"name": "No_Operation", "arg": [], "func": [(0, None, ())]}
        attack = {"name": "Attack_Unit", "arg": ["tag"], "func": [(12, None, ())]}
        unsupported = {
            "name": "Unsupported",
            "arg": ["string"],
            "func": [(12, None, ())],
        }
        self.config = SimpleNamespace(
            AGENTS={
                name: {
                    "team": [{"name": team_name, "unit_type": [311]}],
                    "action": {311: [no_op, attack, unsupported]},
                }
            }
        )

    def get_text_c_inp(self) -> None:
        return None

    def get_text_o(self, obs: Any) -> str:
        assert obs is self.team_unit_obs_list[0]
        self.text_observation_calls += 1
        return f"observation from {self.name}"

    def get_func_a(self, raw_text_a: str) -> tuple[list[Any], dict[str, Any]]:
        self.action_translation_calls += 1
        self.action_text = raw_text_a
        return [[{"name": "translated"}]], {}


class StubBroker:
    def __init__(self) -> None:
        self.submissions: list[tuple[Any, str]] = []

    def submit(self, agent: Any, obs: Any, text_observation: str) -> str:
        del obs
        self.submissions.append((agent, text_observation))
        return "Actions:\n    Team A:\n        <No_Operation()>"


class UpstreamStyleBase:
    def __init__(self) -> None:
        self.broker: RuntimeDecisionBroker = StubBroker()
        self.lock = threading.Lock()
        self.is_waiting = False
        self.first_action = False
        self.action_lists: list[Any] = []
        self.calls: list[str] = []

    def get_text_c_inp(self) -> None:
        self.calls.append("communication")

    def get_text_o(self, obs: Any) -> str:
        del obs
        self.calls.append("observation")
        return "upstream observation"

    def get_func_a(self, raw_text_a: str) -> tuple[list[Any], dict[str, Any]]:
        assert "No_Operation" in raw_text_a
        self.calls.append("translation")
        return [[{"name": "translated"}]], {}


class MROAgent(RuntimeQueryMixin, UpstreamStyleBase):
    pass


class TerminalAgent:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._episode_reported = False
        self.runtime_client = SimpleNamespace(close=lambda: events.append("close"))

    def _report_episode(self, obs: Any) -> None:
        del obs
        self.events.append("episode-end")
        self._episode_reported = True


class FakeRuntime:
    def __init__(self, *, planner_pending: bool = False) -> None:
        self.tick_calls = 0
        self.planner_pending = planner_pending
        self.execution_reports: list[dict[str, Any]] = []
        self.episode_results: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def tick(self, observation: dict[str, Any]) -> dict[str, Any]:
        ObservationEnvelope.model_validate(observation)
        self.tick_calls += 1
        return {
            "protocol_version": "1.0",
            "run_id": observation["run_id"],
            "episode_id": observation["episode_id"],
            "step_id": observation["step_id"],
            "decision_id": "decision-worker",
            "strategic_goal": "harass",
            "summary": "attack the visible worker",
            "planner_pending": self.planner_pending,
            "commands": [
                {
                    "command_id": "command-attack",
                    "actor": "AgentA/A",
                    "name": "Attack_Unit",
                    "arguments": [311],
                    "priority": 50,
                    "ttl_game_loops": 20,
                    "created_game_loop": observation["game_loop"],
                    "source": "planner",
                    "preconditions": {},
                }
            ],
            "rejected_commands": [],
        }

    def execution(self, report: dict[str, Any]) -> None:
        self.execution_reports.append(report)

    def end_episode(self, result: dict[str, Any]) -> None:
        self.episode_results.append(result)


def _fake_timestep(*, alerts: list[int] | None = None) -> Any:
    available_actions = [0, 12]
    observation = SimpleNamespace(
        player=SimpleNamespace(
            minerals=375,
            vespene=100,
            food_used=34,
            food_cap=46,
            food_workers=20,
            food_army=14,
        ),
        raw_units=[
            _unit(4300734465, 311, 1, 42, 38, 112, 204),
            _unit(4294967297, 59, 1, 30, 30, 1000, 255),
            _unit(4316463105, 104, 4, 52, 32, 30, 191),
        ],
        production_queue=[SimpleNamespace(ability_id=141, build_progress=50)],
        upgrades=[84],
        alerts=[3] if alerts is None else alerts,
        available_actions=available_actions,
        game_loop=[224],
    )
    return SimpleNamespace(observation=observation, reward=0, last=lambda: False)


def _unit(
    tag: int,
    unit_type: int,
    alliance: int,
    x: int,
    y: int,
    health: int,
    health_ratio: int,
) -> Any:
    return SimpleNamespace(
        tag=tag,
        unit_type=unit_type,
        alliance=alliance,
        x=x,
        y=y,
        health=health,
        health_ratio=health_ratio,
        energy=0,
        order_length=0,
    )


def _episode_result() -> dict[str, Any]:
    return {
        "protocol_version": "1.0",
        "run_id": "run-worker",
        "episode_id": "episode-worker",
        "scenario": "pvz_task1_level1",
        "seed": 0,
        "outcome": "victory",
        "score": 1.0,
        "steps": 3,
        "metrics": {},
        "failure_reason": None,
    }
