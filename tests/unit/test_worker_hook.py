from __future__ import annotations

import importlib
import inspect
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rtscortex_llm_pysc2.addon import ADDON_SPECS
from rtscortex_llm_pysc2.broker import PrimitiveDispatch, SharedDecisionBroker
from rtscortex_llm_pysc2.coordinator import BridgeCoordinator
from rtscortex_llm_pysc2.extractor import (
    TimeStepExtractor,
    build_screen_candidates,
    current_team_order,
    nexus_placement_footprint_is_visible,
    production_source_tag,
    screen_build_position_is_legal,
    semantic_argument_candidates,
)
from rtscortex_llm_pysc2.hook import RuntimeDecisionBroker, RuntimeQueryMixin
from rtscortex_llm_pysc2.observation import ObservationMapper
from rtscortex_llm_pysc2.production import PRODUCTION_SPECS
from rtscortex_llm_pysc2.routing import RoutedActionBatch, RoutedCommand
from rtscortex_llm_pysc2.worker import (
    RTSCortexLLMAgent,
    RTSCortexMainAgent,
    WorkerSettings,
    _apply_scenario_bootstrap,
    _candidate_dispatch_failure,
    _canonical_pysc2_arguments,
    _execution_team_name,
    _finish_terminal,
    _isolate_next_action,
    _pending_plan_idle_delay,
    _prime_deterministic_gas_rebalance,
    _producer_is_visible,
    _production_source_invalid_reason,
    _rebind_builder_to_selected_worker,
    _refresh_build_action_position,
    _release_runtime_observation_barrier,
    _replace_screen_action_position,
    _requires_terran_production_chain,
    _resolve_build_action_position,
    _run_with_auto_worker_management_guard,
    _scenario_config,
    _semantic_target_failure,
    _should_block_gas_rebalance,
    _translated_build_position,
    _translation_failure_code,
    _upstream_replaced_production_with_noop,
    _worker_player_race,
)

from rtscortex.contracts import ObservationEnvelope


def test_timestep_extractor_produces_json_safe_five_part_snapshot() -> None:
    agents = {"CombatGroup7": FakeAgent("CombatGroup7", "Adept-1", _fake_timestep(), StubBroker())}
    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={311: "Adept", 59: "Nexus", 104: "Drone"},
        upgrade_names={84: "WarpGateResearch"},
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
    assert envelope.state.upgrades == ["WarpGateResearch"]
    assert envelope.available_actions[1].argument_names == ["tag"]
    assert envelope.available_actions[1].argument_types == ["tag"]
    assert "Unsupported" not in {action.name for action in envelope.available_actions}


@pytest.mark.parametrize(
    ("build_progress", "order_length", "expected_status"),
    [
        (0.5, 0, "constructing"),
        (50, 1, "constructing"),
        (1.0, 0, "idle"),
        (100, 1, "active"),
    ],
)
def test_timestep_extractor_marks_incomplete_structures_as_constructing(
    build_progress: float,
    order_length: int,
    expected_status: str,
) -> None:
    timestep = _fake_timestep()
    nexus = timestep.observation.raw_units[1]
    nexus.build_progress = build_progress
    nexus.order_length = order_length
    agent = FakeAgent("CombatGroup7", "Adept-1", timestep, StubBroker())

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={59: "Nexus"},
        building_types=(59,),
    ).extract(timestep, {"CombatGroup7": agent}, {"CombatGroup7": ""}, step_id=3)

    nexus_snapshot = next(unit for unit in snapshot["units"] if unit["unit_type"] == "Nexus")
    assert nexus_snapshot["status"] == expected_status


def test_timestep_extractor_keeps_missing_build_progress_idle() -> None:
    timestep = _fake_timestep()
    agent = FakeAgent("CombatGroup7", "Adept-1", timestep, StubBroker())

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={59: "Nexus"},
        building_types=(59,),
    ).extract(timestep, {"CombatGroup7": agent}, {"CombatGroup7": ""}, step_id=3)

    nexus_snapshot = next(unit for unit in snapshot["units"] if unit["unit_type"] == "Nexus")
    assert nexus_snapshot["status"] == "idle"


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
    assert "<Attack_Unit(0x101480001)>" in first.action_text
    assert "Team B:\n        <No_Operation()>" in second.action_text

    dispatch = broker.claim_primitive(
        "AgentA",
        "A",
        "Attack_Unit",
        "Attack_screen",
        final_primitive=True,
        ordinal=0,
        total=1,
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


def test_broker_releases_barrier_when_missing_combat_agent_is_disabled() -> None:
    runtime = FakeRuntime()
    broker = SharedDecisionBroker(
        BridgeCoordinator(runtime),
        TimeStepExtractor("run-worker", "episode-worker"),
        decision_timeout_seconds=1.0,
    )
    timestep = _fake_timestep()
    first = FakeAgent("AgentA", "A", timestep, broker)
    second = FakeAgent("AgentB", "B", timestep, broker)
    departed = FakeAgent("CombatGroup0", "C", timestep, broker)
    for agent in (first, second, departed):
        broker.register(agent)

    errors: list[BaseException] = []

    def submit(agent: FakeAgent) -> None:
        try:
            broker.submit(agent, timestep, f"observation from {agent.name}")
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=submit, args=(agent,)) for agent in (first, second)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)
    departed.enable = False
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert runtime.tick_calls == 1


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


def test_worker_error_episode_preserves_bridge_counters() -> None:
    runtime = FakeRuntime()
    broker = SharedDecisionBroker(
        BridgeCoordinator(runtime),
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    broker.record_unattributed_primitive()
    main_agent = cast(Any, object.__new__(RTSCortexMainAgent))
    main_agent.worker_settings = WorkerSettings(
        run_id="run-worker",
        episode_id="episode-worker",
        socket_path=None,
        runtime_url="http://rtscortex",
        seed=7,
    )
    main_agent.decision_broker = broker
    main_agent.transport_noop_primitives = 4
    main_agent.steps = 12
    main_agent._episode_reported = False

    main_agent._report_error_episode(RuntimeError("bridge failed"))

    assert main_agent._episode_reported is True
    assert runtime.episode_results == [
        {
            "protocol_version": "1.1",
            "run_id": "run-worker",
            "episode_id": "episode-worker",
            "scenario": "pvz_task1_level1",
            "seed": 7,
            "outcome": "error",
            "score": 0.0,
            "steps": 12,
            "metrics": {
                "transport_noop_primitives": 4,
                "unattributed_primitives": 1,
                "candidate_outside_pysc2_dispatches": 0,
                "observation_gap_watchdog_triggers": 0,
            },
            "failure_reason": "RuntimeError: bridge failed",
        }
    ]


def test_worker_max_frame_hook_reports_explicit_truncation() -> None:
    runtime = FakeRuntime()
    broker = SharedDecisionBroker(
        BridgeCoordinator(runtime),
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    main_agent = cast(Any, object.__new__(RTSCortexMainAgent))
    main_agent.worker_settings = WorkerSettings(
        run_id="run-worker",
        episode_id="episode-worker",
        socket_path=None,
        runtime_url="http://rtscortex",
        seed=7,
    )
    main_agent.decision_broker = broker
    main_agent.transport_noop_primitives = 4
    main_agent._episode_reported = False
    main_agent._frame_publisher = None
    closed: list[bool] = []
    main_agent.runtime_client = SimpleNamespace(close=lambda: closed.append(True))

    main_agent.on_episode_truncated(2_500)

    assert main_agent._episode_reported is True
    assert closed == [True]
    assert runtime.episode_results == [
        {
            "protocol_version": "1.1",
            "run_id": "run-worker",
            "episode_id": "episode-worker",
            "scenario": "pvz_task1_level1",
            "seed": 7,
            "outcome": "truncated",
            "score": 0.0,
            "steps": 2_500,
            "metrics": {
                "transport_noop_primitives": 4,
                "unattributed_primitives": 0,
                "candidate_outside_pysc2_dispatches": 0,
                "observation_gap_watchdog_triggers": 0,
            },
            "failure_reason": "max_agent_steps_reached",
        }
    ]


def test_worker_settings_prefer_canonical_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RTSCORTEX_RUN_ID", "run-env")
    monkeypatch.setenv("RTSCORTEX_EPISODE_ID", "episode-env")
    monkeypatch.setenv("RTSCORTEX_RUNTIME_SOCKET", "/tmp/canonical.sock")
    monkeypatch.setenv("RTSCORTEX_SOCKET", "/tmp/legacy.sock")
    monkeypatch.setenv("RTSCORTEX_SCENARIO", "pvz_task1_level1")
    monkeypatch.setenv("RTSCORTEX_SEED", "17")
    monkeypatch.setenv("RTSCORTEX_AGENT_RACE", "terran")
    monkeypatch.setenv("RTSCORTEX_PENDING_PLAN_STEP_DELAY_SECONDS", "0.75")
    monkeypatch.setenv("RTSCORTEX_SIMULATION_SPEED_MULTIPLIER", "0.25")
    monkeypatch.setenv("RTSCORTEX_PAUSE_UNTIL_FIRST_PLAN", "true")
    monkeypatch.setenv("RTSCORTEX_RUNTIME_REQUEST_TIMEOUT_SECONDS", "50")
    monkeypatch.setenv("RTSCORTEX_ACTION_EFFECT_TIMEOUT_GAME_LOOPS", "96")
    monkeypatch.setenv("RTSCORTEX_OBSERVATION_GAP_WATCHDOG_GAME_LOOPS", "448")
    monkeypatch.setenv("RTSCORTEX_OBSERVATION_GAP_HARD_LIMIT_GAME_LOOPS", "1792")

    settings = WorkerSettings.from_environment()

    assert settings.socket_path == "/tmp/canonical.sock"
    assert settings.scenario == "pvz_task1_level1"
    assert settings.seed == 17
    assert settings.agent_race == "terran"
    assert settings.pending_plan_step_delay_seconds == 0.75
    assert settings.simulation_speed_multiplier == 0.25
    assert settings.pause_until_first_plan is True
    assert settings.runtime_request_timeout_seconds == 50.0
    assert settings.action_effect_timeout_game_loops == 96
    assert settings.observation_gap_watchdog_game_loops == 448
    assert settings.observation_gap_hard_limit_game_loops == 1792


def test_production_camera_waits_for_exact_producer_feature_observation() -> None:
    timestep = _fake_timestep()
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.func_list = [(2, object(), ("select", 0xBBB)), (503, object(), ())]
    agent._rtscortex_translation_ordinal = 1
    agent._rtscortex_production_source_tag = 0xBBB
    agent._rtscortex_production_camera_waits = 0
    agent._rtscortex_production_camera_wait_loop = None
    agent.size_screen = 128

    assert agent._wait_for_production_camera("Train_Zealot", timestep) is True
    assert agent._rtscortex_production_camera_waits == 1

    producer = _unit(0xBBB, 62, 1, 32, 32, 500, 255)
    producer.is_on_screen = True
    timestep.observation.feature_units.append(producer)

    assert (
        _producer_is_visible(
            timestep.observation,
            0xBBB,
            size_screen=128,
        )
        is True
    )
    assert agent._wait_for_production_camera("Train_Zealot", timestep) is False
    assert agent._rtscortex_production_camera_waits == 0

    producer.x = 0
    assert (
        _producer_is_visible(
            timestep.observation,
            0xBBB,
            size_screen=128,
        )
        is False
    )


def test_production_camera_settlement_timeout_is_structured_failure() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    command = RoutedCommand(
        command_id="command-camera-timeout",
        actor="Developer/Empty",
        team_name="Empty",
        name="Train_Adept",
        rendered_action="<Train_Adept()>",
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route("Developer", ("Empty",), command, step_id=70),
    )
    camera = broker.claim_primitive(
        "Developer",
        "Empty",
        "Train_Adept",
        "llm_pysc2_move_camera",
        final_primitive=False,
        ordinal=0,
        total=3,
        requested_function_id=573,
        emitted_function_id=573,
    )
    assert camera is not None
    broker.settle_primitive(camera, success=True, game_loop=224)
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.name = "Developer"
    agent.broker = broker
    agent.func_list = [(2, object(), ("select", 0xBBB)), (457, object(), ())]
    agent.action_list = []
    agent._rtscortex_translation_ordinal = 1
    agent._rtscortex_production_source_tag = 0xBBB
    agent._rtscortex_production_camera_waits = 0
    agent._rtscortex_production_camera_wait_loop = None
    agent.size_screen = 128
    agent._rtscortex_semantic_action = {"name": "Train_Adept"}
    agent.team_unit_team_curr = None
    agent.team_unit_tag_curr = None
    agent.team_unit_tag_list = []
    agent.flag_enable_empty_unit_group = True
    timestep = _fake_timestep()

    for offset in range(5):
        timestep.observation.game_loop = [224 + offset]
        assert agent._wait_for_production_camera("Train_Adept", timestep) is True
    broker.end_episode(_episode_result())

    assert len(runtime.execution_reports) == 1
    report = runtime.execution_reports[0]
    assert report["status"] == "failed"
    assert report["failure_code"] == "producer_not_observable"
    assert report["execution_stage"] == "translation"


def test_production_camera_counts_unique_observations_only() -> None:
    timestep = _fake_timestep()
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.func_list = [(2, object(), ("select", 0xBBB)), (503, object(), ())]
    agent._rtscortex_translation_ordinal = 1
    agent._rtscortex_production_source_tag = 0xBBB
    agent._rtscortex_production_camera_waits = 0
    agent._rtscortex_production_camera_wait_loop = None
    agent.size_screen = 128

    for _ in range(5):
        assert agent._wait_for_production_camera("Train_Zealot", timestep) is True

    assert agent._rtscortex_production_camera_waits == 1


def test_production_selection_waits_for_a_new_game_loop() -> None:
    timestep = _fake_timestep()
    timestep.observation.game_loop = [3127]
    producer = SimpleNamespace(tag=0xBBB, alliance=1, is_selected=True)
    timestep.observation.raw_units.append(producer)
    timestep.observation.feature_units.append(producer)
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.func_list = [(94, object(), ("queued",))]
    agent._rtscortex_translation_ordinal = 2
    agent._rtscortex_production_selection_loop = 3127
    agent._rtscortex_production_selection_attempts = 0
    agent._rtscortex_production_source_tag = 0xBBB

    assert agent._wait_for_production_selection("Build_BarracksTechLab", timestep) is True
    assert agent._rtscortex_production_selection_loop == 3127

    timestep.observation.game_loop = [3128]
    assert agent._wait_for_production_selection("Build_BarracksTechLab", timestep) is False
    assert agent._rtscortex_production_selection_loop is None


def test_production_selection_retries_inside_exact_producer_footprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_actions = SimpleNamespace(
        FUNCTIONS=SimpleNamespace(
            select_point=lambda mode, position: SimpleNamespace(
                function=2,
                arguments=[[mode], list(position)],
            )
        )
    )
    original_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_actions if name == "pysc2.lib.actions" else original_import(name),
    )
    timestep = _fake_timestep()
    timestep.observation.game_loop = [3128]
    producer = SimpleNamespace(
        tag=0xBBB,
        alliance=1,
        is_selected=False,
        is_on_screen=True,
        x=66,
        y=66,
        radius=8,
    )
    timestep.observation.raw_units.append(producer)
    timestep.observation.feature_units.append(producer)
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.func_list = [(94, object(), ("queued",))]
    agent._rtscortex_translation_ordinal = 2
    agent._rtscortex_production_selection_loop = 3127
    agent._rtscortex_production_selection_attempts = 0
    agent._rtscortex_production_source_tag = 0xBBB
    agent.size_screen = 128

    assert agent._wait_for_production_selection("Build_BarracksTechLab", timestep) is False
    result = agent._reselect_unconfirmed_production_source(
        "Build_BarracksTechLab",
        timestep,
    )

    assert result is not None
    function_id, function_call = result
    assert function_id == 2
    assert function_call.arguments[1] == [62, 62]
    assert agent._rtscortex_production_selection_attempts == 1
    assert agent._rtscortex_production_selection_loop == 3128


def test_non_production_action_never_uses_producer_selection_barrier() -> None:
    timestep = _fake_timestep()
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.func_list = [(331, object(), ("queued", "screen"))]
    agent._rtscortex_translation_ordinal = 2
    agent._rtscortex_production_selection_loop = 3127

    assert agent._wait_for_production_selection("Move_Screen", timestep) is False


def test_unavailable_screen_build_reselects_exact_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_actions = SimpleNamespace(
        FUNCTIONS=SimpleNamespace(
            select_point=lambda _mode, position: SimpleNamespace(
                function=2,
                arguments=[[0], list(position)],
            )
        )
    )
    original_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_actions if name == "pysc2.lib.actions" else original_import(name),
    )
    timestep = _fake_timestep()
    builder = _unit(0xAAA, 84, 1, 45, 52, 20, 255)
    builder.is_on_screen = True
    timestep.observation.feature_units.append(builder)
    timestep.observation.available_actions = [0, 2]
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.team_unit_tag_curr = 0xAAA
    agent._rtscortex_build_selection_retries = 0
    action = {
        "name": "Build_Gateway_Screen",
        "func": [(57, object(), ("now", [64, 64]))],
    }

    function_id, function_call = agent._reselect_builder_for_unavailable_build(action, timestep)

    assert function_id == 2
    assert function_call.arguments[1] == [45, 52]
    assert agent._rtscortex_build_selection_retries == 1

    timestep.observation.available_actions.append(57)
    assert agent._reselect_builder_for_unavailable_build(action, timestep) is None
    assert agent._rtscortex_build_selection_retries == 0


def test_observation_gap_watchdog_latches_lightweight_observations_after_recovery() -> None:
    broker = SimpleNamespace(last_decision_game_loop=100, triggers=0)
    broker.record_observation_gap_watchdog_trigger = lambda: setattr(
        broker, "triggers", broker.triggers + 1
    )
    agent = cast(Any, object.__new__(RTSCortexMainAgent))
    agent.decision_broker = broker
    agent.worker_settings = WorkerSettings(
        run_id="run-watchdog",
        episode_id="episode-watchdog",
        socket_path=None,
        runtime_url="http://rtscortex",
        seed=0,
        observation_gap_watchdog_game_loops=10,
        observation_gap_hard_limit_game_loops=40,
    )
    agent._observation_watchdog_active = False
    agent._observation_watchdog_preempted = False
    agent._observation_watchdog_baseline_loop = None
    agent._rtscortex_force_runtime_decision = False

    assert agent._update_observation_gap_watchdog(111) is True
    assert agent._rtscortex_force_runtime_decision is True
    assert broker.triggers == 1
    assert agent._update_observation_gap_watchdog(112) is True
    assert broker.triggers == 1

    broker.last_decision_game_loop = 120
    assert agent._update_observation_gap_watchdog(121) is True
    assert agent._rtscortex_force_runtime_decision is True
    assert agent._observation_watchdog_active is False
    assert agent._observation_watchdog_preempted is True
    assert broker.triggers == 1
    assert not _should_block_gas_rebalance(
        effect_verification_blocked=False,
        observation_watchdog_active=agent._observation_watchdog_active,
    )

    broker.last_decision_game_loop = 130
    assert agent._update_observation_gap_watchdog(131) is True
    assert broker.triggers == 1


def test_effect_verification_still_blocks_deterministic_gas_rebalance() -> None:
    assert _should_block_gas_rebalance(
        effect_verification_blocked=True,
        observation_watchdog_active=False,
    )


def test_observation_gap_watchdog_fails_before_unbounded_stall() -> None:
    broker = SimpleNamespace(last_decision_game_loop=100, triggers=0)
    broker.record_observation_gap_watchdog_trigger = lambda: setattr(
        broker, "triggers", broker.triggers + 1
    )
    agent = cast(Any, object.__new__(RTSCortexMainAgent))
    agent.decision_broker = broker
    agent.worker_settings = WorkerSettings(
        run_id="run-watchdog",
        episode_id="episode-watchdog",
        socket_path=None,
        runtime_url="http://rtscortex",
        seed=0,
        observation_gap_watchdog_game_loops=10,
        observation_gap_hard_limit_game_loops=40,
    )
    agent._observation_watchdog_active = False
    agent._observation_watchdog_preempted = False
    agent._observation_watchdog_baseline_loop = None
    agent._rtscortex_force_runtime_decision = False

    with pytest.raises(RuntimeError, match="observation_gap_watchdog_timeout"):
        agent._update_observation_gap_watchdog(141)


def test_observation_gap_watchdog_releases_optional_team_selection_barrier() -> None:
    exact = SimpleNamespace(
        enable=True,
        teams=[{"name": "Probe-1", "select_type": "select", "unit_tags": [0xA, 0xB]}],
        team_unit_tag_list=[],
        team_unit_team_list=[],
        _is_waiting_query=lambda: True,
    )
    grouped = SimpleNamespace(
        enable=True,
        teams=[
            {
                "name": "Zealot-1",
                "select_type": "select_all_type",
                "unit_tags": [0xC, 0xD],
            }
        ],
        team_unit_tag_list=[],
        team_unit_team_list=[],
        _is_waiting_query=lambda: True,
    )
    busy = SimpleNamespace(
        enable=True,
        teams=[{"name": "Busy-1", "select_type": "select", "unit_tags": [0xE]}],
        team_unit_tag_list=[],
        team_unit_team_list=[],
        _is_waiting_query=lambda: False,
    )
    main_agent = SimpleNamespace(
        agents={"Builder": exact, "CombatGroup": grouped, "Busy": busy},
        unit_uid_disappear={0xB},
    )

    _release_runtime_observation_barrier(main_agent)
    _release_runtime_observation_barrier(main_agent)

    assert exact.team_unit_tag_list == [0xA]
    assert exact.team_unit_team_list == ["Probe-1"]
    assert grouped.team_unit_tag_list == [0xC]
    assert grouped.team_unit_team_list == ["Zealot-1"]
    assert busy.team_unit_tag_list == []


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


def test_shared_broker_initial_barrier_waits_for_first_runtime_decision() -> None:
    runtime = BlockingRuntime()
    broker = SharedDecisionBroker(
        BridgeCoordinator(runtime),
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    timestep = _fake_timestep()
    agent = FakeAgent("AgentA", "A", timestep, broker)
    broker.register(agent)
    thread = threading.Thread(target=agent.query, args=(timestep,))

    thread.start()
    assert runtime.entered.wait(timeout=1)
    assert broker.initial_decision_started is True

    runtime.release.set()
    broker.wait_for_initial_decision()
    thread.join(timeout=1)

    assert not thread.is_alive()


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


def test_auto_worker_management_guard_disables_and_restores_upstream_flag() -> None:
    config = SimpleNamespace(
        ENABLE_AUTO_WORKER_MANAGE=True,
        ENABLE_AUTO_WORKER_TRAINING=True,
    )

    def upstream_step() -> str:
        assert config.ENABLE_AUTO_WORKER_MANAGE is False
        assert config.ENABLE_AUTO_WORKER_TRAINING is False
        return "action"

    action = _run_with_auto_worker_management_guard(
        config,
        blocked=True,
        upstream_step=upstream_step,
    )

    assert action == "action"
    assert config.ENABLE_AUTO_WORKER_MANAGE is True
    assert config.ENABLE_AUTO_WORKER_TRAINING is True


def test_auto_worker_management_guard_restores_flag_after_upstream_error() -> None:
    config = SimpleNamespace(
        ENABLE_AUTO_WORKER_MANAGE=True,
        ENABLE_AUTO_WORKER_TRAINING=True,
    )

    def upstream_step() -> None:
        assert config.ENABLE_AUTO_WORKER_MANAGE is False
        assert config.ENABLE_AUTO_WORKER_TRAINING is False
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError, match="upstream failed"):
        _run_with_auto_worker_management_guard(
            config,
            blocked=True,
            upstream_step=upstream_step,
        )

    assert config.ENABLE_AUTO_WORKER_MANAGE is True
    assert config.ENABLE_AUTO_WORKER_TRAINING is True


def test_deterministic_gas_rebalance_selects_nearest_stable_mineral_worker() -> None:
    assimilator = SimpleNamespace(tag=500, x=20.0, y=20.0, build_progress=100)
    workers = [
        SimpleNamespace(tag=30, x=5.0, y=5.0),
        SimpleNamespace(tag=20, x=19.0, y=19.0),
        SimpleNamespace(tag=10, x=19.0, y=19.0),
    ]
    agent = SimpleNamespace(
        config=SimpleNamespace(ENABLE_AUTO_WORKER_MANAGE=True),
        main_loop_lock=False,
        stop_worker=None,
        stop_worker_nexus_tag=None,
        stop_worker_at=None,
        nexus_info_dict={
            "100": {
                "nexus": SimpleNamespace(tag=100),
                "gas_building_1": assimilator,
                "gas_building_2": None,
                "worker_g1_tag_list": [],
                "worker_g2_tag_list": [],
                "worker_m_tag_list": [30, 20, 10],
            }
        },
    )
    observation = SimpleNamespace(raw_units=workers)

    selected = _prime_deterministic_gas_rebalance(
        agent,
        observation,
        blocked=False,
    )

    assert selected is True
    assert agent.stop_worker.tag == 10
    assert agent.stop_worker_nexus_tag == 100
    assert agent.stop_worker_at == "m"


def test_deterministic_gas_rebalance_excludes_reserved_builder_worker() -> None:
    assimilator = SimpleNamespace(tag=500, x=20.0, y=20.0, build_progress=100)
    workers = [
        SimpleNamespace(tag=10, x=19.0, y=19.0),
        SimpleNamespace(tag=20, x=18.0, y=18.0),
    ]
    builder = SimpleNamespace(
        unit_tag_list=[10, 20],
        team_unit_tag_list=[10, 20],
        team_unit_tag_curr=None,
        teams=[{"unit_tags": [10]}],
    )
    agent = SimpleNamespace(
        config=SimpleNamespace(ENABLE_AUTO_WORKER_MANAGE=True),
        main_loop_lock=False,
        stop_worker=None,
        stop_worker_nexus_tag=None,
        stop_worker_at=None,
        agents={"Builder": builder},
        nexus_info_dict={
            "100": {
                "nexus": SimpleNamespace(tag=100),
                "gas_building_1": assimilator,
                "gas_building_2": None,
                "worker_g1_tag_list": [],
                "worker_g2_tag_list": [],
                "worker_m_tag_list": [10, 20],
            }
        },
    )

    selected = _prime_deterministic_gas_rebalance(
        agent,
        SimpleNamespace(raw_units=workers),
        blocked=False,
    )

    assert selected is True
    assert agent.stop_worker.tag == 20
    assert agent._rtscortex_reserved_worker_tags == {10}


def test_deterministic_gas_rebalance_evicts_builder_already_on_gas() -> None:
    assimilator = SimpleNamespace(tag=500, x=20.0, y=20.0, build_progress=100)
    builder_worker = SimpleNamespace(tag=10, x=20.0, y=20.0)
    ordinary_worker = SimpleNamespace(tag=20, x=18.0, y=18.0)
    builder = SimpleNamespace(
        unit_tag_list=[10],
        team_unit_tag_list=[],
        team_unit_tag_curr=None,
        teams=[{"unit_tags": [10]}],
    )
    agent = SimpleNamespace(
        config=SimpleNamespace(ENABLE_AUTO_WORKER_MANAGE=True),
        main_loop_lock=False,
        stop_worker=None,
        stop_worker_nexus_tag=None,
        stop_worker_at=None,
        agents={"Builder": builder},
        nexus_info_dict={
            "100": {
                "nexus": SimpleNamespace(tag=100),
                "gas_building_1": assimilator,
                "gas_building_2": None,
                "worker_g1_tag_list": [10],
                "worker_g2_tag_list": [],
                "worker_m_tag_list": [20],
            }
        },
    )

    selected = _prime_deterministic_gas_rebalance(
        agent,
        SimpleNamespace(raw_units=[builder_worker, ordinary_worker]),
        blocked=False,
    )

    assert selected is True
    assert agent.stop_worker is builder_worker
    assert agent.stop_worker_nexus_tag == 100
    assert agent.stop_worker_at == "g1"


def test_builder_rebinds_to_actual_same_type_worker_selected_by_feature_click() -> None:
    team = {"name": "Builder-SCV-1", "unit_tags": [10]}
    builder = SimpleNamespace(
        _is_executing_actions=lambda: True,
        curr_action_name="Build_Barracks_Screen",
        team_unit_tag_curr=10,
        team_unit_team_curr="Builder-SCV-1",
        teams=[team],
    )
    main_agent = SimpleNamespace(
        agents={"Builder": builder},
        nexus_info_dict={"100": {"worker_g_tag_list": []}},
    )
    observation = SimpleNamespace(
        raw_units=[
            SimpleNamespace(tag=10, alliance=1, unit_type=45, is_selected=False),
            SimpleNamespace(tag=20, alliance=1, unit_type=45, is_selected=True),
        ]
    )

    rebound = _rebind_builder_to_selected_worker(main_agent, observation)

    assert rebound is True
    assert builder.team_unit_tag_curr == 20
    assert team["unit_tags"] == [20]
    assert builder._rtscortex_last_builder_rebind == (10, 20)


def test_builder_never_rebinds_to_selected_gas_worker() -> None:
    team = {"name": "Builder-SCV-1", "unit_tags": [10]}
    builder = SimpleNamespace(
        _is_executing_actions=lambda: True,
        curr_action_name="Build_Barracks_Screen",
        team_unit_tag_curr=10,
        team_unit_team_curr="Builder-SCV-1",
        teams=[team],
    )
    main_agent = SimpleNamespace(
        agents={"Builder": builder},
        nexus_info_dict={"100": {"worker_g1_tag_list": [20]}},
    )
    observation = SimpleNamespace(
        raw_units=[
            SimpleNamespace(tag=10, alliance=1, unit_type=45, is_selected=False),
            SimpleNamespace(tag=20, alliance=1, unit_type=45, is_selected=True),
        ]
    )

    assert _rebind_builder_to_selected_worker(main_agent, observation) is False
    assert builder.team_unit_tag_curr == 10
    assert team["unit_tags"] == [10]


def test_deterministic_gas_rebalance_respects_effect_and_main_loop_guards() -> None:
    builder = SimpleNamespace(
        unit_tag_list=[10],
        team_unit_tag_list=[],
        team_unit_tag_curr=None,
        teams=[{"unit_tags": [10]}],
    )
    agent = SimpleNamespace(
        config=SimpleNamespace(ENABLE_AUTO_WORKER_MANAGE=True),
        main_loop_lock=False,
        stop_worker=None,
        stop_worker_nexus_tag=None,
        agents={"Builder": builder},
        nexus_info_dict={},
    )

    assert (
        _prime_deterministic_gas_rebalance(
            agent,
            SimpleNamespace(raw_units=[]),
            blocked=True,
        )
        is False
    )
    assert agent._rtscortex_reserved_worker_tags == {10}
    agent.main_loop_lock = True
    assert (
        _prime_deterministic_gas_rebalance(
            agent,
            SimpleNamespace(raw_units=[]),
            blocked=False,
        )
        is False
    )
    assert agent._rtscortex_reserved_worker_tags == {10}


def test_translated_build_position_extracts_final_screen_argument() -> None:
    assert _translated_build_position("Build_Nexus_Near", [(90, 70)]) == [90, 70]
    assert _translated_build_position("Build_Pylon_Screen", ["now", [65, 55]]) == [65, 55]
    assert _translated_build_position("Move_Screen", [[65, 55]]) is None
    assert _translated_build_position("Build_Nexus_Near", ["invalid"]) is None


def test_build_dispatch_does_not_mutate_reusable_action_template() -> None:
    template = {
        "name": "Build_Pylon_Screen",
        "arg": ["screen"],
        "func": [(70, None, ("now", [0, 0]))],
    }
    action_list = [
        {
            "name": template["name"],
            "arg": template["arg"],
            "func": template["func"],
        }
    ]

    isolated = _isolate_next_action(action_list)
    _replace_screen_action_position(isolated, [65, 35])

    assert isolated["arg"] == [[65, 35]]
    assert template["arg"] == ["screen"]
    assert action_list[0] is isolated


def test_build_position_resampling_excludes_previously_failed_candidate() -> None:
    observation = SimpleNamespace(
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(0),
        ),
        feature_units=[],
    )
    candidates = build_screen_candidates(observation, "Build_Pylon_Screen")
    assert len(candidates) >= 2
    requested = candidates[0]
    action = {
        "name": "Build_Pylon_Screen",
        "arg": [list(requested)],
        "func": [(70, object(), ("now", list(requested)))],
    }

    resolved = _resolve_build_action_position(
        action,
        observation,
        excluded_positions={(requested[0], requested[1])},
    )

    assert resolved is not None
    assert resolved != requested
    assert resolved in candidates
    assert action["arg"] == [resolved]


def test_build_translation_retry_initializes_optional_screen_provenance() -> None:
    source = inspect.getsource(RTSCortexLLMAgent.get_func)

    assert source.index("provenance = None") < source.index("semantic_action_name =")


def test_timestep_extractor_maps_sc2_attack_alerts() -> None:
    agent = FakeAgent("CombatGroupSmac", "Stalker-1", _fake_timestep(), StubBroker())
    snapshot = TimeStepExtractor("run-worker", "episode-worker").extract(
        _fake_timestep(alerts=[6, 19, 3]),
        {"CombatGroupSmac": agent},
        {"CombatGroupSmac": "under attack"},
        step_id=1,
    )

    assert snapshot["alerts"] == ["building_under_attack", "unit_under_attack", "alert:3"]


def test_timestep_extractor_adds_structured_pylon_screen_candidates() -> None:
    timestep = _fake_timestep()
    timestep.observation.feature_screen = SimpleNamespace(
        buildable=UniformGrid(1),
        pathable=UniformGrid(1),
        player_relative=UniformGrid(0),
        power=UniformGrid(0),
    )
    agent = FakeAgent("Builder", "Builder-Probe-1", timestep, StubBroker())
    agent.config.AGENTS["Builder"]["action"][311].append(
        {"name": "Build_Pylon_Screen", "arg": ["screen"], "func": [(12, None, ())]}
    )

    snapshot = TimeStepExtractor("run-worker", "episode-worker").extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=1,
    )

    pylon = next(
        action
        for action in snapshot["teams"][0]["available_actions"]
        if action["name"] == "Build_Pylon_Screen"
    )
    assert pylon["argument_candidates"]

    assert all(len(candidate) == 1 for candidate in pylon["argument_candidates"])
    assert "RTSCortex Build Candidates" not in snapshot["text_observation"]


def test_timestep_extractor_keeps_build_when_primitive_is_temporarily_unavailable() -> None:
    timestep = _fake_timestep()
    timestep.observation.available_actions = [0]
    timestep.observation.feature_screen = SimpleNamespace(
        buildable=UniformGrid(1),
        pathable=UniformGrid(1),
        player_relative=UniformGrid(0),
        power=UniformGrid(0),
    )
    agent = FakeAgent("Builder", "Builder-Probe-1", timestep, StubBroker())
    agent.config.AGENTS["Builder"]["action"][311].append(
        {"name": "Build_Pylon_Screen", "arg": ["screen"], "func": [(70, None, ())]}
    )

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        action_source_types={70: 311},
    ).extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=1,
    )

    pylon = next(
        action
        for action in snapshot["teams"][0]["available_actions"]
        if action["name"] == "Build_Pylon_Screen"
    )
    assert pylon["argument_candidates"]

    without_probe = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        action_source_types={70: 999},
    ).extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=2,
    )
    assert "Build_Pylon_Screen" not in {
        action["name"] for action in without_probe["teams"][0]["available_actions"]
    }


def test_timestep_extractor_defers_terran_builds_while_reserved_scv_is_constructing() -> None:
    timestep = _fake_timestep()
    scv = timestep.observation.raw_units[0]
    scv.unit_type = 45
    scv.order_length = 1
    scv.order_id_0 = 222
    supply_depot = _unit(0xDEAD, 19, 1, 50, 50, 400, 255)
    supply_depot.build_progress = 100
    timestep.observation.raw_units.append(supply_depot)
    timestep.observation.feature_screen = SimpleNamespace(
        buildable=UniformGrid(1),
        pathable=UniformGrid(1),
        player_relative=UniformGrid(0),
        power=UniformGrid(0),
    )
    agent = FakeAgent("Builder", "Builder-SCV-1", timestep, StubBroker())
    team = agent.config.AGENTS["Builder"]["team"][0]
    team["unit_type"] = [45]
    team["unit_tags"] = [scv.tag]
    agent.config.AGENTS["Builder"]["action"] = {
        45: [
            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
            {
                "name": "Build_SupplyDepot_Screen",
                "arg": ["screen"],
                "func": [(91, None, ())],
            },
            {
                "name": "Build_Barracks_Screen",
                "arg": ["screen"],
                "func": [(42, None, ())],
            },
        ]
    }
    extractor = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={45: "SCV", 19: "SupplyDepot"},
        action_source_types={42: 45, 91: 45},
    )

    busy = extractor.extract(
        timestep,
        {"Builder": agent},
        {"Builder": "building supply"},
        step_id=1,
    )
    scv.order_length = 0
    idle = extractor.extract(
        timestep,
        {"Builder": agent},
        {"Builder": "ready"},
        step_id=2,
    )

    assert [action["name"] for action in busy["teams"][0]["available_actions"]] == ["No_Operation"]
    assert {action["name"] for action in idle["teams"][0]["available_actions"]} == {
        "No_Operation",
        "Build_SupplyDepot_Screen",
        "Build_Barracks_Screen",
    }


def test_timestep_extractor_does_not_apply_scv_construction_lock_to_probe() -> None:
    timestep = _fake_timestep()
    probe = timestep.observation.raw_units[0]
    probe.order_length = 1
    probe.order_id_0 = 35
    timestep.observation.feature_screen = SimpleNamespace(
        buildable=UniformGrid(1),
        pathable=UniformGrid(1),
        player_relative=UniformGrid(0),
        power=UniformGrid(0),
    )
    agent = FakeAgent("Builder", "Builder-Probe-1", timestep, StubBroker())
    agent.config.AGENTS["Builder"]["team"][0]["unit_tags"] = [probe.tag]
    agent.config.AGENTS["Builder"]["action"][311].append(
        {"name": "Build_Pylon_Screen", "arg": ["screen"], "func": [(70, None, ())]}
    )

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={311: "Probe"},
        action_source_types={70: 311},
    ).extract(
        timestep,
        {"Builder": agent},
        {"Builder": "probe may leave the pylon"},
        step_id=1,
    )

    assert "Build_Pylon_Screen" in {
        action["name"] for action in snapshot["teams"][0]["available_actions"]
    }


def test_worker_player_race_prefers_explicit_bridge_race_over_upstream_compatibility() -> None:
    agent = SimpleNamespace(
        rtscortex_player_race="terran",
        config=SimpleNamespace(rtscortex_player_race="protoss"),
        race="protoss",
    )

    assert _worker_player_race(agent) == "terran"
    assert "player_race=self.worker_settings.agent_race" in inspect.getsource(
        RTSCortexMainAgent.__init__
    )
    source = inspect.getsource(RTSCortexLLMAgent.get_func)
    assert "_requires_terran_production_chain(action_name)" in source
    assert _requires_terran_production_chain("Train_Marine") is True
    assert _requires_terran_production_chain("Train_Zealot") is False


def test_terran_production_chain_bypasses_upstream_source_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = {
        "name": "Train_Marine",
        "arg": [],
        "func": [(477, object(), ("queued",))],
    }
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent._rtscortex_production_source_tag = 0xABC
    agent.action_list = [action.copy()]
    agent.func_list = []
    functions = SimpleNamespace(
        llm_pysc2_move_camera=object(),
        select_point=object(),
    )
    monkeypatch.setattr(
        "rtscortex_llm_pysc2.worker.importlib.import_module",
        lambda _name: SimpleNamespace(FUNCTIONS=functions),
    )

    agent._prime_terran_production_chain(action)

    assert [int(item[0]) for item in agent.func_list] == [573, 2, 477]
    assert agent.action_list == []
    assert agent.curr_action_name == "Train_Marine"
    assert agent._rtscortex_translation_total == 3


def test_untracked_transport_noop_clears_semantic_cache_before_next_action() -> None:
    source = inspect.getsource(RTSCortexLLMAgent.get_func)
    marker = 'f"translator primitive for {action_name!r} has no unique active command"'
    untracked_branch = source[source.rindex("if dispatch is None:", 0, source.index(marker)) :]

    assert untracked_branch.index("self._rtscortex_semantic_action = None") < (
        untracked_branch.index("return result")
    )


def test_terran_addon_uses_exact_producer_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    action = {
        "name": "Build_BarracksReactor",
        "arg": [],
        "func": [(73, object(), ("queued",))],
    }
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent._rtscortex_production_source_tag = 0xABC
    agent.action_list = [action.copy()]
    agent.func_list = []
    functions = SimpleNamespace(
        llm_pysc2_move_camera=object(),
        select_point=object(),
    )
    monkeypatch.setattr(
        "rtscortex_llm_pysc2.worker.importlib.import_module",
        lambda _name: SimpleNamespace(FUNCTIONS=functions),
    )
    agent._prime_terran_production_chain(action)

    assert [int(item[0]) for item in agent.func_list] == [573, 2, 73]
    assert _requires_terran_production_chain("Build_BarracksReactor") is True


def test_timestep_extractor_enumerates_stable_pathable_move_and_blink_candidates() -> None:
    timestep = _fake_timestep()
    timestep.observation.feature_screen = SimpleNamespace(pathable=UniformGrid(1))
    timestep.observation.feature_minimap = SimpleNamespace(
        pathable=Grid([[1 for _ in range(128)] for _ in range(128)]),
        player_relative=Grid([[0 for _ in range(128)] for _ in range(128)]),
    )
    agent = FakeAgent("CombatGroup1", "Stalker-1", timestep, StubBroker())
    action_space = agent.config.AGENTS["CombatGroup1"]["action"][311]
    action_space.extend(
        [
            {"name": "Move_Screen", "arg": ["screen"], "func": [(12, None, ())]},
            {"name": "Move_Minimap", "arg": ["minimap"], "func": [(12, None, ())]},
            {"name": "Ability_Blink_Screen", "arg": ["screen"], "func": [(12, None, ())]},
            {
                "name": "Select_Unit_Blink_Screen",
                "arg": ["tag", "screen"],
                "func": [(12, None, ())],
            },
            {"name": "Warp_Stalker_Near", "arg": ["tag"], "func": [(12, None, ())]},
        ]
    )
    extractor = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={311: "Stalker"},
    )

    first = extractor.extract(
        timestep,
        {"CombatGroup1": agent},
        {"CombatGroup1": "stalker observation"},
        step_id=1,
    )
    second = extractor.extract(
        timestep,
        {"CombatGroup1": agent},
        {"CombatGroup1": "stalker observation"},
        step_id=2,
    )
    first_actions = {action["name"]: action for action in first["teams"][0]["available_actions"]}
    second_actions = {action["name"]: action for action in second["teams"][0]["available_actions"]}

    assert first_actions["Move_Minimap"]["argument_candidates"]
    assert (
        first_actions["Move_Minimap"]["argument_candidates"]
        == second_actions["Move_Minimap"]["argument_candidates"]
    )
    assert "Warp_Stalker_Near" not in first_actions
    for action_name in (
        "Move_Screen",
        "Ability_Blink_Screen",
        "Select_Unit_Blink_Screen",
    ):
        action = first_actions[action_name]
        assert action["argument_candidates"]
        assert action["argument_candidates"] == second_actions[action_name]["argument_candidates"]
        assert all(
            len(candidate) == len(action["argument_names"])
            for candidate in action["argument_candidates"]
        )

    mapped = ObservationEnvelope.model_validate(ObservationMapper().map(first))
    move = next(action for action in mapped.available_actions if action.name == "Move_Screen")
    assert move.argument_candidates
    assert all(
        timestep.observation.feature_screen.pathable[position[1]][position[0]] == 1
        for [position] in move.argument_candidates
    )


def test_builder_move_minimap_prioritizes_remote_resource_scouting_target() -> None:
    size = 64
    pathable = [[1 for _ in range(size)] for _ in range(size)]
    player_relative = [[0 for _ in range(size)] for _ in range(size)]
    visibility = [[0 for _ in range(size)] for _ in range(size)]
    player_relative[8][8] = 1
    visibility[8][8] = 2
    resource_pixels = ((46, 46), (48, 46), (50, 46), (47, 48), (49, 48))
    for x, y in resource_pixels:
        player_relative[y][x] = 3
        pathable[y][x] = 0

    screen_size = 128
    power = [[0 for _ in range(screen_size)] for _ in range(screen_size)]
    for y in range(48, 81):
        for x in range(48, 81):
            power[y][x] = 1

    timestep = _fake_timestep()
    timestep.observation.feature_minimap = SimpleNamespace(
        pathable=Grid(pathable),
        player_relative=Grid(player_relative),
        visibility_map=Grid(visibility),
    )
    timestep.observation.feature_screen = SimpleNamespace(
        pathable=UniformGrid(1),
        power=Grid(power),
    )
    timestep.observation.feature_units = [
        SimpleNamespace(
            tag=4300734465,
            x=63,
            y=64,
            radius=2.0,
            alliance=1,
            is_on_screen=True,
        ),
        SimpleNamespace(
            tag=4316463105,
            x=64,
            y=48,
            radius=3.0,
            alliance=3,
            is_on_screen=True,
        ),
    ]
    agent = FakeAgent("Builder", "Builder-Probe-1", timestep, StubBroker())
    agent.config.AGENTS["Builder"]["action"][311].extend(
        [
            {"name": "Move_Minimap", "arg": ["minimap"], "func": [(12, None, ())]},
            {"name": "Move_Screen", "arg": ["screen"], "func": [(12, None, ())]},
        ]
    )
    extractor = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={60: "Pylon", 62: "Gateway"},
    )

    opening = extractor.extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=0,
    )
    opening_actions = {action["name"] for action in opening["teams"][0]["available_actions"]}
    assert {"Move_Minimap", "Move_Screen"}.isdisjoint(opening_actions)

    timestep.observation.raw_units.append(
        SimpleNamespace(unit_type=60, alliance=1, build_progress=1.0)
    )
    local_movement = extractor.extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=1,
    )
    local_actions = {
        action["name"]: action for action in local_movement["teams"][0]["available_actions"]
    }
    assert "Move_Screen" in local_actions
    assert "Move_Minimap" not in local_actions
    local_candidates = local_actions["Move_Screen"]["argument_candidates"]
    assert local_candidates
    assert all(power[position[1]][position[0]] == 1 for [position] in local_candidates)

    timestep.observation.feature_units[0].x = 48
    timestep.observation.feature_units[0].y = 48
    timestep.observation.feature_units[1].x = 80
    timestep.observation.feature_units[1].y = 80
    moved_units = extractor.extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=2,
    )
    moved_actions = {
        action["name"]: action for action in moved_units["teams"][0]["available_actions"]
    }
    assert moved_actions["Move_Screen"]["argument_candidates"] == local_candidates

    timestep.observation.raw_units.append(
        SimpleNamespace(unit_type=62, alliance=1, build_progress=1.0)
    )

    first = extractor.extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=3,
    )
    second = extractor.extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=4,
    )
    first_action = next(
        action
        for action in first["teams"][0]["available_actions"]
        if action["name"] == "Move_Minimap"
    )
    second_action = next(
        action
        for action in second["teams"][0]["available_actions"]
        if action["name"] == "Move_Minimap"
    )

    assert first_action["argument_candidates"] == second_action["argument_candidates"]
    assert len(first_action["argument_candidates"]) == 1
    [resource_target] = first_action["argument_candidates"][0]
    assert (resource_target[0] - 48) ** 2 + (resource_target[1] - 47) ** 2 <= 4
    assert pathable[resource_target[1]][resource_target[0]] == 1
    assert all(
        pathable[position[1]][position[0]] == 1
        for [position] in first_action["argument_candidates"]
    )

    for y in range(40, 56):
        for x in range(40, 56):
            visibility[y][x] = 1
    after_scout = extractor.extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=5,
    )
    after_scout_action = next(
        action
        for action in after_scout["teams"][0]["available_actions"]
        if action["name"] == "Move_Minimap"
    )
    assert all(
        (position[0] - 48) ** 2 + (position[1] - 47) ** 2 > 64
        for [position] in after_scout_action["argument_candidates"]
    )


def test_timestep_extractor_hides_candidates_for_unavailable_build_actions() -> None:
    timestep = _fake_timestep()
    timestep.observation.feature_screen = SimpleNamespace(
        buildable=UniformGrid(1),
        pathable=UniformGrid(1),
        player_relative=UniformGrid(0),
        power=UniformGrid(1),
    )
    agent = FakeAgent("Builder", "Builder-Probe-1", timestep, StubBroker())

    snapshot = TimeStepExtractor("run-worker", "episode-worker").extract(
        timestep,
        {"Builder": agent},
        {"Builder": "builder observation"},
        step_id=1,
    )

    assert "Build_Pylon_Screen" not in {
        action["name"] for action in snapshot["teams"][0]["available_actions"]
    }


def test_build_candidates_skip_occupied_full_footprints() -> None:
    player_relative = [[0 for _ in range(128)] for _ in range(128)]
    player_relative[65][65] = 1
    observation = SimpleNamespace(
        feature_units=[
            SimpleNamespace(x=55, y=65, is_on_screen=True, unit_type=84),
            SimpleNamespace(x=65, y=55, is_on_screen=True, unit_type=59),
        ],
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=Grid(player_relative),
            power=UniformGrid(0),
        ),
    )

    candidates = build_screen_candidates(observation, "Build_Pylon_Screen")

    assert [65, 65] not in candidates  # SELF layer at the preferred center.
    assert [55, 65] not in candidates  # Explicit Probe position.
    assert [65, 55] not in candidates  # Explicit structure position.
    assert candidates[0] == [65, 75]


def test_build_candidates_dilate_feature_unit_radius_around_footprint() -> None:
    observation = SimpleNamespace(
        feature_units=[
            SimpleNamespace(
                x=72,
                y=65,
                radius=2.0,
                alliance=1,
                is_on_screen=True,
            )
        ],
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(0),
        ),
    )

    candidates = build_screen_candidates(observation, "Build_Pylon_Screen")

    assert [65, 65] not in candidates


def test_terran_production_build_reserves_right_side_addon_footprint() -> None:
    observation = SimpleNamespace(
        feature_units=[
            SimpleNamespace(
                x=80,
                y=65,
                radius=0.5,
                alliance=1,
                is_on_screen=True,
            )
        ],
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(0),
        ),
        raw_units=[
            SimpleNamespace(
                alliance=1,
                unit_type=19,
                build_progress=100,
            ),
            SimpleNamespace(
                alliance=1,
                unit_type=18,
                build_progress=100,
            ),
        ],
    )

    assert not screen_build_position_is_legal(
        observation,
        "Build_Barracks_Screen",
        [65, 65],
        unit_names={18: "CommandCenter", 19: "SupplyDepot"},
    )
    assert screen_build_position_is_legal(
        observation,
        "Build_EngineeringBay_Screen",
        [65, 65],
        unit_names={18: "CommandCenter", 19: "SupplyDepot"},
    )


def test_terran_production_build_requires_buildable_addon_footprint() -> None:
    buildable = [[1 for _ in range(128)] for _ in range(128)]
    buildable[65][80] = 0
    observation = SimpleNamespace(
        feature_units=[],
        feature_screen=SimpleNamespace(
            buildable=Grid(buildable),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(0),
        ),
        raw_units=[
            SimpleNamespace(
                alliance=1,
                unit_type=21,
                build_progress=100,
            )
        ],
    )

    assert not screen_build_position_is_legal(
        observation,
        "Build_Factory_Screen",
        [65, 65],
        unit_names={21: "Barracks"},
    )
    buildable[65][80] = 1
    assert screen_build_position_is_legal(
        observation,
        "Build_Factory_Screen",
        [65, 65],
        unit_names={21: "Barracks"},
    )


def test_terran_production_build_keeps_future_addon_clear_of_geyser() -> None:
    anchor = SimpleNamespace(
        tag=0xAAA,
        alliance=1,
        unit_type=18,
        x=20,
        y=20,
        radius=0.5,
        is_on_screen=True,
    )
    geyser = SimpleNamespace(
        tag=0xBBB,
        alliance=3,
        unit_type=342,
        x=63.5,
        y=59.5,
    )
    observation = SimpleNamespace(
        feature_units=[anchor],
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(0),
        ),
        raw_units=[
            SimpleNamespace(
                tag=0xAAA,
                alliance=1,
                unit_type=18,
                x=50.0,
                y=50.0,
                build_progress=100,
            ),
            SimpleNamespace(
                tag=0xAAC,
                alliance=1,
                unit_type=19,
                x=50.0,
                y=52.0,
                build_progress=100,
            ),
            geyser,
        ],
    )
    unit_names = {18: "CommandCenter", 19: "SupplyDepot", 342: "VespeneGeyser"}

    assert not screen_build_position_is_legal(
        observation,
        "Build_Barracks_Screen",
        [65, 65],
        unit_names=unit_names,
    )
    geyser.x = 70.0
    geyser.y = 70.0
    assert screen_build_position_is_legal(
        observation,
        "Build_Barracks_Screen",
        [65, 65],
        unit_names=unit_names,
    )


def test_new_build_avoids_existing_terran_producer_addon_reservation() -> None:
    observation = SimpleNamespace(
        feature_units=[
            SimpleNamespace(
                x=55,
                y=65,
                radius=1.5,
                alliance=1,
                unit_type=21,
                is_on_screen=True,
            )
        ],
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(0),
        ),
        raw_units=[
            SimpleNamespace(
                alliance=1,
                unit_type=19,
                build_progress=100,
            )
        ],
    )

    assert not screen_build_position_is_legal(
        observation,
        "Build_Barracks_Screen",
        [70, 65],
        unit_names={19: "SupplyDepot", 21: "Barracks"},
    )
    assert screen_build_position_is_legal(
        observation,
        "Build_Barracks_Screen",
        [30, 65],
        unit_names={19: "SupplyDepot", 21: "Barracks"},
    )


def test_feature_screen_radius_is_not_scaled_twice_for_build_occupancy() -> None:
    observation = SimpleNamespace(
        feature_units=[
            SimpleNamespace(
                x=64,
                y=64,
                radius=14.0,
                alliance=1,
                is_on_screen=True,
            )
        ],
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(0),
        ),
    )

    candidates = build_screen_candidates(observation, "Build_Pylon_Screen")

    assert candidates
    assert all(abs(x - 64) > 14 or abs(y - 64) > 14 for x, y in candidates)


def test_worker_refreshes_volatile_build_position_at_execution_time() -> None:
    observation = SimpleNamespace(
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(0),
        )
    )
    action = {
        "name": "Build_Pylon_Screen",
        "arg": ["screen"],
        "func": [(70, object(), ("now", [54.0, 68.0]))],
    }

    refreshed = _refresh_build_action_position(action, observation)

    assert refreshed is True
    assert action["func"][0][2] == ("now", [60, 70])


def test_build_candidates_use_pysc2_row_major_coordinates() -> None:
    buildable = [[0 for _ in range(128)] for _ in range(128)]
    pathable = [[0 for _ in range(128)] for _ in range(128)]
    player_relative = [[0 for _ in range(128)] for _ in range(128)]
    for row in range(64, 76):
        for column in range(59, 71):
            buildable[row][column] = 1
            pathable[row][column] = 1
    observation = SimpleNamespace(
        feature_screen=SimpleNamespace(
            buildable=Grid(buildable),
            pathable=Grid(pathable),
            player_relative=Grid(player_relative),
            power=UniformGrid(0),
        )
    )

    assert [65, 70] in build_screen_candidates(observation, "Build_Pylon_Screen")

    transposed_buildable = [[0 for _ in range(128)] for _ in range(128)]
    transposed_pathable = [[0 for _ in range(128)] for _ in range(128)]
    for row in range(59, 71):
        for column in range(64, 76):
            transposed_buildable[row][column] = 1
            transposed_pathable[row][column] = 1
    transposed_observation = SimpleNamespace(
        feature_screen=SimpleNamespace(
            buildable=Grid(transposed_buildable),
            pathable=Grid(transposed_pathable),
            player_relative=Grid(player_relative),
            power=UniformGrid(0),
        )
    )

    transposed_candidates = build_screen_candidates(
        transposed_observation,
        "Build_Pylon_Screen",
    )
    assert [65, 70] not in transposed_candidates
    assert [70, 65] in transposed_candidates


def test_gateway_candidates_use_row_major_power_plane() -> None:
    buildable = [[0 for _ in range(128)] for _ in range(128)]
    pathable = [[0 for _ in range(128)] for _ in range(128)]
    for row in range(82, 99):
        for column in range(57, 74):
            buildable[row][column] = 1
            pathable[row][column] = 1
    row_major_power = [[0 for _ in range(128)] for _ in range(128)]
    for row in range(82, 99):
        for column in range(57, 74):
            row_major_power[row][column] = 1
    row_major_observation = SimpleNamespace(
        feature_screen=SimpleNamespace(
            buildable=Grid(buildable),
            pathable=Grid(pathable),
            player_relative=UniformGrid(0),
            power=Grid(row_major_power),
        )
    )

    assert [65, 90] in build_screen_candidates(
        row_major_observation,
        "Build_Gateway_Screen",
    )

    transposed_power = [[0 for _ in range(128)] for _ in range(128)]
    transposed_power[65][90] = 1
    transposed_observation = SimpleNamespace(
        feature_screen=SimpleNamespace(
            buildable=Grid(buildable),
            pathable=Grid(pathable),
            player_relative=UniformGrid(0),
            power=Grid(transposed_power),
        )
    )

    assert build_screen_candidates(transposed_observation, "Build_Gateway_Screen") == []


def test_cybernetics_core_candidates_require_completed_gateway_and_power() -> None:
    observation = SimpleNamespace(
        player_common=SimpleNamespace(minerals=500),
        raw_units=[],
        feature_units=[],
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(1),
        ),
    )
    unit_names = {62: "Gateway"}

    assert (
        semantic_argument_candidates(
            observation,
            "Build_CyberneticsCore_Screen",
            unit_names=unit_names,
        )
        == []
    )

    observation.raw_units.append(
        SimpleNamespace(unit_type=62, alliance=1, build_progress=100, x=30, y=30)
    )
    candidates = semantic_argument_candidates(
        observation,
        "Build_CyberneticsCore_Screen",
        unit_names=unit_names,
    )

    assert candidates
    assert all(len(candidate) == 1 and len(candidate[0]) == 2 for candidate in candidates)


def test_stargate_candidates_require_completed_core_and_full_resource_cost() -> None:
    observation = SimpleNamespace(
        player_common=SimpleNamespace(minerals=150, vespene=150),
        raw_units=[],
        feature_units=[],
        feature_screen=SimpleNamespace(
            buildable=UniformGrid(1),
            pathable=UniformGrid(1),
            player_relative=UniformGrid(0),
            power=UniformGrid(1),
        ),
    )
    unit_names = {72: "CyberneticsCore"}

    assert (
        semantic_argument_candidates(
            observation,
            "Build_Stargate_Screen",
            unit_names=unit_names,
        )
        == []
    )

    core = _unit(0xDEF, 72, 1, 36, 35, 500, 255)
    core.build_progress = 50
    observation.raw_units.append(core)
    assert (
        semantic_argument_candidates(
            observation,
            "Build_Stargate_Screen",
            unit_names=unit_names,
        )
        == []
    )

    core.build_progress = 100
    observation.player_common.minerals = 149
    assert (
        semantic_argument_candidates(
            observation,
            "Build_Stargate_Screen",
            unit_names=unit_names,
        )
        == []
    )

    observation.player_common.minerals = 150
    observation.player_common.vespene = 149
    assert (
        semantic_argument_candidates(
            observation,
            "Build_Stargate_Screen",
            unit_names=unit_names,
        )
        == []
    )

    observation.player_common.vespene = 150
    candidates = semantic_argument_candidates(
        observation,
        "Build_Stargate_Screen",
        unit_names=unit_names,
    )
    assert candidates
    assert all(len(candidate) == 1 and len(candidate[0]) == 2 for candidate in candidates)


def test_shield_battery_requires_core_power_and_clear_full_footprint() -> None:
    buildable = [[0 for _ in range(128)] for _ in range(128)]
    pathable = [[0 for _ in range(128)] for _ in range(128)]
    power = [[0 for _ in range(128)] for _ in range(128)]
    for y in range(59, 71):
        for x in range(59, 71):
            buildable[y][x] = 1
            pathable[y][x] = 1
            power[y][x] = 1
    observation = SimpleNamespace(
        player_common=SimpleNamespace(minerals=100, vespene=0),
        raw_units=[],
        feature_units=[],
        feature_screen=SimpleNamespace(
            buildable=Grid(buildable),
            pathable=Grid(pathable),
            player_relative=UniformGrid(0),
            power=Grid(power),
        ),
    )
    unit_names = {72: "CyberneticsCore"}

    assert (
        semantic_argument_candidates(
            observation,
            "Build_ShieldBattery_Screen",
            unit_names=unit_names,
        )
        == []
    )

    core = _unit(0xC0E, 72, 1, 30, 30, 500, 255)
    core.build_progress = 100
    observation.raw_units.append(core)
    assert [65, 65] in build_screen_candidates(
        observation,
        "Build_ShieldBattery_Screen",
    )
    assert screen_build_position_is_legal(
        observation,
        "Build_ShieldBattery_Screen",
        [65, 65],
        unit_names=unit_names,
    )
    assert not screen_build_position_is_legal(
        observation,
        "Build_ShieldBattery_Screen",
        [5, 5],
        unit_names=unit_names,
    )

    pathable[65][65] = 0
    assert not screen_build_position_is_legal(
        observation,
        "Build_ShieldBattery_Screen",
        [65, 65],
        unit_names=unit_names,
    )
    pathable[65][65] = 1
    power[65][65] = 0
    assert not screen_build_position_is_legal(
        observation,
        "Build_ShieldBattery_Screen",
        [65, 65],
        unit_names=unit_names,
    )
    power[65][65] = 1
    observation.feature_units.append(SimpleNamespace(x=65, y=65, radius=0.5, is_on_screen=True))
    assert not screen_build_position_is_legal(
        observation,
        "Build_ShieldBattery_Screen",
        [65, 65],
        unit_names=unit_names,
    )


def test_assimilator_candidates_use_nearby_unoccupied_raw_geyser() -> None:
    nexus = SimpleNamespace(
        tag=1,
        unit_type=59,
        alliance=1,
        build_progress=100,
        x=10,
        y=10,
    )
    near_geyser = SimpleNamespace(tag=100, unit_type=342, alliance=3, x=15, y=10)
    far_geyser = SimpleNamespace(tag=101, unit_type=342, alliance=3, x=30, y=30)
    observation = SimpleNamespace(
        player_common=SimpleNamespace(minerals=500),
        raw_units=[nexus, near_geyser, far_geyser],
        feature_units=[
            SimpleNamespace(
                tag=100,
                unit_type=342,
                alliance=3,
                is_on_screen=True,
                x=60,
                y=60,
            ),
            SimpleNamespace(
                tag=101,
                unit_type=342,
                alliance=3,
                is_on_screen=True,
                x=80,
                y=80,
            ),
        ],
    )
    unit_names = {59: "Nexus", 61: "Assimilator", 342: "VespeneGeyser"}

    assert semantic_argument_candidates(
        observation,
        "Build_Assimilator_Near",
        unit_names=unit_names,
    ) == [[100]]

    observation.feature_units = []
    assert semantic_argument_candidates(
        observation,
        "Build_Assimilator_Near",
        unit_names=unit_names,
    ) == [[100]]

    observation.raw_units.append(SimpleNamespace(tag=2, unit_type=61, alliance=1, x=15, y=10))
    assert (
        semantic_argument_candidates(
            observation,
            "Build_Assimilator_Near",
            unit_names=unit_names,
        )
        == []
    )


def _nexus_candidate_observation() -> tuple[
    SimpleNamespace,
    list[SimpleNamespace],
    dict[str, UniformGrid],
]:
    raw_offsets = [
        (7, 0),
        (5, 5),
        (0, 7),
        (-5, 5),
        (-7, 0),
        (-5, -5),
        (0, -7),
        (5, -5),
    ]
    screen_offsets = [
        (40, 0),
        (28, 28),
        (0, 40),
        (-28, 28),
        (-40, 0),
        (-28, -28),
        (0, -40),
        (28, -28),
    ]
    raw_resources = [
        SimpleNamespace(
            tag=101 + index,
            unit_type=341,
            alliance=3,
            x=50 + offset_x,
            y=50 + offset_y,
        )
        for index, (offset_x, offset_y) in enumerate(raw_offsets)
    ]
    feature_resources = [
        SimpleNamespace(
            tag=101 + index,
            unit_type=341,
            alliance=3,
            is_on_screen=True,
            display_type=1,
            x=65 + offset_x,
            y=65 + offset_y,
        )
        for index, (offset_x, offset_y) in enumerate(screen_offsets)
    ]
    main_resources = [
        SimpleNamespace(
            tag=index,
            unit_type=341,
            alliance=3,
            x=8 + index % 3,
            y=8 + index // 3,
        )
        for index in range(1, 7)
    ]
    planes = {
        "visibility_map": UniformGrid(2),
        "buildable": UniformGrid(1),
        "pathable": UniformGrid(1),
        "player_relative": UniformGrid(0),
    }
    return (
        SimpleNamespace(
            player_common=SimpleNamespace(minerals=500),
            raw_units=[
                SimpleNamespace(
                    tag=10,
                    unit_type=59,
                    alliance=1,
                    build_progress=100,
                    x=10,
                    y=10,
                ),
                *main_resources,
                *raw_resources,
            ],
            feature_units=feature_resources,
            feature_screen=SimpleNamespace(**planes),
        ),
        feature_resources,
        planes,
    )


def test_nexus_candidates_require_a_current_legal_near_placement() -> None:
    observation, _, _ = _nexus_candidate_observation()

    assert semantic_argument_candidates(
        observation,
        "Build_Nexus_Near",
        unit_names={59: "Nexus", 341: "MineralField"},
    ) == [[101]]


def test_nexus_candidates_require_scouted_visible_resource_cluster() -> None:
    observation, feature_resources, _ = _nexus_candidate_observation()
    observation.feature_units = []

    assert (
        semantic_argument_candidates(
            observation,
            "Build_Nexus_Near",
            unit_names={59: "Nexus", 341: "MineralField"},
        )
        == []
    )

    observation.feature_units = feature_resources
    assert semantic_argument_candidates(
        observation,
        "Build_Nexus_Near",
        unit_names={59: "Nexus", 341: "MineralField"},
    ) == [[101]]


def test_nexus_candidate_requires_the_exact_anchor_to_be_currently_visible() -> None:
    observation, feature_resources, _ = _nexus_candidate_observation()
    observation.feature_units = [unit for unit in feature_resources if unit.tag != 101]
    unit_names = {59: "Nexus", 341: "MineralField"}

    assert (
        semantic_argument_candidates(
            observation,
            "Build_Nexus_Near",
            unit_names=unit_names,
        )
        == []
    )

    observation.feature_units.append(feature_resources[0])
    assert semantic_argument_candidates(
        observation,
        "Build_Nexus_Near",
        unit_names=unit_names,
    ) == [[101]]


def test_nexus_candidates_exclude_resources_near_enemy_hatchery() -> None:
    observation, _, _ = _nexus_candidate_observation()
    without_hatchery = SimpleNamespace(**vars(observation))
    assert semantic_argument_candidates(
        without_hatchery,
        "Build_Nexus_Near",
        unit_names={59: "Nexus", 86: "Hatchery", 341: "MineralField"},
    )

    observation.raw_units.append(
        SimpleNamespace(
            tag=20,
            unit_type=86,
            alliance=4,
            build_progress=100,
            x=50,
            y=50,
        )
    )
    observation.feature_units.append(
        SimpleNamespace(
            tag=20,
            unit_type=86,
            alliance=4,
            is_on_screen=True,
            display_type=1,
            x=65,
            y=65,
        )
    )

    assert (
        semantic_argument_candidates(
            observation,
            "Build_Nexus_Near",
            unit_names={59: "Nexus", 86: "Hatchery", 341: "MineralField"},
        )
        == []
    )


@pytest.mark.parametrize(
    ("plane_name", "invalid_value"),
    [
        ("buildable", 0),
        ("pathable", 0),
        ("player_relative", 1),
        ("visibility_map", 1),
    ],
)
def test_nexus_candidate_requires_a_clear_visible_full_footprint(
    plane_name: str,
    invalid_value: int,
) -> None:
    observation, _, _ = _nexus_candidate_observation()
    setattr(observation.feature_screen, plane_name, UniformGrid(invalid_value))

    assert (
        semantic_argument_candidates(
            observation,
            "Build_Nexus_Near",
            unit_names={59: "Nexus", 341: "MineralField"},
        )
        == []
    )


def test_nexus_candidate_rejects_a_clearance_solution_at_the_screen_edge() -> None:
    size = 24
    resources = [
        SimpleNamespace(
            tag=index + 1,
            unit_type=341,
            alliance=3,
            is_on_screen=True,
            display_type=1,
            x=8,
            y=4 + index * 2,
        )
        for index in range(5)
    ]
    observation = SimpleNamespace(
        player_common=SimpleNamespace(minerals=500),
        raw_units=[
            SimpleNamespace(
                tag=10,
                unit_type=59,
                alliance=1,
                build_progress=100,
                x=-20,
                y=-20,
            ),
            *resources,
        ],
        feature_units=[
            *resources,
            SimpleNamespace(
                tag=20,
                unit_type=59,
                alliance=1,
                is_on_screen=True,
                display_type=1,
                x=15,
                y=8,
            ),
        ],
        feature_screen=SimpleNamespace(
            visibility_map=Grid([[2 for _ in range(size)] for _ in range(size)]),
            buildable=Grid([[1 for _ in range(size)] for _ in range(size)]),
            pathable=Grid([[1 for _ in range(size)] for _ in range(size)]),
            player_relative=Grid([[0 for _ in range(size)] for _ in range(size)]),
        ),
    )

    assert (
        semantic_argument_candidates(
            observation,
            "Build_Nexus_Near",
            unit_names={59: "Nexus", 341: "MineralField"},
        )
        == []
    )


def test_nexus_final_footprint_requires_current_visibility() -> None:
    visibility = [[2 for _ in range(128)] for _ in range(128)]
    observation = SimpleNamespace(feature_screen=SimpleNamespace(visibility_map=Grid(visibility)))

    assert nexus_placement_footprint_is_visible(observation, [64, 64])

    visibility[64][64] = 1
    assert not nexus_placement_footprint_is_visible(observation, [64, 64])

    visibility[64][64] = 2
    visibility[51][51] = 0
    assert not nexus_placement_footprint_is_visible(observation, [64, 64])

    visibility[51][51] = 2
    visibility[20][20] = 0
    assert nexus_placement_footprint_is_visible(observation, [64, 64])
    assert not nexus_placement_footprint_is_visible(observation, [4, 4])


def test_worker_semantic_revalidation_distinguishes_enemy_and_build_targets() -> None:
    friendly = SimpleNamespace(tag=7, alliance=1, is_on_screen=True)
    enemy = SimpleNamespace(tag=8, alliance=4, is_on_screen=True)
    observation = SimpleNamespace(feature_units=[friendly, enemy], raw_units=[])

    assert _semantic_target_failure(
        {"name": "Attack_Unit", "arg": [7], "func": []},
        observation,
        {},
    ) == ("friendly_target", "target 0x7 is not an enemy")
    assert (
        _semantic_target_failure(
            {"name": "Attack_Unit", "arg": [8], "func": []},
            observation,
            {},
        )
        is None
    )
    assert _translation_failure_code("area not pathable", "Build_Pylon_Screen") == ("not_pathable")
    assert (
        _translation_failure_code(
            "no complete footprint with valid resource clearance",
            "Build_Nexus_Near",
        )
        == "invalid_expansion_anchor"
    )


def test_worker_rejects_screen_arguments_outside_current_candidate_domain() -> None:
    observation = SimpleNamespace(
        feature_screen=SimpleNamespace(pathable=UniformGrid(1)),
        feature_units=[
            SimpleNamespace(
                tag=0xABC,
                unit_type=74,
                alliance=1,
                is_on_screen=True,
                x=64,
                y=64,
            )
        ],
    )
    candidates = semantic_argument_candidates(
        observation,
        "Ability_Blink_Screen",
        unit_names={74: "Stalker"},
    )
    assert candidates
    legal_position = candidates[0][0]

    assert (
        _semantic_target_failure(
            {"name": "Ability_Blink_Screen", "arg": [legal_position], "func": []},
            observation,
            {74: "Stalker"},
        )
        is None
    )
    assert _semantic_target_failure(
        {"name": "Ability_Blink_Screen", "arg": [[1, 1]], "func": []},
        observation,
        {74: "Stalker"},
    ) == (
        "candidate_invalidated",
        "Ability_Blink_Screen arguments are outside the current candidate set",
    )
    assert _translation_failure_code("area needs power", "Build_Gateway_Screen") == ("need_power")


def test_worker_revalidates_move_minimap_against_scouting_candidates() -> None:
    observation = SimpleNamespace(
        feature_minimap=SimpleNamespace(
            pathable=Grid([[1 for _ in range(128)] for _ in range(128)]),
            player_relative=Grid([[0 for _ in range(128)] for _ in range(128)]),
        )
    )
    candidates = semantic_argument_candidates(
        observation,
        "Move_Minimap",
        unit_names={},
    )
    assert candidates
    legal_position = candidates[0][0]

    assert (
        _semantic_target_failure(
            {"name": "Move_Minimap", "arg": [legal_position], "func": []},
            observation,
            {},
        )
        is None
    )
    assert _semantic_target_failure(
        {"name": "Move_Minimap", "arg": [[1, 1]], "func": []},
        observation,
        {},
    ) == (
        "candidate_invalidated",
        "Move_Minimap arguments are outside the current candidate set",
    )


def test_worker_detects_an_accepted_primitive_outside_current_candidates() -> None:
    observation = _fake_timestep().observation
    enemy_tag = observation.feature_units[-1].tag
    own_tag = observation.feature_units[0].tag

    assert (
        _candidate_dispatch_failure(
            {"name": "Attack_Unit", "arg": [hex(enemy_tag)], "func": []},
            observation,
            {104: "Drone", 311: "Adept"},
            final_primitive=True,
            translated_position=None,
        )
        is None
    )
    failure = _candidate_dispatch_failure(
        {"name": "Attack_Unit", "arg": [hex(own_tag)], "func": []},
        observation,
        {104: "Drone", 311: "Adept"},
        final_primitive=True,
        translated_position=None,
    )
    assert failure is not None
    assert "current candidate set" in failure


def test_candidate_outside_dispatch_counter_is_persisted_and_fails_command(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    metrics_path = tmp_path / "worker.metrics.json"
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
        metrics_path=str(metrics_path),
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route(
            "Combat",
            ("Army-1",),
            RoutedCommand(
                command_id="command-outside",
                actor="Combat/Army-1",
                team_name="Army-1",
                name="Attack_Unit",
                rendered_action="<Attack_Unit(0xdef)>",
                requested_arguments=("0xdef",),
                resolved_arguments=("0xdef",),
            ),
            step_id=55,
        ),
    )
    dispatch = broker.claim_primitive(
        "Combat",
        "Army-1",
        "Attack_Unit",
        "Attack_screen",
        final_primitive=True,
        ordinal=0,
        total=1,
        requested_function_id=12,
        emitted_function_id=12,
    )
    assert dispatch is not None
    assert broker.metrics()["candidate_outside_pysc2_dispatches"] == 0
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "unattributed_primitives": 0,
        "candidate_outside_pysc2_dispatches": 0,
        "observation_gap_watchdog_triggers": 0,
    }

    with pytest.raises(RuntimeError, match="outside the current candidate set"):
        broker.reject_candidate_outside_dispatch(
            dispatch,
            "Attack_Unit target is outside the current candidate set",
            game_loop=900,
        )

    assert broker.metrics()["candidate_outside_pysc2_dispatches"] == 1
    assert (
        json.loads(metrics_path.read_text(encoding="utf-8"))["candidate_outside_pysc2_dispatches"]
        == 1
    )
    assert len(runtime.execution_reports) == 1
    report = runtime.execution_reports[0]
    assert report["command_id"] == "command-outside"
    assert report["status"] == "failed"
    assert report["execution_stage"] == "translation"
    assert report["failure_code"] == "bridge_integrity_error"
    assert report["primitive_trace"][0] == {
        "function_name": "Attack_screen",
        "requested_function_id": 12,
        "emitted_function_id": 12,
        "origin": "translator",
        "ordinal": 0,
        "total": 1,
        "game_loop": 900,
        "accepted": False,
        "failure_code": "bridge_integrity_error",
        "detail": "Attack_Unit target is outside the current candidate set",
    }


def test_pysc2_argument_normalization_matches_symbolic_and_encoded_enums() -> None:
    assert _canonical_pysc2_arguments(331, ["now", (64, 48)]) == (0, (64, 48))
    assert _canonical_pysc2_arguments(331, [[0], [64, 48]]) == (0, (64, 48))
    assert _canonical_pysc2_arguments(331, ["queued", [48, 64]]) == (1, (48, 64))


def test_main_agent_detects_candidate_argument_mutation_before_pysc2() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route(
            "Combat",
            ("Army-1",),
            RoutedCommand(
                command_id="command-mutated",
                actor="Combat/Army-1",
                team_name="Army-1",
                name="Attack_Unit",
                rendered_action="<Attack_Unit(0xdef)>",
                requested_arguments=("0xdef",),
                resolved_arguments=("0xdef",),
            ),
            step_id=56,
        ),
    )
    dispatch = broker.claim_primitive(
        "Combat",
        "Army-1",
        "Attack_Unit",
        "Attack_screen",
        final_primitive=True,
        ordinal=0,
        total=1,
        requested_function_id=12,
        emitted_function_id=12,
    )
    assert dispatch is not None
    upstream_agent = SimpleNamespace(
        _rtscortex_translation_attempt={
            "dispatch": dispatch,
            "emitted_function_id": 12,
            "expected_arguments": [[0], [40, 40]],
            "candidate_constrained": True,
        }
    )
    main_agent = cast(Any, object.__new__(RTSCortexMainAgent))
    main_agent._pending_primitive = None
    main_agent.AGENT_NAMES = ["Combat"]
    main_agent.agent_id = 0
    main_agent.agents = {"Combat": upstream_agent}
    main_agent.decision_broker = broker
    action = SimpleNamespace(function=12, arguments=[[0], [80, 80]])
    observation = SimpleNamespace(observation=SimpleNamespace(game_loop=[901]))

    with pytest.raises(RuntimeError, match="changed translator arguments"):
        main_agent._capture_primitive(action, observation)

    assert broker.metrics()["candidate_outside_pysc2_dispatches"] == 1
    assert len(runtime.execution_reports) == 1
    assert runtime.execution_reports[0]["command_id"] == "command-mutated"
    assert runtime.execution_reports[0]["failure_code"] == "bridge_integrity_error"


def test_timestep_extractor_exposes_developer_empty_team_actions() -> None:
    timestep = _fake_timestep()
    gateway = _unit(0xABC, 62, 1, 35, 35, 500, 255)
    gateway.build_progress = 100
    gateway.active = 0
    timestep.observation.raw_units.append(gateway)
    train = {"name": "Train_Zealot", "arg": [], "func": [(100, None, ())]}
    agent = SimpleNamespace(
        name="Developer",
        flag_enable_empty_unit_group=True,
        team_unit_team_list=[],
        team_unit_obs_list=[timestep],
        config=SimpleNamespace(
            AGENTS={
                "Developer": {
                    "team": [{"name": "Empty", "unit_type": []}],
                    "action": {"EmptyGroup": [train]},
                }
            }
        ),
    )

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={62: "Gateway"},
        action_source_types={100: 62},
    ).extract(
        timestep,
        {"Developer": agent},
        {"Developer": "production overview"},
        step_id=1,
    )

    assert current_team_order(agent) == ("Empty",)
    assert snapshot["teams"] == [
        {
            "agent_name": "Developer",
            "team_name": "Empty",
            "available_actions": [
                {
                    "name": "No_Operation",
                    "argument_names": [],
                    "argument_types": [],
                    "argument_candidates": None,
                },
                {
                    "name": "Train_Zealot",
                    "argument_names": [],
                    "argument_types": [],
                    "argument_candidates": None,
                },
            ],
        }
    ]


def test_current_team_order_deduplicates_logical_select_all_team() -> None:
    agent = SimpleNamespace(
        flag_enable_empty_unit_group=False,
        team_unit_team_list=["Zealot-1", "Zealot-1", "Stalker-1"],
    )

    assert current_team_order(agent) == ("Zealot-1", "Stalker-1")


def test_timestep_extractor_hides_production_action_without_source_structure() -> None:
    timestep = _fake_timestep()
    train = {"name": "Train_Zealot", "arg": [], "func": [(100, None, ())]}
    agent = SimpleNamespace(
        name="Developer",
        flag_enable_empty_unit_group=True,
        team_unit_team_list=[],
        team_unit_obs_list=[timestep],
        config=SimpleNamespace(
            AGENTS={
                "Developer": {
                    "team": [{"name": "Empty", "unit_type": []}],
                    "action": {"EmptyGroup": [train]},
                }
            }
        ),
    )

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        action_source_types={100: 999},
    ).extract(
        timestep,
        {"Developer": agent},
        {"Developer": "production overview"},
        step_id=1,
    )

    assert snapshot["teams"][0]["available_actions"] == [
        {
            "name": "No_Operation",
            "argument_names": [],
            "argument_types": [],
            "argument_candidates": None,
        }
    ]


@pytest.mark.parametrize(
    ("build_progress", "active", "order_length"),
    [(20, 0, 0), (100, 1, 0), (100, 0, 1)],
)
def test_timestep_extractor_hides_train_until_gateway_is_complete_and_idle(
    build_progress: int,
    active: int,
    order_length: int,
) -> None:
    timestep = _fake_timestep()
    gateway = _unit(0xABC, 62, 1, 35, 35, 500, 255)
    gateway.build_progress = build_progress
    gateway.active = active
    gateway.order_length = order_length
    timestep.observation.raw_units.append(gateway)
    train = {"name": "Train_Zealot", "arg": [], "func": [(100, None, ())]}
    agent = _developer_agent(timestep, [train])

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={62: "Gateway"},
        action_source_types={100: 62},
    ).extract(
        timestep,
        {"Developer": agent},
        {"Developer": "production overview"},
        step_id=1,
    )

    assert [action["name"] for action in snapshot["teams"][0]["available_actions"]] == [
        "No_Operation"
    ]


def test_train_stalker_requires_completed_cybernetics_core() -> None:
    timestep = _fake_timestep()
    gateway = _unit(0xABC, 62, 1, 35, 35, 500, 255)
    gateway.build_progress = 100
    gateway.active = 0
    timestep.observation.raw_units.append(gateway)
    stalker = {"name": "Train_Stalker", "arg": [], "func": [(101, None, ())]}
    agent = _developer_agent(timestep, [stalker])
    extractor = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={62: "Gateway", 72: "CyberneticsCore"},
        action_source_types={101: 62},
    )

    without_core = extractor.extract(
        timestep,
        {"Developer": agent},
        {"Developer": "production overview"},
        step_id=1,
    )
    core = _unit(0xDEF, 72, 1, 36, 35, 500, 255)
    core.build_progress = 100
    timestep.observation.raw_units.append(core)
    with_core = extractor.extract(
        timestep,
        {"Developer": agent},
        {"Developer": "production overview"},
        step_id=2,
    )

    assert [action["name"] for action in without_core["teams"][0]["available_actions"]] == [
        "No_Operation"
    ]
    assert [action["name"] for action in with_core["teams"][0]["available_actions"]] == [
        "No_Operation",
        "Train_Stalker",
    ]


def test_research_warpgate_requires_its_full_resource_cost() -> None:
    timestep = _fake_timestep()
    core = _unit(0xDEF, 72, 1, 36, 35, 500, 255)
    core.build_progress = 100
    core.active = 0
    timestep.observation.raw_units.append(core)
    research = {"name": "Research_WarpGate", "arg": [], "func": [(428, None, ())]}
    agent = _developer_agent(timestep, [research])
    extractor = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={72: "CyberneticsCore"},
        action_source_types={428: 72},
    )

    timestep.observation.player.vespene = 0
    without_gas = extractor.extract(
        timestep,
        {"Developer": agent},
        {"Developer": "production overview"},
        step_id=1,
    )
    timestep.observation.player.vespene = 50
    with_full_cost = extractor.extract(
        timestep,
        {"Developer": agent},
        {"Developer": "production overview"},
        step_id=2,
    )

    assert [action["name"] for action in without_gas["teams"][0]["available_actions"]] == [
        "No_Operation"
    ]
    assert [action["name"] for action in with_full_cost["teams"][0]["available_actions"]] == [
        "No_Operation",
        "Research_WarpGate",
    ]


def test_production_source_resolver_returns_the_idle_completed_structure_tag() -> None:
    timestep = _fake_timestep()
    busy_gateway = _unit(0xAAA, 62, 1, 35, 35, 500, 255)
    busy_gateway.build_progress = 100
    busy_gateway.active = 1
    idle_gateway = _unit(0xBBB, 62, 1, 36, 35, 500, 255)
    idle_gateway.build_progress = 100
    idle_gateway.active = 0
    timestep.observation.raw_units.extend([busy_gateway, idle_gateway])

    assert (
        production_source_tag(
            timestep.observation,
            {"name": "Train_Zealot", "func": [(100, None, ())]},
            unit_names={62: "Gateway"},
            action_source_types={100: 62},
        )
        == 0xBBB
    )


def test_production_source_follows_upstream_raw_order_instead_of_tag_order() -> None:
    timestep = _fake_timestep()
    first = _unit(0xBBB, 62, 1, 36, 35, 500, 255)
    second = _unit(0xAAA, 62, 1, 37, 35, 500, 255)
    for gateway in (first, second):
        gateway.build_progress = 100
        gateway.active = 0
    timestep.observation.raw_units.extend([first, second])

    resolved = production_source_tag(
        timestep.observation,
        {"name": "Train_Zealot", "func": [(503, None, ())]},
        unit_names={62: "Gateway"},
        action_source_types={503: 62},
    )

    assert resolved == 0xBBB


def test_train_registry_pins_multirace_worker_actions_and_raw_orders() -> None:
    assert {action: spec.raw_order_id for action, spec in PRODUCTION_SPECS.items()} == {
        "Train_Zealot": 49,
        "Train_Stalker": 50,
        "Train_Adept": 54,
        "Train_Phoenix": 55,
        "Train_VoidRay": 57,
        "Train_Oracle": 58,
        "Train_Marine": 511,
        "Train_Marauder": 510,
        "Train_Hellion": 506,
        "Train_SiegeTank": 521,
        "Train_Medivac": 512,
        "Train_VikingFighter": 525,
    }


def test_addon_source_resolver_requires_idle_unattached_exact_producer() -> None:
    spec = ADDON_SPECS["Build_BarracksReactor"]
    timestep = _fake_timestep()
    timestep.observation.player.minerals = spec.minerals
    timestep.observation.player.vespene = spec.vespene
    attached = _unit(0xAAA, 21, 1, 35, 35, 500, 255)
    attached.build_progress = 100
    attached.active = 0
    attached.add_on_tag = 0xBAD
    available = _unit(0xBBB, 21, 1, 36, 35, 500, 255)
    available.build_progress = 100
    available.active = 0
    available.add_on_tag = 0
    timestep.observation.raw_units.extend([attached, available])

    resolved = production_source_tag(
        timestep.observation,
        {
            "name": spec.action_name,
            "func": [(spec.feature_function_id, None, ("queued",))],
        },
        unit_names={21: "Barracks"},
        action_source_types={spec.feature_function_id: 21},
    )

    assert resolved == 0xBBB

    assert (
        production_source_tag(
            timestep.observation,
            {
                "name": spec.action_name,
                "func": [(spec.feature_function_id, None, ("queued",))],
            },
            unit_names={21: "Barracks"},
            action_source_types={spec.feature_function_id: 21},
            excluded_source_tags={0xBBB},
        )
        is None
    )


def test_production_source_rejects_new_structure_false_completion_frame() -> None:
    spec = PRODUCTION_SPECS["Train_Marine"]
    timestep = _fake_timestep()
    timestep.observation.player.minerals = spec.minerals
    incomplete = _unit(0xBBB, 21, 1, 36, 35, 55, 28)
    incomplete.health_max = 1000
    incomplete.build_progress = 100
    incomplete.active = 0
    incomplete.order_length = 0
    timestep.observation.raw_units.append(incomplete)

    assert (
        production_source_tag(
            timestep.observation,
            {"name": spec.action_name, "func": [(spec.feature_function_id, None, ())]},
            unit_names={21: "Barracks"},
            action_source_types={spec.feature_function_id: 21},
        )
        is None
    )


def test_addon_unavailable_function_has_placement_failure_code() -> None:
    assert (
        _translation_failure_code(
            "function Build_Reactor_Barracks_quick is not available",
            "Build_BarracksReactor",
        )
        == "no_legal_addon_placement"
    )


@pytest.mark.parametrize("action_name", ["Train_Phoenix", "Train_Oracle"])
def test_stargate_train_availability_uses_registry_costs(action_name: str) -> None:
    spec = PRODUCTION_SPECS[action_name]
    timestep = _fake_timestep()
    timestep.observation.player.minerals = spec.minerals
    timestep.observation.player.vespene = spec.vespene
    timestep.observation.player.food_used = 10
    timestep.observation.player.food_cap = 10 + spec.supply
    stargate = _unit(0x57A, 67, 1, 36, 35, 500, 255)
    stargate.build_progress = 100
    stargate.active = 0
    timestep.observation.raw_units.append(stargate)

    assert (
        production_source_tag(
            timestep.observation,
            {"name": action_name, "func": [(spec.feature_function_id, None, ())]},
            unit_names={67: "Stargate"},
            action_source_types={spec.feature_function_id: 67},
        )
        == 0x57A
    )

    timestep.observation.player.vespene -= 1
    assert (
        production_source_tag(
            timestep.observation,
            {"name": action_name, "func": [(spec.feature_function_id, None, ())]},
            unit_names={67: "Stargate"},
            action_source_types={spec.feature_function_id: 67},
        )
        is None
    )


def test_extractor_projects_production_queue_with_exact_producer_tag() -> None:
    timestep = _fake_timestep()
    gateway = _unit(0xBBB, 62, 1, 36, 35, 500, 255)
    gateway.build_progress = 100
    gateway.active = 1
    gateway.order_length = 1
    gateway.order_id_0 = 49
    gateway.order_progress_0 = 0.25
    timestep.observation.raw_units.append(gateway)
    timestep.observation.production_queue = [SimpleNamespace(ability_id=916, build_progress=25)]

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={62: "Gateway"},
    ).extract(timestep, {}, {}, step_id=1)

    assert snapshot["production_queue"] == [
        {"name": "Train_Zealot", "producer_tag": 0xBBB, "progress": 0.25}
    ]


def test_translator_source_tag_replaces_stale_preflight_provenance() -> None:
    timestep = _fake_timestep()
    first = _unit(0xBBB, 62, 1, 36, 35, 500, 255)
    second = _unit(0xAAA, 62, 1, 37, 35, 500, 255)
    for gateway in (first, second):
        gateway.build_progress = 100
        gateway.active = 0
    timestep.observation.raw_units.extend([first, second])
    broker = SharedDecisionBroker(
        BridgeCoordinator(FakeRuntime()),
        TimeStepExtractor(
            "run-worker",
            "episode-worker",
            unit_names={62: "Gateway"},
            action_source_types={503: 62},
        ),
    )
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.name = "Developer"
    agent.broker = broker
    agent.unit_names = {62: "Gateway"}
    agent.team_unit_team_curr = None
    agent.team_unit_tag_curr = None
    agent.team_unit_tag_list = []
    agent.flag_enable_empty_unit_group = True
    action = {"name": "Train_Zealot", "func": [(503, None, ())]}

    assert agent._reject_unavailable_production_action(action, timestep) is False
    assert agent._rtscortex_production_source_tag == 0xBBB
    producer_tag, failure = agent._validated_production_source_tag(
        "Train_Zealot",
        {
            "ordinal": 0,
            "requested_function_id": 573,
            "requested_arguments": [0xAAA],
            "resolved_arguments": [(39.0, 68.0)],
        },
    )

    assert producer_tag == 0xAAA
    assert failure is None
    assert agent._rtscortex_production_source_tag == 0xAAA


def test_missing_production_provenance_publishes_one_translation_failure() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    command = RoutedCommand(
        command_id="command-train-provenance",
        actor="Developer/Empty",
        team_name="Empty",
        name="Train_Zealot",
        rendered_action="<Train_Zealot()>",
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route("Developer", ("Empty",), command, step_id=68),
    )
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.name = "Developer"
    agent.broker = broker
    agent._rtscortex_production_source_tag = None
    agent.team_unit_team_curr = None
    agent.team_unit_tag_curr = None
    agent.team_unit_tag_list = []
    agent.flag_enable_empty_unit_group = True
    agent.func_list = []
    agent._rtscortex_semantic_action = {"name": "Train_Zealot"}
    timestep = _fake_timestep()

    producer_tag, failure = agent._validated_production_source_tag(
        "Train_Zealot",
        {
            "ordinal": 0,
            "requested_function_id": 573,
            "requested_arguments": [],
            "resolved_arguments": [(1, 2)],
        },
    )
    assert producer_tag is None
    assert failure is not None
    failure_code, reason = failure
    agent._settle_production_command_failure(
        "Train_Zealot",
        timestep,
        failure_code=failure_code,
        reason=reason,
    )
    broker.end_episode(_episode_result())

    assert len(runtime.execution_reports) == 1
    report = runtime.execution_reports[0]
    assert report["status"] == "failed"
    assert report["execution_stage"] == "translation"
    assert report["failure_code"] == "production_provenance_missing"


def test_disappearing_producer_after_camera_select_is_one_predispatch_failure() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    command = RoutedCommand(
        command_id="command-train-invalidated",
        actor="Developer/Empty",
        team_name="Empty",
        name="Train_Zealot",
        rendered_action="<Train_Zealot()>",
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route("Developer", ("Empty",), command, step_id=69),
    )
    for ordinal, function_name, function_id in (
        (0, "llm_pysc2_move_camera", 573),
        (1, "select_point", 2),
    ):
        dispatch = broker.claim_primitive(
            "Developer",
            "Empty",
            "Train_Zealot",
            function_name,
            final_primitive=False,
            ordinal=ordinal,
            total=3,
            requested_function_id=function_id,
            emitted_function_id=function_id,
        )
        assert dispatch is not None
        broker.settle_primitive(dispatch, success=True, game_loop=224 + ordinal)

    timestep = _fake_timestep()
    reason = _production_source_invalid_reason(
        timestep.observation,
        0xBBB,
        PRODUCTION_SPECS["Train_Zealot"],
        {62: "Gateway"},
    )
    assert reason is not None
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.name = "Developer"
    agent.broker = broker
    agent.team_unit_team_curr = None
    agent.team_unit_tag_curr = None
    agent.team_unit_tag_list = []
    agent.flag_enable_empty_unit_group = True
    agent.func_list = []
    agent._rtscortex_semantic_action = {"name": "Train_Zealot"}
    agent._rtscortex_production_source_tag = 0xBBB
    agent._settle_production_command_failure(
        "Train_Zealot",
        timestep,
        failure_code="production_source_invalidated",
        reason=reason,
    )
    broker.end_episode(_episode_result())

    assert len(runtime.execution_reports) == 1
    report = runtime.execution_reports[0]
    assert report["status"] == "failed"
    assert report["execution_stage"] == "pre_dispatch"
    assert report["failure_code"] == "production_source_invalidated"
    assert len(report["primitive_trace"]) == 3


@pytest.mark.parametrize(
    ("action_name", "minerals", "vespene", "food_used", "food_cap", "expected"),
    [
        ("Train_Zealot", 100, 0, 13, 15, True),
        ("Train_Zealot", 99, 0, 13, 15, False),
        ("Train_Zealot", 100, 0, 14, 15, False),
        ("Train_Stalker", 125, 50, 13, 15, True),
        ("Train_Stalker", 124, 50, 13, 15, False),
        ("Train_Stalker", 125, 49, 13, 15, False),
        ("Train_Stalker", 125, 50, 14, 15, False),
    ],
)
def test_known_simple64_production_cost_boundaries(
    action_name: str,
    minerals: int,
    vespene: int,
    food_used: int,
    food_cap: int,
    expected: bool,
) -> None:
    timestep = _fake_timestep()
    timestep.observation.player.minerals = minerals
    timestep.observation.player.vespene = vespene
    timestep.observation.player.food_used = food_used
    timestep.observation.player.food_cap = food_cap
    gateway = _unit(0xABC, 62, 1, 35, 35, 500, 255)
    gateway.build_progress = 100
    gateway.active = 0
    timestep.observation.raw_units.append(gateway)
    if action_name == "Train_Stalker":
        core = _unit(0xDEF, 72, 1, 36, 35, 500, 255)
        core.build_progress = 100
        timestep.observation.raw_units.append(core)

    resolved = production_source_tag(
        timestep.observation,
        {"name": action_name, "func": [(100, None, ())]},
        unit_names={62: "Gateway", 72: "CyberneticsCore"},
        action_source_types={100: 62},
    )

    assert (resolved == 0xABC) is expected


@pytest.mark.parametrize(
    ("minerals", "vespene", "food_used", "food_cap", "core_complete", "expected"),
    [
        (100, 25, 13, 15, True, True),
        (99, 25, 13, 15, True, False),
        (100, 24, 13, 15, True, False),
        (100, 25, 14, 15, True, False),
        (100, 25, 13, 15, False, False),
    ],
)
def test_train_adept_requires_full_cost_supply_core_and_gateway_source(
    minerals: int,
    vespene: int,
    food_used: int,
    food_cap: int,
    core_complete: bool,
    expected: bool,
) -> None:
    timestep = _fake_timestep()
    timestep.observation.player.minerals = minerals
    timestep.observation.player.vespene = vespene
    timestep.observation.player.food_used = food_used
    timestep.observation.player.food_cap = food_cap
    gateway = _unit(0xADE, 62, 1, 35, 35, 500, 255)
    gateway.build_progress = 100
    gateway.active = 0
    stargate = _unit(0x57A, 67, 1, 36, 35, 500, 255)
    stargate.build_progress = 100
    stargate.active = 0
    timestep.observation.raw_units.extend([gateway, stargate])
    if core_complete:
        core = _unit(0xC0E, 72, 1, 37, 35, 500, 255)
        core.build_progress = 100
        timestep.observation.raw_units.append(core)

    resolved = production_source_tag(
        timestep.observation,
        {"name": "Train_Adept", "func": [(457, None, ())]},
        unit_names={62: "Gateway", 67: "Stargate", 72: "CyberneticsCore"},
        action_source_types={457: 62},
    )

    assert (resolved == 0xADE) is expected


@pytest.mark.parametrize(
    ("minerals", "vespene", "food_used", "food_cap", "expected"),
    [
        (250, 150, 11, 15, True),
        (249, 150, 11, 15, False),
        (250, 149, 11, 15, False),
        (250, 150, 12, 15, False),
    ],
)
def test_train_voidray_requires_full_cost_supply_and_stargate_source(
    minerals: int,
    vespene: int,
    food_used: int,
    food_cap: int,
    expected: bool,
) -> None:
    timestep = _fake_timestep()
    timestep.observation.player.minerals = minerals
    timestep.observation.player.vespene = vespene
    timestep.observation.player.food_used = food_used
    timestep.observation.player.food_cap = food_cap
    gateway = _unit(0x6A7, 62, 1, 35, 35, 500, 255)
    gateway.build_progress = 100
    gateway.active = 0
    stargate = _unit(0x57A, 67, 1, 36, 35, 500, 255)
    stargate.build_progress = 100
    stargate.active = 0
    timestep.observation.raw_units.extend([gateway, stargate])

    resolved = production_source_tag(
        timestep.observation,
        {"name": "Train_VoidRay", "func": [(500, None, ())]},
        unit_names={62: "Gateway", 67: "Stargate"},
        action_source_types={500: 67},
    )

    assert (resolved == 0x57A) is expected


def test_full_supply_hides_zealot_despite_completed_idle_gateway() -> None:
    timestep = _fake_timestep()
    timestep.observation.player.minerals = 500
    timestep.observation.player.food_used = timestep.observation.player.food_cap
    gateway = _unit(0xABC, 62, 1, 35, 35, 500, 255)
    gateway.build_progress = 100
    gateway.active = 0
    timestep.observation.raw_units.append(gateway)
    train = {"name": "Train_Zealot", "arg": [], "func": [(100, None, ())]}
    agent = _developer_agent(timestep, [train])

    snapshot = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={62: "Gateway"},
        action_source_types={100: 62},
    ).extract(
        timestep,
        {"Developer": agent},
        {"Developer": "production overview"},
        step_id=1,
    )

    assert [action["name"] for action in snapshot["teams"][0]["available_actions"]] == [
        "No_Operation"
    ]


def test_unknown_production_action_keeps_source_only_availability() -> None:
    timestep = _fake_timestep()
    timestep.observation.player.minerals = 0
    timestep.observation.player.vespene = 0
    timestep.observation.player.food_used = timestep.observation.player.food_cap
    gateway = _unit(0xABC, 62, 1, 35, 35, 500, 255)
    gateway.build_progress = 100
    gateway.active = 0
    timestep.observation.raw_units.append(gateway)

    assert (
        production_source_tag(
            timestep.observation,
            {"name": "Train_Unknown", "func": [(100, None, ())]},
            unit_names={62: "Gateway"},
            action_source_types={100: 62},
        )
        == 0xABC
    )


def test_worker_closes_unavailable_production_command_and_frees_actor() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    extractor = TimeStepExtractor(
        "run-worker",
        "episode-worker",
        unit_names={62: "Gateway"},
        action_source_types={100: 62},
    )
    broker = SharedDecisionBroker(coordinator, extractor)
    command = RoutedCommand(
        command_id="command-train-unavailable",
        actor="Developer/Empty",
        team_name="Empty",
        name="Train_Zealot",
        rendered_action="<Train_Zealot()>",
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route("Developer", ("Empty",), command, step_id=68),
    )
    timestep = _fake_timestep()
    gateway = _unit(0xABC, 62, 1, 35, 35, 100, 51)
    gateway.build_progress = 20
    gateway.active = 0
    timestep.observation.raw_units.append(gateway)
    agent = cast(Any, object.__new__(RTSCortexLLMAgent))
    agent.name = "Developer"
    agent.broker = broker
    agent.unit_names = {62: "Gateway"}
    agent.team_unit_team_curr = None
    agent.team_unit_tag_curr = None
    agent.team_unit_tag_list = []
    agent.flag_enable_empty_unit_group = True
    agent.action_list = [{"name": "Train_Zealot", "arg": [], "func": [(100, None, ())]}]
    agent.func_list = []
    agent._rtscortex_semantic_action = agent.action_list[0]

    assert agent._reject_unavailable_production_action(agent.action_list[0], timestep)

    assert agent.action_list == []
    assert agent.func_list == []
    assert len(runtime.execution_reports) == 1
    assert runtime.execution_reports[0]["command_id"] == "command-train-unavailable"
    assert runtime.execution_reports[0]["status"] == "failed"
    assert runtime.execution_reports[0]["execution_stage"] == "pre_dispatch"
    assert runtime.execution_reports[0]["failure_code"] == "production_source_unavailable"
    assert broker.command_id_for("Developer", "Empty", "Train_Zealot") is None


def test_upstream_train_to_noop_rewrite_is_detected_by_original_action_name() -> None:
    assert _upstream_replaced_production_with_noop(
        "Train_Zealot",
        {"action_name": "No_Operation", "requested_function_id": 0, "accepted": True},
    )
    assert not _upstream_replaced_production_with_noop(
        "Train_Zealot",
        {"action_name": "Train_Zealot", "requested_function_id": 503, "accepted": True},
    )
    assert not _upstream_replaced_production_with_noop(
        "Move_Screen",
        {"action_name": "No_Operation", "requested_function_id": 0, "accepted": True},
    )


def test_empty_team_is_used_for_developer_primitive_tracking() -> None:
    agent = SimpleNamespace(
        flag_enable_empty_unit_group=True,
        team_unit_tag_list=[],
        team_unit_team_curr=None,
    )

    assert _execution_team_name(agent) == "Empty"


def test_broker_preserves_explicit_developer_team_name() -> None:
    broker = SharedDecisionBroker(
        BridgeCoordinator(FakeRuntime()),
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    broker._command_queues[("Developer", "WarpGate-1", "Train_Zealot")].append(  # noqa: SLF001
        "command-train"
    )

    agent = SimpleNamespace(
        flag_enable_empty_unit_group=True,
        team_unit_tag_list=[],
        team_unit_tag_curr=0xABC,
        team_unit_team_curr="WarpGate-1",
    )
    assert _execution_team_name(agent) == "WarpGate-1"
    dispatch = broker.claim_primitive(
        "Developer",
        _execution_team_name(agent),
        "Train_Zealot",
        "Train_Zealot_quick",
        final_primitive=True,
        ordinal=0,
        total=1,
    )

    assert dispatch is not None
    assert dispatch.command_id == "command-train"


def test_broker_reject_command_publishes_one_exact_terminal_report() -> None:
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
    agent = FakeAgent("AgentA", "A", timestep, broker)
    broker.register(agent)
    agent.query(timestep)

    dispatch = broker.reject_command(
        "AgentA",
        "A",
        "Attack_Unit",
        failure_code="actor_not_available",
    )
    assert dispatch == PrimitiveDispatch(
        command_id="command-attack",
        function_name="pre_dispatch",
        final_primitive=True,
        ordinal=0,
        total=1,
        failure_code="actor_not_available",
    )

    broker.settle_primitive(
        dispatch,
        success=False,
        failure_reason="team head unit is unavailable before action translation",
        game_loop=927,
    )
    broker.end_episode(_episode_result())

    assert len(runtime.execution_reports) == 1
    report = runtime.execution_reports[0]
    assert report["command_id"] == "command-attack"
    assert report["action_name"] == "Attack_Unit"
    assert report["actor"] == "AgentA/A"
    assert report["status"] == "failed"
    assert report["execution_stage"] == "pre_dispatch"
    assert report["failure_code"] == "actor_not_available"
    assert report["primitive_trace"] == [
        {
            "function_name": "pre_dispatch",
            "requested_function_id": None,
            "emitted_function_id": None,
            "origin": "translator",
            "ordinal": 0,
            "total": 1,
            "game_loop": 927,
            "accepted": False,
            "failure_code": "actor_not_available",
            "detail": "team head unit is unavailable before action translation",
        }
    ]
    assert (
        broker.reject_command(
            "AgentA",
            "A",
            "Attack_Unit",
            failure_code="actor_not_available",
        )
        is None
    )


def test_worker_consumes_upstream_abort_marker_with_exact_identity() -> None:
    calls: list[tuple[Any, ...]] = []
    dispatch = PrimitiveDispatch(
        command_id="command-move",
        function_name="pre_dispatch",
        final_primitive=True,
        ordinal=0,
        total=1,
        failure_code="actor_not_available",
    )

    class AbortBroker:
        def reject_command(
            self,
            agent_name: str,
            team_name: str,
            action_name: str,
            *,
            failure_code: str,
        ) -> PrimitiveDispatch:
            calls.append(("reject", agent_name, team_name, action_name, failure_code))
            return dispatch

        def settle_primitive(
            self,
            value: PrimitiveDispatch,
            *,
            success: bool,
            failure_reason: str,
            game_loop: int,
        ) -> None:
            calls.append(("settle", value, success, failure_reason, game_loop))

    agent = SimpleNamespace(
        name="Builder",
        last_execution_abort={
            "team_name": "Builder-Probe-1",
            "action_name": "Move_Screen",
            "actor_tag": 0x101480001,
            "failure_code": "actor_not_available",
            "failure_reason": (
                "team head unit disappearance was confirmed before action translation"
            ),
        },
    )
    main_agent = cast(Any, object.__new__(RTSCortexMainAgent))
    main_agent.agents = {"Builder": agent}
    main_agent.decision_broker = AbortBroker()
    observation = SimpleNamespace(observation=SimpleNamespace(game_loop=[927]))

    main_agent._consume_execution_aborts(observation)

    assert agent.last_execution_abort is None
    assert calls == [
        (
            "reject",
            "Builder",
            "Builder-Probe-1",
            "Move_Screen",
            "actor_not_available",
        ),
        (
            "settle",
            dispatch,
            False,
            "team head unit disappearance was confirmed before action translation: "
            "Builder/Builder-Probe-1 (tag 0x101480001)",
            927,
        ),
    ]


def test_worker_keeps_transiently_missing_actor_pending_without_abort_marker() -> None:
    class NoAbortBroker:
        def reject_command(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("transient actor disappearance must not terminate a command")

    agent = SimpleNamespace(name="Builder", last_execution_abort=None)
    main_agent = cast(Any, object.__new__(RTSCortexMainAgent))
    main_agent.agents = {"Builder": agent}
    main_agent.decision_broker = NoAbortBroker()
    observation = SimpleNamespace(observation=SimpleNamespace(game_loop=[927]))

    main_agent._consume_execution_aborts(observation)

    assert agent.last_execution_abort is None


def test_broker_attributes_builder_failure_and_combat_success_in_same_step() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route(
            "Builder",
            ("Builder-Probe-1",),
            RoutedCommand(
                command_id="command-builder",
                actor="Builder/Builder-Probe-1",
                team_name="Builder-Probe-1",
                name="Build_Pylon_Screen",
                rendered_action="<Build_Pylon_Screen([65,65])>",
                requested_arguments=([65, 65],),
                resolved_arguments=([65, 65],),
            ),
            step_id=902,
        ),
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route(
            "Combat",
            ("Army-1",),
            RoutedCommand(
                command_id="command-combat",
                actor="Combat/Army-1",
                team_name="Army-1",
                name="Attack_Unit",
                rendered_action="<Attack_Unit(0xdef)>",
                requested_arguments=("0xdef",),
                resolved_arguments=("0xdef",),
            ),
            step_id=902,
        ),
    )

    builder = broker.reject_command(
        "Builder",
        "Builder-Probe-1",
        "Build_Pylon_Screen",
        failure_code="candidate_invalidated",
    )
    combat = broker.claim_primitive(
        "Combat",
        "Army-1",
        "Attack_Unit",
        "Attack_screen",
        final_primitive=True,
        ordinal=0,
        total=1,
        requested_function_id=12,
        emitted_function_id=12,
    )

    assert builder is not None
    assert combat is not None
    broker.settle_primitive(
        builder,
        success=False,
        failure_reason="screen placement candidate became stale",
        game_loop=20811,
    )
    broker.settle_primitive(combat, success=True, game_loop=20811)

    reports = {report["command_id"]: report for report in runtime.execution_reports}
    assert set(reports) == {"command-builder", "command-combat"}
    assert reports["command-builder"]["actor"] == "Builder/Builder-Probe-1"
    assert reports["command-builder"]["execution_stage"] == "pre_dispatch"
    assert reports["command-builder"]["failure_code"] == "candidate_invalidated"
    assert reports["command-builder"]["primitive_trace"][0]["function_name"] == "pre_dispatch"
    assert reports["command-combat"]["actor"] == "Combat/Army-1"
    assert reports["command-combat"]["status"] == "succeeded"
    assert reports["command-combat"]["pysc2_function"] == "Attack_screen"
    assert reports["command-combat"]["primitive_trace"][0]["failure_code"] is None


def test_broker_keeps_same_attack_action_isolated_by_explicit_team() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    route = _bridge_route(
        "Combat",
        ("Alpha", "Beta"),
        RoutedCommand(
            command_id="attack-alpha",
            actor="Combat/Alpha",
            team_name="Alpha",
            name="Attack_Unit",
            rendered_action="<Attack_Unit(0xaaa)>",
            requested_arguments=("0xaaa",),
            resolved_arguments=("0xaaa",),
        ),
        RoutedCommand(
            command_id="attack-beta",
            actor="Combat/Beta",
            team_name="Beta",
            name="Attack_Unit",
            rendered_action="<Attack_Unit(0xbbb)>",
            requested_arguments=("0xbbb",),
            resolved_arguments=("0xbbb",),
        ),
        step_id=77,
    )
    _register_bridge_route(broker, coordinator, route)

    beta = broker.claim_primitive(
        "Combat",
        "Beta",
        "Attack_Unit",
        "Attack_screen",
        final_primitive=True,
        ordinal=0,
        total=1,
    )
    alpha = broker.claim_primitive(
        "Combat",
        "Alpha",
        "Attack_Unit",
        "Attack_screen",
        final_primitive=True,
        ordinal=0,
        total=1,
    )

    assert beta is not None and beta.command_id == "attack-beta"
    assert alpha is not None and alpha.command_id == "attack-alpha"
    broker.settle_primitive(beta, success=True, game_loop=900)
    broker.settle_primitive(alpha, success=True, game_loop=900)

    reports = {report["command_id"]: report for report in runtime.execution_reports}
    assert reports["attack-alpha"]["actor"] == "Combat/Alpha"
    assert reports["attack-alpha"]["requested_arguments"] == ["0xaaa"]
    assert reports["attack-beta"]["actor"] == "Combat/Beta"
    assert reports["attack-beta"]["requested_arguments"] == ["0xbbb"]


def test_step_902_camera_only_nexus_is_cancelled_without_builder_disappeared() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route(
            "Builder",
            ("Builder-Probe-1",),
            RoutedCommand(
                command_id="command-camera-only",
                actor="Builder/Builder-Probe-1",
                team_name="Builder-Probe-1",
                name="Build_Nexus_Near",
                rendered_action="<Build_Nexus_Near(0x100)>",
                requested_arguments=("0x100",),
                resolved_arguments=("0x100",),
            ),
            step_id=902,
        ),
    )

    camera = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        "Build_Nexus_Near",
        "llm_pysc2_move_camera",
        final_primitive=True,
        origin="orchestration",
        requested_function_id=573,
        emitted_function_id=573,
    )

    assert camera is not None
    assert camera.final_primitive is False
    assert camera.requested_function_id == 573
    broker.settle_primitive(camera, success=True, game_loop=20811)
    broker.end_episode(_episode_result())

    assert len(runtime.execution_reports) == 1
    report = runtime.execution_reports[0]
    assert report["command_id"] == "command-camera-only"
    assert report["status"] == "cancelled"
    assert report["execution_stage"] == "episode_end"
    assert report["failure_code"] == "episode_ended"
    assert "builder disappeared" not in (report["failure_reason"] or "")
    assert report["primitive_trace"] == [
        {
            "function_name": "llm_pysc2_move_camera",
            "requested_function_id": 573,
            "emitted_function_id": 573,
            "origin": "orchestration",
            "ordinal": None,
            "total": None,
            "game_loop": 20811,
            "accepted": True,
            "failure_code": None,
            "detail": None,
        }
    ]


def test_worker_maps_next_action_result_to_pysc2_rejection_and_clears_chain() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route(
            "Builder",
            ("Builder-Probe-1",),
            RoutedCommand(
                command_id="command-chain",
                actor="Builder/Builder-Probe-1",
                team_name="Builder-Probe-1",
                name="Build_Pylon_Screen",
                rendered_action="<Build_Pylon_Screen([65,65])>",
                requested_arguments=([65, 65],),
                resolved_arguments=([65, 65],),
            ),
            step_id=44,
        ),
    )
    first = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        "Build_Pylon_Screen",
        "select_point",
        final_primitive=False,
        ordinal=0,
        total=2,
        requested_function_id=2,
        emitted_function_id=2,
    )
    assert first is not None
    upstream_agent = SimpleNamespace(func_list=[("remaining",)])
    main_agent = cast(Any, object.__new__(RTSCortexMainAgent))
    main_agent._pending_primitive = first
    main_agent._pending_primitive_agent = upstream_agent
    main_agent.decision_broker = broker
    observation = SimpleNamespace(
        observation=SimpleNamespace(game_loop=[1000], action_result=[1]),
    )

    main_agent._settle_previous_primitive(observation)

    assert upstream_agent.func_list == []
    assert main_agent._pending_primitive is None
    assert main_agent._pending_primitive_agent is None
    assert broker._active_commands == {}  # noqa: SLF001
    assert len(runtime.execution_reports) == 1
    report = runtime.execution_reports[0]
    assert report["command_id"] == "command-chain"
    assert report["status"] == "failed"
    assert report["execution_stage"] == "pysc2_acceptance"
    assert report["failure_code"] == "pysc2_rejected"
    assert report["primitive_trace"] == [
        {
            "function_name": "select_point",
            "requested_function_id": 2,
            "emitted_function_id": 2,
            "origin": "translator",
            "ordinal": 0,
            "total": 2,
            "game_loop": 1000,
            "accepted": False,
            "failure_code": "pysc2_rejected",
            "detail": "PySC2 action result 1",
        }
    ]


def test_worker_anchors_production_selection_barrier_to_acceptance_observation() -> None:
    calls: list[tuple[PrimitiveDispatch, bool, str | None, int]] = []

    class Broker:
        def settle_primitive(
            self,
            dispatch: PrimitiveDispatch,
            *,
            success: bool,
            failure_reason: str | None,
            game_loop: int,
        ) -> None:
            calls.append((dispatch, success, failure_reason, game_loop))

    dispatch = PrimitiveDispatch(
        command_id="command-techlab",
        function_name="select_point",
        final_primitive=False,
        ordinal=1,
        total=3,
        requested_function_id=2,
        emitted_function_id=2,
    )
    upstream_agent = cast(
        Any,
        SimpleNamespace(
            curr_action_name="Build_BarracksTechLab",
            func_list=[(94, object(), ("queued",))],
            _rtscortex_translation_ordinal=2,
            _rtscortex_production_selection_loop=3126,
            _rtscortex_active_build_route=None,
        ),
    )
    main_agent = cast(Any, object.__new__(RTSCortexMainAgent))
    main_agent._pending_primitive = dispatch
    main_agent._pending_primitive_agent = upstream_agent
    main_agent.decision_broker = Broker()
    timestep = SimpleNamespace(
        observation=SimpleNamespace(game_loop=[3127], action_result=[]),
    )

    main_agent._settle_previous_primitive(timestep)

    assert upstream_agent._rtscortex_production_selection_loop == 3127
    assert RTSCortexLLMAgent._wait_for_production_selection(
        upstream_agent,
        "Build_BarracksTechLab",
        timestep,
    ) is True
    assert calls == [(dispatch, True, None, 3127)]
    assert main_agent._pending_primitive is None
    assert main_agent._pending_primitive_agent is None


def test_broker_requires_a_contiguous_translator_sequence() -> None:
    runtime = FakeRuntime()
    coordinator = BridgeCoordinator(runtime)
    broker = SharedDecisionBroker(
        coordinator,
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    _register_bridge_route(
        broker,
        coordinator,
        _bridge_route(
            "Builder",
            ("Builder-Probe-1",),
            RoutedCommand(
                command_id="command-pylon",
                actor="Builder/Builder-Probe-1",
                team_name="Builder-Probe-1",
                name="Build_Pylon_Screen",
                rendered_action="<Build_Pylon_Screen([65,65])>",
                requested_arguments=([65, 65],),
                resolved_arguments=([65, 65],),
            ),
            step_id=44,
        ),
    )

    with pytest.raises(RuntimeError, match="must begin at ordinal 0"):
        broker.claim_primitive(
            "Builder",
            "Builder-Probe-1",
            "Build_Pylon_Screen",
            "Build_Pylon_screen",
            final_primitive=True,
            ordinal=1,
            total=2,
        )

    assert len(runtime.execution_reports) == 1
    report = runtime.execution_reports[0]
    assert report["command_id"] == "command-pylon"
    assert report["status"] == "failed"
    assert report["execution_stage"] == "translation"
    assert report["failure_code"] == "bridge_integrity_error"
    assert report["failure_reason"] == "translator sequence must begin at ordinal 0"
    assert report["primitive_trace"] == [
        {
            "function_name": "Build_Pylon_screen",
            "requested_function_id": None,
            "emitted_function_id": None,
            "origin": "translator",
            "ordinal": 1,
            "total": 2,
            "game_loop": None,
            "accepted": False,
            "failure_code": "bridge_integrity_error",
            "detail": "translator sequence must begin at ordinal 0",
        }
    ]
    assert broker._active_commands == {}  # noqa: SLF001
    assert not broker._command_queues  # noqa: SLF001
    broker.end_episode(_episode_result())
    assert len(runtime.execution_reports) == 1


@pytest.mark.parametrize(
    ("action_name", "function_name", "function_id", "command_id"),
    [
        ("Build_Nexus_Near", "Build_Nexus_screen", 65, "command-nexus"),
        (
            "Build_Assimilator_Near",
            "Build_Assimilator_screen",
            40,
            "command-assimilator",
        ),
    ],
)
def test_broker_treats_near_build_camera_settlement_noop_as_non_final(
    action_name: str,
    function_name: str,
    function_id: int,
    command_id: str,
) -> None:
    broker = SharedDecisionBroker(
        BridgeCoordinator(FakeRuntime()),
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    broker._command_queues[("Builder", "Builder-Probe-1", action_name)].append(  # noqa: SLF001
        command_id
    )

    camera = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        action_name,
        "llm_pysc2_move_camera",
        final_primitive=False,
        ordinal=0,
        total=3,
        requested_function_id=573,
        emitted_function_id=573,
    )
    settlement = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        action_name,
        "no_op",
        final_primitive=False,
        ordinal=1,
        total=3,
        requested_function_id=0,
        emitted_function_id=0,
    )
    final = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        action_name,
        function_name,
        final_primitive=True,
        ordinal=2,
        total=3,
        requested_function_id=function_id,
        emitted_function_id=function_id,
    )

    assert camera is not None and camera.final_primitive is False
    assert settlement is not None and settlement.final_primitive is False
    assert final is not None and final.final_primitive is True
    assert {camera.command_id, settlement.command_id, final.command_id} == {command_id}


def test_broker_attributes_pretranslator_orchestration_without_claiming_command() -> None:
    broker = SharedDecisionBroker(
        BridgeCoordinator(FakeRuntime()),
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    broker._command_queues[("Builder", "Builder-Probe-1", "Build_Pylon_Screen")].append(  # noqa: SLF001
        "command-pylon"
    )

    before_translation = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        "Build_Pylon_Screen",
        "move_camera",
        final_primitive=True,
        origin="orchestration",
    )
    assert before_translation is not None
    assert before_translation.command_id == "command-pylon"
    assert before_translation.final_primitive is False
    assert list(
        broker._command_queues[("Builder", "Builder-Probe-1", "Build_Pylon_Screen")]  # noqa: SLF001
    ) == ["command-pylon"]
    assert broker._active_commands == {}  # noqa: SLF001
    first = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        "Build_Pylon_Screen",
        "select_point",
        final_primitive=False,
        ordinal=0,
        total=2,
    )
    assert first is not None
    orchestration = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        "Build_Pylon_Screen",
        "move_camera",
        final_primitive=True,
        origin="orchestration",
    )
    assert orchestration is not None
    assert orchestration.command_id == "command-pylon"
    assert orchestration.final_primitive is False

    final = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        "Build_Pylon_Screen",
        "Build_Pylon_screen",
        final_primitive=True,
        ordinal=1,
        total=2,
        requested_function_id=65,
        emitted_function_id=65,
    )
    assert final is not None
    assert final.final_primitive is True


def test_broker_rejects_ambiguous_teamless_primitive_ownership() -> None:
    broker = SharedDecisionBroker(
        BridgeCoordinator(FakeRuntime()),
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    broker._command_queues[("Combat", "A", "Attack_Unit")].append("attack-a")  # noqa: SLF001
    broker._command_queues[("Combat", "B", "Attack_Unit")].append("attack-b")  # noqa: SLF001

    with pytest.raises(RuntimeError, match="ownership is ambiguous"):
        broker.claim_primitive(
            "Combat",
            None,
            "Attack_Unit",
            "Attack_screen",
            final_primitive=True,
            ordinal=0,
            total=1,
        )
    assert broker.metrics()["unattributed_primitives"] == 1

    with pytest.raises(RuntimeError, match="orchestration primitive ownership is ambiguous"):
        broker.claim_primitive(
            "Combat",
            None,
            "Attack_Unit",
            "move_camera",
            final_primitive=False,
            origin="orchestration",
        )
    assert broker.metrics()["unattributed_primitives"] == 2


def test_broker_forwards_raw_observations_for_deferred_effect_verification() -> None:
    coordinator = EffectRecordingCoordinator()
    broker = SharedDecisionBroker(
        cast(Any, coordinator),
        TimeStepExtractor("run-worker", "episode-worker"),
    )
    broker._command_queues[("Builder", "Builder-Probe-1", "Build_Pylon_Screen")].append(  # noqa: SLF001
        "command-pylon"
    )
    dispatch = broker.claim_primitive(
        "Builder",
        "Builder-Probe-1",
        "Build_Pylon_Screen",
        "Build_Pylon_screen",
        final_primitive=True,
        ordinal=0,
        total=1,
    )
    assert dispatch is not None
    observation = SimpleNamespace(game_loop=[225])

    broker.prepare_effect(dispatch, observation, builder_tag=0xABC)
    broker.settle_primitive(dispatch, success=True, game_loop=225)
    broker.observe_effects(observation)

    assert coordinator.calls == [
        ("prepare", "command-pylon", observation, 0xABC, None),
        ("primitive", "command-pylon", "Build_Pylon_screen", True),
        ("complete", "command-pylon", 225),
        ("observe", observation),
    ]


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


def test_worker_selects_rtscortex_melee_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMeleeConfig:
        def __init__(self) -> None:
            self.AGENTS: dict[str, Any] = {}
            self.reset_args: dict[str, Any] | None = None

        def reset_llm(self, **kwargs: Any) -> None:
            self.reset_args = kwargs

    config = FakeMeleeConfig()
    no_op_function = object()

    def fake_import(name: str) -> Any:
        if name == "rtscortex_llm_pysc2.melee":
            return SimpleNamespace(RTSCortexMeleeConfig=lambda: config)
        if name == "pysc2.lib.actions":
            return SimpleNamespace(FUNCTIONS=SimpleNamespace(no_op=no_op_function))
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    selected = _scenario_config("Simple64")

    assert selected is config
    assert config.reset_args == {
        "model_name": "gpt-3.5-turbo",
        "api_base": "http://127.0.0.1",
        "api_key": "rtscortex-unused",
    }


def test_worker_selects_rtscortex_terran_melee_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTerranMeleeConfig:
        def __init__(self) -> None:
            self.AGENTS: dict[str, Any] = {}
            self.reset_args: dict[str, Any] | None = None

        def reset_llm(self, **kwargs: Any) -> None:
            self.reset_args = kwargs

    config = FakeTerranMeleeConfig()
    no_op_function = object()

    def fake_import(name: str) -> Any:
        if name == "rtscortex_llm_pysc2.terran_melee":
            return SimpleNamespace(RTSCortexTerranMeleeConfig=lambda: config)
        if name == "pysc2.lib.actions":
            return SimpleNamespace(FUNCTIONS=SimpleNamespace(no_op=no_op_function))
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    selected = _scenario_config("Simple64", agent_race="terran")

    assert selected is config
    assert config.reset_args == {
        "model_name": "gpt-3.5-turbo",
        "api_base": "http://127.0.0.1",
        "api_key": "rtscortex-unused",
    }


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
            "protocol_version": "1.1",
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
                    "arguments": ["0x101480001"],
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


class BlockingRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def tick(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.entered.set()
        assert self.release.wait(timeout=1)
        return super().tick(observation)


class EffectRecordingCoordinator:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def prepare_effect(
        self,
        command_id: str,
        observation: Any,
        *,
        builder_tag: int | None,
        producer_tag: int | None = None,
    ) -> None:
        self.calls.append(("prepare", command_id, observation, builder_tag, producer_tag))

    def record_primitive(
        self,
        command_id: str,
        function_name: str,
        *,
        success: bool,
        failure_reason: str | None = None,
        **metadata: Any,
    ) -> None:
        del failure_reason, metadata
        self.calls.append(("primitive", command_id, function_name, success))

    def complete_command(
        self,
        command_id: str,
        *,
        game_loop: int | None = None,
    ) -> None:
        self.calls.append(("complete", command_id, game_loop))

    def observe_effects(self, observation: Any) -> list[dict[str, Any]]:
        self.calls.append(("observe", observation))
        return []


class UniformGrid:
    shape = (128, 128)

    def __init__(self, value: int) -> None:
        self.value = value

    def __getitem__(self, _index: int) -> UniformGrid:
        return self

    def __eq__(self, other: object) -> bool:
        return self.value == other


class Grid:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values
        self.shape = (len(values), len(values[0]))

    def __getitem__(self, index: int) -> list[int]:
        return self.values[index]


def _fake_timestep(*, alerts: list[int] | None = None) -> Any:
    available_actions = [0, 12]
    raw_units = [
        _unit(4300734465, 311, 1, 42, 38, 112, 204),
        _unit(4294967297, 59, 1, 30, 30, 1000, 255),
        _unit(4316463105, 104, 4, 52, 32, 30, 191),
    ]
    observation = SimpleNamespace(
        player=SimpleNamespace(
            minerals=375,
            vespene=100,
            food_used=34,
            food_cap=46,
            food_workers=20,
            food_army=14,
        ),
        raw_units=raw_units,
        feature_units=raw_units,
        production_queue=[SimpleNamespace(ability_id=141, build_progress=50)],
        upgrades=[84],
        alerts=[3] if alerts is None else alerts,
        available_actions=available_actions,
        game_loop=[224],
    )
    return SimpleNamespace(observation=observation, reward=0, last=lambda: False)


def _developer_agent(timestep: Any, actions: list[dict[str, Any]]) -> Any:
    return SimpleNamespace(
        name="Developer",
        flag_enable_empty_unit_group=True,
        team_unit_team_list=[],
        team_unit_obs_list=[timestep],
        config=SimpleNamespace(
            AGENTS={
                "Developer": {
                    "team": [{"name": "Empty", "unit_type": []}],
                    "action": {"EmptyGroup": actions},
                }
            }
        ),
    )


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
        "protocol_version": "1.1",
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


def _bridge_route(
    agent_name: str,
    team_order: tuple[str, ...],
    *commands: RoutedCommand,
    step_id: int,
) -> RoutedActionBatch:
    return RoutedActionBatch(
        protocol_version="1.1",
        run_id="run-worker",
        episode_id="episode-worker",
        step_id=step_id,
        decision_id=f"decision-{step_id}-{agent_name}",
        agent_name=agent_name,
        team_order=team_order,
        commands=commands,
        action_text="Actions:",
    )


def _register_bridge_route(
    broker: SharedDecisionBroker,
    coordinator: BridgeCoordinator,
    route: RoutedActionBatch,
) -> None:
    coordinator.tracker.register(route)
    for command in route.commands:
        coordinator.effect_verifier.track(command)
        broker._command_queues[(route.agent_name, command.team_name, command.name)].append(  # noqa: SLF001
            command.command_id
        )
