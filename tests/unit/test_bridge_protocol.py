from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from rtscortex_llm_pysc2.coordinator import BridgeCoordinator
from rtscortex_llm_pysc2.execution import ExecutionTracker
from rtscortex_llm_pysc2.observation import ObservationMapper, canonical_actor, split_actor
from rtscortex_llm_pysc2.routing import ActionRouter, RoutedActionBatch

from rtscortex.contracts import ActionBatch, ExecutionReport, ObservationEnvelope
from rtscortex.runtime.validation import ActionValidator

FIXTURES = Path(__file__).parents[1] / "fixtures" / "llm_pysc2"
TEAM_ORDER = ["Adept-1", "AdeptPhase-1", "DarkTemplar-1"]


def load_fixture(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((FIXTURES / name).read_text()))


def test_observation_mapper_matches_versioned_fixture() -> None:
    envelope = ObservationMapper().map(load_fixture("observation_snapshot.json"))

    assert envelope == load_fixture("observation_envelope.json")
    validated = ObservationEnvelope.model_validate(envelope)
    assert validated.state.economy.army_supply == 14
    assert validated.available_actions[0].name == "No_Operation"
    assert validated.available_actions[0].actor_scopes == [
        "CombatGroup7/Adept-1",
        "CombatGroup7/AdeptPhase-1",
        "CombatGroup7/DarkTemplar-1",
    ]
    assert validated.available_actions[1].argument_types == ["tag"]


def test_actor_scope_is_canonical_agent_and_team() -> None:
    assert canonical_actor("CombatGroup7", "Adept-1") == "CombatGroup7/Adept-1"
    assert split_actor("CombatGroup7/Adept-1") == ("CombatGroup7", "Adept-1")

    with pytest.raises(ValueError, match="expected 'agent/team'"):
        split_actor("Adept-1")


def test_router_matches_ordered_upstream_team_fixture() -> None:
    observation = load_fixture("observation_envelope.json")
    batch = load_fixture("action_batch.json")
    validated_observation = ObservationEnvelope.model_validate(observation)
    validated_batch = ActionBatch.model_validate(batch)

    validation = ActionValidator(max_actions=3).validate(
        validated_batch.commands, validated_observation
    )
    assert validation.rejected == []

    route = ActionRouter().route(
        batch,
        agent_name="CombatGroup7",
        team_order=TEAM_ORDER,
        available_actions=observation["available_actions"],
    )

    assert route.to_dict() == load_fixture("routed_actions.json")
    assert route.commands[0].team_name == "Adept-1"
    assert "Team AdeptPhase-1:\n        <No_Operation()>" in route.action_text
    assert "<Attack_Unit(0x101480001)>" in route.action_text


def test_router_rejects_team_missing_from_current_positional_order() -> None:
    observation = load_fixture("observation_envelope.json")
    batch = load_fixture("action_batch.json")

    with pytest.raises(ValueError, match="absent from the current team order"):
        ActionRouter().route(
            batch,
            agent_name="CombatGroup7",
            team_order=["Adept-1", "AdeptPhase-1"],
            available_actions=observation["available_actions"],
        )


def test_execution_tracker_aggregates_multiple_pysc2_primitives() -> None:
    route = _fixture_route()
    tracker = ExecutionTracker()
    tracker.register(route)
    tracker.record_primitive(
        "command-adept-attack",
        "llm_pysc2_move_camera",
        success=False,
        latency_ms=2.5,
        failure_reason="camera retry",
        origin="orchestration",
    )
    tracker.record_primitive(
        "command-adept-attack",
        "Attack_screen",
        success=True,
        latency_ms=4.0,
        origin="translator",
        ordinal=0,
        total=1,
        requested_function_id=12,
        emitted_function_id=12,
    )

    report = tracker.complete("command-adept-attack")

    validated = ExecutionReport.model_validate(report)
    assert validated.success is True
    assert validated.pysc2_function == "llm_pysc2_move_camera -> Attack_screen"
    assert validated.latency_ms == 6.5
    assert validated.primitive_trace[0].origin.value == "orchestration"
    assert validated.primitive_trace[1].requested_function_id == 12
    assert validated.primitive_trace[1].emitted_function_id == 12


def test_coordinator_calls_runtime_once_and_reports_execution() -> None:
    runtime = FakeRuntime(load_fixture("action_batch.json"))
    coordinator = BridgeCoordinator(runtime)

    decision = coordinator.decide(
        load_fixture("observation_snapshot.json"), {"CombatGroup7": TEAM_ORDER}
    )
    coordinator.record_primitive(
        "command-dark-attack", "Attack_screen", success=False, failure_reason="target hidden"
    )
    report = coordinator.complete_command("command-dark-attack")

    assert runtime.tick_calls == 1
    assert report is not None
    assert decision.observation == load_fixture("observation_envelope.json")
    assert (
        decision.action_text("CombatGroup7") == load_fixture("routed_actions.json")["action_text"]
    )
    assert report["success"] is False
    assert runtime.execution_reports == [report]


def test_coordinator_defers_build_report_until_raw_state_confirms_effect() -> None:
    snapshot = _build_snapshot()
    runtime = FakeRuntime(_build_batch())
    coordinator = BridgeCoordinator(runtime)

    coordinator.decide(snapshot, {"Builder": ["Builder-Probe-1"]})
    coordinator.prepare_effect(
        "command-pylon",
        _raw_effect_observation(game_loop=224, minerals=250),
        builder_tag=0xABC,
    )
    coordinator.record_primitive(
        "command-pylon", "Build_Pylon_screen", success=True, latency_ms=4.0
    )

    report = coordinator.complete_command("command-pylon", game_loop=225)

    assert report is None
    assert runtime.execution_reports == []

    reports = coordinator.observe_effects(
        _raw_effect_observation(
            game_loop=246,
            minerals=150,
            structures=["Nexus", "Pylon"],
        )
    )

    assert len(reports) == 1
    assert reports[0]["success"] is True
    assert reports[0]["pysc2_function"] == "Build_Pylon_screen"
    assert runtime.execution_reports == reports
    assert coordinator.observe_effects(_raw_effect_observation(game_loop=268, minerals=175)) == []


def test_coordinator_does_not_report_one_structure_for_two_commands() -> None:
    runtime = FakeRuntime(_concurrent_build_batch())
    coordinator = BridgeCoordinator(runtime)
    coordinator.decide(
        _concurrent_build_snapshot(),
        {"Builder": ["Builder-Probe-1", "Builder-Probe-2"]},
    )
    baseline = _raw_effect_observation(game_loop=224, minerals=400)
    for command_id in ("command-pylon-a", "command-pylon-b"):
        coordinator.prepare_effect(command_id, baseline, builder_tag=0xABC)
        coordinator.record_primitive(command_id, "Build_Pylon_screen", success=True)
        assert coordinator.complete_command(command_id, game_loop=225) is None

    one_new_pylon = _raw_effect_observation(
        game_loop=246,
        minerals=300,
        structures=["Nexus", "Pylon"],
    )

    reports = coordinator.observe_effects(one_new_pylon)

    assert len(reports) == 1
    assert reports[0]["command_id"] == "command-pylon-a"
    assert reports[0]["success"] is True
    assert coordinator.observe_effects(one_new_pylon) == []
    assert len(runtime.execution_reports) == 1

    timeout_reports = coordinator.observe_effects(
        _raw_effect_observation(
            game_loop=337,
            minerals=300,
            structures=["Nexus", "Pylon"],
        )
    )

    assert len(timeout_reports) == 1
    assert timeout_reports[0]["command_id"] == "command-pylon-b"
    assert timeout_reports[0]["success"] is False
    assert timeout_reports[0]["effect_evidence"]["observed_structure_tag"] is None
    assert len(runtime.execution_reports) == 2


def test_coordinator_reports_deferred_build_once_at_episode_end() -> None:
    runtime = FakeRuntime(_build_batch())
    coordinator = BridgeCoordinator(runtime)
    coordinator.decide(_build_snapshot(), {"Builder": ["Builder-Probe-1"]})
    coordinator.prepare_effect(
        "command-pylon",
        _raw_effect_observation(game_loop=224, minerals=250),
        builder_tag=0xABC,
    )
    coordinator.record_primitive("command-pylon", "Build_Pylon_screen", success=True)
    assert coordinator.complete_command("command-pylon", game_loop=225) is None

    coordinator.end_episode({"outcome": "draw"})

    assert len(runtime.execution_reports) == 1
    assert runtime.execution_reports[0]["command_id"] == "command-pylon"
    assert runtime.execution_reports[0]["success"] is False
    assert runtime.execution_reports[0]["failure_reason"].startswith(
        "episode ended before gameplay effect was confirmed"
    )


def test_coordinator_defers_move_minimap_until_builder_starts_moving() -> None:
    runtime = FakeRuntime(_move_batch())
    coordinator = BridgeCoordinator(runtime)
    coordinator.decide(_move_snapshot(), {"Builder": ["Builder-Probe-1"]})
    coordinator.prepare_effect(
        "command-move",
        _raw_move_observation(game_loop=224, center=(8, 8)),
        builder_tag=0xABC,
    )
    coordinator.record_primitive("command-move", "Move_minimap", success=True)

    assert coordinator.complete_command("command-move", game_loop=225) is None
    assert coordinator.effect_verifier.blocks_auto_worker_management is False
    assert runtime.execution_reports == []
    assert coordinator.observe_effects(_raw_move_observation(game_loop=300, center=(48, 48))) == []

    reports = coordinator.observe_effects(
        _raw_move_observation(game_loop=320, center=(8, 8), builder_position=(31.5, 30))
    )

    assert len(reports) == 1
    report = ExecutionReport.model_validate(reports[0])
    assert report.status.value == "succeeded"
    assert report.execution_stage is not None
    assert report.execution_stage.value == "effect_verification"
    assert report.effect_evidence is not None
    assert report.effect_evidence.target_type == "Move_Minimap"
    assert report.effect_evidence.target_position == (48.0, 48.0)
    assert report.effect_evidence.confirmed_game_loop == 320
    assert report.effect_evidence.baseline_builder_position == (30.0, 30.0)
    assert report.effect_evidence.observed_builder_position == (31.5, 30.0)
    assert report.effect_evidence.builder_displacement == 1.5


def test_coordinator_reports_in_transit_move_as_unconfirmed_at_episode_end() -> None:
    runtime = FakeRuntime(_move_batch())
    coordinator = BridgeCoordinator(runtime)
    coordinator.decide(_move_snapshot(), {"Builder": ["Builder-Probe-1"]})
    coordinator.prepare_effect(
        "command-move",
        _raw_move_observation(game_loop=224, center=(8, 8)),
        builder_tag=0xABC,
    )
    coordinator.record_primitive("command-move", "Move_minimap", success=True)
    assert coordinator.complete_command("command-move", game_loop=225) is None

    coordinator.end_episode({"outcome": "draw"})

    assert len(runtime.execution_reports) == 1
    report = ExecutionReport.model_validate(runtime.execution_reports[0])
    assert report.status.value == "unconfirmed"
    assert report.execution_stage is not None
    assert report.execution_stage.value == "episode_end"
    assert report.failure_code == "episode_ended_unconfirmed"


def test_coordinator_rejects_duplicate_dispatch_from_runtime() -> None:
    runtime = FakeRuntime(load_fixture("action_batch.json"))
    coordinator = BridgeCoordinator(runtime)
    snapshot = load_fixture("observation_snapshot.json")

    first = coordinator.decide(snapshot, {"CombatGroup7": TEAM_ORDER})
    assert len(first.routes["CombatGroup7"].commands) == 3
    with pytest.raises(RuntimeError, match="duplicate command dispatch invariant"):
        coordinator.decide(snapshot, {"CombatGroup7": TEAM_ORDER})


def test_coordinator_reports_pending_commands_before_episode_end() -> None:
    runtime = FakeRuntime(load_fixture("action_batch.json"))
    coordinator = BridgeCoordinator(runtime)
    coordinator.decide(load_fixture("observation_snapshot.json"), {"CombatGroup7": TEAM_ORDER})

    coordinator.end_episode({"outcome": "defeat"})

    assert len(runtime.execution_reports) == 3
    assert all(report["success"] is False for report in runtime.execution_reports)
    assert all(
        report["failure_reason"] == "episode ended before command completion"
        for report in runtime.execution_reports
    )
    assert runtime.episode_results == [{"outcome": "defeat"}]


def _fixture_route() -> RoutedActionBatch:
    observation = load_fixture("observation_envelope.json")
    return ActionRouter().route(
        load_fixture("action_batch.json"),
        agent_name="CombatGroup7",
        team_order=TEAM_ORDER,
        available_actions=observation["available_actions"],
    )


def _build_snapshot() -> dict[str, Any]:
    snapshot = load_fixture("observation_snapshot.json")
    snapshot["teams"] = [
        {
            "agent_name": "Builder",
            "team_name": "Builder-Probe-1",
            "available_actions": [
                {
                    "name": "Build_Pylon_Screen",
                    "argument_names": ["screen"],
                    "argument_types": ["position"],
                    "argument_candidates": [[[65, 65]]],
                }
            ],
        }
    ]
    return snapshot


def _move_snapshot() -> dict[str, Any]:
    snapshot = load_fixture("observation_snapshot.json")
    snapshot["teams"] = [
        {
            "agent_name": "Builder",
            "team_name": "Builder-Probe-1",
            "available_actions": [
                {
                    "name": "Move_Minimap",
                    "argument_names": ["minimap"],
                    "argument_types": ["position"],
                    "argument_candidates": [[[48, 48]]],
                }
            ],
        }
    ]
    return snapshot


def _concurrent_build_snapshot() -> dict[str, Any]:
    snapshot = _build_snapshot()
    action = snapshot["teams"][0]["available_actions"][0]
    snapshot["teams"] = [
        {
            "agent_name": "Builder",
            "team_name": team_name,
            "available_actions": [action],
        }
        for team_name in ("Builder-Probe-1", "Builder-Probe-2")
    ]
    return snapshot


def _build_batch() -> dict[str, Any]:
    return {
        "protocol_version": "1.1",
        "run_id": "run-fixture",
        "episode_id": "episode-pvz-task1",
        "step_id": 7,
        "decision_id": "decision-build",
        "strategic_goal": "Build supply",
        "summary": "Build one Pylon",
        "planner_pending": False,
        "idle_reason": None,
        "commands": [
            {
                "command_id": "command-pylon",
                "actor": "Builder/Builder-Probe-1",
                "name": "Build_Pylon_Screen",
                "arguments": [[65, 65]],
                "priority": 50,
                "ttl_game_loops": 32,
                "created_game_loop": 224,
                "source": "planner",
                "preconditions": {},
            }
        ],
        "rejected_commands": [],
    }


def _move_batch() -> dict[str, Any]:
    return {
        "protocol_version": "1.1",
        "run_id": "run-fixture",
        "episode_id": "episode-pvz-task1",
        "step_id": 7,
        "decision_id": "decision-move",
        "strategic_goal": "Scout expansion",
        "summary": "Move the Builder to a remote resource cluster",
        "planner_pending": False,
        "idle_reason": None,
        "commands": [
            {
                "command_id": "command-move",
                "actor": "Builder/Builder-Probe-1",
                "name": "Move_Minimap",
                "arguments": [[48, 48]],
                "priority": 50,
                "ttl_game_loops": 112,
                "created_game_loop": 224,
                "source": "planner",
                "preconditions": {},
            }
        ],
        "rejected_commands": [],
    }


def _concurrent_build_batch() -> dict[str, Any]:
    batch = _build_batch()
    template = batch["commands"][0]
    batch["commands"] = [
        {
            **template,
            "command_id": command_id,
            "actor": f"Builder/{team_name}",
        }
        for command_id, team_name in (
            ("command-pylon-a", "Builder-Probe-1"),
            ("command-pylon-b", "Builder-Probe-2"),
        )
    ]
    return batch


def _raw_effect_observation(
    *,
    game_loop: int,
    minerals: int,
    structures: list[str] | None = None,
) -> dict[str, Any]:
    raw_units = [
        {
            "tag": 0xABC,
            "unit_type": "Probe",
            "alliance": 1,
            "order_length": 1,
            "order_id_0": 295,
            "is_selected": True,
            "build_progress": 100,
            "x": 30,
            "y": 30,
        }
    ]
    raw_units.extend(
        {
            "tag": index + 1,
            "unit_type": name,
            "alliance": 1,
            "order_length": 0,
            "is_selected": False,
            "build_progress": 50 if name == "Pylon" else 100,
            "x": 31.875 if name == "Pylon" else 25,
            "y": 30 if name == "Pylon" else 25,
        }
        for index, name in enumerate(structures or ["Nexus"])
    )
    return {
        "game_loop": [game_loop],
        "player_common": {"minerals": minerals},
        "raw_units": raw_units,
        "feature_units": [
            {"tag": 0xABC, "x": 55, "y": 65, "is_on_screen": True},
            *[
                {
                    "tag": unit["tag"],
                    "x": 65 if unit["unit_type"] == "Pylon" else 30,
                    "y": 65 if unit["unit_type"] == "Pylon" else 30,
                    "is_on_screen": True,
                }
                for unit in raw_units
                if unit["unit_type"] != "Probe"
            ],
        ],
    }


def _raw_move_observation(
    *,
    game_loop: int,
    center: tuple[int, int],
    builder_position: tuple[float, float] = (30, 30),
    builder_orders: list[int] | None = None,
) -> dict[str, Any]:
    observation = _raw_effect_observation(game_loop=game_loop, minerals=250)
    builder = observation["raw_units"][0]
    builder["x"], builder["y"] = builder_position
    orders = [295] if builder_orders is None else builder_orders
    builder["order_length"] = len(orders)
    for index in range(4):
        builder.pop(f"order_id_{index}", None)
    for index, order in enumerate(orders):
        builder[f"order_id_{index}"] = order
    camera = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(center[1] - 1, center[1] + 2):
        for x in range(center[0] - 1, center[0] + 2):
            camera[y][x] = 1
    observation["feature_minimap"] = {"camera": camera}
    return observation


class FakeRuntime:
    def __init__(self, batch: dict[str, Any]) -> None:
        self.batch = batch
        self.tick_calls = 0
        self.execution_reports: list[dict[str, Any]] = []
        self.episode_results: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def tick(self, observation: dict[str, Any]) -> dict[str, Any]:
        ObservationEnvelope.model_validate(observation)
        self.tick_calls += 1
        return self.batch

    def execution(self, report: dict[str, Any]) -> None:
        self.execution_reports.append(report)

    def end_episode(self, result: dict[str, Any]) -> None:
        self.episode_results.append(result)
