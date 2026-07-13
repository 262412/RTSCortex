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
    assert "<Attack_Unit(0x100580001)>" in route.action_text


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
        "command-adept-attack", "llm_pysc2_move_camera", success=True, latency_ms=2.5
    )
    tracker.record_primitive("command-adept-attack", "Attack_screen", success=True, latency_ms=4.0)

    report = tracker.complete("command-adept-attack")

    validated = ExecutionReport.model_validate(report)
    assert validated.success is True
    assert validated.pysc2_function == "llm_pysc2_move_camera -> Attack_screen"
    assert validated.latency_ms == 6.5


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
    assert decision.observation == load_fixture("observation_envelope.json")
    assert (
        decision.action_text("CombatGroup7") == load_fixture("routed_actions.json")["action_text"]
    )
    assert report["success"] is False
    assert runtime.execution_reports == [report]


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
