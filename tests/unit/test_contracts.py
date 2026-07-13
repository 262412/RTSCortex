from __future__ import annotations

import pytest
from pydantic import ValidationError

from rtscortex.contracts import (
    ActionArgumentType,
    ActionBatch,
    ActionCommand,
    ActionSource,
    AvailableAction,
    EpisodeOutcome,
    EpisodeResult,
    ExecutionReport,
    ObservationEnvelope,
    UnitState,
)
from tests.helpers import make_observation


def test_observation_round_trip_is_lossless() -> None:
    observation = make_observation(alerts=["under_attack"])
    restored = ObservationEnvelope.model_validate_json(observation.model_dump_json())
    assert restored == observation


def test_contract_rejects_unknown_protocol_version() -> None:
    payload = make_observation().model_dump(mode="json")
    payload["protocol_version"] = "2.0"
    with pytest.raises(ValidationError):
        ObservationEnvelope.model_validate(payload)


def test_unit_health_is_bounded() -> None:
    with pytest.raises(ValidationError):
        UnitState(
            unit_id="bad",
            unit_type="Adept",
            alliance="self",
            health_fraction=1.1,
        )


def test_available_action_rejects_mismatched_argument_schema() -> None:
    with pytest.raises(ValidationError):
        AvailableAction(
            name="Move",
            argument_names=["position"],
            argument_types=[ActionArgumentType.POSITION, ActionArgumentType.BOOLEAN],
        )


def test_action_contract_round_trip_and_version_check() -> None:
    command = ActionCommand(
        command_id="command-1",
        actor="CombatGroup7/Adept-1",
        name="Attack_Unit",
        arguments=["0x1001"],
        created_game_loop=32,
        source=ActionSource.PLANNER,
    )
    batch = ActionBatch(
        run_id="run-1",
        episode_id="episode-1",
        step_id=2,
        decision_id="decision-1",
        commands=[command],
    )

    assert ActionBatch.model_validate_json(batch.model_dump_json()) == batch
    payload = batch.model_dump(mode="json")
    payload["protocol_version"] = "2.0"
    with pytest.raises(ValidationError):
        ActionBatch.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("priority", 101), ("ttl_game_loops", 0), ("created_game_loop", -1)],
)
def test_action_command_rejects_invalid_limits(field: str, value: int) -> None:
    payload = {
        "command_id": "command-1",
        "actor": "CombatGroup7/Adept-1",
        "name": "No_Operation",
        "created_game_loop": 0,
        "source": "fallback",
        field: value,
    }

    with pytest.raises(ValidationError):
        ActionCommand.model_validate(payload)


def test_execution_contract_round_trip_and_invalid_samples() -> None:
    report = ExecutionReport(
        run_id="run-1",
        episode_id="episode-1",
        step_id=2,
        command_id="command-1",
        success=True,
        pysc2_function="Attack_screen",
        latency_ms=2.5,
    )

    assert ExecutionReport.model_validate_json(report.model_dump_json()) == report
    for invalid in (
        {**report.model_dump(mode="json"), "protocol_version": "2.0"},
        {**report.model_dump(mode="json"), "latency_ms": -0.1},
    ):
        with pytest.raises(ValidationError):
            ExecutionReport.model_validate(invalid)


def test_episode_result_contract_round_trip_and_invalid_samples() -> None:
    result = EpisodeResult(
        run_id="run-1",
        episode_id="episode-1",
        scenario="pvz_task1_level1",
        seed=7,
        outcome=EpisodeOutcome.VICTORY,
        steps=10,
    )

    assert EpisodeResult.model_validate_json(result.model_dump_json()) == result
    for invalid in (
        {**result.model_dump(mode="json"), "protocol_version": "2.0"},
        {**result.model_dump(mode="json"), "steps": -1},
        {**result.model_dump(mode="json"), "unexpected": True},
    ):
        with pytest.raises(ValidationError):
            EpisodeResult.model_validate(invalid)
