from __future__ import annotations

import pytest
from pydantic import ValidationError

from rtscortex.contracts import (
    ActionArgumentType,
    ActionBatch,
    ActionCommand,
    ActionSource,
    AvailableAction,
    EffectEvidence,
    EpisodeOutcome,
    EpisodeResult,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
    IdleReason,
    ObservationEnvelope,
    PrimitiveTraceEntry,
    UnitState,
)
from tests.helpers import make_observation


def test_observation_round_trip_is_lossless() -> None:
    observation = make_observation(alerts=["under_attack"])
    restored = ObservationEnvelope.model_validate_json(observation.model_dump_json())
    assert restored == observation
    assert restored.protocol_version == "1.1"


def test_v1_contracts_read_legacy_payloads_but_write_v1_1_by_default() -> None:
    legacy = make_observation().model_dump(mode="json")
    legacy["protocol_version"] = "1.0"

    restored = ObservationEnvelope.model_validate(legacy)

    assert restored.protocol_version == "1.0"
    assert make_observation().protocol_version == "1.1"


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
    with pytest.raises(ValidationError):
        AvailableAction(name="Move", argument_names=["position"])


def test_v11_requires_complete_candidates_for_all_tag_and_position_actions() -> None:
    payload = make_observation().model_dump(mode="json")
    payload["available_actions"].append(
        {
            "name": "Move_Screen",
            "argument_names": ["screen"],
            "argument_types": ["position"],
            "actor_scopes": ["army"],
            "argument_candidates": None,
        }
    )

    with pytest.raises(ValidationError, match="tag and position actions"):
        ObservationEnvelope.model_validate(payload)

    payload["protocol_version"] = "1.0"
    assert ObservationEnvelope.model_validate(payload).protocol_version == "1.0"


def test_available_action_normalizes_complete_argument_candidates() -> None:
    action = AvailableAction(
        name="Attack_Unit",
        argument_names=["tag"],
        argument_types=[ActionArgumentType.TAG],
        argument_candidates=[[0xABC], ["0xDEF"]],
    )
    placement = AvailableAction(
        name="Build_Pylon_Screen",
        argument_names=["screen"],
        argument_types=[ActionArgumentType.POSITION],
        argument_candidates=[[[65, 90]], [(70, 90)]],
    )

    assert action.argument_candidates == [["0xabc"], ["0xdef"]]
    assert placement.argument_candidates == [[[65, 90]], [[70, 90]]]


@pytest.mark.parametrize(
    "candidates",
    [
        [["tag"]],
        [[-1]],
        [[True]],
        ["0xabc"],
        [["0xabc", "extra"]],
    ],
)
def test_available_action_rejects_invalid_argument_candidates(
    candidates: list[object],
) -> None:
    with pytest.raises(ValidationError):
        AvailableAction.model_validate(
            {
                "name": "Attack_Unit",
                "argument_names": ["tag"],
                "argument_types": [ActionArgumentType.TAG],
                "argument_candidates": candidates,
            }
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
    assert batch.planner_pending is False
    assert batch.idle_reason is None
    payload = batch.model_dump(mode="json")
    payload["protocol_version"] = "2.0"
    with pytest.raises(ValidationError):
        ActionBatch.model_validate(payload)


def test_action_batch_has_strong_idle_reason() -> None:
    batch = ActionBatch(
        run_id="run-1",
        episode_id="episode-1",
        step_id=2,
        decision_id="decision-1",
        idle_reason=IdleReason.WAITING_FOR_PLANNER,
    )

    assert batch.idle_reason is IdleReason.WAITING_FOR_PLANNER
    with pytest.raises(ValidationError):
        ActionBatch.model_validate(
            {**batch.model_dump(mode="json"), "idle_reason": "unknown_idle_reason"}
        )


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
        action_name="Attack_Unit",
        actor="CombatGroup7/Adept-1",
        source=ActionSource.PLANNER,
        requested_arguments=["0x1"],
        resolved_arguments=["0x1"],
        status=ExecutionStatus.SUCCEEDED,
        execution_stage=ExecutionStage.PYSC2_ACCEPTANCE,
        pysc2_function="Attack_screen",
        latency_ms=2.5,
    )

    assert ExecutionReport.model_validate_json(report.model_dump_json()) == report
    assert report.status is ExecutionStatus.SUCCEEDED
    for invalid in (
        {**report.model_dump(mode="json"), "protocol_version": "2.0"},
        {**report.model_dump(mode="json"), "latency_ms": -0.1},
    ):
        with pytest.raises(ValidationError):
            ExecutionReport.model_validate(invalid)


def test_execution_report_v1_1_fields_and_legacy_defaults() -> None:
    legacy = ExecutionReport.model_validate(
        {
            "protocol_version": "1.0",
            "run_id": "run-1",
            "episode_id": "episode-1",
            "step_id": 2,
            "command_id": "command-1",
            "success": False,
            "failure_reason": "legacy failure",
        }
    )
    current = ExecutionReport(
        run_id="run-1",
        episode_id="episode-1",
        step_id=3,
        command_id="command-2",
        success=False,
        action_name="Build_Pylon_Screen",
        actor="Builder/Probe-1",
        source=ActionSource.PLANNER,
        requested_arguments=[[65, 90]],
        resolved_arguments=[[66, 90]],
        status=ExecutionStatus.UNCONFIRMED,
        execution_stage=ExecutionStage.EPISODE_END,
        failure_code="effect_timeout",
        primitive_trace=[
            PrimitiveTraceEntry.model_validate(
                {
                    "function_name": "Build_Pylon_screen",
                    "origin": "translator",
                    "ordinal": 1,
                    "total": 2,
                    "game_loop": 80,
                    "accepted": True,
                    "detail": "accepted by PySC2",
                }
            )
        ],
        effect_evidence=EffectEvidence.model_validate(
            {
                "target_type": "Pylon",
                "target_position": [66, 90],
                "builder_tag": "0xabc",
                "baseline_structure_tags": ["0x1"],
                "observed_structure_tag": "0x2",
                "dispatched_loop": 80,
                "accepted_loop": 81,
                "confirmed_loop": 92,
                "order_seen": True,
                "order_last_seen_game_loop": 220,
                "post_order_grace_game_loops": 32,
                "mineral_delta": 100,
                "elapsed_game_loops": 12,
                "base_timeout_game_loops": 112,
                "effective_timeout_game_loops": 448,
                "active_order_extension": True,
            }
        ),
    )

    assert legacy.status is ExecutionStatus.FAILED
    assert legacy.execution_stage is None
    assert current.protocol_version == "1.1"
    assert current.primitive_trace[0].origin.value == "translator"
    assert current.primitive_trace[0].function == "Build_Pylon_screen"
    assert current.primitive_trace[0].raw_reason == "accepted by PySC2"
    assert current.effect_evidence is not None
    assert current.effect_evidence.target_position == (66.0, 90.0)
    assert current.effect_evidence.new_structure_tag == "0x2"
    assert current.effect_evidence.dispatch_game_loop == 80
    assert current.effect_evidence.elapsed_game_loops == 12
    assert current.effect_evidence.effective_timeout_game_loops == 448
    assert current.effect_evidence.active_order_extension is True
    assert current.effect_evidence.order_last_seen_game_loop == 220
    assert current.effect_evidence.post_order_grace_game_loops == 32


def test_effect_evidence_accepts_move_start_provenance() -> None:
    evidence = EffectEvidence(
        target_type="Move_Minimap",
        target_position=(16, 21),
        builder_tag="0xabc",
        baseline_builder_position=(51, 69),
        observed_builder_position=(50, 69),
        builder_displacement=1.0,
        move_order_seen=True,
    )

    assert EffectEvidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_effect_evidence_round_trips_production_provenance_without_breaking_legacy() -> None:
    evidence = EffectEvidence(
        effect_kind="production",
        producer_tag="0xabc",
        producer_type="Stargate",
        producer_observed_type="Stargate",
        expected_unit_type="VoidRay",
        expected_order_id=57,
        baseline_unit_tags=["0x1"],
        new_unit_tag="0x2",
        baseline_producer_orders=[16],
        producer_orders=[57],
        production_order_seen=True,
        confirmation_kind="producer_order",
        accepted_game_loop=100,
        confirmed_game_loop=108,
    )

    assert EffectEvidence.model_validate_json(evidence.model_dump_json()) == evidence
    legacy = EffectEvidence(target_type="Pylon", new_structure_tag="0x10")
    assert legacy.effect_kind is None
    assert legacy.baseline_unit_tags == []
    assert legacy.production_order_seen is False
    assert legacy.producer_consumed is False
    assert legacy.confirmation_kind is None

    with pytest.raises(ValidationError):
        EffectEvidence(effect_kind="accepted")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EffectEvidence(confirmation_kind="acceptance_only")  # type: ignore[arg-type]

    zerg = EffectEvidence(
        effect_kind="production",
        requested_producer_tag="0x200",
        producer_tag="0x100",
        producer_type="Larva",
        producer_observed_type="Cocoon",
        producer_consumed=True,
        expected_unit_type="Zergling",
        expected_order_id=528,
        confirmation_kind="producer_morph",
    )
    assert EffectEvidence.model_validate_json(zerg.model_dump_json()) == zerg


def test_effect_evidence_accepts_terran_addon_confirmation() -> None:
    evidence = EffectEvidence(
        effect_kind="addon",
        target_type="BarracksReactor",
        producer_tag="0xabc",
        producer_type="Barracks",
        expected_order_id=208,
        new_structure_tag="0xdef",
        confirmation_kind="new_structure",
    )

    assert EffectEvidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_effect_evidence_accepts_zerg_structure_morph_confirmation() -> None:
    evidence = EffectEvidence(
        effect_kind="morph",
        target_type="Lair",
        target_tag="0xabc",
        producer_tag="0xabc",
        producer_type="Hatchery",
        producer_observed_type="Lair",
        expected_order_id=388,
        confirmation_kind="source_morph",
        source_build_progress=0.1,
    )

    assert EffectEvidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_effect_evidence_accepts_zerg_larva_inject_confirmation() -> None:
    evidence = EffectEvidence(
        effect_kind="inject",
        target_type="Hatchery",
        target_tag="0xabc",
        builder_tag="0xdef",
        producer_tag="0xdef",
        producer_type="Queen",
        expected_order_id=315,
        baseline_target_buff_ids=[],
        target_buff_ids=[11],
        confirmation_kind="target_buff",
    )

    assert EffectEvidence.model_validate_json(evidence.model_dump_json()) == evidence


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
