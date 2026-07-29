from __future__ import annotations

from pathlib import Path

from rtscortex.evaluation.engineering import (
    REQUIRED_ENGINEERING_GATES,
    EngineeringAccumulator,
    build_engineering_gate_report,
)
from rtscortex.memory import StoredEvent


def _event(
    event_id: int,
    event_type: str,
    payload: dict[str, object],
) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="run",
        episode_id="episode",
        step_id=event_id,
        event_type=event_type,
        created_at=f"2026-07-28T00:00:{event_id:02d}+00:00",
        payload=payload,
    )


def test_missing_required_engineering_evidence_fails_closed(tmp_path: Path) -> None:
    report = build_engineering_gate_report(
        [],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=None,
    )

    assert report["accepted"] is False
    assert set(report["gates"]) == set(REQUIRED_ENGINEERING_GATES)
    assert "effective_loops_per_second" in report["missing_required_metrics"]
    assert "natural_run_disk_reduction_ratio" in report["missing_required_metrics"]


def test_build_gates_use_effect_evidence_not_api_acceptance(tmp_path: Path) -> None:
    events = [
        _event(
            1,
            "execution",
            {
                "protocol_version": "1.1",
                "run_id": "run",
                "episode_id": "episode",
                "step_id": 1,
                "command_id": "build-1",
                "success": False,
                "action_name": "Build_Pylon_Screen",
                "actor": "Builder/Empty",
                "status": "failed",
                "execution_stage": "effect_verification",
                "failure_code": "no_build_start_evidence",
                "primitive_trace": [
                    {
                        "function": "Build_Pylon_pt",
                        "origin": "translator",
                        "ordinal": 0,
                        "total": 1,
                        "accepted": True,
                    }
                ],
                "effect_evidence": {
                    "effect_kind": "build",
                    "target_position": [30.0, 40.0],
                    "validated_target_position": [30.0, 40.0],
                    "emitted_target_position": [30.0, 40.0],
                    "builder_tag": "0x1",
                    "build_started": False,
                },
            },
        )
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["build_start_coverage"] == 0.0
    assert report["metrics"]["build_confirmation_rate"] == 0.0
    assert report["metrics"]["build_failure_rate"] == 1.0
    assert report["gates"]["build_start_coverage"]["passed"] is False
    assert report["gates"]["build_confirmation_rate"]["passed"] is False
    assert report["gates"]["build_failure_rate"]["passed"] is False


def test_performance_and_disk_metrics_are_thresholded(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_bytes(b"x" * 400)
    events = [
        _event(1, "observation", {"game_loop": 1, "state": {}}),
        _event(
            2,
            "event_store_performance",
            {
                "subscriber_callbacks_under_durable_lock": 0,
            },
        ),
        _event(3, "observation", {"game_loop": 2, "state": {}}),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=400.0,
    )

    assert report["metrics"]["effective_loops_per_second"] == 1.0
    assert report["metrics"]["natural_run_disk_reduction_ratio"] == 2.0
    assert report["gates"]["effective_loops_per_second"]["passed"] is False
    assert report["gates"]["natural_run_disk_reduction_ratio"]["passed"] is False


def test_zero_production_exposure_is_not_confirmation_success(tmp_path: Path) -> None:
    report = build_engineering_gate_report(
        [],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["minimum_production_exposure"] is False
    assert report["metrics"]["production_confirmation_complete"] is None
    assert report["gates"]["production_confirmation_complete"]["passed"] is False


def test_zero_build_exposure_is_not_reported_as_full_coverage(tmp_path: Path) -> None:
    report = build_engineering_gate_report(
        [],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["minimum_build_exposure"] is False
    assert report["metrics"]["build_start_coverage"] is None
    assert report["metrics"]["build_confirmation_rate"] is None
    assert report["metrics"]["build_failure_rate"] is None


def test_missing_recovery_evidence_does_not_pass_recovery_gate(tmp_path: Path) -> None:
    report = build_engineering_gate_report(
        [],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["recovery_evidence_present"] is False
    assert report["metrics"]["checkpoint_tail_recovery_bounded"] is None
    assert report["gates"]["recovery_evidence_present"]["passed"] is False


def test_two_actors_one_engagement_is_not_attack_redispatch(tmp_path: Path) -> None:
    events = [
        _attack_dispatch(1, actor="CombatGroup1", command_id="attack-a"),
        _attack_dispatch(2, actor="CombatGroup2", command_id="attack-b"),
        _event(
            3,
            "execution",
            {
                "command_id": "attack-b",
                "action_name": "Attack_Unit",
                "status": "cancelled",
                "failure_code": "engagement_target_eliminated",
                "effect_evidence": {
                    "effect_kind": "combat",
                    "confirmation_kind": "satisfied_by_peer",
                    "engagement_id": "engagement:shared",
                },
            },
        ),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["unchanged_attack_redispatch_count"] == 0


def test_same_actor_same_target_equivalent_order_is_redispatch(tmp_path: Path) -> None:
    events = [
        _attack_dispatch(1, actor="CombatGroup1", command_id="attack-a"),
        _attack_dispatch(2, actor="CombatGroup1", command_id="attack-b"),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["unchanged_attack_redispatch_count"] == 1
    assert report["gates"]["unchanged_attack_redispatch_zero"]["passed"] is False


def test_same_actor_retry_after_terminal_is_new_attack_attempt(tmp_path: Path) -> None:
    events = [
        _attack_dispatch(1, actor="CombatGroup1", command_id="attack-a"),
        _event(
            2,
            "execution",
            {
                "command_id": "attack-a",
                "action_name": "Attack_Unit",
                "status": "failed",
                "failure_code": "combat_target_lost",
            },
        ),
        _attack_dispatch(3, actor="CombatGroup1", command_id="attack-b"),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["unchanged_attack_redispatch_count"] == 0


def test_cross_type_overlapping_footprint_fails_gate(tmp_path: Path) -> None:
    events = [
        _ledger_transition(
            1,
            reservation_id="placement:gateway",
            structure_type="Gateway",
            cells=[[30, 30], [31, 30]],
        ),
        _ledger_transition(
            2,
            reservation_id="placement:pylon",
            structure_type="Pylon",
            cells=[[31, 30], [32, 30]],
        ),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["overlap_count"] == 1
    assert report["gates"]["cross_type_footprint_overlap_zero"]["passed"] is False


def test_nonspatial_failure_permanently_suppressing_cell_fails_gate(
    tmp_path: Path,
) -> None:
    event = _ledger_transition(
        1,
        reservation_id="placement:gateway",
        structure_type="Gateway",
        cells=[[30, 30]],
        next_state="permanent_invalid",
        failure_class="nonspatial",
        actor_failure=True,
    )

    report = build_engineering_gate_report(
        [event],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["nonspatial_quarantine_count"] == 1
    assert report["gates"]["nonspatial_quarantine_zero"]["passed"] is False


def test_preacceptance_invalid_footprint_redispatch_fails_gate(tmp_path: Path) -> None:
    events = [
        _ledger_transition(
            1,
            reservation_id="placement:first",
            structure_type="Gateway",
            cells=[[30, 30]],
            next_state="permanent_invalid",
            failure_class="spatial_permanent",
        ),
        _ledger_transition(
            2,
            reservation_id="placement:second",
            structure_type="Pylon",
            cells=[[30, 30]],
        ),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["invalid_redispatch_count"] == 1
    assert report["gates"]["invalid_footprint_redispatch_zero"]["passed"] is False


def test_placement_ledger_rejects_declared_state_that_does_not_match_history(
    tmp_path: Path,
) -> None:
    reserved = _ledger_transition(
        1,
        reservation_id="placement:one",
        structure_type="Pylon",
        cells=[[30, 30]],
    )
    invalid_release = _event(
        2,
        "placement_ledger_transition",
        {
            **reserved.payload,
            "previous_state": "occupied",
            "next_state": "released",
            "game_loop": 2,
        },
    )

    report = build_engineering_gate_report(
        [reserved, invalid_release],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["invalid_transition_count"] == 1
    assert report["gates"]["placement_ledger_transitions_valid"]["passed"] is False


def test_defense_pending_inventory_counts_against_cap(tmp_path: Path) -> None:
    event = _event(
        1,
        "defense_inventory_evaluated",
        {
            "item_type": "ShieldBattery",
            "completed": 4,
            "constructing_or_training": 1,
            "queued": 0,
            "reserved": 1,
            "dispatched_not_terminal": 1,
            "effective_count": 6,
            "hard_cap": 4,
            "decision": "over_cap",
        },
    )

    report = build_engineering_gate_report(
        [event],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["defense_inventory_cap_violation_count"] == 1
    assert report["gates"]["defense_inventory_within_cap"]["passed"] is False


def test_internally_inconsistent_defense_inventory_fails_closed(
    tmp_path: Path,
) -> None:
    event = _event(
        1,
        "defense_inventory_evaluated",
        {
            "item_type": "Phoenix",
            "completed": 6,
            "constructing_or_training": 1,
            "queued": 0,
            "reserved": 0,
            "dispatched_not_terminal": 1,
            "effective_count": 6,
            "hard_cap": 6,
            "decision": "at_cap",
        },
    )

    report = build_engineering_gate_report(
        [event],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["defense_inventory_invalid_count"] == 1
    assert report["gates"]["defense_inventory_within_cap"]["passed"] is False


def test_engineering_accumulator_discards_high_frequency_observations() -> None:
    accumulator = EngineeringAccumulator()
    for event_id in range(1_000):
        accumulator.ingest(
            StoredEvent(
                event_id=event_id,
                run_id="run",
                episode_id="episode",
                step_id=event_id,
                event_type="observation",
                created_at="2026-07-28T00:00:00+00:00",
                payload={"game_loop": event_id},
            )
        )

    assert accumulator.event_count == 1_000
    assert accumulator.events == []
    assert accumulator.max_game_loop == 999


def _attack_dispatch(
    event_id: int,
    *,
    actor: str,
    command_id: str,
) -> StoredEvent:
    return _event(
        event_id,
        "command_lifecycle",
        {
            "status": "dispatched",
            "command": {
                "command_id": command_id,
                "operation_id": "operation:" + "a" * 64,
                "name": "Attack_Unit",
                "actor": actor,
                "arguments": ["0xdead"],
            },
        },
    )


def _ledger_transition(
    event_id: int,
    *,
    reservation_id: str,
    structure_type: str,
    cells: list[list[int]],
    next_state: str = "reserved",
    failure_class: str | None = None,
    actor_failure: bool = False,
) -> StoredEvent:
    return _event(
        event_id,
        "placement_ledger_transition",
        {
            "reservation_id": reservation_id,
            "structure_type": structure_type,
            "footprint_cells": cells,
            "previous_state": "unreserved",
            "next_state": next_state,
            "failure_class": failure_class,
            "actor_failure": actor_failure,
            "game_loop": event_id,
            "release_reason": failure_class,
        },
    )
