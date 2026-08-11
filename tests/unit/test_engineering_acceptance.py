from __future__ import annotations

from pathlib import Path

from rtscortex_llm_pysc2.extractor import BUILD_SPECS

from rtscortex.evaluation.engineering import (
    REQUIRED_ENGINEERING_GATES,
    EngineeringAccumulator,
    build_engineering_gate_report,
)
from rtscortex.memory import StoredEvent
from rtscortex.placement import (
    CANONICAL_PLACEMENT_SPECS,
    canonical_footprint_cells,
)


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


def test_engineering_accumulator_is_hard_bounded() -> None:
    accumulator = EngineeringAccumulator(retention_limit=2)

    for event_id in range(1, 5):
        accumulator.ingest(_event(event_id, "execution", {"command_id": str(event_id)}))

    assert len(accumulator.events) == 2
    assert accumulator.retention_overflow_count == 2


def test_terminal_collapse_non_recovery_macro_dispatch_fails_closed(
    tmp_path: Path,
) -> None:
    report = build_engineering_gate_report(
        [
            _event(
                1,
                "command_lifecycle",
                {
                    "command": {"command_id": "macro-build"},
                    "status": "dispatched",
                },
            ),
            _event(
                2,
                "command_lineage",
                {
                    "command_id": "macro-build",
                    "lineage": {
                        "command_id": "macro-build",
                        "source_role": "macro",
                    },
                    "macro_plan_id": "macro-plan:unsafe",
                    "semantic_action": "BUILD PYLON",
                    "terminal_collapse": True,
                    "townhall_recovery": False,
                },
            ),
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["terminal_collapse_non_recovery_macro_dispatch_count"] == 1
    assert report["diagnostics"]["terminal_collapse_non_recovery_macro_dispatch_count"] == 1
    assert report["gates"]["terminal_collapse_non_recovery_macro_dispatch_count"]["passed"] is False
    assert report["accepted"] is False


def test_terminal_collapse_lineage_without_dispatch_is_not_counted(
    tmp_path: Path,
) -> None:
    report = build_engineering_gate_report(
        [
            _event(
                1,
                "command_lifecycle",
                {
                    "command": {"command_id": "macro-build"},
                    "status": "superseded",
                },
            ),
            _event(
                2,
                "command_lineage",
                {
                    "command_id": "macro-build",
                    "lineage": {
                        "command_id": "macro-build",
                        "source_role": "macro",
                    },
                    "terminal_collapse": True,
                    "townhall_recovery": False,
                },
            ),
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["terminal_collapse_non_recovery_macro_dispatch_count"] == 0
    assert report["diagnostics"]["terminal_collapse_macro_lineage_unknown_count"] == 0
    assert report["gates"]["terminal_collapse_non_recovery_macro_dispatch_count"]["passed"] is True


def test_terminal_collapse_dispatch_guard_violation_fails_gate(tmp_path: Path) -> None:
    report = build_engineering_gate_report(
        [
            _event(
                1,
                "terminal_collapse_non_recovery_macro_dispatch",
                {
                    "command_id": "guarded-macro-build",
                    "reason": "terminal_collapse_non_recovery_macro_dispatch",
                },
            )
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["terminal_collapse_non_recovery_macro_dispatch_count"] == 1
    assert report["diagnostics"]["terminal_collapse_macro_dispatch_guard_violation_count"] == 1
    assert report["gates"]["terminal_collapse_non_recovery_macro_dispatch_count"]["passed"] is False


def test_terminal_collapse_raw_boundary_violation_fails_gate(tmp_path: Path) -> None:
    report = build_engineering_gate_report(
        [
            _event(
                1,
                "execution",
                {
                    "command_id": "queued-macro-build",
                    "status": "failed",
                    "execution_stage": "raw_pre_dispatch",
                    "failure_code": "terminal_collapse_non_recovery_macro_dispatch",
                },
            )
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["terminal_collapse_non_recovery_macro_dispatch_count"] == 1
    assert report["diagnostics"]["terminal_collapse_macro_raw_boundary_violation_count"] == 1
    assert report["gates"]["terminal_collapse_non_recovery_macro_dispatch_count"]["passed"] is False


def test_missing_terminal_collapse_lineage_classification_fails_closed(
    tmp_path: Path,
) -> None:
    report = build_engineering_gate_report(
        [
            _event(
                1,
                "command_lifecycle",
                {
                    "command": {"command_id": "legacy-macro-build"},
                    "status": "dispatched",
                },
            ),
            _event(
                2,
                "command_lineage",
                {
                    "command_id": "legacy-macro-build",
                    "lineage": {
                        "command_id": "legacy-macro-build",
                        "source_role": "macro",
                    },
                },
            ),
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["terminal_collapse_non_recovery_macro_dispatch_count"] is None
    assert report["diagnostics"]["terminal_collapse_macro_lineage_unknown_count"] == 1
    assert report["gates"]["terminal_collapse_non_recovery_macro_dispatch_count"]["passed"] is False


def test_no_terminal_collapse_non_recovery_macro_dispatch_passes_zero_gate(
    tmp_path: Path,
) -> None:
    report = build_engineering_gate_report(
        [],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["terminal_collapse_non_recovery_macro_dispatch_count"] == 0
    assert report["gates"]["terminal_collapse_non_recovery_macro_dispatch_count"]["passed"] is True


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
                    "requested_target_position": [30.25, 40.25],
                    "final_validated_target_position": [30.0, 40.0],
                    "validated_target_position": [30.0, 40.0],
                    "emitted_target_position": [30.0, 40.0],
                    "verified_target_position": [30.0, 40.0],
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


def test_semantic_build_diagnostics_join_command_lifecycle_and_raw_placement_payloads(
    tmp_path: Path,
) -> None:
    events = [
        _semantic_build_dispatch(
            1,
            command_id="build-a1",
            operation_id="operation:a",
            attempt_id="attempt:a1",
            status="dispatched",
        ),
        _semantic_build_transition(
            2,
            command_id="build-a1",
            operation_id="operation:a",
            target_state_revision="revision:a",
            next_state="temporary_suppressed",
            failure_class="spatial_retryable",
            operation_circuit_open=True,
        ),
        _semantic_build_execution(
            3,
            command_id="build-a1",
            operation_id="operation:a",
            attempt_id="attempt:a1",
            status="failed",
            failure_code="no_build_start_evidence",
            failure_classification="dynamic_target_obstruction",
        ),
        _semantic_build_dispatch(
            4,
            command_id="build-a2",
            operation_id="operation:a",
            attempt_id="attempt:a2",
            status="dispatched",
        ),
        _semantic_build_transition(
            5,
            command_id="build-a2",
            operation_id="operation:a",
            target_state_revision="revision:b",
            next_state="occupied",
        ),
        _semantic_build_execution(
            6,
            command_id="build-a2",
            operation_id="operation:a",
            attempt_id="attempt:a2",
            status="succeeded",
        ),
        _semantic_build_dispatch(
            7,
            command_id="build-b1",
            operation_id="operation:b",
            attempt_id="attempt:b1",
            status="dispatched",
        ),
        _semantic_build_transition(
            8,
            command_id="build-b1",
            operation_id="operation:b",
            target_state_revision="revision:c",
            next_state="released",
            failure_class="nonspatial",
        ),
        _semantic_build_execution(
            9,
            command_id="build-b1",
            operation_id="operation:b",
            attempt_id="attempt:b1",
            status="failed",
            failure_code="builder_unavailable",
            failure_classification="builder_not_ready",
        ),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    diagnostics = report["diagnostics"]
    assert diagnostics["semantic_build_operation_max_failure_streak"] == 1
    assert diagnostics["semantic_build_circuit_breaker_count"] == 1
    assert diagnostics["semantic_build_cross_revision_retry_count"] == 1
    assert diagnostics["semantic_build_failure_classification"] == {
        "builder_not_ready": 1,
        "dynamic_target_obstruction": 1,
    }
    assert diagnostics["semantic_build_operations"]["operation:a"] == {
        "accepted_attempt_count": 2,
        "no_start_failure_count": 1,
        "max_no_start_failure_streak": 1,
        "circuit_breaker_count": 1,
        "cross_revision_retry_count": 1,
        "failure_classification": {"dynamic_target_obstruction": 1},
    }
    assert diagnostics["semantic_build_operations"]["operation:b"] == {
        "accepted_attempt_count": 1,
        "no_start_failure_count": 0,
        "max_no_start_failure_streak": 0,
        "circuit_breaker_count": 0,
        "cross_revision_retry_count": 0,
        "failure_classification": {"builder_not_ready": 1},
    }
    assert report["gates"]["semantic_build_failure_streak_bounded"]["passed"] is True


def test_semantic_build_failure_streak_gate_fails_closed_above_bound(tmp_path: Path) -> None:
    events = []
    for event_id in range(1, 5):
        command_id = f"build-{event_id}"
        events.extend(
            [
                _semantic_build_dispatch(
                    event_id * 3 - 2,
                    command_id=command_id,
                    operation_id="operation:streak",
                    attempt_id=f"attempt:s{event_id}",
                    status="dispatched",
                ),
                _semantic_build_execution(
                    event_id * 3 - 1,
                    command_id=command_id,
                    operation_id="operation:streak",
                    attempt_id=f"attempt:s{event_id}",
                    status="failed",
                    failure_code="no_build_start_evidence",
                ),
            ]
        )

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["semantic_build_operation_max_failure_streak"] == 4
    assert report["gates"]["semantic_build_failure_streak_bounded"]["passed"] is False


def test_semantic_build_streak_ignores_pre_dispatch_rejections(tmp_path: Path) -> None:
    events = [
        _semantic_build_execution(
            event_id,
            command_id=f"pre-dispatch-{event_id}",
            operation_id="operation:pre-dispatch",
            attempt_id=f"attempt:pre-{event_id}",
            status="failed",
            failure_code="builder_not_ready",
            execution_stage="pre_dispatch",
        )
        for event_id in range(1, 5)
    ]
    events.append(
        _semantic_build_execution(
            5,
            command_id="accepted-build",
            operation_id="operation:pre-dispatch",
            attempt_id="attempt:accepted",
            status="succeeded",
        )
    )

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["semantic_build_operation_max_failure_streak"] == 0
    assert report["gates"]["semantic_build_failure_streak_bounded"]["passed"] is True


def test_build_started_effect_missing_is_not_a_no_start_streak(tmp_path: Path) -> None:
    events = [
        _semantic_build_execution(
            1,
            command_id="started-but-missing",
            operation_id="operation:started",
            attempt_id="attempt:started",
            status="failed",
            failure_code="build_started_effect_missing",
            failure_classification="gameplay_effect_missing_after_start",
        )
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["semantic_build_operation_max_failure_streak"] == 0
    assert report["diagnostics"]["semantic_build_failure_classification"] == {
        "gameplay_effect_missing_after_start": 1
    }


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


def test_health_delta_collision_ignores_failed_or_unconfirmed_evidence(tmp_path: Path) -> None:
    events = [
        _combat_execution(
            1,
            status="failed",
            confirmation_kind="target_damaged",
            engagement_id="engagement:failed",
        ),
        _combat_execution(
            2,
            status="unconfirmed",
            confirmation_kind="target_damaged",
            engagement_id="engagement:unconfirmed",
        ),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["health_delta_engagement_collision_count"] == 0
    assert report["gates"]["health_delta_engagement_attribution_valid"]["passed"] is True


def test_confirmed_health_evidence_requires_complete_attribution(tmp_path: Path) -> None:
    for field, value in (
        ("confirmed_game_loop", None),
        ("engagement_id", None),
        ("target_tag", None),
        ("observed_target_health", 100.0),
        ("target_health_delta", None),
    ):
        evidence = _combat_execution(1, confirmation_kind="target_damaged").payload[
            "effect_evidence"
        ]
        evidence[field] = value

        report = build_engineering_gate_report(
            [_event(1, "execution", {**_combat_execution(1).payload, "effect_evidence": evidence})],
            run_dir=tmp_path,
            natural_run_baseline_bytes_per_loop=100.0,
        )

        assert report["diagnostics"]["health_delta_engagement_collision_count"] == 1
        assert report["gates"]["health_delta_engagement_attribution_valid"]["passed"] is False


def test_same_confirmed_health_transition_in_distinct_engagements_collides(
    tmp_path: Path,
) -> None:
    events = [
        _combat_execution(1, engagement_id="engagement:first"),
        _combat_execution(2, engagement_id="engagement:second"),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["health_delta_engagement_collision_count"] == 1
    assert report["gates"]["health_delta_engagement_attribution_valid"]["passed"] is False


def test_target_removed_confirmation_is_grouped_even_without_health_delta(tmp_path: Path) -> None:
    events = [
        _combat_execution(
            1,
            confirmation_kind="target_removed",
            observed_target_health=100.0,
            target_health_delta=0.0,
            engagement_id="engagement:first",
        ),
        _combat_execution(
            2,
            confirmation_kind="target_removed",
            observed_target_health=100.0,
            target_health_delta=0.0,
            engagement_id="engagement:second",
        ),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["health_delta_engagement_collision_count"] == 1
    assert report["gates"]["health_delta_engagement_attribution_valid"]["passed"] is False


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


def test_unchanged_temporarily_failed_target_redispatch_fails_gate(tmp_path: Path) -> None:
    events = [
        _ledger_transition(
            1,
            reservation_id="placement:first",
            structure_type="Assimilator",
            cells=[[30, 30]],
            action_name="Build_Assimilator_Near",
            target_state_revision="state-a",
        ),
        _ledger_transition(
            2,
            reservation_id="placement:first",
            structure_type="Assimilator",
            cells=[[30, 30]],
            action_name="Build_Assimilator_Near",
            previous_state="reserved",
            next_state="temporary_suppressed",
            failure_class="spatial_retryable",
            target_state_revision="state-a",
        ),
        _ledger_transition(
            3,
            reservation_id="placement:second",
            structure_type="Assimilator",
            cells=[[30, 30]],
            action_name="Build_Assimilator_Near",
            target_state_revision="state-a",
        ),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["unchanged_failed_target_redispatch_count"] == 1
    assert report["gates"]["unchanged_failed_target_redispatch_zero"]["passed"] is False


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
            "transition_id": "placement-transition:" + "2" * 64,
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
            "current_batch_selected": 0,
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
            "current_batch_selected": 0,
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


def _combat_execution(
    event_id: int,
    *,
    status: str = "succeeded",
    confirmation_kind: str | None = "target_damaged",
    target_tag: str | None = "0xdead",
    confirmed_game_loop: int | None = 108,
    engagement_id: str | None = "engagement:shared",
    baseline_target_health: float | None = 100.0,
    observed_target_health: float | None = 90.0,
    target_health_delta: float | None = 10.0,
) -> StoredEvent:
    return _event(
        event_id,
        "execution",
        {
            "command_id": f"attack-{event_id}",
            "action_name": "Attack_Unit",
            "status": status,
            "execution_stage": "effect_verification",
            "effect_evidence": {
                "effect_kind": "combat",
                "confirmation_kind": confirmation_kind,
                "target_tag": target_tag,
                "confirmed_game_loop": confirmed_game_loop,
                "engagement_id": engagement_id,
                "baseline_target_health": baseline_target_health,
                "observed_target_health": observed_target_health,
                "target_health_delta": target_health_delta,
            },
        },
    )


def _semantic_build_dispatch(
    event_id: int,
    *,
    command_id: str,
    operation_id: str,
    attempt_id: str,
    status: str,
) -> StoredEvent:
    return _event(
        event_id,
        "command_lifecycle",
        {
            "status": status,
            "command": {
                "command_id": command_id,
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "name": "Build_Pylon_Screen",
                "actor": "Builder/Builder-Probe-1",
                "arguments": [[20, 20]],
            },
        },
    )


def _semantic_build_transition(
    event_id: int,
    *,
    command_id: str,
    operation_id: str,
    target_state_revision: str,
    next_state: str,
    failure_class: str | None = None,
    operation_circuit_open: bool = False,
) -> StoredEvent:
    return _event(
        event_id,
        "placement_ledger_transition",
        {
            "command_id": command_id,
            "action_name": "Build_Pylon_Screen",
            "transition_id": f"placement-transition:{event_id:064x}",
            "transition": {
                "reservation_id": f"placement:{command_id}",
                "structure_type": "Pylon",
                "footprint_cells": [[19, 19], [19, 20], [20, 19], [20, 20]],
                "previous_state": "reserved",
                "next_state": next_state,
                "failure_class": failure_class,
                "game_loop": event_id,
                "target_state_revision": target_state_revision,
                "operation_circuit_open": operation_circuit_open,
            },
            "operation_id": operation_id,
            "operation_circuit_open": operation_circuit_open,
        },
    )


def _semantic_build_execution(
    event_id: int,
    *,
    command_id: str,
    operation_id: str,
    attempt_id: str,
    status: str,
    failure_code: str | None = None,
    failure_classification: str | None = None,
    execution_stage: str = "effect_verification",
) -> StoredEvent:
    return _event(
        event_id,
        "execution",
        {
            "command_id": command_id,
            "operation_id": operation_id,
            "attempt_id": attempt_id,
            "action_name": "Build_Pylon_Screen",
            "status": status,
            "failure_code": failure_code,
            "execution_stage": execution_stage,
            "effect_evidence": {
                "effect_kind": "build",
                "reservation_id": f"placement:{command_id}",
                "failure_class": None,
                "failure_classification": failure_classification,
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
    command_id: str | None = None,
    previous_state: str = "unreserved",
    game_loop: int | None = None,
    action_name: str | None = None,
    builder_tag: str | None = "0xb1",
    builder_lease_state: str | None = None,
    target_state_revision: str | None = None,
) -> StoredEvent:
    resolved_command_id = command_id or f"command:{reservation_id}"
    resolved_lease_state = (
        builder_lease_state
        if builder_lease_state is not None
        else "acquired"
        if next_state == "reserved" and builder_tag
        else "released"
        if builder_tag
        and next_state
        in {
            "occupied",
            "released",
            "temporary_suppressed",
            "permanent_invalid",
        }
        else None
    )
    return _event(
        event_id,
        "placement_ledger_transition",
        {
            "command_id": resolved_command_id,
            "action_name": action_name or f"Build_{structure_type}_Screen",
            "transition_id": f"placement-transition:{event_id:064x}",
            "builder_tag": builder_tag,
            "builder_lease_state": resolved_lease_state,
            "reservation_id": reservation_id,
            "structure_type": structure_type,
            "footprint_cells": cells,
            "previous_state": previous_state,
            "next_state": next_state,
            "failure_class": failure_class,
            "actor_failure": actor_failure,
            "game_loop": event_id if game_loop is None else game_loop,
            "release_reason": failure_class,
            "target_state_revision": target_state_revision,
        },
    )


def _accepted_build(
    event_id: int,
    *,
    command_id: str,
    reservation_id: str,
    status: str = "succeeded",
    action_name: str = "Build_Pylon_Screen",
    builder_tag: str | None = "0xb1",
    target_position: tuple[float, float] = (22.0, 24.0),
    occupied_grid_cells: list[list[int]] | None = None,
) -> StoredEvent:
    spec = CANONICAL_PLACEMENT_SPECS[action_name]
    cells = (
        sorted(canonical_footprint_cells(target_position, spec))
        if occupied_grid_cells is None
        else occupied_grid_cells
    )
    return _event(
        event_id,
        "execution",
        {
            "command_id": command_id,
            "action_name": action_name,
            "status": status,
            "execution_stage": "effect_verification",
            "primitive_trace": [
                {
                    "origin": "translator",
                    "accepted": True,
                }
            ],
            "effect_evidence": {
                "effect_kind": "build",
                "reservation_id": reservation_id,
                "target_position": list(target_position),
                "requested_target_position": list(target_position),
                "final_validated_target_position": list(target_position),
                "validated_target_position": list(target_position),
                "emitted_target_position": list(target_position),
                "verified_target_position": list(target_position),
                "builder_tag": builder_tag,
                "footprint_width": spec.footprint + (2 if spec.reserves_addon_space else 0),
                "footprint_height": spec.footprint,
                "occupied_grid_cells": cells,
                "build_started": True,
            },
        },
    )


def test_every_accepted_build_requires_complete_ledger_chain(tmp_path: Path) -> None:
    events = [
        _ledger_transition(
            1,
            reservation_id="placement:one",
            structure_type="Pylon",
            cells=[[21, 23], [22, 23], [21, 24], [22, 24]],
            command_id="build-one",
        ),
        _ledger_transition(
            2,
            reservation_id="placement:one",
            structure_type="Pylon",
            cells=[[21, 23], [22, 23], [21, 24], [22, 24]],
            command_id="build-one",
            previous_state="reserved",
            next_state="occupied",
        ),
        _ledger_transition(
            3,
            reservation_id="placement:one",
            structure_type="Pylon",
            cells=[[21, 23], [22, 23], [21, 24], [22, 24]],
            command_id="build-one",
            previous_state="occupied",
            next_state="released",
        ),
        _accepted_build(4, command_id="build-one", reservation_id="placement:one"),
        _accepted_build(5, command_id="build-two", reservation_id="placement:two"),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["placement_ledger_coverage"] == 0.5
    assert report["gates"]["placement_ledger_coverage"]["passed"] is False


def test_one_ledger_chain_cannot_cover_multiple_accepted_builds(tmp_path: Path) -> None:
    events = [
        _ledger_transition(
            1,
            reservation_id="placement:shared",
            structure_type="Pylon",
            cells=[[21, 23], [22, 23], [21, 24], [22, 24]],
            command_id="build-one",
        ),
        _ledger_transition(
            2,
            reservation_id="placement:shared",
            structure_type="Pylon",
            cells=[[21, 23], [22, 23], [21, 24], [22, 24]],
            command_id="build-one",
            previous_state="reserved",
            next_state="occupied",
        ),
        _ledger_transition(
            3,
            reservation_id="placement:shared",
            structure_type="Pylon",
            cells=[[21, 23], [22, 23], [21, 24], [22, 24]],
            command_id="build-one",
            previous_state="occupied",
            next_state="released",
        ),
        _accepted_build(4, command_id="build-one", reservation_id="placement:shared"),
        _accepted_build(5, command_id="build-two", reservation_id="placement:shared"),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["accepted_build_ledger_count"] == 1
    assert report["metrics"]["placement_ledger_coverage"] == 0.5


def test_placement_ledger_rejects_identity_mutation_and_nonmonotonic_time(
    tmp_path: Path,
) -> None:
    events = [
        _ledger_transition(
            20,
            reservation_id="placement:one",
            structure_type="Pylon",
            cells=[[21, 23], [22, 23], [21, 24], [22, 24]],
            command_id="build-one",
            game_loop=20,
        ),
        _ledger_transition(
            21,
            reservation_id="placement:one",
            structure_type="Gateway",
            cells=[[21, 23], [22, 23], [23, 23]],
            command_id="build-one",
            previous_state="reserved",
            next_state="released",
            game_loop=19,
        ),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["invalid_transition_count"] >= 2
    assert report["gates"]["placement_ledger_transitions_valid"]["passed"] is False


def test_terminal_episode_rejects_orphan_reservation(tmp_path: Path) -> None:
    report = build_engineering_gate_report(
        [
            _ledger_transition(
                1,
                reservation_id="placement:orphan",
                structure_type="Pylon",
                cells=[[21, 23], [22, 23], [21, 24], [22, 24]],
                command_id="build-orphan",
            )
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["orphan_reservation_count"] == 1
    assert report["gates"]["placement_ledger_terminal_complete"]["passed"] is False


def test_accepted_build_without_ledger_lease_fails_coverage(tmp_path: Path) -> None:
    events = _complete_pylon_chain(builder_tag=None)

    report = build_engineering_gate_report(
        [*events, _accepted_build(4, command_id="build-one", reservation_id="placement:one")],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["builder_lease_complete_count"] == 0
    assert report["metrics"]["placement_ledger_coverage"] == 0.0
    assert report["gates"]["builder_lease_complete"]["passed"] is False


def test_ledger_builder_must_match_effect_builder(tmp_path: Path) -> None:
    events = _complete_pylon_chain(builder_tag="0xb2")

    report = build_engineering_gate_report(
        [*events, _accepted_build(4, command_id="build-one", reservation_id="placement:one")],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["builder_lease_complete_count"] == 0
    assert report["gates"]["builder_lease_complete"]["passed"] is False


def test_ledger_builder_tag_cannot_change_mid_chain(tmp_path: Path) -> None:
    events = _complete_pylon_chain(builder_tag="0xb1")
    events[1].payload["builder_tag"] = "0xb2"

    report = build_engineering_gate_report(
        [*events, _accepted_build(4, command_id="build-one", reservation_id="placement:one")],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["builder_lease_complete_count"] == 0
    assert report["gates"]["builder_lease_complete"]["passed"] is False


def test_empty_builder_tag_does_not_count_as_lease(tmp_path: Path) -> None:
    events = _complete_pylon_chain(builder_tag="")

    report = build_engineering_gate_report(
        [
            *events,
            _accepted_build(
                4,
                command_id="build-one",
                reservation_id="placement:one",
                builder_tag="",
            ),
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["builder_lease_complete_count"] == 0
    assert report["gates"]["builder_lease_complete"]["passed"] is False


def test_ledger_cells_must_match_canonical_action_footprint(tmp_path: Path) -> None:
    events = _complete_pylon_chain(
        builder_tag="0xb1",
        cells=[[22, 24]],
    )

    report = build_engineering_gate_report(
        [*events, _accepted_build(4, command_id="build-one", reservation_id="placement:one")],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["canonical_footprint_mismatch_count"] == 1
    assert report["gates"]["placement_ledger_canonical_footprint_complete"]["passed"] is False


def test_ledger_structure_type_must_match_build_action(tmp_path: Path) -> None:
    events = _complete_pylon_chain(
        builder_tag="0xb1",
        structure_type="Gateway",
    )

    report = build_engineering_gate_report(
        [*events, _accepted_build(4, command_id="build-one", reservation_id="placement:one")],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["canonical_footprint_mismatch_count"] == 1
    assert report["gates"]["placement_ledger_canonical_footprint_complete"]["passed"] is False


def test_ledger_cells_must_match_effect_evidence_cells(tmp_path: Path) -> None:
    events = _complete_pylon_chain(builder_tag="0xb1")

    report = build_engineering_gate_report(
        [
            *events,
            _accepted_build(
                4,
                command_id="build-one",
                reservation_id="placement:one",
                occupied_grid_cells=[[22, 24]],
            ),
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["diagnostics"]["canonical_footprint_mismatch_count"] == 1
    assert report["gates"]["placement_ledger_canonical_footprint_complete"]["passed"] is False


def test_terran_addon_cells_are_verified_not_only_self_reported(tmp_path: Path) -> None:
    action_name = "Build_Barracks_Screen"
    spec = CANONICAL_PLACEMENT_SPECS[action_name]
    canonical_cells = sorted(canonical_footprint_cells((22.0, 24.0), spec))
    events = _complete_pylon_chain(
        builder_tag="0xb1",
        cells=[list(cell) for cell in canonical_cells[:-1]],
        structure_type="Barracks",
        action_name=action_name,
    )

    report = build_engineering_gate_report(
        [
            *events,
            _accepted_build(
                4,
                command_id="build-one",
                reservation_id="placement:one",
                action_name=action_name,
            ),
        ],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert len(canonical_cells) == 15
    assert report["diagnostics"]["canonical_footprint_mismatch_count"] == 1
    assert report["gates"]["placement_ledger_canonical_footprint_complete"]["passed"] is False


def test_acceptance_placement_specs_match_bridge_build_specs() -> None:
    assert set(CANONICAL_PLACEMENT_SPECS) == set(BUILD_SPECS)
    for action_name, bridge_spec in BUILD_SPECS.items():
        acceptance_spec = CANONICAL_PLACEMENT_SPECS[action_name]
        assert acceptance_spec.structure_type == bridge_spec.target_structure
        assert acceptance_spec.footprint == bridge_spec.footprint
        assert acceptance_spec.reserves_addon_space == bridge_spec.reserves_addon_space


def _complete_pylon_chain(
    *,
    builder_tag: str | None,
    cells: list[list[int]] | None = None,
    structure_type: str = "Pylon",
    action_name: str = "Build_Pylon_Screen",
) -> list[StoredEvent]:
    resolved_cells = cells or [[21, 23], [22, 23], [21, 24], [22, 24]]
    return [
        _ledger_transition(
            1,
            reservation_id="placement:one",
            structure_type=structure_type,
            cells=resolved_cells,
            command_id="build-one",
            action_name=action_name,
            builder_tag=builder_tag,
        ),
        _ledger_transition(
            2,
            reservation_id="placement:one",
            structure_type=structure_type,
            cells=resolved_cells,
            command_id="build-one",
            previous_state="reserved",
            next_state="occupied",
            action_name=action_name,
            builder_tag=builder_tag,
        ),
        _ledger_transition(
            3,
            reservation_id="placement:one",
            structure_type=structure_type,
            cells=resolved_cells,
            command_id="build-one",
            previous_state="occupied",
            next_state="released",
            action_name=action_name,
            builder_tag=builder_tag,
        ),
    ]
