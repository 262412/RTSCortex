from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rtscortex.contracts import EpisodeOutcome, EpisodeResult
from rtscortex.evaluation.metrics import (
    aggregate_episode_metrics,
    compute_episode_metrics,
    compute_execution_metrics,
)
from rtscortex.memory import StoredEvent, read_event_log


def _event(
    event_id: int,
    event_type: str,
    payload: dict[str, object],
    *,
    second: int = 1,
    step_id: int | None = None,
) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="run-1",
        episode_id="episode-1",
        step_id=event_id if step_id is None else step_id,
        event_type=event_type,
        created_at=f"2026-01-01T00:00:0{second}+00:00",
        payload=payload,
    )


def test_episode_metrics_cover_runtime_telemetry() -> None:
    events = [
        _event(1, "observation", {}, second=0),
        _event(
            2,
            "decision",
            {
                "batch": {"rejected_commands": ["unknown action"]},
                "reflex_latency_ms": 1.0,
                "tick_latency_ms": 4.0,
                "preemptions": [{"actor": "army"}, {"actor": "worker"}],
            },
        ),
        _event(
            3,
            "decision",
            {
                "batch": {"rejected_commands": []},
                "reflex_latency_ms": 3.0,
                "tick_latency_ms": 8.0,
                "preemptions": [],
            },
        ),
        _event(4, "execution", {"success": True}),
        _event(5, "execution", {"success": False}),
        _event(6, "planner_cycle", {"latency_ms": 10.0}),
        _event(7, "planner_cycle", {"latency_ms": 20.0}),
        _event(
            8,
            "module_result",
            {
                "model_call": True,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        ),
        _event(
            9,
            "module_result",
            {
                "model_call": True,
                "usage": {
                    "prompt_tokens": 300,
                    "completion_tokens": 100,
                    "total_tokens": 400,
                },
            },
        ),
        _event(
            10,
            "plan_accepted",
            {
                "is_revision": False,
                "plan_age_game_loops": 20,
                "accepted_game_loop": 100,
            },
        ),
        _event(
            11,
            "plan_accepted",
            {
                "is_revision": True,
                "plan_age_game_loops": 30,
                "accepted_game_loop": 220,
            },
        ),
        _event(
            12,
            "plan_accepted",
            {
                "is_revision": False,
                "plan_age_game_loops": 10,
                "accepted_game_loop": 350,
            },
        ),
        _event(13, "episode_result", {}, second=5),
    ]
    result = EpisodeResult(
        run_id="run-1",
        episode_id="episode-1",
        scenario="test",
        seed=7,
        outcome=EpisodeOutcome.TRUNCATED,
        steps=2,
        failure_reason="max_steps_reached",
    )

    metrics = compute_episode_metrics(
        events,
        result,
        prompt_cost_per_million_tokens=2.0,
        completion_cost_per_million_tokens=4.0,
    )

    assert metrics.action_attempts == 2
    assert metrics.action_successes == 1
    assert metrics.action_success_rate == 0.5
    assert metrics.execution.meaningful_commands == 2
    assert metrics.execution.meaningful_successes == 1
    assert metrics.execution.completed_execution_success_rate == 0.5
    assert metrics.illegal_actions == 1
    assert metrics.illegal_action_rate == pytest.approx(1 / 3)
    assert metrics.planner_latency_ms_p50 == 15.0
    assert metrics.planner_latency_ms_p95 == 19.5
    assert metrics.reflex_latency_ms_p50 == 2.0
    assert metrics.reflex_latency_ms_p95 == pytest.approx(2.9)
    assert metrics.tick_latency_ms_p50 == 6.0
    assert metrics.tick_latency_ms_p95 == pytest.approx(7.8)
    assert metrics.model_requests == 2
    assert metrics.prompt_tokens == 400
    assert metrics.completion_tokens == 150
    assert metrics.total_tokens == 550
    assert metrics.model_cost_usd == pytest.approx(0.0014)
    assert metrics.reflex_preemptions == 2
    assert metrics.plans_accepted == 3
    assert metrics.plan_revisions == 1
    assert metrics.plan_revision_rate == 0.5
    assert metrics.plan_age_game_loops_p50 == 20.0
    assert metrics.plan_age_game_loops_p95 == 29.0
    assert metrics.plan_accept_gap_game_loops_p50 == 125.0
    assert metrics.plan_accept_gap_game_loops_p95 == 129.5
    assert metrics.plan_accept_gap_samples == 2
    assert metrics.episode_duration_seconds == 5.0
    assert metrics.failure_reason == "max_steps_reached"

    aggregate = aggregate_episode_metrics([metrics, metrics])
    assert aggregate["action_attempts"] == 4
    assert aggregate["model_requests"] == 4
    assert aggregate["model_cost_usd"] == pytest.approx(0.0028)
    assert aggregate["failure_reasons"] == {"max_steps_reached": 2}
    assert aggregate["plan_accept_gap_game_loops_p50"] == 125.0
    assert aggregate["plan_accept_gap_samples"] == 4
    assert aggregate["execution"]["meaningful_commands"] == 4
    assert aggregate["execution"]["meaningful_action_success_rate"] == 0.5


def test_execution_metrics_separate_control_noops_and_terminal_states() -> None:
    decision = _event(
        1,
        "decision",
        {
            "batch": {
                "planner_pending": True,
                "idle_reason": "waiting_for_planner",
                "commands": [
                    {
                        "command_id": "noop-1",
                        "name": "No_Operation",
                        "actor": "global",
                        "source": "fallback",
                    },
                    {
                        "command_id": "build-1",
                        "name": "Build_Pylon_Screen",
                        "actor": "Builder/Probe-1",
                        "source": "planner",
                    },
                    {
                        "command_id": "attack-1",
                        "name": "Attack_Unit",
                        "actor": "CombatGroup/Zealot-1",
                        "source": "planner",
                    },
                ],
                "rejected_commands": ["invalid-1: target is not an enemy"],
            }
        },
    )
    events = [
        decision,
        _event(
            2,
            "execution",
            {"command_id": "noop-1", "success": True, "pysc2_function": "no_op"},
        ),
        _event(
            3,
            "execution",
            {
                "command_id": "build-1",
                "action_name": "Build_Pylon_Screen",
                "actor": "Builder/Probe-1",
                "success": True,
                "status": "succeeded",
                "execution_stage": "effect_verification",
                "effect_evidence": {"new_structure_tag": "0xabc"},
            },
        ),
        _event(
            4,
            "execution",
            {
                "command_id": "attack-1",
                "success": False,
                "status": "failed",
                "execution_stage": "pre_dispatch",
                "failure_code": "friendly_target",
            },
        ),
        _event(
            5,
            "execution",
            {
                "command_id": "attack-2",
                "action_name": "Attack_Unit",
                "success": False,
                "status": "unconfirmed",
                "execution_stage": "episode_end",
                "failure_code": "effect_timeout",
            },
        ),
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.execution_reports == 4
    assert metrics.control_noops == 1
    assert metrics.meaningful_commands == 3
    assert metrics.meaningful_successes == 1
    assert metrics.meaningful_failures == 1
    assert metrics.meaningful_unconfirmed == 1
    assert metrics.meaningful_action_success_rate == pytest.approx(1 / 3)
    assert metrics.completed_execution_success_rate == 0.5
    assert metrics.terminal_backlog_rate == pytest.approx(1 / 3)
    assert metrics.failure_by_code == {"friendly_target": 1, "effect_timeout": 1}
    assert metrics.build_funnel == {
        "proposed": 1,
        "candidate_validated": 1,
        "translator_accepted": 1,
        "pysc2_accepted": 1,
        "effect_confirmed": 1,
    }
    assert metrics.build_effect_confirmed_rate == 1.0
    assert metrics.build_effect_timeout_rate == 0.0
    assert metrics.build_pre_dispatch_rejection_rate == 0.0
    assert metrics.builder_attack_commands == 0
    assert metrics.friendly_target_attacks == 1
    assert metrics.command_by_action_actor["Attack_Unit / CombatGroup/Zealot-1"] == 1
    assert metrics.failure_by_action_stage_code["Attack_Unit / pre_dispatch / friendly_target"] == 1
    assert metrics.planner_pending_decisions == 1
    assert metrics.idle_reason_counts == {"waiting_for_planner": 1}
    assert metrics.unique_validation_rejected_command_ids == 1


def test_execution_metrics_expose_hard_invariant_counters() -> None:
    events = [
        _event(
            1,
            "module_result",
            {
                "module": "planning",
                "output": {
                    "plan": {
                        "proposed_actions": [
                            {"actor": "global", "name": "No_Operation", "arguments": []}
                        ]
                    }
                },
            },
        ),
        _event(
            2,
            "execution",
            {
                "command_id": "build-1",
                "action_name": "Build_Pylon_Screen",
                "actor": "Builder/Probe-1",
                "success": False,
                "status": "failed",
                "execution_stage": "translation",
                "failure_code": "translator_rejected",
                "primitive_trace": [
                    {
                        "origin": "orchestration",
                        "requested_function_id": 573,
                        "emitted_function_id": 573,
                        "accepted": True,
                    }
                ],
            },
        ),
        _event(
            3,
            "episode_result",
            {
                "metrics": {
                    "transport_noop_primitives": 11,
                    "unattributed_primitives": 2,
                    "candidate_outside_pysc2_dispatches": 3,
                }
            },
        ),
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.planner_noop_proposals == 1
    assert metrics.generic_translation_failures == 1
    assert metrics.upstream_placement_rejections == 1
    assert metrics.unattributed_primitives == 2
    assert metrics.candidate_outside_pysc2_dispatches == 3
    assert metrics.orchestration_573_terminal_reports == 1


def test_raw_planner_proposals_drive_build_and_attack_safety_metrics() -> None:
    own_tag = "0xabc"
    observation: dict[str, object] = {
        "state": {
            "own_units": [{"unit_id": own_tag, "alliance": "self"}],
            "own_structures": [],
            "visible_enemies": [{"unit_id": "0xdef", "alliance": "enemy"}],
        }
    }
    dispatched_friendly_attack = {
        "command_id": "run-1:episode-1:1:planner:3",
        "name": "Attack_Unit",
        "actor": "CombatGroup/Zealot-1",
        "arguments": [own_tag],
        "source": "planner",
    }
    events = [
        _event(1, "observation", observation, step_id=1),
        _event(
            2,
            "module_result",
            {
                "module": "planning",
                "output": {
                    "plan": {
                        "proposed_actions": [
                            {
                                "actor": "Builder/Probe-1",
                                "name": "Build_Pylon_Screen",
                                "arguments": [[60, 40]],
                            },
                            {
                                "actor": "Builder/Probe-1",
                                "name": "Build_Gateway_Screen",
                                "arguments": [[64, 40]],
                            },
                            {
                                "actor": "Builder/Probe-1",
                                "name": "Attack_Unit",
                                "arguments": ["0xdef"],
                            },
                            {
                                "actor": "CombatGroup/Zealot-1",
                                "name": "Attack_Unit",
                                "arguments": [own_tag.upper()],
                            },
                        ]
                    }
                },
            },
            step_id=1,
        ),
        _event(
            3,
            "command_lifecycle",
            {"command": dispatched_friendly_attack, "status": "dispatched"},
        ),
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.build_funnel["proposed"] == 2
    assert metrics.planner_proposal_audit_complete is True
    assert metrics.planner_proposal_audited_results == metrics.planner_module_results == 1
    assert metrics.planner_unsafe_attack_proposals == 2
    assert metrics.planner_builder_attack_proposals == 1
    assert metrics.planner_friendly_target_attack_proposals == 1
    assert metrics.planner_unsafe_attack_rejected_before_dispatch == 1
    assert metrics.planner_unsafe_attack_dispatched == 1
    assert metrics.builder_attack_commands == 0
    assert metrics.friendly_target_attacks == 1


def test_attack_proposal_audit_is_incomplete_without_source_observation() -> None:
    events = [
        _event(
            1,
            "module_result",
            {
                "module": "planning",
                "output": {
                    "plan": {
                        "proposed_actions": [
                            {
                                "actor": "CombatGroup/Zealot-1",
                                "name": "Attack_Unit",
                                "arguments": ["0xabc"],
                            }
                        ]
                    }
                },
            },
        )
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.planner_module_results == 1
    assert metrics.planner_proposal_audited_results == 0
    assert metrics.planner_proposal_audit_complete is False


def test_aggregate_meaningful_rate_uses_global_counts() -> None:
    base = compute_episode_metrics(
        [],
        EpisodeResult(
            run_id="run-1",
            episode_id="episode-1",
            scenario="Simple64",
            seed=0,
            outcome=EpisodeOutcome.TRUNCATED,
        ),
    )
    perfect = replace(
        base,
        execution=replace(
            base.execution,
            meaningful_commands=1,
            meaningful_successes=1,
            meaningful_failures=0,
        ),
    )
    large_failure = replace(
        base,
        execution=replace(
            base.execution,
            meaningful_commands=9,
            meaningful_successes=0,
            meaningful_failures=9,
        ),
    )

    aggregate = aggregate_episode_metrics([perfect, large_failure])

    assert aggregate["execution"]["meaningful_action_success_rate"] == 0.1


def test_build_funnel_requires_a_final_attributed_translator_primitive() -> None:
    events = [
        _event(
            1,
            "decision",
            {
                "batch": {
                    "commands": [
                        {
                            "command_id": "cancelled-before-translation",
                            "name": "Build_Pylon_Screen",
                            "actor": "Builder/Probe-1",
                            "source": "planner",
                        },
                        {
                            "command_id": "accepted-before-end",
                            "name": "Build_Pylon_Screen",
                            "actor": "Builder/Probe-1",
                            "source": "planner",
                        },
                    ],
                    "rejected_commands": [],
                }
            },
        ),
        _event(
            2,
            "execution",
            {
                "command_id": "cancelled-before-translation",
                "action_name": "Build_Pylon_Screen",
                "success": False,
                "status": "cancelled",
                "execution_stage": "episode_end",
                "failure_code": "episode_ended",
                "primitive_trace": [
                    {
                        "function_name": "move_camera",
                        "origin": "orchestration",
                        "accepted": True,
                    }
                ],
            },
        ),
        _event(
            3,
            "execution",
            {
                "command_id": "accepted-before-end",
                "action_name": "Build_Pylon_Screen",
                "success": False,
                "status": "unconfirmed",
                "execution_stage": "episode_end",
                "failure_code": "episode_ended_unconfirmed",
                "primitive_trace": [
                    {
                        "function_name": "Build_Pylon_screen",
                        "origin": "translator",
                        "ordinal": 0,
                        "total": 1,
                        "accepted": True,
                    }
                ],
            },
        ),
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.build_funnel == {
        "proposed": 2,
        "candidate_validated": 2,
        "translator_accepted": 1,
        "pysc2_accepted": 1,
        "effect_confirmed": 0,
    }


def test_build_funnel_distinguishes_proposed_from_candidate_validated() -> None:
    proposed = [
        {
            "command_id": command_id,
            "name": "Build_Pylon_Screen",
            "actor": "Builder/Probe-1",
            "source": "planner",
        }
        for command_id in ("valid-build", "rejected-build")
    ]
    events = [
        _event(
            1,
            "decision",
            {
                "planner_candidates": proposed,
                "validated_candidates": [proposed[0]],
                "batch": {
                    "commands": [proposed[0]],
                    "rejected_commands": ["rejected-build: no legal placement"],
                },
            },
        )
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.build_funnel["proposed"] == 2
    assert metrics.build_funnel["candidate_validated"] == 1


def test_build_effect_timeout_and_predispatch_rates_use_build_command_denominators() -> None:
    commands = [
        {
            "command_id": command_id,
            "name": "Build_Pylon_Screen",
            "actor": "Builder/Probe-1",
            "source": "planner",
        }
        for command_id in ("effect-timeout", "pre-dispatch")
    ]
    events = [
        _event(
            1,
            "decision",
            {
                "planner_candidates": commands,
                "validated_candidates": commands,
                "batch": {"commands": commands, "rejected_commands": []},
            },
        ),
        _event(
            2,
            "execution",
            {
                **commands[0],
                "success": False,
                "status": "failed",
                "execution_stage": "effect_verification",
                "failure_code": "target_not_created",
                "primitive_trace": [
                    {
                        "function_name": "Build_Pylon_screen",
                        "origin": "translator",
                        "ordinal": 0,
                        "total": 1,
                        "accepted": True,
                    }
                ],
            },
        ),
        _event(
            3,
            "execution",
            {
                **commands[1],
                "success": False,
                "status": "failed",
                "execution_stage": "pre_dispatch",
                "failure_code": "no_legal_placement",
            },
        ),
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.build_funnel["pysc2_accepted"] == 1
    assert metrics.build_effect_timeouts == 1
    assert metrics.build_effect_timeout_rate == 1.0
    assert metrics.build_effect_confirmed_rate == 0.0
    assert metrics.build_pre_dispatch_rejections == 1
    assert metrics.build_pre_dispatch_rejection_rate == 0.5


def test_terminal_coverage_uses_dispatched_lifecycle_not_execution_count() -> None:
    command_a = {"command_id": "a", "name": "Attack_Unit"}
    command_b = {"command_id": "b", "name": "Attack_Unit"}
    events = [
        _event(1, "command_lifecycle", {"command": command_a, "status": "dispatched"}),
        _event(2, "command_lifecycle", {"command": command_b, "status": "dispatched"}),
        _event(3, "command_lifecycle", {"command": command_a, "status": "dispatched"}),
        _event(
            4,
            "execution",
            {
                "command_id": "a",
                "action_name": "Attack_Unit",
                "success": False,
                "status": "failed",
                "execution_stage": "pysc2_acceptance",
                "failure_code": "pysc2_rejected",
            },
        ),
        _event(
            5,
            "execution",
            {
                "command_id": "a",
                "action_name": "Attack_Unit",
                "success": False,
                "status": "failed",
                "execution_stage": "pysc2_acceptance",
                "failure_code": "pysc2_rejected",
            },
        ),
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.dispatched_commands == 2
    assert metrics.terminal_commands_reported == 1
    assert metrics.missing_terminal_reports == 1
    assert metrics.duplicate_terminal_reports == 1
    assert metrics.duplicate_dispatches == 1
    assert metrics.terminal_report_coverage == 0.5
    assert metrics.failure_classification_coverage == 1.0


def test_v11_pending_episode_end_cancellation_is_not_counted_as_dispatched() -> None:
    command = {
        "command_id": "pending",
        "name": "Build_Pylon_Screen",
        "actor": "Builder/Probe-1",
        "arguments": [[60, 40]],
        "source": "planner",
    }
    events = [
        _event(1, "command_lifecycle", {"command": command, "status": "pending"}),
        _event(
            2,
            "execution",
            {
                "protocol_version": "1.1",
                "command_id": "pending",
                "action_name": "Build_Pylon_Screen",
                "actor": "Builder/Probe-1",
                "source": "planner",
                "requested_arguments": [[60, 40]],
                "success": False,
                "status": "cancelled",
                "execution_stage": "episode_end",
                "failure_code": "episode_ended_before_dispatch",
            },
        ),
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.known_lifecycle_commands == 1
    assert metrics.dispatched_commands == 0
    assert metrics.terminal_commands_reported == 0
    assert metrics.unexpected_terminal_reports == 0


def test_unexpected_terminal_report_is_counted_explicitly() -> None:
    command = {"command_id": "known", "name": "Attack_Unit", "actor": "CombatGroup/a"}
    report = {
        "protocol_version": "1.1",
        "action_name": "Attack_Unit",
        "actor": "CombatGroup/a",
        "source": "planner",
        "success": False,
        "status": "failed",
        "execution_stage": "pysc2_acceptance",
        "failure_code": "pysc2_rejected",
    }
    events = [
        _event(1, "command_lifecycle", {"command": command, "status": "dispatched"}),
        _event(2, "execution", {"command_id": "known", **report}),
        _event(3, "execution", {"command_id": "rogue", **report}),
    ]

    metrics = compute_execution_metrics(events)

    assert metrics.known_lifecycle_commands == 1
    assert metrics.dispatched_commands == 1
    assert metrics.execution_reports == 2
    assert metrics.unexpected_terminal_reports == 1


def test_legacy_full_match_characterization_uses_semantic_command_names() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "legacy_full_match_characterization.jsonl"
    metrics = compute_execution_metrics(list(read_event_log(fixture)))

    assert metrics.decision_count == 946
    assert metrics.fallback_decisions == 754
    assert metrics.execution_reports == 903
    assert metrics.dispatched_commands == 903
    assert metrics.legacy_successes == 759
    assert metrics.legacy_action_success_rate == pytest.approx(759 / 903)
    assert metrics.control_noops == 782
    assert metrics.control_noop_successes == 729
    assert metrics.meaningful_commands == 121
    assert metrics.meaningful_successes == 30
    assert metrics.meaningful_failures == 73
    assert metrics.meaningful_cancelled == 18
    assert metrics.meaningful_action_success_rate == pytest.approx(30 / 121)
    assert metrics.completed_meaningful_commands == 103
    assert metrics.completed_execution_success_rate == pytest.approx(30 / 103)
    assert metrics.status_counts["cancelled"] == 71
    assert metrics.unique_validation_rejected_command_ids == 39
    assert metrics.planner_noop_proposals == 32
    assert metrics.generic_translation_failures == 72
