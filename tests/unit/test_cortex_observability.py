from __future__ import annotations

import pytest

from rtscortex.evaluation import compute_cortex_observability, render_timeline
from rtscortex.memory import StoredEvent


def _event(event_id: int, event_type: str, payload: dict[str, object]) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="cortex-run",
        episode_id="episode-0",
        step_id=event_id,
        event_type=event_type,
        created_at=f"2026-01-01T00:00:{event_id:02d}+00:00",
        payload=payload,
    )


def test_cortex_observability_projects_lineage_and_executor_invariants() -> None:
    events = [
        _event(0, "macro_plan_accepted", {"plan_id": "plan-1"}),
        _event(
            1,
            "candidate_set_built",
            {
                "intent_id": "intent-1",
                "candidates": [
                    {"candidate_id": "candidate-1", "action_name": "Build_Pylon_Screen"},
                    {"candidate_id": "candidate-2", "action_name": "Build_Gateway_Screen"},
                ],
            },
        ),
        _event(
            2,
            "intent_emitted",
            {"intent_id": "intent-1", "role": "macro", "action_name": "Build_Pylon_Screen"},
        ),
        _event(
            3,
            "executor_selection",
            {
                "executor_id": "deterministic",
                "intent_id": "intent-1",
                "selection_id": "selection-1",
                "selected_candidate_id": "candidate-1",
                "latency_ms": 1.0,
            },
        ),
        _event(
            4,
            "executor_selection",
            {
                "executor_id": "tiny-ranker",
                "intent_id": "intent-1",
                "selection_id": "selection-2",
                "selected_candidate_id": "outside-domain",
                "latency_ms": 3.0,
                "fallback": True,
            },
        ),
        _event(
            5,
            "executor_selection",
            {"executor_id": "deterministic", "status": "abstain"},
        ),
        _event(
            6,
            "command_lifecycle",
            {"command": {"command_id": "command-1"}, "status": "dispatched"},
        ),
        _event(
            7,
            "command_lifecycle",
            {"command_id": "command-2", "status": "dispatched"},
        ),
        _event(
            8,
            "command_lineage",
            {
                "command_id": "command-1",
                "intent_id": "intent-1",
                "candidate_id": "candidate-1",
                "selection_id": "selection-1",
                "macro_plan_id": "plan-1",
                "source_role": "macro",
            },
        ),
        _event(9, "specialist_failed", {"role": "macro", "model_id": "hima-a"}),
        _event(10, "specialist_recovered", {"role": "macro", "model_id": "hima-a"}),
        _event(
            11,
            "situation_assessed",
            {
                "phase": "combat",
                "threat_level": "critical",
                "threat_score": 9.5,
                "threat_evidence": ["building_under_attack", "empty_army_overwhelmed"],
            },
        ),
    ]

    metrics = compute_cortex_observability(events)

    assert metrics.observed is True
    assert metrics.intent_counts == {"macro": 1}
    assert metrics.executor_counts == {"deterministic": 2, "tiny-ranker": 1}
    assert metrics.executor_selections == 2
    assert metrics.executor_abstentions == 1
    assert metrics.executor_fallbacks == 1
    assert metrics.executor_candidate_violations == 1
    assert metrics.lineage_integrity_violations == 0
    assert metrics.executor_latency_ms_p50 == 2.0
    assert metrics.executor_latency_ms_p95 == pytest.approx(2.9)
    assert metrics.dispatched_commands == 2
    assert metrics.lineage_commands == 1
    assert metrics.missing_lineage_commands == 1
    assert metrics.orphan_lineage_commands == 0
    assert metrics.command_lineage_coverage == 0.5
    assert metrics.specialist_failure_counts == {"macro": 1}
    assert metrics.specialist_recovery_counts == {"macro": 1}
    assert metrics.threat_level_counts == {"critical": 1}
    assert metrics.max_threat_score == 9.5
    assert metrics.threat_evidence_coverage == 1.0
    assert metrics.threat_evidence_counts == {
        "building_under_attack": 1,
        "empty_army_overwhelmed": 1,
    }


def test_cortex_timeline_renders_semantic_pipeline_events() -> None:
    events = [
        _event(
            1,
            "situation_assessed",
            {
                "source_kind": "deterministic",
                "assessment": {
                    "game_phase": "early",
                    "threat_level": "low",
                    "threat_score": 1.5,
                    "threat_evidence": ["visible_enemy_contact"],
                    "army_readiness": "not_ready",
                },
            },
        ),
        _event(
            2,
            "macro_plan_accepted",
            {
                "model_id": "hima-a",
                "plan": {
                    "plan_id": "plan-1",
                    "steps": [{"action": "Pylon"}, {"action": "Gateway"}],
                },
                "runtime_frontier": "Build_Pylon_Screen",
            },
        ),
        _event(
            3,
            "intent_emitted",
            {
                "intent_id": "intent-1",
                "role": "macro",
                "intent": {"action_names": ["Build_Pylon_Screen"]},
            },
        ),
        _event(
            4,
            "candidate_set_built",
            {
                "intent_id": "intent-1",
                "candidates": [
                    {"candidate_id": "candidate-1", "action_name": "Build_Pylon_Screen"}
                ],
            },
        ),
        _event(
            5,
            "executor_selection",
            {
                "executor_id": "deterministic",
                "intent_id": "intent-1",
                "selection_id": "selection-1",
                "selected_candidate_id": "candidate-1",
                "latency_ms": 1.5,
            },
        ),
        _event(
            6,
            "command_lineage",
            {
                "command_id": "command-1",
                "macro_plan_id": "plan-1",
                "intent_id": "intent-1",
                "candidate_id": "candidate-1",
                "selection_id": "selection-1",
                "source_role": "macro",
            },
        ),
        _event(
            7,
            "macro_plan_rejected",
            {"plan_id": "plan-2", "model_id": "hima-a", "reason": "parse_error"},
        ),
        _event(
            8,
            "specialist_failed",
            {"role": "macro", "model_id": "hima-a", "message": "request timed out"},
        ),
        _event(9, "specialist_recovered", {"role": "macro", "model_id": "hima-a"}),
    ]

    report = render_timeline(events)

    assert "### Cortex observability" in report
    assert "`hima`/`hima-a`" in report
    assert "| 0 | 0 | 1 | 0/0 | 0 | 2 | 0 |" in report
    assert "Situation assessed by `deterministic`: phase `early`" in report
    assert "score `1.5`" in report
    assert "evidence `visible_enemy_contact`" in report
    assert "Macro plan `plan-1` from `hima-a` accepted with `2` steps" in report
    assert "macro intent `intent-1`: `Build_Pylon_Screen`" in report
    assert "Built `1` executable candidates" in report
    assert "Executor `deterministic` selected `candidate-1` in 1.50 ms" in report
    assert "plan `plan-1` → intent `intent-1` → candidate `candidate-1`" in report
    assert "Macro plan `plan-2` from `hima-a` rejected: parse_error" in report
    assert "Specialist `macro`/`hima-a` failed: request timed out" in report
    assert "Specialist `macro`/`hima-a` recovered" in report


def test_legacy_journal_has_non_applicable_cortex_projection() -> None:
    event = _event(1, "planner_cycle", {"status": "completed", "latency_ms": 1.0})

    metrics = compute_cortex_observability([event])
    report = render_timeline([event])

    assert metrics.observed is False
    assert metrics.as_dict()["event_counts"] == {}
    assert "### Cortex observability" not in report


def test_cortex_invariants_are_scoped_to_the_intent_and_selection() -> None:
    events = [
        _event(
            1,
            "candidate_set_built",
            {
                "intent_id": "intent-a",
                "candidates": [{"candidate_id": "candidate-shared"}],
            },
        ),
        _event(
            2,
            "candidate_set_built",
            {
                "intent_id": "intent-b",
                "candidates": [{"candidate_id": "candidate-b"}],
            },
        ),
        _event(
            3,
            "executor_selection",
            {
                "intent_id": "intent-b",
                "selection_id": "selection-b",
                "selected_candidate_id": "candidate-shared",
                "executor_id": "broken-executor",
            },
        ),
        _event(
            4,
            "command_lineage",
            {
                "command_id": "command-b",
                "intent_id": "intent-b",
                "selection_id": "selection-b",
                "candidate_id": "candidate-shared",
            },
        ),
        _event(
            5,
            "command_lifecycle",
            {"command_id": "command-b", "status": "dispatched"},
        ),
    ]

    metrics = compute_cortex_observability(events)

    assert metrics.executor_candidate_violations == 1
    assert metrics.lineage_integrity_violations == 1
    assert metrics.missing_lineage_commands == 1
    assert metrics.command_lineage_coverage == 0.0


def test_duplicate_command_lineage_is_not_folded_into_valid_coverage() -> None:
    lineage: dict[str, object] = {
        "command_id": "command-a",
        "intent_id": "intent-a",
        "selection_id": "selection-a",
        "candidate_id": "candidate-a",
        "source_role": "reflex",
        "macro_plan_id": None,
    }
    events = [
        _event(
            0,
            "intent_emitted",
            {"intent_id": "intent-a", "role": "reflex"},
        ),
        _event(
            1,
            "candidate_set_built",
            {
                "intent_id": "intent-a",
                "candidates": [{"candidate_id": "candidate-a"}],
            },
        ),
        _event(
            2,
            "executor_selection",
            {
                "intent_id": "intent-a",
                "selection_id": "selection-a",
                "selected_candidate_id": "candidate-a",
                "executor_id": "executor-a",
            },
        ),
        _event(3, "command_lineage", lineage),
        _event(4, "command_lineage", lineage),
        _event(
            5,
            "command_lifecycle",
            {"command_id": "command-a", "status": "dispatched"},
        ),
    ]

    metrics = compute_cortex_observability(events)

    assert metrics.duplicate_lineage_commands == 1
    assert metrics.lineage_integrity_violations == 1
    assert metrics.missing_lineage_commands == 1
    assert metrics.command_lineage_coverage == 0.0


def test_macro_lineage_requires_emitted_intent_and_accepted_plan() -> None:
    events = [
        _event(
            1,
            "candidate_set_built",
            {
                "intent_id": "intent-a",
                "candidates": [{"candidate_id": "candidate-a"}],
            },
        ),
        _event(
            2,
            "executor_selection",
            {
                "intent_id": "intent-a",
                "selection_id": "selection-a",
                "selected_candidate_id": "candidate-a",
                "executor_id": "executor-a",
            },
        ),
        _event(
            3,
            "command_lineage",
            {
                "command_id": "command-a",
                "intent_id": "intent-a",
                "selection_id": "selection-a",
                "candidate_id": "candidate-a",
                "source_role": "macro",
                "macro_plan_id": "plan-ghost",
            },
        ),
        _event(
            4,
            "command_lifecycle",
            {"command_id": "command-a", "status": "dispatched"},
        ),
    ]

    metrics = compute_cortex_observability(events)

    assert metrics.lineage_integrity_violations == 1
    assert metrics.missing_lineage_commands == 1
    assert metrics.command_lineage_coverage == 0.0
