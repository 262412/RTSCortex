from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from rtscortex.memory import StoredEvent
from rtscortex.playbook import (
    PlaybookCondition,
    PlaybookRetryGuardBinding,
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    analyze_typed_retry_coverage,
    playbook_predicate_fingerprint,
    playbook_rule_fingerprint,
)


def _rule() -> PlaybookRule:
    return PlaybookRule(
        rule_id="playbook-rule:retry",
        canonical_key="retry",
        category=PlaybookRuleCategory.EXECUTION_GUARD,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.AVOID,
        strength=PlaybookRuleStrength.SOFT,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("Attack_Unit",),
        confidence=0.95,
        retry_guard=PlaybookRetryGuardBinding(
            failure_code="combat_actor_order_unbound",
            max_age_game_loops=112,
        ),
    )


def _events(
    *,
    signature: str | None = None,
    feedback_signature: str | None = None,
    operation_id: str = "operation:" + "1" * 64,
    lineage_operation_id: str | None = None,
    attempt_ordinal: int = 1,
    lineage_attempt_ordinal: int | None = None,
    game_loop: int = 60,
    terminal_game_loop: int = 40,
    feedback_failure_code: str = "combat_actor_order_unbound",
    predicate_fingerprint: str | None = None,
    include_lineage: bool = True,
) -> tuple[StoredEvent, ...]:
    rule = _rule()
    rule_fingerprint = playbook_rule_fingerprint(rule)
    predicate_fingerprint = predicate_fingerprint or playbook_predicate_fingerprint(rule)
    signature = signature or "a" * 64
    feedback_signature = feedback_signature or signature
    events: list[StoredEvent] = [
        StoredEvent(
            event_id=1,
            run_id="run",
            episode_id="episode-0",
            step_id=1,
            event_type="playbook_case_recorded",
            created_at="2026-08-09T00:00:00+00:00",
            payload={
                "command_id": "command:failure",
                "semantic_action": "Attack_Unit",
                "retry_feedback": {
                    "action_name": "Attack_Unit",
                    "failure_code": feedback_failure_code,
                    "candidate_signature": feedback_signature,
                    "operation_id": operation_id,
                    "attempt_ordinal": 0,
                    "terminal_game_loop": terminal_game_loop,
                    "max_age_game_loops": 112,
                },
            },
        ),
    ]
    if include_lineage:
        events.append(
            StoredEvent(
                event_id=2,
                run_id="run",
                episode_id="episode-0",
                step_id=2,
                event_type="command_lineage",
                created_at="2026-08-09T00:00:00+00:00",
                payload={
                    "command_id": "command:retry",
                    "lineage": {
                        "candidate_id": "candidate:retry",
                        "operation_id": lineage_operation_id or operation_id,
                        "attempt_ordinal": (
                            attempt_ordinal
                            if lineage_attempt_ordinal is None
                            else lineage_attempt_ordinal
                        ),
                        "selected_game_loop": game_loop,
                    },
                },
            )
        )
    events.append(
        StoredEvent(
            event_id=3,
            run_id="run",
            episode_id="episode-0",
            step_id=3,
            event_type="playbook_rule_applied",
            created_at="2026-08-09T00:00:00+00:00",
            payload={
                "application_id": "application:retry",
                "rule_id": rule.rule_id,
                "run_id": "run",
                "episode_id": "episode-0",
                "step_id": 3,
                "game_loop": game_loop,
                "target_kind": "candidate",
                "target_id": "candidate:retry",
                "action_name": "Attack_Unit",
                "counterfactual_signature": signature,
                "rule_fingerprint": rule_fingerprint,
                "predicate_fingerprint": predicate_fingerprint,
                "matched": True,
                "blocked": False,
                "reason": "rule_scored",
            },
        )
    )
    return tuple(events)


def test_typed_coverage_requires_runtime_identity_and_counts_opportunity() -> None:
    rule = _rule()
    coverage = analyze_typed_retry_coverage(
        _events(), rule=rule, context_values={"agent_race": "protoss"}
    )

    assert coverage.typed_retry_opportunity_count == 1
    assert coverage.typed_retry_application_count == 1
    assert coverage.typed_retry_coverage_unavailable is False
    assert coverage.as_dict()["opportunity_bindings"] == [
        {
            "opportunity_id": f"operation:{'1' * 64}|{'a' * 64}|1|60",
            "failure_code": "combat_actor_order_unbound",
            "candidate_signature": "a" * 64,
            "operation_id": "operation:" + "1" * 64,
            "next_attempt_ordinal": 1,
            "candidate_game_loop": 60,
            "terminal_game_loop": 40,
            "freshness_age_game_loops": 20,
            "max_age_game_loops": 112,
            "rule_fingerprint": playbook_rule_fingerprint(rule),
            "predicate_fingerprint": playbook_predicate_fingerprint(rule),
        }
    ]


def test_typed_application_binds_lineage_recorded_after_guard_evaluation() -> None:
    events = list(_events())
    lineage_index = next(
        index for index, event in enumerate(events) if event.event_type == "command_lineage"
    )
    application_index = next(
        index for index, event in enumerate(events) if event.event_type == "playbook_rule_applied"
    )
    events[lineage_index] = replace(events[lineage_index], event_id=4)
    events[application_index] = replace(events[application_index], event_id=2)

    coverage = analyze_typed_retry_coverage(
        events,
        rule=_rule(),
        context_values={"agent_race": "protoss"},
    )

    assert coverage.typed_retry_opportunity_count == 1
    assert coverage.typed_retry_application_count == 1
    assert coverage.typed_retry_coverage_unavailable is False


def test_typed_coverage_does_not_use_broad_shadow_count_for_signature_mismatch() -> None:
    coverage = analyze_typed_retry_coverage(
        _events(signature="b" * 64, feedback_signature="a" * 64),
        rule=_rule(),
        context_values={"agent_race": "protoss"},
    )

    assert coverage.typed_retry_opportunity_count == 0
    assert coverage.typed_retry_application_count == 0


@pytest.mark.parametrize(
    "event_kwargs",
    [
        {"feedback_failure_code": "different_failure"},
        {"lineage_operation_id": "operation:" + "2" * 64},
        {"lineage_attempt_ordinal": 2},
        {"game_loop": 200},
        {"game_loop": 30},
    ],
)
def test_typed_coverage_rejects_each_runtime_identity_mismatch(
    event_kwargs: dict[str, Any],
) -> None:
    coverage = analyze_typed_retry_coverage(
        _events(**event_kwargs),
        rule=_rule(),
        context_values={"agent_race": "protoss"},
    )

    assert coverage.typed_retry_opportunity_count == 0
    assert coverage.typed_retry_application_count == 0


def test_typed_coverage_fails_closed_when_operation_lineage_is_missing() -> None:
    coverage = analyze_typed_retry_coverage(
        _events(include_lineage=False),
        rule=_rule(),
        context_values={"agent_race": "protoss"},
    )

    assert coverage.typed_retry_opportunity_count == 0
    assert coverage.typed_retry_application_count == 0
    assert coverage.typed_retry_coverage_unavailable is True
    assert "missing_operation_lineage" in coverage.unavailable_reasons


def test_typed_coverage_fingerprint_mismatch_is_not_an_opportunity() -> None:
    coverage = analyze_typed_retry_coverage(
        _events(predicate_fingerprint="f" * 64),
        rule=_rule(),
        context_values={"agent_race": "protoss"},
    )

    assert coverage.typed_retry_opportunity_count == 0
    assert coverage.typed_retry_application_count == 0
    assert coverage.typed_retry_coverage_unavailable is True
    assert "predicate_fingerprint_mismatch" in coverage.unavailable_reasons


def test_unrelated_rule_application_cannot_poison_parent_coverage() -> None:
    unrelated = StoredEvent(
        event_id=99,
        run_id="run",
        episode_id="episode-0",
        step_id=99,
        event_type="playbook_rule_applied",
        created_at="2026-08-09T00:00:00+00:00",
        payload={
            "application_id": "application:unrelated",
            "rule_id": "playbook-rule:unrelated",
            "target_kind": "candidate",
            "target_id": "candidate:unrelated",
            "counterfactual_signature": "not-a-fingerprint",
            "rule_fingerprint": "not-a-fingerprint",
            "predicate_fingerprint": "not-a-fingerprint",
        },
    )
    coverage = analyze_typed_retry_coverage(
        (*_events(), unrelated),
        rule=_rule(),
        context_values={"agent_race": "protoss"},
    )

    assert coverage.typed_retry_opportunity_count == 1
    assert coverage.typed_retry_application_count == 1
    assert coverage.typed_retry_coverage_unavailable is False


def test_typed_coverage_replays_candidate_set_without_rule_application() -> None:
    rule = _rule()
    operation_id = "operation:" + "2" * 64
    events = (
        StoredEvent(
            event_id=1,
            run_id="run",
            episode_id="episode-0",
            step_id=10,
            event_type="situation_assessed",
            created_at="2026-08-09T00:00:00+00:00",
            payload={"assessment_id": "assessment:1", "phase": "combat"},
        ),
        StoredEvent(
            event_id=2,
            run_id="run",
            episode_id="episode-0",
            step_id=10,
            event_type="candidate_set_built",
            created_at="2026-08-09T00:00:00+00:00",
            payload={
                "role": "offense",
                "candidates": [
                    {
                        "candidate_id": "candidate:retry",
                        "action_name": "Attack_Unit",
                        "actor": "CombatGroup7/Adept-1",
                        "arguments": ["enemy:1"],
                    }
                ],
            },
        ),
        StoredEvent(
            event_id=3,
            run_id="run",
            episode_id="episode-0",
            step_id=10,
            event_type="execution",
            created_at="2026-08-09T00:00:00+00:00",
            payload={
                "command_id": "command:failure",
                "action_name": "Attack_Unit",
                "actor": "CombatGroup7/Adept-1",
                "failure_code": "combat_actor_order_unbound",
                "operation_id": operation_id,
                "attempt_ordinal": 0,
                "requested_arguments": ["enemy:1"],
                "step_id": 40,
            },
        ),
        StoredEvent(
            event_id=4,
            run_id="run",
            episode_id="episode-0",
            step_id=20,
            event_type="command_lineage",
            created_at="2026-08-09T00:00:00+00:00",
            payload={
                "lineage": {
                    "candidate_id": "candidate:retry",
                    "operation_id": operation_id,
                    "attempt_ordinal": 1,
                    "situation_assessment_id": "assessment:1",
                    "selected_game_loop": 60,
                }
            },
        ),
    )
    coverage = analyze_typed_retry_coverage(
        events,
        rule=rule,
        context_values={"agent_race": "protoss"},
    )
    assert coverage.typed_retry_opportunity_count == 1
    assert coverage.typed_retry_application_count == 0
    assert coverage.typed_retry_coverage_unavailable is False


def test_execution_feedback_uses_effect_terminal_game_loop_for_freshness() -> None:
    rule = _rule()
    operation_id = "operation:" + "4" * 64
    events = list(
        _events(
            include_lineage=False,
        )
    )
    events = [
        event
        for event in events
        if event.event_type != "playbook_case_recorded"
        and event.event_type != "playbook_rule_applied"
    ]
    events.extend(
        [
            StoredEvent(
                event_id=1,
                run_id="run",
                episode_id="episode-0",
                step_id=10,
                event_type="candidate_set_built",
                created_at="2026-08-09T00:00:00+00:00",
                payload={
                    "role": "offense",
                    "candidates": [
                        {
                            "candidate_id": "candidate:retry",
                            "action_name": "Attack_Unit",
                            "actor": "CombatGroup7/Adept-1",
                            "arguments": ["enemy:1"],
                        }
                    ],
                },
            ),
            StoredEvent(
                event_id=2,
                run_id="run",
                episode_id="episode-0",
                step_id=40,
                event_type="execution",
                created_at="2026-08-09T00:00:00+00:00",
                payload={
                    "action_name": "Attack_Unit",
                    "actor": "CombatGroup7/Adept-1",
                    "failure_code": "combat_actor_order_unbound",
                    "operation_id": operation_id,
                    "attempt_ordinal": 0,
                    "requested_arguments": ["enemy:1"],
                    "step_id": 40,
                    "effect_evidence": {"accepted_game_loop": 41},
                },
            ),
            StoredEvent(
                event_id=3,
                run_id="run",
                episode_id="episode-0",
                step_id=50,
                event_type="command_lineage",
                created_at="2026-08-09T00:00:00+00:00",
                payload={
                    "lineage": {
                        "candidate_id": "candidate:retry",
                        "operation_id": operation_id,
                        "attempt_ordinal": 1,
                        "selected_game_loop": 153,
                    }
                },
            ),
        ]
    )
    coverage = analyze_typed_retry_coverage(
        tuple(events), rule=rule, context_values={"agent_race": "protoss"}
    )

    assert coverage.typed_retry_opportunity_count == 1


def test_successful_bound_action_is_not_incomplete_terminal_feedback() -> None:
    events = _events()
    success = StoredEvent(
        event_id=4,
        run_id="run",
        episode_id="episode-0",
        step_id=4,
        event_type="execution",
        created_at="2026-08-09T00:00:00+00:00",
        payload={
            "action_name": "Attack_Unit",
            "actor": "CombatGroup7/Adept-1",
            "status": "succeeded",
            "failure_code": None,
            "operation_id": "operation:" + "3" * 64,
            "attempt_ordinal": 0,
            "requested_arguments": ["enemy:1"],
            "step_id": 50,
        },
    )
    coverage = analyze_typed_retry_coverage(
        (*events, success),
        rule=_rule(),
        context_values={"agent_race": "protoss"},
    )
    assert coverage.typed_retry_coverage_unavailable is False
    assert "incomplete_terminal_feedback" not in coverage.unavailable_reasons


def test_typed_coverage_fails_closed_when_dynamic_predicate_context_is_missing() -> None:
    original = _rule()
    rule = _rule().model_copy(
        update={"conditions": (PlaybookCondition(field="threat_level", value="high"),)}
    )
    coverage = analyze_typed_retry_coverage(
        _events(),
        rule=rule,
        expected_rule_fingerprint=playbook_rule_fingerprint(original),
        expected_predicate_fingerprint=playbook_predicate_fingerprint(original),
        context_values={"agent_race": "protoss"},
    )
    assert coverage.typed_retry_opportunity_count == 0
    assert coverage.typed_retry_coverage_unavailable is True
    assert "missing_retry_predicate_inputs" in coverage.unavailable_reasons
