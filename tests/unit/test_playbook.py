from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rtscortex.contracts import (
    ActionSource,
    EpisodeOutcome,
    EpisodeResult,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
)
from rtscortex.cortex import (
    ArmyReadiness,
    EconomyStatus,
    GamePhase,
    ResourceClaim,
    RoleId,
    SituationAssessment,
    StrategicIntent,
    ThreatLevel,
)
from rtscortex.memory import EventStore
from rtscortex.playbook import (
    CortexPlaybookReviewer,
    DecisionQuality,
    LessonStatus,
    PlaybookCondition,
    PlaybookContext,
    PlaybookIntentGuard,
    PlaybookQuery,
    PlaybookRule,
    PlaybookRuleApplication,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleKind,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    PlaybookStore,
)


def test_playbook_public_import_succeeds_in_cold_interpreter() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "from rtscortex.playbook import CortexPlaybookReviewer"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _episode_events(
    root: Path,
    run_id: str,
    *,
    seed: int = 0,
) -> tuple[EventStore, EpisodeResult]:
    store = EventStore(root / f"{run_id}.sqlite3", root / f"{run_id}.jsonl")
    store.append_event(
        run_id=run_id,
        episode_id="episode",
        step_id=100,
        event_type="situation_assessed",
        payload={
            "game_loop": 100,
            "phase": "technology",
            "threat_level": "none",
            "economy_status": "stable",
            "army_readiness": "forming",
            "own_force": {"estimated_resource_value": 200, "total_units": 2},
            "visible_enemy_force": {"estimated_resource_value": 0, "total_units": 0},
            "bases": {"own_base_count": 1, "own_production_capacity": 1},
            "scouting": {"enemy_visible": False},
        },
    )
    store.append_event(
        run_id=run_id,
        episode_id="episode",
        step_id=120,
        event_type="command_lineage",
        payload={
            "command_id": f"{run_id}:command",
            "macro_plan_id": f"{run_id}:plan",
            "semantic_action": "BUILD STARGATE",
            "lineage": {
                "source_role": "macro",
                "responsibility": "technology",
                "selected_game_loop": 120,
            },
        },
    )
    store.append_event(
        run_id=run_id,
        episode_id="episode",
        step_id=121,
        event_type="execution",
        payload=ExecutionReport(
            run_id=run_id,
            episode_id="episode",
            step_id=121,
            command_id=f"{run_id}:command",
            success=True,
            action_name="Build_Stargate_Screen",
            actor="Builder/Probe-1",
            source=ActionSource.PLANNER,
            requested_arguments=[[65, 90]],
            status=ExecutionStatus.SUCCEEDED,
            execution_stage=ExecutionStage.EFFECT_VERIFICATION,
        ),
    )
    store.append_event(
        run_id=run_id,
        episode_id="episode",
        step_id=620,
        event_type="situation_assessed",
        payload={
            "game_loop": 620,
            "phase": "technology",
            "threat_level": "none",
            "economy_status": "stable",
            "army_readiness": "forming",
            "own_force": {"estimated_resource_value": 550, "total_units": 4},
            "visible_enemy_force": {"estimated_resource_value": 0, "total_units": 0},
            "bases": {"own_base_count": 1, "own_production_capacity": 2},
            "scouting": {"enemy_visible": False},
        },
    )
    return store, EpisodeResult(
        run_id=run_id,
        episode_id="episode",
        scenario="Simple64",
        seed=seed,
        outcome=EpisodeOutcome.VICTORY,
        steps=620,
    )


def test_playbook_promotes_only_repeated_outcome_backed_experience(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=2)

    first_store, first_result = _episode_events(tmp_path, "run-1")
    first_cases, first_lessons = reviewer.review_episode(
        first_store.events_after("run-1", 0, 100, episode_id="episode"),
        first_result,
        agent_race="protoss",
        opponent_race="zerg",
    )
    second_store, second_result = _episode_events(tmp_path, "run-2", seed=1)
    _, second_lessons = reviewer.review_episode(
        second_store.events_after("run-2", 0, 100, episode_id="episode"),
        second_result,
        agent_race="protoss",
        opponent_race="zerg",
    )

    assert first_cases[0].quality is DecisionQuality.ADVANTAGE_GAINED
    assert first_lessons[0].status is LessonStatus.CANDIDATE
    assert second_lessons[0].status is LessonStatus.PROMOTED
    selection = playbook.retrieve(
        PlaybookQuery(
            context=PlaybookContext(
                agent_race="protoss",
                opponent_race="zerg",
                phase=GamePhase.TECHNOLOGY,
                map_name="Simple64",
            )
        )
    )
    assert selection.lesson_ids == (second_lessons[0].lesson_id,)
    candidate_rules = [
        rule for rule in playbook.rules() if rule.status is PlaybookRuleStatus.CANDIDATE
    ]
    assert len(candidate_rules) == 1
    assert candidate_rules[0].strength is PlaybookRuleStrength.ADVISORY
    assert candidate_rules[0].source_run_ids == ("run-1", "run-2")

    first_store.close()
    second_store.close()
    playbook.close()


def _persistent_threat_episode(
    root: Path,
    run_id: str,
    *,
    seed: int,
) -> tuple[EventStore, EpisodeResult]:
    store = EventStore(root / f"{run_id}.sqlite3", root / f"{run_id}.jsonl")
    for step_id, game_loop in ((10, 1_000), (20, 1_224)):
        store.append_event(
            run_id=run_id,
            episode_id="episode",
            step_id=step_id,
            event_type="situation_assessed",
            payload={
                "game_loop": game_loop,
                "phase": "combat",
                "threat_level": "high",
                "economy_status": "stable",
                "army_readiness": "ready",
                "own_force": {"estimated_resource_value": 800, "total_units": 8},
                "visible_enemy_force": {
                    "estimated_resource_value": 700,
                    "total_units": 7,
                },
                "bases": {"own_base_count": 2, "own_production_capacity": 4},
                "scouting": {"enemy_visible": True},
            },
        )
    return store, EpisodeResult(
        run_id=run_id,
        episode_id="episode",
        scenario="Simple64",
        seed=seed,
        outcome=EpisodeOutcome.DEFEAT,
        steps=20,
    )


def test_strategic_consequence_iterates_into_next_match_intent_scoring(
    tmp_path: Path,
) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=2)
    for run_id, seed in (("threat-run-1", 0), ("threat-run-2", 0)):
        store, result = _persistent_threat_episode(tmp_path, run_id, seed=seed)
        reviewer.review_episode(
            store.events_after(run_id, 0, 100, episode_id="episode"),
            result,
            agent_race="protoss",
            opponent_race="zerg",
        )
        store.close()

    repeated_same_seed = next(
        rule
        for rule in playbook.rules()
        if rule.evidence.get("consequence_type") == "threat_unanswered"
    )
    assert repeated_same_seed.status is PlaybookRuleStatus.CANDIDATE
    repeated_lesson = next(
        lesson
        for lesson in playbook.lessons()
        if lesson.consequence_type is not None
        and lesson.consequence_type.value == "threat_unanswered"
    )
    assert repeated_lesson.status is LessonStatus.CANDIDATE

    for index in range(48):
        playbook.record_rule_application(
            PlaybookRuleApplication(
                application_id=f"shadow:{index}",
                rule_id=repeated_same_seed.rule_id,
                run_id="shadow-run",
                episode_id="shadow-episode",
                step_id=index,
                game_loop=index,
                target_kind="intent",
                target_id=f"intent:{index}",
                matched=True,
                reason="candidate_shadow_match",
            )
        )

    store, result = _persistent_threat_episode(tmp_path, "threat-run-3", seed=1)
    reviewer.review_episode(
        store.events_after("threat-run-3", 0, 100, episode_id="episode"),
        result,
        agent_race="protoss",
        opponent_race="zerg",
    )
    store.close()

    rule = next(
        rule
        for rule in playbook.rules()
        if rule.evidence.get("consequence_type") == "threat_unanswered"
    )
    assert rule.status is PlaybookRuleStatus.ACTIVE
    assert rule.strength is PlaybookRuleStrength.SOFT
    assert rule.effect is PlaybookRuleEffect.PREFER
    assert rule.role_ids == ("defense",)
    assert set(rule.source_seeds) == {0, 1}
    assert rule.contradiction_count == 0
    promoted_lesson = next(
        lesson
        for lesson in playbook.lessons()
        if lesson.consequence_type is not None
        and lesson.consequence_type.value == "threat_unanswered"
    )
    assert promoted_lesson.status is LessonStatus.PROMOTED
    assert {(condition.field, condition.value) for condition in rule.conditions} >= {
        ("phase", "combat"),
        ("threat_level", "high"),
        ("army_readiness", "ready"),
    }

    situation = SituationAssessment(
        assessment_id="assessment:next-match",
        run_id="next-run",
        episode_id="episode",
        step_id=1,
        game_loop=1_300,
        valid_until_game_loop=1_316,
        phase=GamePhase.COMBAT,
        threat_level=ThreatLevel.HIGH,
        economy_status=EconomyStatus.STABLE,
        army_readiness=ArmyReadiness.READY,
        source_kind="deterministic",
        source_id="test",
        source_version="1",
    )
    context = PlaybookContext(
        agent_race="protoss",
        opponent_race="zerg",
        phase=GamePhase.COMBAT,
        map_name="Simple64",
    )
    defense = StrategicIntent(
        intent_id="intent:defense",
        continuity_key="defense:respond",
        run_id="next-run",
        episode_id="episode",
        step_id=1,
        created_game_loop=1_300,
        role=RoleId.DEFENSE,
        objective="answer the threat",
        desired_effect="remove the threat",
        action_names=("Move_Minimap",),
        resource_claim=ResourceClaim(reservation_game_loops=16),
        source_id="test",
        source_version="1",
    )
    offense = defense.model_copy(
        update={
            "intent_id": "intent:offense",
            "continuity_key": "offense:push",
            "role": RoleId.OFFENSE,
        }
    )
    guard = PlaybookIntentGuard()
    selected_rules = playbook.rules_for_guard(context=context)

    defense_result = guard.evaluate(
        defense,
        context=context,
        situation=situation,
        rules=selected_rules,
        game_loop=1_300,
        mode="active",
    )
    offense_result = guard.evaluate(
        offense,
        context=context,
        situation=situation,
        rules=selected_rules,
        game_loop=1_300,
        mode="active",
    )

    assert defense_result.score_delta == 0.5
    assert defense_result.rule_ids == (rule.rule_id,)
    assert offense_result.score_delta == 0.0
    assert offense_result.rule_ids == ()
    playbook.close()


def test_playbook_does_not_apply_unscoped_contradiction_to_typed_rule(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=2)
    for seed, run_id in enumerate(("run-1", "run-2")):
        store, result = _episode_events(tmp_path, run_id, seed=seed)
        reviewer.review_episode(
            store.events_after(run_id, 0, 100, episode_id="episode"),
            result,
            agent_race="protoss",
            opponent_race="zerg",
        )
        store.close()

    third_store, third_result = _episode_events(tmp_path, "run-3")
    third_store.append_event(
        run_id="run-3",
        episode_id="episode",
        step_id=3,
        event_type="macro_plan_rejected",
        payload={
            "classification": "illegal_action",
            "reason": "illegal_runtime_frontier",
            "proposal": {"steps": [{"canonical_action": "BUILD STARGATE"}]},
        },
    )
    reviewer.review_episode(
        third_store.events_after("run-3", 0, 100, episode_id="episode"),
        third_result,
        agent_race="protoss",
        opponent_race="zerg",
    )

    preferred = next(
        rule
        for rule in playbook.rules()
        if rule.action_names == ("BUILD STARGATE",) and rule.effect.value == "prefer"
    )
    assert preferred.contradiction_count == 0
    assert preferred.contradiction_seeds == ()
    third_store.close()
    playbook.close()


def test_playbook_does_not_promote_bridge_failure_as_strategy(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=1)
    store, result = _episode_events(tmp_path, "run")
    # Replace the successful journal with a separate failed command so the reviewer sees
    # both an outcome-backed success and a Bridge diagnostic case.
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=3,
        event_type="command_lineage",
        payload={
            "command_id": "failed-command",
            "semantic_action": "BUILD NEXUS",
            "lineage": {"source_role": "macro"},
        },
    )
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=4,
        event_type="execution",
        payload=ExecutionReport(
            run_id="run",
            episode_id="episode",
            step_id=4,
            command_id="failed-command",
            success=False,
            action_name="Build_Nexus_Near",
            actor="Builder/Probe-1",
            source=ActionSource.PLANNER,
            status=ExecutionStatus.FAILED,
            execution_stage=ExecutionStage.TRANSLATION,
            failure_code="invalid_expansion_anchor",
        ),
    )

    cases, lessons = reviewer.review_episode(
        store.events_after("run", 0, 100, episode_id="episode"),
        result,
        agent_race="protoss",
        opponent_race="zerg",
    )

    failed = next(case for case in cases if case.command_id == "failed-command")
    assert failed.quality is DecisionQuality.EXECUTION_ERROR
    assert all(lesson.recommended_action != "BUILD NEXUS" for lesson in lessons)
    guard = next(
        lesson for lesson in lessons if lesson.rule_kind is PlaybookRuleKind.EXECUTION_GUARD
    )
    assert guard.status is LessonStatus.PROMOTED
    assert guard.avoid_action == "BUILD NEXUS"
    executable_guard = next(
        rule for rule in playbook.rules() if rule.action_names == ("BUILD NEXUS",)
    )
    assert executable_guard.status is PlaybookRuleStatus.CANDIDATE

    store.close()
    playbook.close()


def test_playbook_repairs_unscoped_v2_execution_candidate(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=1)
    store, result = _episode_events(tmp_path, "run")
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=3,
        event_type="command_lineage",
        payload={
            "command_id": "failed-command",
            "semantic_action": "BUILD NEXUS",
            "lineage": {"source_role": "macro"},
        },
    )
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=4,
        event_type="execution",
        payload=ExecutionReport(
            run_id="run",
            episode_id="episode",
            step_id=4,
            command_id="failed-command",
            success=False,
            action_name="Build_Nexus_Near",
            actor="Builder/Probe-1",
            source=ActionSource.PLANNER,
            status=ExecutionStatus.FAILED,
            execution_stage=ExecutionStage.TRANSLATION,
            failure_code="invalid_expansion_anchor",
        ),
    )
    cases, _ = reviewer.review_episode(
        store.events_after("run", 0, 100, episode_id="episode"),
        result,
        agent_race="protoss",
        opponent_race="zerg",
    )
    failed = next(case for case in cases if case.command_id == "failed-command")
    playbook.upsert_rule(
        PlaybookRule(
            rule_id="playbook-rule:unscoped",
            canonical_key="unscoped",
            category=PlaybookRuleCategory.EXECUTION_GUARD,
            conditions=(),
            effect=PlaybookRuleEffect.AVOID,
            status=PlaybookRuleStatus.CANDIDATE,
            strength=PlaybookRuleStrength.ADVISORY,
            confidence=0.75,
            source_case_ids=(failed.case_id,),
            source_run_ids=(failed.run_id,),
        )
    )

    CortexPlaybookReviewer(playbook, promotion_support=1)

    rules = playbook.rules()
    repaired_source = next(rule for rule in rules if rule.canonical_key == "unscoped")
    assert repaired_source.status is PlaybookRuleStatus.RETIRED
    assert repaired_source.evidence["repair_reason"] == "unscoped_execution_candidate"
    assert any(rule.action_names == ("BUILD NEXUS",) for rule in rules)
    store.close()
    playbook.close()


def test_playbook_records_illegal_macro_proposal_as_cortex_error(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=1)
    store, result = _episode_events(tmp_path, "run")
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="macro_plan_rejected",
        payload={
            "classification": "illegal_action",
            "reason": "illegal_runtime_frontier",
            "proposal": {"steps": [{"canonical_action": "RESEARCH PSIONIC STORM"}]},
        },
    )

    cases, lessons = reviewer.review_episode(
        store.events_after("run", 0, 100, episode_id="episode"),
        result,
        agent_race="protoss",
        opponent_race="zerg",
    )

    rejected = next(case for case in cases if case.command_id.startswith("proposal:"))
    assert rejected.quality is DecisionQuality.STRATEGIC_ERROR
    assert rejected.failure_owner.value == "cortex"
    assert rejected.semantic_action == "RESEARCH PSIONIC STORM"
    warning = next(lesson for lesson in lessons if lesson.avoid_action is not None)
    assert warning.rule_kind is PlaybookRuleKind.STRATEGY
    assert warning.avoid_action == "RESEARCH PSIONIC STORM"

    store.close()
    playbook.close()


def test_playbook_does_not_learn_temporarily_deferred_macro_proposal(
    tmp_path: Path,
) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=1)
    store, result = _episode_events(tmp_path, "run")
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="macro_plan_rejected",
        payload={
            "classification": "mapped_deferred",
            "reason": "missing_prerequisite_gateway",
            "proposal": {"steps": [{"canonical_action": "TRAIN PROBE"}]},
        },
    )
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="macro_plan_rejected",
        payload={
            "classification": "unsupported_by_runtime",
            "reason": "not_implemented",
            "proposal": {"steps": [{"canonical_action": "TRAIN TEMPEST"}]},
        },
    )

    cases, _ = reviewer.review_episode(
        store.events_after("run", 0, 100, episode_id="episode"),
        result,
        agent_race="protoss",
        opponent_race="zerg",
    )

    assert all(not case.command_id.startswith("proposal:") for case in cases)
    assert all("TRAIN PROBE" not in rule.action_names for rule in playbook.rules())
    assert all("TRAIN TEMPEST" not in rule.action_names for rule in playbook.rules())

    store.close()
    playbook.close()


def test_playbook_guard_rule_cap_is_applied_after_context_filter(tmp_path: Path) -> None:
    store = PlaybookStore(tmp_path / "playbook.sqlite3")
    for index in range(8):
        store.upsert_rule(
            PlaybookRule(
                rule_id=f"rule:terran:{index}",
                canonical_key=f"terran:{index}",
                category=PlaybookRuleCategory.EXECUTION_GUARD,
                conditions=(PlaybookCondition(field="agent_race", value="terran"),),
                effect=PlaybookRuleEffect.AVOID,
                strength=PlaybookRuleStrength.SOFT,
                status=PlaybookRuleStatus.ACTIVE,
                action_names=("Train_Marine",),
                confidence=0.95,
            )
        )
    protoss = PlaybookRule(
        rule_id="rule:protoss",
        canonical_key="protoss",
        category=PlaybookRuleCategory.EXECUTION_GUARD,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.AVOID,
        strength=PlaybookRuleStrength.SOFT,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("Build_Pylon_Screen",),
        confidence=0.8,
    )
    store.upsert_rule(protoss)

    selected = store.rules_for_guard(
        context=PlaybookContext(
            agent_race="protoss",
            opponent_race="zerg",
            phase=GamePhase.EARLY,
            map_name="Simple64",
        )
    )

    assert selected == (protoss,)
    store.close()


def test_playbook_promotes_repeated_producer_failure_as_compact_execution_rule(
    tmp_path: Path,
) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=2)
    updates = []
    for run_id in ("producer-run-1", "producer-run-2"):
        store = EventStore(tmp_path / f"{run_id}.sqlite3", tmp_path / f"{run_id}.jsonl")
        store.append_event(
            run_id=run_id,
            episode_id="episode",
            step_id=0,
            event_type="situation_assessed",
            payload={"phase": "production"},
        )
        store.append_event(
            run_id=run_id,
            episode_id="episode",
            step_id=1,
            event_type="command_lineage",
            payload={
                "command_id": f"{run_id}:command",
                "semantic_action": "TRAIN ADEPT",
                "lineage": {"source_role": "macro"},
            },
        )
        store.append_event(
            run_id=run_id,
            episode_id="episode",
            step_id=2,
            event_type="execution",
            payload=ExecutionReport(
                run_id=run_id,
                episode_id="episode",
                step_id=2,
                command_id=f"{run_id}:command",
                success=False,
                action_name="Train_Adept",
                actor="Developer/Empty",
                source=ActionSource.PLANNER,
                status=ExecutionStatus.FAILED,
                execution_stage=ExecutionStage.TRANSLATION,
                failure_code="producer_not_observable",
            ),
        )
        result = EpisodeResult(
            run_id=run_id,
            episode_id="episode",
            scenario="Simple64",
            seed=0,
            outcome=EpisodeOutcome.DEFEAT,
            steps=2,
        )
        _, lessons = reviewer.review_episode(
            store.events_after(run_id, 0, 100, episode_id="episode"),
            result,
            agent_race="protoss",
            opponent_race="zerg",
        )
        updates.extend(lessons)
        store.close()

    guard = updates[-1]
    assert guard.rule_kind is PlaybookRuleKind.EXECUTION_GUARD
    assert guard.status is LessonStatus.PROMOTED
    assert guard.support_count == 2
    assert "fresh feature observation" in guard.statement
    executable_guard = next(
        rule for rule in playbook.rules() if rule.action_names == ("TRAIN ADEPT",)
    )
    assert executable_guard.status is PlaybookRuleStatus.CANDIDATE
    assert executable_guard.strength is PlaybookRuleStrength.ADVISORY
    selection = playbook.retrieve(
        PlaybookQuery(
            context=PlaybookContext(
                agent_race="protoss",
                opponent_race="terran",
                phase=GamePhase.COMBAT,
                map_name="AnotherMap",
            )
        )
    )
    assert guard.lesson_id in selection.lesson_ids
    playbook.close()


def test_playbook_quarantines_legacy_soft_execution_penalty(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    playbook.upsert_rule(
        PlaybookRule(
            rule_id="rule:unsafe-execution-penalty",
            canonical_key="unsafe-execution-penalty",
            category=PlaybookRuleCategory.EXECUTION_GUARD,
            conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
            effect=PlaybookRuleEffect.AVOID,
            strength=PlaybookRuleStrength.SOFT,
            status=PlaybookRuleStatus.ACTIVE,
            action_names=("Attack_Unit",),
            confidence=0.95,
        )
    )

    CortexPlaybookReviewer(playbook)

    rule = next(
        rule
        for rule in playbook.rules()
        if rule.canonical_key == "unsafe-execution-penalty"
    )
    assert rule.status is PlaybookRuleStatus.SUSPENDED
    assert rule.strength is PlaybookRuleStrength.ADVISORY
    assert rule.evidence["suspension_reason"] == "missing_typed_failure_precondition"
    playbook.close()
