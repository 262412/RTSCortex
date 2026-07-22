from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rtscortex.contracts import EconomyState, ObservationEnvelope, SC2State, UnitState
from rtscortex.cortex import (
    CandidateFeatures,
    DeterministicSituationAnalyzer,
    ExecutableCandidate,
    GamePhase,
    IntentArbiter,
    IntentDecisionStatus,
    ResourceClaim,
    RoleAgentContext,
    RoleAgentCoordinator,
    RoleId,
    StrategicIntent,
    StrategicIntentAdapter,
)
from rtscortex.cortex.models import IntentTarget, MacroIntent, ReflexIntent
from rtscortex.playbook import (
    LessonStatus,
    PlaybookCandidateGuard,
    PlaybookCondition,
    PlaybookContext,
    PlaybookIntentGuard,
    PlaybookLesson,
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleLifecycle,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    PlaybookStore,
)
from rtscortex.races import ActionDomain, RaceId, built_in_race_profiles, race_profile


def _observation(*, minerals: int = 200, vespene: int = 0) -> ObservationEnvelope:
    return ObservationEnvelope(
        run_id="run",
        episode_id="episode",
        step_id=1,
        game_loop=32,
        state=SC2State(
            economy=EconomyState(
                minerals=minerals,
                vespene=vespene,
                supply_used=12,
                supply_cap=15,
                workers=12,
                army_supply=2,
            ),
            own_units=[
                UnitState(
                    unit_id="0xprobe",
                    unit_type="Probe",
                    alliance="self",
                    position=(10, 10),
                )
            ],
            own_structures=[
                UnitState(
                    unit_id="0xnexus",
                    unit_type="Nexus",
                    alliance="self",
                    position=(10, 10),
                )
            ],
        ),
    )


def _intent(
    identity: str,
    role: RoleId,
    *,
    minerals: int = 0,
    emergency: bool = False,
    actor: str = "",
) -> StrategicIntent:
    return StrategicIntent(
        intent_id=f"intent:{identity}",
        continuity_key=f"{role.value}:{identity}",
        run_id="run",
        episode_id="episode",
        step_id=1,
        created_game_loop=32,
        role=role,
        objective=identity,
        desired_effect=identity,
        action_names=(identity,),
        actor_scopes=() if not actor else (actor,),
        resource_claim=ResourceClaim(minerals=minerals, reservation_game_loops=16),
        emergency=emergency,
        urgency=1.0 if emergency else 0.5,
        source_id="test",
        source_version="1",
    )


def test_builtin_race_profiles_are_complete_and_isolated() -> None:
    profiles = built_in_race_profiles()

    assert tuple(profile.race for profile in profiles) == tuple(RaceId)
    assert race_profile("protoss").data.worker_type == "Probe"
    assert race_profile("terran").data.worker_type == "SCV"
    assert race_profile("zerg").data.worker_type == "Drone"
    assert not any(
        "Pylon" in action
        for profile in profiles[1:]
        for mapping in profile.data.macro_action_mappings
        for action in mapping.runtime_actions
    )


@pytest.mark.parametrize(
    ("race", "action_name", "expected_domain", "expected_producers"),
    [
        ("protoss", "Build_Stargate_Screen", ActionDomain.PRODUCTION, ("Probe",)),
        ("protoss", "Research_WarpGate", ActionDomain.TECHNOLOGY, ("CyberneticsCore",)),
        ("terran", "Build_Factory_Screen", ActionDomain.PRODUCTION, ("SCV",)),
        ("terran", "Build_BarracksTechLab", ActionDomain.TECHNOLOGY, ("Barracks",)),
        ("terran", "Build_MissileTurret_Screen", ActionDomain.DEFENSE, ("SCV",)),
        ("zerg", "Morph_Lair", ActionDomain.TECHNOLOGY, ("Hatchery",)),
        ("zerg", "Build_HydraliskDen_Screen", ActionDomain.PRODUCTION, ("Drone",)),
        ("zerg", "Build_SporeCrawler_Screen", ActionDomain.DEFENSE, ("Drone",)),
    ],
)
def test_race_profiles_lock_role_ownership_and_producer_semantics(
    race: str,
    action_name: str,
    expected_domain: ActionDomain,
    expected_producers: tuple[str, ...],
) -> None:
    profile = race_profile(race)

    assert profile.domain_for_action(action_name) is expected_domain
    assert profile.producers_for_action(action_name) == expected_producers


def test_race_profile_capabilities_match_implemented_live_readiness() -> None:
    protoss = race_profile("protoss").data.capability_snapshot()
    terran = race_profile("terran").data.capability_snapshot()
    zerg = race_profile("zerg").data.capability_snapshot()

    assert protoss["runtime_mapping_ready"] is True
    assert protoss["live_worker_ready"] is True
    assert terran["macro_contract_ready"] is True
    assert terran["runtime_mapping_ready"] is True
    assert terran["live_worker_ready"] is True
    assert zerg["macro_contract_ready"] is True
    assert zerg["runtime_mapping_ready"] is True
    assert zerg["live_worker_ready"] is True
    assert zerg["effect_verification_kinds"] == [
        "build",
        "production",
        "morph",
        "inject",
        "move",
    ]


def test_situation_v2_keeps_unobserved_map_facts_unknown() -> None:
    assessment = DeterministicSituationAnalyzer().assess(_observation())

    assert assessment.spatial.map_control_fraction is None
    assert assessment.spatial.threat_eta_seconds is None
    assert assessment.scouting.enemy_visible is False
    assert assessment.information_gaps == ["enemy_force_not_visible"]
    assert all(fact.source and 0 <= fact.confidence <= 1 for fact in assessment.facts)


def test_intent_arbiter_conserves_decisions_and_resources() -> None:
    observation = _observation(minerals=150)
    intents = (
        _intent("Build_Gateway", RoleId.PRODUCTION, minerals=150),
        _intent("Build_Pylon", RoleId.ECONOMY, minerals=100),
        _intent("Retreat", RoleId.RETREAT, emergency=True, actor="army"),
    )

    result = IntentArbiter().arbitrate(intents, observation)

    assert len(result.decisions) == len(intents)
    selected = sum(
        decision.status is IntentDecisionStatus.SELECTED for decision in result.decisions
    )
    assert selected == 2
    assert result.agenda.reserved_resources.minerals <= 150
    assert intents[2].intent_id in result.selected_intent_ids


def test_only_actual_defense_reflex_is_an_emergency() -> None:
    observation = _observation()
    adapter = StrategicIntentAdapter(race_profile("protoss"))
    static_defense = adapter.adapt(
        MacroIntent(
            intent_id="macro-defense",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective="add static defense",
            action_names=["Build_ShieldBattery_Screen"],
            target=IntentTarget(),
            ttl_game_loops=112,
            source_id="hima",
            source_version="revision",
            macro_plan_id="plan",
        )
    )
    defense_reflex = adapter.adapt(
        ReflexIntent(
            intent_id="reflex-defense",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective="defend the main base",
            action_names=["Move_Minimap"],
            ttl_game_loops=8,
            source_id="reflex",
            source_version="1",
        )
    )

    assert static_defense.role is RoleId.DEFENSE
    assert static_defense.emergency is False
    assert defense_reflex.role is RoleId.DEFENSE
    assert defense_reflex.emergency is True


def test_zerg_queen_controller_routes_inject_to_economy_and_creep_to_defense() -> None:
    observation = _observation()
    assessment = DeterministicSituationAnalyzer().assess(observation)
    profile = race_profile("zerg")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))
    intents = (
        ReflexIntent(
            intent_id="inject",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective="Maintain deterministic Zerg larva production",
            action_names=["Effect_InjectLarva"],
            actor_scopes=["CombatGroup1/Queen-1"],
            ttl_game_loops=8,
            source_id="zerg-controller",
            source_version="1",
        ),
        ReflexIntent(
            intent_id="creep",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective="Extend deterministic Zerg creep coverage",
            action_names=["Build_CreepTumor_Queen_Screen"],
            actor_scopes=["CombatGroup1/Queen-1"],
            ttl_game_loops=8,
            source_id="zerg-controller",
            source_version="1",
        ),
    )

    routed = coordinator.evaluate(
        RoleAgentContext(
            observation=observation,
            situation=assessment,
            source_intents=intents,
        )
    )

    assert routed["inject"].role is RoleId.ECONOMY
    assert routed["creep"].role is RoleId.DEFENSE
    assert routed["inject"].emergency is False
    assert routed["creep"].emergency is False


def test_legacy_playbook_migration_is_advisory_and_non_blocking(tmp_path: Path) -> None:
    store = PlaybookStore(tmp_path / "playbook.sqlite3")
    assert store.rules_for_guard() == ()
    store.close()


def test_playbook_hard_rule_requires_active_mode_to_block() -> None:
    rule = PlaybookRule(
        rule_id="rule:no-stop",
        canonical_key="no-stop",
        category=PlaybookRuleCategory.ENGINE_INVARIANT,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.FORBID,
        strength=PlaybookRuleStrength.HARD,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("Stop",),
        confidence=1.0,
    )
    observation = _observation()
    situation = DeterministicSituationAnalyzer().assess(observation)
    candidate = ExecutableCandidate(
        candidate_id="candidate:" + "0" * 64,
        observation_fingerprint="0" * 64,
        intent_id="intent:test",
        action_name="Stop",
        actor="army",
        features=CandidateFeatures(
            action_rank=0,
            actor_rank=0,
            argument_rank=0,
            compile_ordinal=0,
        ),
    )
    context = PlaybookContext(
        agent_race="protoss",
        opponent_race="zerg",
        phase=situation.phase,
        map_name="Simple64",
    )
    guard = PlaybookCandidateGuard()

    shadow = guard.evaluate(
        candidate,
        role="offense",
        context=context,
        situation=situation,
        rules=(rule,),
        run_id="run",
        episode_id="episode",
        step_id=1,
        game_loop=32,
        mode="shadow",
    )
    active = guard.evaluate(
        candidate,
        role="offense",
        context=context,
        situation=situation,
        rules=(rule,),
        run_id="run",
        episode_id="episode",
        step_id=1,
        game_loop=32,
        mode="active",
    )

    assert shadow.blocked is False
    assert shadow.applications[0].reason == "shadow_would_block"
    assert active.blocked is True


def test_playbook_soft_intent_score_is_observed_but_not_applied_in_shadow() -> None:
    rule = PlaybookRule(
        rule_id="rule:prefer-defense",
        canonical_key="prefer-defense",
        category=PlaybookRuleCategory.TACTICAL_RESPONSE,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.PREFER,
        strength=PlaybookRuleStrength.SOFT,
        status=PlaybookRuleStatus.ACTIVE,
        role_ids=("defense",),
        confidence=0.9,
    )
    observation = _observation()
    situation = DeterministicSituationAnalyzer().assess(observation)
    intent = _intent("Move_Minimap", RoleId.DEFENSE)
    context = PlaybookContext(
        agent_race="protoss",
        opponent_race="zerg",
        phase=situation.phase,
        map_name="Simple64",
    )
    guard = PlaybookIntentGuard()

    shadow = guard.evaluate(
        intent,
        context=context,
        situation=situation,
        rules=(rule,),
        game_loop=observation.game_loop,
        mode="shadow",
    )
    active = guard.evaluate(
        intent,
        context=context,
        situation=situation,
        rules=(rule,),
        game_loop=observation.game_loop,
        mode="active",
    )

    assert shadow.score_delta == 0.0
    assert shadow.applications[0].score_delta == 0.5
    assert active.score_delta == 0.5


def test_playbook_matches_hima_semantic_build_action_to_runtime_candidate() -> None:
    rule = PlaybookRule(
        rule_id="rule:placement",
        canonical_key="placement",
        category=PlaybookRuleCategory.EXECUTION_GUARD,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.AVOID,
        strength=PlaybookRuleStrength.SOFT,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("BUILD PYLON",),
        confidence=0.9,
    )
    candidate = ExecutableCandidate(
        candidate_id="candidate:" + "1" * 64,
        observation_fingerprint="0" * 64,
        intent_id="intent:pylon",
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        arguments=[[64, 64]],
        features=CandidateFeatures(
            action_rank=0,
            actor_rank=0,
            argument_rank=0,
            compile_ordinal=0,
        ),
    )
    situation = DeterministicSituationAnalyzer().assess(_observation(), ())

    result = PlaybookCandidateGuard().evaluate(
        candidate,
        role="economy",
        context=PlaybookContext(
            agent_race="protoss",
            opponent_race="zerg",
            phase=GamePhase.EARLY,
            map_name="Simple64",
        ),
        situation=situation,
        rules=(rule,),
        run_id="run",
        episode_id="episode",
        step_id=1,
        game_loop=32,
        mode="active",
    )

    assert result.rule_ids == (rule.rule_id,)
    assert result.score_delta == -0.5


def test_playbook_promotion_rejects_insufficient_evidence() -> None:
    rule = PlaybookRule(
        rule_id="rule:test",
        canonical_key="test",
        category=PlaybookRuleCategory.EXECUTION_GUARD,
        conditions=(),
        effect=PlaybookRuleEffect.AVOID,
        strength=PlaybookRuleStrength.ADVISORY,
        status=PlaybookRuleStatus.CANDIDATE,
        action_names=("Build_Gateway_Screen",),
        confidence=0.9,
        source_run_ids=("run-1",),
    )

    with pytest.raises(ValueError, match="two runs"):
        PlaybookRuleLifecycle().promote_to_soft(rule)


def test_playbook_promotion_rejects_untyped_execution_penalty() -> None:
    rule = PlaybookRule(
        rule_id="rule:test",
        canonical_key="test",
        category=PlaybookRuleCategory.EXECUTION_GUARD,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.AVOID,
        strength=PlaybookRuleStrength.ADVISORY,
        status=PlaybookRuleStatus.CANDIDATE,
        action_names=("Attack_Unit",),
        confidence=0.9,
        source_run_ids=("run-1", "run-2"),
        source_seeds=(0, 1),
    )

    with pytest.raises(ValueError, match="typed failure precondition"):
        PlaybookRuleLifecycle().promote_to_soft(rule)


def test_playbook_hard_promotion_rejects_only_censored_sources() -> None:
    rule = PlaybookRule(
        rule_id="rule:censored",
        canonical_key="censored",
        category=PlaybookRuleCategory.TACTICAL_RESPONSE,
        conditions=(PlaybookCondition(field="threat_level", value="low"),),
        effect=PlaybookRuleEffect.AVOID,
        strength=PlaybookRuleStrength.SOFT,
        status=PlaybookRuleStatus.ACTIVE,
        role_ids=("retreat",),
        confidence=0.95,
        source_run_ids=("run-0", "run-1", "run-2"),
        source_seeds=(0, 1, 2),
        censored_source_run_ids=("run-0", "run-1", "run-2"),
        censored_source_seeds=(0, 1, 2),
        code_revision="revision",
        sc2_patch="4.10",
        shadow_state_count=48,
    )

    with pytest.raises(ValueError, match="uncensored"):
        PlaybookRuleLifecycle().promote_to_hard(
            rule,
            current_code_revision="revision",
            current_sc2_patch="4.10",
        )


def test_playbook_contradictions_require_distinct_seeds() -> None:
    rule = PlaybookRule(
        rule_id="rule:test",
        canonical_key="test",
        category=PlaybookRuleCategory.EXECUTION_GUARD,
        conditions=(),
        effect=PlaybookRuleEffect.AVOID,
        strength=PlaybookRuleStrength.SOFT,
        status=PlaybookRuleStatus.ACTIVE,
        confidence=0.9,
    )
    lifecycle = PlaybookRuleLifecycle()

    once = lifecycle.record_contradiction(rule, seed=7)
    duplicate = lifecycle.record_contradiction(once, seed=7)
    suspended = lifecycle.record_contradiction(duplicate, seed=8)
    retired = lifecycle.record_contradiction(suspended, seed=9)

    assert duplicate == once
    assert once.contradiction_count == 1
    assert suspended.status is PlaybookRuleStatus.SUSPENDED
    assert retired.status is PlaybookRuleStatus.RETIRED
    assert retired.contradiction_seeds == (7, 8, 9)


def test_playbook_canonical_upsert_merges_lineage_without_inflating_support(
    tmp_path: Path,
) -> None:
    store = PlaybookStore(tmp_path / "playbook.sqlite3")
    base = PlaybookRule(
        rule_id="rule:first",
        canonical_key="same-condition-and-effect",
        category=PlaybookRuleCategory.MATCHUP_STRATEGY,
        conditions=(),
        effect=PlaybookRuleEffect.PREFER,
        strength=PlaybookRuleStrength.SOFT,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("Train_Adept",),
        confidence=0.8,
        support_count=2,
        source_run_ids=("run-1",),
    )
    store.upsert_rule(base)
    store.upsert_rule(
        base.model_copy(
            update={
                "rule_id": "rule:second",
                "source_run_ids": ("run-2",),
            }
        )
    )

    rules = store.rules()
    assert len(rules) == 1
    assert rules[0].support_count == 2
    assert rules[0].source_run_ids == ("run-1", "run-2")
    store.close()


def test_playbook_v2_migrates_legacy_lessons_as_advisory_with_backup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "playbook.sqlite3"
    lesson = PlaybookLesson(
        lesson_id="lesson:legacy",
        signature="legacy-signature",
        context=PlaybookContext(
            agent_race="protoss",
            opponent_race="zerg",
            phase=GamePhase.EARLY,
            map_name="Simple64",
        ),
        statement="Prefer a supply provider before the cap.",
        recommended_action="BUILD PYLON",
        status=LessonStatus.PROMOTED,
        confidence=0.9,
        support_count=4,
        contradiction_count=0,
        source_case_ids=("case:1",),
        source_episode_ids=("run-1/episode-1",),
    )
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE playbook_lessons (
            lesson_id TEXT PRIMARY KEY,
            signature TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO playbook_lessons VALUES (?, ?, ?)",
        (lesson.lesson_id, lesson.signature, lesson.model_dump_json()),
    )
    connection.commit()
    connection.close()

    store = PlaybookStore(path)
    rule = store.rules()[0]
    assert rule.status is PlaybookRuleStatus.LEGACY
    assert rule.strength is PlaybookRuleStrength.ADVISORY
    assert (tmp_path / "playbook.pre-v2.sqlite3").is_file()
    store.close()
