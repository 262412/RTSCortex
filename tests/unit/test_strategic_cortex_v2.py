from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rtscortex.contracts import (
    ActionArgumentType,
    ActionCommand,
    ActionSource,
    AvailableAction,
    EconomyState,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
    ObservationEnvelope,
    ProductionItem,
    SC2State,
    UnitState,
)
from rtscortex.cortex import (
    AttemptKey,
    CandidateFeatures,
    DeterministicSituationAnalyzer,
    ExecutableCandidate,
    GamePhase,
    IntentArbiter,
    IntentDecisionStatus,
    OperationKey,
    ResourceClaim,
    RoleAgentContext,
    RoleAgentCoordinator,
    RoleId,
    StrategicIntent,
    StrategicIntentAdapter,
    ThreatLevel,
)
from rtscortex.cortex.models import IntentTarget, MacroIntent, ReflexIntent, TacticalIntent
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


def _terminal_collapse_defense_observation(
    *,
    own_structures: list[UnitState] | None = None,
    immediate_action: str | None = None,
) -> ObservationEnvelope:
    actions = [
        AvailableAction(
            name="Build_Pylon_Screen",
            argument_names=["screen"],
            argument_types=[ActionArgumentType.POSITION],
            actor_scopes=["Builder/Builder-Probe-1"],
            argument_candidates=[[[64, 64]]],
        ),
        AvailableAction(
            name="Attack_Unit",
            argument_names=["tag"],
            argument_types=[ActionArgumentType.TAG],
            actor_scopes=["Builder/Builder-Probe-1"],
            argument_candidates=[["0xe1"]],
        ),
    ]
    if immediate_action is not None:
        actions.insert(
            0,
            AvailableAction(
                name=immediate_action,
                actor_scopes=["Developer/Empty"],
            ),
        )
    return ObservationEnvelope(
        run_id="run",
        episode_id="terminal-collapse",
        step_id=1,
        game_loop=32,
        state=SC2State(
            economy=EconomyState(
                minerals=500,
                vespene=300,
                supply_used=1,
                supply_cap=15,
                workers=1,
                army_supply=0,
            ),
            own_units=[
                UnitState(
                    unit_id="0xprobe",
                    unit_type="Probe",
                    alliance="self",
                    position=(10, 10),
                )
            ],
            own_structures=[] if own_structures is None else own_structures,
            visible_enemies=[
                UnitState(
                    unit_id="0xe1",
                    unit_type="Zergling",
                    alliance="enemy",
                    position=(12, 10),
                )
            ],
        ),
        available_actions=actions,
    )


def _intent(
    identity: str,
    role: RoleId,
    *,
    minerals: int = 0,
    emergency: bool = False,
    actor: str = "",
    semantic_target_key: str = "global:none",
    dependency_intent_ids: tuple[str, ...] = (),
    dependency_semantic_keys: tuple[str, ...] = (),
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
        semantic_target_key=semantic_target_key,
        resource_claim=ResourceClaim(minerals=minerals, reservation_game_loops=16),
        dependency_intent_ids=dependency_intent_ids,
        dependency_semantic_keys=dependency_semantic_keys,
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


def test_operation_identity_survives_ticks_while_attempt_identity_remains_unique() -> None:
    operation = OperationKey(
        run_id="run",
        episode_id="episode",
        role="offense",
        action_family="Attack_Unit",
        semantic_actor="CombatGroup1/Zealot-1",
        target_key="enemy:0xbeef",
    )

    first_attempt = AttemptKey(
        operation_id=operation.operation_id,
        command_id="command-at-step-10",
    )
    second_attempt = AttemptKey(
        operation_id=operation.operation_id,
        command_id="command-at-step-11",
    )

    assert first_attempt.operation_id == second_attempt.operation_id
    assert first_attempt.attempt_id != second_attempt.attempt_id


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


def test_disjoint_focus_fire_intents_survive_role_arbitration() -> None:
    observation = _observation()
    first = _intent("Attack_Unit:first", RoleId.FOCUS_FIRE, actor="CombatGroup1")
    second = _intent("Attack_Unit:second", RoleId.FOCUS_FIRE, actor="CombatGroup2")

    result = IntentArbiter().arbitrate((first, second), observation)

    assert set(result.selected_intent_ids) == {first.intent_id, second.intent_id}
    assert all(decision.status is IntentDecisionStatus.SELECTED for decision in result.decisions)


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


def test_defense_agent_independently_responds_to_a_critical_threat() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0xe1",
                            unit_type="Drone",
                            alliance="enemy",
                            position=(12, 10),
                        )
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=[
                        "CombatGroup0/Zealot-1",
                        "Builder/Builder-Probe-1",
                    ],
                    argument_candidates=[["0xe1"]],
                )
            ],
        }
    )
    assessment = (
        DeterministicSituationAnalyzer()
        .assess(observation)
        .model_copy(update={"threat_level": ThreatLevel.CRITICAL, "threat_score": 12.0})
    )
    coordinator = RoleAgentCoordinator(
        race_profile("protoss"),
        StrategicIntentAdapter(race_profile("protoss")),
    )
    context = RoleAgentContext(
        observation=observation,
        situation=assessment,
        source_intents=(),
    )

    source_intents = coordinator.propose_defense_intents(context)
    assert len(source_intents) == 1
    assert source_intents[0].action_names == ["Attack_Unit"]
    assert source_intents[0].target.unit_tag == "0xe1"

    routed = coordinator.evaluate(
        RoleAgentContext(
            observation=observation,
            situation=assessment,
            source_intents=source_intents,
        )
    )
    strategic = routed[source_intents[0].intent_id]
    assert strategic.role is RoleId.DEFENSE
    assert strategic.emergency is True
    assert strategic.urgency == 1.0


def test_defense_agent_compiles_race_profile_emergency_options_and_resource_claims() -> None:
    base = _observation(minerals=500, vespene=300)
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0xe1",
                            unit_type="Mutalisk",
                            alliance="enemy",
                            position=(12, 10),
                        )
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Train_Phoenix",
                    actor_scopes=["Developer/Empty"],
                ),
                AvailableAction(
                    name="Train_Stalker",
                    actor_scopes=["Developer/Empty"],
                ),
                AvailableAction(
                    name="Build_ShieldBattery_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Builder-Probe-1"],
                    argument_candidates=[[[64, 64]]],
                ),
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["Builder/Builder-Probe-1"],
                    argument_candidates=[["0xe1"]],
                ),
            ],
        }
    )
    assessment = (
        DeterministicSituationAnalyzer()
        .assess(observation)
        .model_copy(update={"threat_level": ThreatLevel.CRITICAL, "threat_score": 20.0})
    )
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))

    source_intents = coordinator.propose_defense_intents(
        RoleAgentContext(observation, assessment, ())
    )
    routed = coordinator.evaluate(RoleAgentContext(observation, assessment, source_intents))
    actions = {intent.action_names[0] for intent in routed.values()}

    assert {"Train_Phoenix", "Train_Stalker", "Build_ShieldBattery_Screen"} <= actions
    assert "Attack_Unit" in actions
    phoenix = next(
        intent for intent in routed.values() if intent.action_names == ("Train_Phoenix",)
    )
    assert phoenix.role is RoleId.DEFENSE
    assert phoenix.emergency is True
    assert phoenix.resource_claim.minerals == 150
    assert phoenix.resource_claim.vespene == 100
    assert phoenix.mutually_exclusive_groups == ("defense-emergency-response",)


def test_terminal_collapse_suppresses_prerequisite_but_keeps_worker_defense() -> None:
    observation = _terminal_collapse_defense_observation()
    assessment = (
        DeterministicSituationAnalyzer().assess(observation).model_copy(update={"facts": []})
    )
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))

    source_intents = coordinator.propose_defense_intents(
        RoleAgentContext(observation, assessment, ())
    )

    assert assessment.army_readiness.value == "empty"
    assert assessment.bases.own_base_count == 0
    assert assessment.bases.own_production_capacity == 0
    assert [intent.action_names[0] for intent in source_intents] == ["Attack_Unit"]
    diagnostics = coordinator.drain_defense_diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0] == {
        "state": "defense_prerequisite_suppressed_terminal_collapse",
        "reason": "defense_prerequisite_suppressed_terminal_collapse",
        "army_readiness": "empty",
        "own_base_count": 0,
        "own_production_capacity": 0,
        "threat_level": "critical",
        "source_lineage": {
            "source_id": "deterministic-defense-agent",
            "source_version": "2.1.0",
            "situation_assessment_id": assessment.assessment_id,
            "situation_source_kind": "deterministic",
            "situation_source_id": assessment.source_id,
            "situation_source_version": assessment.source_version,
        },
    }

    coordinator.propose_defense_intents(
        RoleAgentContext(
            observation.model_copy(update={"step_id": 2, "game_loop": 33}),
            assessment,
            (),
        )
    )
    assert coordinator.drain_defense_diagnostics() == ()


@pytest.mark.parametrize(
    ("structure_type", "expected_base_count", "expected_production_capacity"),
    [
        ("Nexus", 1, 0),
        ("Gateway", 0, 1),
    ],
)
def test_empty_army_with_surviving_base_or_production_keeps_emergency_prerequisite(
    structure_type: str,
    expected_base_count: int,
    expected_production_capacity: int,
) -> None:
    observation = _terminal_collapse_defense_observation(
        own_structures=[
            UnitState(
                unit_id="0xstructure",
                unit_type=structure_type,
                alliance="self",
                position=(10, 10),
            )
        ]
    )
    assessment = (
        DeterministicSituationAnalyzer()
        .assess(observation)
        .model_copy(update={"threat_level": ThreatLevel.CRITICAL, "facts": []})
    )
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))

    source_intents = coordinator.propose_defense_intents(
        RoleAgentContext(observation, assessment, ())
    )

    assert assessment.army_readiness.value == "empty"
    assert assessment.bases.own_base_count == expected_base_count
    assert assessment.bases.own_production_capacity == expected_production_capacity
    assert "Build_Pylon_Screen" in {intent.action_names[0] for intent in source_intents}
    assert coordinator.drain_defense_diagnostics() == ()


def test_terminal_collapse_keeps_immediate_defense_priority_without_suppression() -> None:
    observation = _terminal_collapse_defense_observation(immediate_action="Train_Stalker")
    assessment = DeterministicSituationAnalyzer().assess(observation)
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))

    source_intents = coordinator.propose_defense_intents(
        RoleAgentContext(observation, assessment, ())
    )
    actions = [intent.action_names[0] for intent in source_intents]

    assert "Train_Stalker" in actions
    assert "Build_Pylon_Screen" not in actions
    assert "Attack_Unit" in actions
    assert coordinator.drain_defense_diagnostics() == ()


def test_defense_unit_production_stops_at_race_saturation_target() -> None:
    base = _observation(minerals=500, vespene=300)
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "own_units": [
                        *base.state.own_units,
                        *[
                            UnitState(
                                unit_id=f"0xphoenix{index}",
                                unit_type="Phoenix",
                                alliance="self",
                            )
                            for index in range(5)
                        ],
                    ],
                    "production_queue": [ProductionItem(name="Phoenix", producer_id="0xstargate")],
                    "visible_enemies": [
                        UnitState(
                            unit_id="0xe1",
                            unit_type="Mutalisk",
                            alliance="enemy",
                            position=(12, 10),
                        )
                    ],
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Train_Phoenix",
                    actor_scopes=["Developer/Empty"],
                )
            ],
        }
    )
    assessment = (
        DeterministicSituationAnalyzer()
        .assess(observation)
        .model_copy(update={"threat_level": ThreatLevel.CRITICAL, "threat_score": 20.0})
    )
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))

    source_intents = coordinator.propose_defense_intents(
        RoleAgentContext(observation, assessment, ())
    )

    assert all(intent.action_names != ["Train_Phoenix"] for intent in source_intents)


def test_defense_inventory_deduplicates_dispatched_effect_already_observed() -> None:
    base = _observation()
    structures = [
        UnitState(
            unit_id=f"0xbattery{index}",
            unit_type="ShieldBattery",
            alliance="self",
            status=("constructing" if index == 3 else "ready"),
        )
        for index in range(4)
    ]
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={"own_structures": [*base.state.own_structures, *structures]}
            )
        }
    )
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))

    inventory = coordinator.defense_inventory_evaluations(
        observation,
        active_commands=(("Build_ShieldBattery_Screen", "dispatched"),),
    )
    battery = next(item for item in inventory if item["item_type"] == "ShieldBattery")

    assert battery["completed"] == 3
    assert battery["constructing_or_training"] == 1
    assert battery["dispatched_not_terminal"] == 1
    assert battery["dispatches_not_already_observed"] == 0
    assert battery["effective_count"] == 4
    assert battery["hard_cap"] == 4


def test_undispatched_defense_proposal_does_not_hold_actor() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0xe1",
                            unit_type="Drone",
                            alliance="enemy",
                            position=(12, 10),
                        )
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup0/Zealot-1"],
                    argument_candidates=[["0xe1"]],
                )
            ],
        }
    )
    assessment = (
        DeterministicSituationAnalyzer()
        .assess(observation)
        .model_copy(update={"threat_level": ThreatLevel.CRITICAL, "threat_score": 12.0})
    )
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))

    first = coordinator.propose_defense_intents(RoleAgentContext(observation, assessment, ()))
    assert len(first) == 1
    undispatched = coordinator.propose_defense_intents(
        RoleAgentContext(
            observation.model_copy(update={"step_id": 2, "game_loop": 33}),
            assessment,
            (),
        )
    )
    assert len(undispatched) == 1
    coordinator.record_dispatch(
        ActionCommand(
            command_id="defense-command",
            actor="CombatGroup0/Zealot-1",
            name="Attack_Unit",
            arguments=["0xe1"],
            created_game_loop=33,
            ttl_game_loops=8,
            source=ActionSource.REFLEX,
        ),
        responsibility="defense",
        game_loop=33,
    )

    transition = coordinator.record_execution(
        ExecutionReport(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=3,
            command_id="defense-command",
            success=False,
            action_name="Attack_Unit",
            actor="CombatGroup0/Zealot-1",
            source=ActionSource.REFLEX,
            requested_arguments=["0xe1"],
            status=ExecutionStatus.FAILED,
            execution_stage=ExecutionStage.EFFECT_VERIFICATION,
            failure_code="effect_timeout",
        ),
        responsibility="defense",
        game_loop=40,
    )
    assert transition is not None
    assert transition["state"] == "defense_response_cooldown"
    assert (
        coordinator.propose_defense_intents(
            RoleAgentContext(
                observation.model_copy(update={"step_id": 4, "game_loop": 71}),
                assessment,
                (),
            )
        )
        == ()
    )
    assert (
        len(
            coordinator.propose_defense_intents(
                RoleAgentContext(
                    observation.model_copy(update={"step_id": 5, "game_loop": 72}),
                    assessment,
                    (),
                )
            )
        )
        == 1
    )


def test_defense_holding_expires_under_sustained_threat() -> None:
    base = _observation()
    actor = "CombatGroup0/Zealot-1"
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0xe1",
                            unit_type="Drone",
                            alliance="enemy",
                            position=(12, 10),
                        )
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Move_Minimap",
                    argument_names=["minimap"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=[actor, "Builder/Builder-Probe-1"],
                    argument_candidates=[[[12, 12]]],
                )
            ],
        }
    )
    assessment = (
        DeterministicSituationAnalyzer()
        .assess(observation)
        .model_copy(update={"threat_level": ThreatLevel.CRITICAL, "threat_score": 12.0})
    )
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))

    first = coordinator.propose_defense_intents(RoleAgentContext(observation, assessment, ()))

    assert [intent.actor_scopes[0] for intent in first] == [actor]
    coordinator.record_dispatch(
        ActionCommand(
            command_id="successful-defense-move",
            actor=actor,
            name="Move_Minimap",
            arguments=[[12, 12]],
            created_game_loop=32,
            ttl_game_loops=8,
            source=ActionSource.REFLEX,
        ),
        responsibility="defense",
        game_loop=32,
    )
    transition = coordinator.record_execution(
        ExecutionReport(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=2,
            command_id="successful-defense-move",
            success=True,
            action_name="Move_Minimap",
            actor=actor,
            source=ActionSource.REFLEX,
            requested_arguments=[[12, 12]],
            status=ExecutionStatus.SUCCEEDED,
            execution_stage=ExecutionStage.EFFECT_VERIFICATION,
        ),
        responsibility="defense",
        game_loop=32,
    )

    assert transition is not None
    assert transition["state"] == "defense_response_holding"
    still_threatened = observation.model_copy(update={"step_id": 3, "game_loop": 100})
    assert (
        coordinator.propose_defense_intents(RoleAgentContext(still_threatened, assessment, ()))
        == ()
    )
    expired_holding = observation.model_copy(update={"step_id": 4, "game_loop": 145})
    assert (
        len(coordinator.propose_defense_intents(RoleAgentContext(expired_holding, assessment, ())))
        == 1
    )

    clear_assessment = assessment.model_copy(
        update={"threat_level": ThreatLevel.NONE, "threat_score": 0.0}
    )
    assert (
        coordinator.propose_defense_intents(
            RoleAgentContext(still_threatened, clear_assessment, ())
        )
        == ()
    )
    renewed = observation.model_copy(update={"step_id": 5, "game_loop": 1_001})
    assert len(coordinator.propose_defense_intents(RoleAgentContext(renewed, assessment, ()))) == 1


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


def test_tactical_role_agents_take_single_owner_lineage() -> None:
    observation = _observation()
    profile = race_profile("protoss")
    coordinator = RoleAgentCoordinator(profile, StrategicIntentAdapter(profile))
    intents = (
        TacticalIntent(
            intent_id="attack",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective="Focus fire the visible enemy",
            action_names=["Attack_Unit"],
            actor_scopes=["CombatGroup0/Zealot-1"],
            source_id="combined-tactical-policy",
            source_version="1",
        ),
        TacticalIntent(
            intent_id="advance",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective="Advance to the enemy base",
            action_names=["Move_Minimap"],
            actor_scopes=["CombatGroup0/Zealot-1"],
            source_id="combined-tactical-policy",
            source_version="1",
        ),
        TacticalIntent(
            intent_id="retreat",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective="Retreat this actor to safety",
            action_names=["Move_Minimap"],
            actor_scopes=["CombatGroup0/Zealot-1"],
            source_id="combined-tactical-policy",
            source_version="1",
        ),
    )

    owned = coordinator.own_tactical_intents(intents)

    assert [intent.source_id for intent in owned] == [
        "deterministic-focus-fire-agent",
        "deterministic-offense-agent",
        "deterministic-retreat-agent",
    ]


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


def test_different_actor_same_context_does_not_match_counterfactual() -> None:
    first, second = _candidate_counterfactual_keys(
        actor_a="CombatGroup1",
        actor_b="CombatGroup2",
        arguments_a=["0xabc"],
        arguments_b=["0xabc"],
    )

    assert first != second


def test_different_target_same_context_does_not_match_counterfactual() -> None:
    first, second = _candidate_counterfactual_keys(
        actor_a="CombatGroup1",
        actor_b="CombatGroup1",
        arguments_a=["0xabc"],
        arguments_b=["0xdef"],
    )

    assert first != second


def test_equivalent_intent_matches_across_run_local_operation_ids() -> None:
    rule = PlaybookRule(
        rule_id="rule:avoid-adept",
        canonical_key="avoid-adept",
        category=PlaybookRuleCategory.TACTICAL_RESPONSE,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.FORBID,
        strength=PlaybookRuleStrength.HARD,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("Train_Adept",),
        confidence=1.0,
    )
    first = _intent("Train_Adept", RoleId.PRODUCTION).model_copy(
        update={
            "run_id": "active-run",
            "episode_id": "active-episode",
            "operation_id": "operation:" + "a" * 64,
            "continuity_key": "active-local-continuity",
        }
    )
    second = first.model_copy(
        update={
            "run_id": "shadow-run",
            "episode_id": "shadow-episode",
            "operation_id": "operation:" + "b" * 64,
            "continuity_key": "shadow-local-continuity",
        }
    )
    situation = DeterministicSituationAnalyzer().assess(_observation())
    context = PlaybookContext(
        agent_race="protoss",
        opponent_race="zerg",
        phase=situation.phase,
        map_name="Simple64",
    )
    guard = PlaybookIntentGuard()

    keys = [
        guard.evaluate(
            intent,
            context=context,
            situation=situation,
            rules=(rule,),
            game_loop=32,
            behavior_before_hash="f" * 64,
            mode="shadow",
        )
        .applications[0]
        .counterfactual_key
        for intent in (first, second)
    ]

    assert keys[0] == keys[1]


def test_intent_counterfactual_differs_for_different_target() -> None:
    first = _intent(
        "Attack_Unit",
        RoleId.FOCUS_FIRE,
        actor="CombatGroup8",
        semantic_target_key="unit:0x100",
    )
    second = first.model_copy(update={"semantic_target_key": "unit:0x200"})

    assert _intent_counterfactual_key(first) != _intent_counterfactual_key(second)


def test_intent_counterfactual_differs_for_different_semantic_operation() -> None:
    first = _intent(
        "Move_Minimap",
        RoleId.OFFENSE,
        actor="CombatGroup8",
        semantic_target_key="position:10,20",
    )
    second = first.model_copy(
        update={
            "action_names": ("Attack_Unit",),
            "semantic_target_key": "unit:0x100",
        }
    )

    assert _intent_counterfactual_key(first) != _intent_counterfactual_key(second)


def test_dependency_identity_not_only_dependency_count() -> None:
    first = _intent(
        "Train_VoidRay",
        RoleId.PRODUCTION,
        dependency_intent_ids=("intent:local-a",),
        dependency_semantic_keys=("technology:stargate",),
    )
    second = first.model_copy(
        update={
            "dependency_intent_ids": ("intent:local-b",),
            "dependency_semantic_keys": ("economy:second-gas",),
        }
    )

    assert _intent_counterfactual_key(first) != _intent_counterfactual_key(second)


def _intent_counterfactual_key(intent: StrategicIntent) -> str | None:
    rule = PlaybookRule(
        rule_id="rule:no-operation",
        canonical_key="no-operation",
        category=PlaybookRuleCategory.TACTICAL_RESPONSE,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.FORBID,
        strength=PlaybookRuleStrength.HARD,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=intent.action_names,
        confidence=1.0,
    )
    situation = DeterministicSituationAnalyzer().assess(_observation())
    result = PlaybookIntentGuard().evaluate(
        intent,
        context=PlaybookContext(
            agent_race="protoss",
            opponent_race="zerg",
            phase=situation.phase,
            map_name="Simple64",
        ),
        situation=situation,
        rules=(rule,),
        game_loop=32,
        behavior_before_hash="f" * 64,
        mode="shadow",
    )
    return result.applications[0].counterfactual_key


def _candidate_counterfactual_keys(
    *,
    actor_a: str,
    actor_b: str,
    arguments_a: list[str],
    arguments_b: list[str],
) -> tuple[str | None, str | None]:
    rule = PlaybookRule(
        rule_id="rule:no-attack",
        canonical_key="no-attack",
        category=PlaybookRuleCategory.TACTICAL_RESPONSE,
        conditions=(PlaybookCondition(field="agent_race", value="protoss"),),
        effect=PlaybookRuleEffect.FORBID,
        strength=PlaybookRuleStrength.HARD,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("Attack_Unit",),
        confidence=1.0,
    )
    situation = DeterministicSituationAnalyzer().assess(_observation())
    context = PlaybookContext(
        agent_race="protoss",
        opponent_race="zerg",
        phase=situation.phase,
        map_name="Simple64",
    )
    guard = PlaybookCandidateGuard()
    keys: list[str | None] = []
    for index, (actor, arguments) in enumerate(((actor_a, arguments_a), (actor_b, arguments_b))):
        candidate = ExecutableCandidate(
            candidate_id=f"candidate:{index:064x}",
            observation_fingerprint="0" * 64,
            intent_id="intent:focus-fire",
            action_name="Attack_Unit",
            actor=actor,
            arguments=arguments,
            features=CandidateFeatures(
                action_rank=0,
                actor_rank=index,
                argument_rank=0,
                compile_ordinal=index,
            ),
        )
        result = guard.evaluate(
            candidate,
            role="focus_fire",
            context=context,
            situation=situation,
            rules=(rule,),
            run_id="run",
            episode_id="episode",
            step_id=1,
            game_loop=32,
            behavior_before_hash="f" * 64,
            mode="shadow",
        )
        keys.append(result.applications[0].counterfactual_key)
    return keys[0], keys[1]


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
        category=PlaybookRuleCategory.ENGINE_INVARIANT,
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
