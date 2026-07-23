from __future__ import annotations

import pytest

from rtscortex.contracts import (
    ActionArgumentType,
    ActionSource,
    AvailableAction,
    EconomyState,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
    ObservationEnvelope,
    SC2State,
    UnitState,
)
from rtscortex.cortex import (
    CandidateCompilationError,
    CandidateCompiler,
    CandidateSelectionStatus,
    DeterministicCandidateExecutor,
    DeterministicSituationAnalyzer,
    DeterministicTacticalAgent,
    EconomyStatus,
    FastExecutorContext,
    GamePhase,
    IntentTarget,
    IntentTargetKind,
    MacroIntent,
    ReflexIntent,
    ResourcePressure,
    TacticalIntent,
    ThreatLevel,
)
from rtscortex.progress import (
    GoalProgressItem,
    GoalProgressReport,
    GoalProgressStatus,
    GoalRequirementKind,
)


def _observation() -> ObservationEnvelope:
    return ObservationEnvelope(
        run_id="run-1",
        episode_id="episode-1",
        step_id=4,
        game_loop=64,
        state=SC2State(
            economy=EconomyState(
                minerals=150,
                supply_used=12,
                supply_cap=23,
                army_supply=4,
            ),
            own_units=[
                UnitState(
                    unit_id="0x10",
                    unit_type="Adept",
                    alliance="self",
                )
            ],
            visible_enemies=[
                UnitState(
                    unit_id="0x20",
                    unit_type="Zergling",
                    alliance="enemy",
                )
            ],
        ),
        available_actions=[
            AvailableAction(
                name="Build_Pylon_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Probe-1", "Builder/Probe-2"],
                argument_candidates=[[[65, 90]], [[70, 90]]],
            ),
            AvailableAction(
                name="Attack_Unit",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["CombatGroup/Army-1"],
                argument_candidates=[["0x10"], ["0x20"]],
            ),
            AvailableAction(
                name="Move_Minimap",
                argument_names=["minimap"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["CombatGroup/Army-1"],
                argument_candidates=[[[20, 30]]],
            ),
        ],
    )


def _macro_intent(*, action_names: list[str] | None = None) -> MacroIntent:
    return MacroIntent(
        intent_id="intent-macro-1",
        run_id="run-1",
        episode_id="episode-1",
        step_id=4,
        created_game_loop=64,
        objective="Establish supply",
        action_names=action_names or ["Build_Pylon_Screen"],
        actor_scopes=["Builder/Probe-1", "Builder/Probe-2"],
        target=IntentTarget(
            kind=IntentTargetKind.PRODUCTION,
            structure_type="Pylon",
        ),
        ttl_game_loops=112,
        source_id="hima-protoss-a",
        source_version="pinned-revision",
        macro_plan_id="plan-1",
    )


def test_candidate_compiler_enumerates_only_exact_available_domain() -> None:
    compiler = CandidateCompiler()

    context = compiler.compile(
        _observation(),
        _macro_intent(),
        busy_actors=("Builder/Probe-1",),
    )
    repeated = compiler.compile(
        _observation(),
        _macro_intent(),
        busy_actors=("Builder/Probe-1",),
    )

    assert [(candidate.actor, candidate.arguments) for candidate in context.candidates] == [
        ("Builder/Probe-2", [[65, 90]]),
        ("Builder/Probe-2", [[70, 90]]),
    ]
    assert [candidate.candidate_id for candidate in repeated.candidates] == [
        candidate.candidate_id for candidate in context.candidates
    ]
    assert [candidate.features.compile_ordinal for candidate in context.candidates] == [
        0,
        1,
    ]
    assert all(
        candidate.observation_fingerprint == context.candidates[0].observation_fingerprint
        for candidate in context.candidates
    )
    assert FastExecutorContext.model_validate_json(context.model_dump_json()) == context


def test_candidate_compiler_filters_friendly_attack_targets() -> None:
    observation = _observation()
    intent = TacticalIntent(
        intent_id="intent-tactical-1",
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        created_game_loop=observation.game_loop,
        objective="Pressure visible enemy",
        action_names=["Attack_Unit"],
        actor_scopes=["CombatGroup/Army-1"],
        target=IntentTarget(kind=IntentTargetKind.ENEMY),
        source_id="deterministic-tactical",
        source_version="0.1.0",
    )

    context = CandidateCompiler().compile(observation, intent)

    assert [candidate.arguments for candidate in context.candidates] == [["0x20"]]


def test_dead_enemy_is_not_a_situation_threat_or_attack_candidate() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0x20",
                            unit_type="Zergling",
                            alliance="enemy",
                            health_fraction=0.0,
                        )
                    ]
                }
            )
        }
    )
    intent = TacticalIntent(
        intent_id="intent-dead-target",
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        created_game_loop=observation.game_loop,
        objective="Do not attack a dead target",
        action_names=["Attack_Unit"],
        actor_scopes=["CombatGroup/Army-1"],
        target=IntentTarget(kind=IntentTargetKind.ENEMY, unit_type="Zergling"),
        source_id="test",
        source_version="1",
    )

    assessment = DeterministicSituationAnalyzer().assess(observation)

    assert assessment.threat_level is ThreatLevel.NONE
    assert assessment.visible_enemy_force.total_units == 0
    assert CandidateCompiler().compile(observation, intent).candidates == []


def test_retreat_state_is_actor_local_and_cools_down_after_arrival() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "own_units": [
                        UnitState(
                            unit_id="0x10",
                            unit_type="Adept",
                            alliance="self",
                            position=(50.0, 50.0),
                            health_fraction=0.2,
                        ),
                        UnitState(
                            unit_id="0x11",
                            unit_type="VoidRay",
                            alliance="self",
                            position=(48.0, 50.0),
                        ),
                    ],
                    "own_structures": [
                        UnitState(
                            unit_id="0x12",
                            unit_type="Nexus",
                            alliance="self",
                            position=(10.0, 10.0),
                        )
                    ],
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Move_Minimap",
                    argument_names=["minimap"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=[
                        "CombatGroup7/Adept-1",
                        "CombatGroup8/VoidRay-1",
                    ],
                    argument_candidates=[[[90, 90]], [[12, 12]]],
                ),
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=[
                        "CombatGroup7/Adept-1",
                        "CombatGroup8/VoidRay-1",
                    ],
                    argument_candidates=[["0x20"]],
                ),
            ],
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
        retreat_cooldown_game_loops=112,
    )

    first = agent.evaluate(
        observation,
        DeterministicSituationAnalyzer().assess(observation),
    )
    next_tick = observation.model_copy(update={"step_id": 5, "game_loop": 65})
    second = agent.evaluate(
        next_tick,
        DeterministicSituationAnalyzer().assess(next_tick),
    )
    arrived = next_tick.model_copy(
        update={
            "step_id": 6,
            "game_loop": 80,
            "state": next_tick.state.model_copy(
                update={
                    "own_units": [
                        next_tick.state.own_units[0].model_copy(
                            update={"position": (12.0, 12.0)}
                        ),
                        next_tick.state.own_units[1],
                    ]
                }
            ),
        }
    )
    third = agent.evaluate(
        arrived,
        DeterministicSituationAnalyzer().assess(arrived),
    )

    assert {(item.actor_scopes[0], item.action_names[0]) for item in first} == {
        ("CombatGroup7/Adept-1", "Move_Minimap"),
        ("CombatGroup8/VoidRay-1", "Attack_Unit"),
    }
    assert [(item.actor_scopes[0], item.action_names[0]) for item in second] == [
        ("CombatGroup8/VoidRay-1", "Attack_Unit")
    ]
    assert [(item.actor_scopes[0], item.action_names[0]) for item in third] == [
        ("CombatGroup8/VoidRay-1", "Attack_Unit")
    ]


def test_tactical_agent_focuses_one_target_and_reacquires_when_it_disappears() -> None:
    observation = _observation().model_copy(
        update={
            "state": _observation().state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0x20",
                            unit_type="Zergling",
                            alliance="enemy",
                            health_fraction=0.2,
                        ),
                        UnitState(
                            unit_id="0x21",
                            unit_type="VoidRay",
                            alliance="enemy",
                        ),
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup/Army-1", "CombatGroup/Army-2"],
                    argument_candidates=[["0x20"], ["0x21"]],
                )
            ],
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
    )
    assessment = DeterministicSituationAnalyzer().assess(observation)

    first = agent.evaluate(observation, assessment)

    assert len(first) == 2
    assert {intent.target.unit_type for intent in first} == {"VoidRay"}
    contexts = [CandidateCompiler().compile(observation, intent) for intent in first]
    assert all(context.candidates[0].arguments == ["0x21"] for context in contexts)

    next_observation = observation.model_copy(
        update={
            "step_id": 5,
            "game_loop": 65,
            "state": observation.state.model_copy(
                update={"visible_enemies": [observation.state.visible_enemies[0]]}
            ),
            "available_actions": [
                observation.available_actions[0].model_copy(
                    update={"argument_candidates": [["0x20"]]}
                )
            ],
        }
    )
    second = agent.evaluate(
        next_observation,
        DeterministicSituationAnalyzer().assess(next_observation),
    )

    assert second
    assert second[0].target.unit_type == "Zergling"
    assert second[0].objective.startswith("Reacquire")


def test_tactical_agent_quarantines_repeated_actor_target_failure() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup/Army-1"],
                    argument_candidates=[["0x20"], ["0x21"]],
                )
            ],
            "state": base.state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0x20",
                            unit_type="VoidRay",
                            alliance="enemy",
                        ),
                        UnitState(
                            unit_id="0x21",
                            unit_type="Zergling",
                            alliance="enemy",
                        ),
                    ]
                }
            ),
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
        target_retry_limit=2,
        target_quarantine_game_loops=112,
    )
    [first] = agent.evaluate(
        observation,
        DeterministicSituationAnalyzer().assess(observation),
    )
    assert first.target.unit_tag == "0x20"
    failure = ExecutionReport(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        command_id="attack-1",
        success=False,
        action_name="Attack_Unit",
        actor="CombatGroup/Army-1",
        source=ActionSource.PLANNER,
        requested_arguments=["0x20"],
        resolved_arguments=["0x20"],
        status=ExecutionStatus.FAILED,
        execution_stage=ExecutionStage.EFFECT_VERIFICATION,
        failure_code="combat_effect_not_observed",
    )

    first_failure = agent.record_execution(failure, game_loop=64)
    second_failure = agent.record_execution(failure, game_loop=72)
    next_observation = observation.model_copy(update={"step_id": 5, "game_loop": 80})
    [next_intent] = agent.evaluate(
        next_observation,
        DeterministicSituationAnalyzer().assess(next_observation),
    )

    assert first_failure is not None and first_failure["state"] == "retryable"
    assert second_failure is not None and second_failure["state"] == "quarantined"
    assert next_intent.target.unit_tag == "0x21"


def test_tactical_agent_attacks_current_screen_structure_when_units_are_last_known() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0x20",
                            unit_type="Zergling",
                            alliance="enemy",
                        ),
                        UnitState(
                            unit_id="0x30",
                            unit_type="Hatchery",
                            alliance="enemy",
                        ),
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup/Army-1"],
                    argument_candidates=[["0x30"]],
                )
            ],
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
    )

    [intent] = agent.evaluate(
        observation,
        DeterministicSituationAnalyzer().assess(observation),
    )
    context = CandidateCompiler().compile(observation, intent)

    assert intent.target.unit_tag == "0x30"
    assert intent.target.unit_type == "Hatchery"
    assert "enemy structure" in intent.objective
    assert [candidate.arguments for candidate in context.candidates] == [["0x30"]]


def test_last_known_enemy_without_screen_target_triggers_reacquire_move() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Move_Minimap",
                    argument_names=["minimap"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["CombatGroup/Army-1"],
                    argument_candidates=[[[20, 30]], [[50, 50]], [[10, 10]]],
                )
            ]
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
        reacquire_cooldown_game_loops=16,
    )

    [first] = agent.evaluate(
        observation,
        DeterministicSituationAnalyzer().assess(observation),
    )
    later = observation.model_copy(update={"step_id": 5, "game_loop": 80})
    [second] = agent.evaluate(
        later,
        DeterministicSituationAnalyzer().assess(later),
    )

    assert first.target.region == "reacquire"
    assert first.target.position == (20, 30)
    assert second.target.position == (50, 50)
    assert CandidateCompiler().compile(later, second).candidates[0].arguments == [
        [50, 50]
    ]


def test_offense_navigation_uses_actor_centroid_and_obsoletes_arrived_waypoint() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "own_units": [
                        UnitState(
                            unit_id="0x10",
                            unit_type="Adept",
                            alliance="self",
                            minimap_position=(5.0, 5.0),
                        ),
                        UnitState(
                            unit_id="0x11",
                            unit_type="VoidRay",
                            alliance="self",
                            minimap_position=(6.0, 6.0),
                        ),
                    ],
                    "visible_enemies": [],
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Move_Minimap",
                    argument_names=["minimap"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=[
                        "CombatGroup7/Adept-1",
                        "CombatGroup8/VoidRay-1",
                    ],
                    argument_candidates=[[[20, 30]], [[50, 50]], [[10, 10]]],
                )
            ],
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
        reacquire_cooldown_game_loops=16,
    )
    first = agent.evaluate(
        observation,
        DeterministicSituationAnalyzer().assess(observation),
    )
    arrived = observation.model_copy(
        update={
            "step_id": 5,
            "game_loop": 80,
            "state": observation.state.model_copy(
                update={
                    "own_units": [
                        observation.state.own_units[0].model_copy(
                            update={"minimap_position": (20.0, 30.0)}
                        ),
                        observation.state.own_units[1],
                    ]
                }
            ),
        }
    )
    second = agent.evaluate(
        arrived,
        DeterministicSituationAnalyzer().assess(arrived),
    )

    assert {intent.actor_scopes[0] for intent in first} == {
        "CombatGroup7/Adept-1",
        "CombatGroup8/VoidRay-1",
    }
    assert [(intent.actor_scopes[0], intent.target.position) for intent in second] == [
        ("CombatGroup7/Adept-1", (50, 50))
    ]


def test_offense_search_prioritizes_last_known_enemy_structure_waypoint() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "own_units": [
                        UnitState(
                            unit_id="0x10",
                            unit_type="Adept",
                            alliance="self",
                            minimap_position=(5.0, 5.0),
                        )
                    ],
                    "visible_enemies": [
                        UnitState(
                            unit_id="0x30",
                            unit_type="Hatchery",
                            alliance="enemy",
                            position=(90.0, 90.0),
                            minimap_position=(48.0, 48.0),
                        )
                    ],
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Move_Minimap",
                    argument_names=["minimap"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["CombatGroup7/Adept-1"],
                    argument_candidates=[[[20, 30]], [[50, 50]], [[10, 10]]],
                )
            ],
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
    )

    [intent] = agent.evaluate(
        observation,
        DeterministicSituationAnalyzer().assess(observation),
    )

    assert intent.target.position == (50, 50)
    assert "last-known enemy structure" in intent.objective


def test_tactical_retreat_selects_reserved_home_minimap_candidate() -> None:
    observation = _observation().model_copy(
        update={
            "state": _observation().state.model_copy(
                update={
                    "own_units": [
                        UnitState(
                            unit_id="0x10",
                            unit_type="Adept",
                            alliance="self",
                            health_fraction=0.2,
                        )
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Move_Minimap",
                    argument_names=["minimap"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["CombatGroup/Army-1"],
                    argument_candidates=[[[90, 90]], [[12, 12]]],
                )
            ],
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
    )

    [intent] = agent.evaluate(
        observation,
        DeterministicSituationAnalyzer().assess(observation),
    )
    context = CandidateCompiler().compile(observation, intent)

    assert intent.target.kind is IntentTargetKind.RETREAT_REGION
    assert [candidate.arguments for candidate in context.candidates] == [[[12, 12]]]


def test_tactical_reacquire_move_is_suppressed_until_cooldown_expires() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(update={"visible_enemies": []}),
            "available_actions": [
                action for action in base.available_actions if action.name == "Move_Minimap"
            ],
        }
    )
    agent = DeterministicTacticalAgent(
        retreat_health_threshold=0.3,
        minimum_advance_army_supply=4,
        reacquire_cooldown_game_loops=112,
    )

    first = agent.evaluate(
        observation,
        DeterministicSituationAnalyzer().assess(observation),
    )
    during_cooldown = observation.model_copy(update={"step_id": 5, "game_loop": 100})
    second = agent.evaluate(
        during_cooldown,
        DeterministicSituationAnalyzer().assess(during_cooldown),
    )
    after_cooldown = observation.model_copy(update={"step_id": 6, "game_loop": 176})
    third = agent.evaluate(
        after_cooldown,
        DeterministicSituationAnalyzer().assess(after_cooldown),
    )

    assert len(first) == 1
    assert second == []
    assert third == []


def test_executor_prefers_goal_advancing_candidate_and_materializes_wire_command() -> None:
    observation = _observation()
    intent = TacticalIntent(
        intent_id="intent-tactical-2",
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        created_game_loop=observation.game_loop,
        objective="Advance while preserving the measurable goal",
        action_names=["Move_Minimap", "Attack_Unit"],
        actor_scopes=["CombatGroup/Army-1"],
        source_id="deterministic-tactical",
        source_version="0.1.0",
        ttl_game_loops=16,
    )
    goal_progress = GoalProgressReport(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        game_loop=observation.game_loop,
        goal_id="pressure",
        strategic_goal="Attack the visible enemy",
        status=GoalProgressStatus.ACTIONABLE,
        missing=[
            GoalProgressItem(
                requirement_id="attack",
                kind=GoalRequirementKind.UNIT,
                target="Zergling",
                required_count=1,
                current_count=0,
            )
        ],
        advancing_actions=["Attack_Unit"],
        unique_next_action="Attack_Unit",
    )
    compiler = CandidateCompiler()
    context = compiler.compile(observation, intent, goal_progress=goal_progress)

    selection = DeterministicCandidateExecutor().select(context)
    command = compiler.materialize(context, selection, command_id="command-cortex-1")

    assert selection.status is CandidateSelectionStatus.SELECTED
    assert command.name == "Attack_Unit"
    assert command.arguments == ["0x20"]
    assert command.source is ActionSource.PLANNER
    assert command.created_game_loop == observation.game_loop
    assert command.ttl_game_loops == intent.ttl_game_loops


def test_reflex_intent_materializes_with_reflex_source() -> None:
    observation = _observation()
    intent = ReflexIntent(
        intent_id="intent-reflex-1",
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        created_game_loop=observation.game_loop,
        objective="Retreat immediately",
        action_names=["Move_Minimap"],
        actor_scopes=["CombatGroup/Army-1"],
        target=IntentTarget(kind=IntentTargetKind.RETREAT_REGION),
        priority=95,
        source_id="low-health-reflex",
        source_version="0.1.0",
    )
    compiler = CandidateCompiler()
    context = compiler.compile(observation, intent)
    selection = DeterministicCandidateExecutor().select(context)

    command = compiler.materialize(context, selection, command_id="command-reflex-1")

    assert command.source is ActionSource.REFLEX
    assert command.priority == 95


def test_executor_abstains_and_compiler_rejects_non_selected_outcome() -> None:
    observation = _observation()
    context = CandidateCompiler().compile(
        observation,
        _macro_intent(action_names=["Train_Oracle"]),
    )

    selection = DeterministicCandidateExecutor().select(context)

    assert selection.status is CandidateSelectionStatus.ABSTAINED
    assert selection.fallback_reason == "no_legal_candidate"
    with pytest.raises(CandidateCompilationError, match="abstained"):
        CandidateCompiler().materialize(context, selection, command_id="command-impossible")


def test_candidate_compiler_never_invents_an_actor_scope() -> None:
    observation = _observation().model_copy(
        update={"available_actions": [AvailableAction(name="Build_Pylon_Screen", actor_scopes=[])]}
    )

    context = CandidateCompiler().compile(observation, _macro_intent())

    assert context.candidates == []


def test_candidate_compiler_rejects_stale_intent() -> None:
    intent = _macro_intent().model_copy(update={"created_game_loop": 63})

    with pytest.raises(CandidateCompilationError, match="stale"):
        CandidateCompiler().compile(_observation(), intent)


def test_deterministic_situation_analyzer_reports_explicit_provenance() -> None:
    assessment = DeterministicSituationAnalyzer(valid_for_game_loops=4).assess(_observation())

    assert assessment.phase is GamePhase.EARLY
    assert assessment.threat_level is ThreatLevel.LOW
    assert assessment.economy_status is EconomyStatus.STABLE
    assert assessment.threats == ["Zergling"]
    assert assessment.valid_until_game_loop == 68
    assert assessment.source_kind == "deterministic"


def test_situation_classifies_large_army_and_independent_resource_pressure() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=20,
                        vespene=817,
                        supply_used=104,
                        supply_cap=120,
                        workers=24,
                        army_supply=80,
                    ),
                    "visible_enemies": [],
                    "own_structures": [
                        UnitState(
                            unit_id="0x50",
                            unit_type="Stargate",
                            alliance="self",
                        )
                    ],
                    "production_queue": [],
                }
            )
        }
    )

    assessment = DeterministicSituationAnalyzer().assess(observation)

    assert assessment.phase is GamePhase.COMBAT
    assert assessment.economy_status is EconomyStatus.CONSTRAINED
    assert assessment.mineral_pressure is ResourcePressure.STARVED
    assert assessment.gas_pressure is ResourcePressure.FLOATING
    facts = {fact.name: fact.evidence for fact in assessment.facts}
    assert facts["mineral_pressure"] == ("starved",)
    assert facts["gas_pressure"] == ("floating",)


def test_situation_threat_uses_alerts_force_ratio_and_hysteresis() -> None:
    base = _observation()
    analyzer = DeterministicSituationAnalyzer(threat_hysteresis_game_loops=32)
    attacked = base.model_copy(
        update={
            "alerts": ["unit_under_attack"],
            "state": base.state.model_copy(
                update={
                    "own_structures": [
                        UnitState(
                            unit_id="0xnexus",
                            unit_type="Nexus",
                            alliance="self",
                            position=(10.0, 10.0),
                        )
                    ],
                    "own_units": [],
                    "visible_enemies": [
                        UnitState(
                            unit_id="0xenemy",
                            unit_type="Roach",
                            alliance="enemy",
                            position=(14.0, 10.0),
                        )
                    ],
                }
            ),
        }
    )

    first = analyzer.assess(attacked)
    persisted = analyzer.assess(
        attacked.model_copy(
            update={
                "step_id": attacked.step_id + 1,
                "game_loop": attacked.game_loop + 16,
                "alerts": [],
                "state": attacked.state.model_copy(
                    update={
                        "own_units": [
                            UnitState(
                                unit_id=f"0xstalker-{index}",
                                unit_type="Stalker",
                                alliance="self",
                            )
                            for index in range(5)
                        ],
                        "visible_enemies": [
                            UnitState(
                                unit_id="0xenemy",
                                unit_type="Roach",
                                alliance="enemy",
                            )
                        ],
                    }
                ),
            }
        )
    )
    temporarily_unseen = analyzer.assess(
        attacked.model_copy(
            update={
                "step_id": attacked.step_id + 2,
                "game_loop": attacked.game_loop + 24,
                "alerts": [],
                "state": attacked.state.model_copy(update={"visible_enemies": []}),
            }
        )
    )
    cleared = analyzer.assess(
        attacked.model_copy(
            update={
                "step_id": attacked.step_id + 3,
                "game_loop": attacked.game_loop + 40,
                "alerts": [],
                "state": attacked.state.model_copy(update={"visible_enemies": []}),
            }
        )
    )

    assert first.threat_level is ThreatLevel.CRITICAL
    assert first.threat_score >= 7.0
    assert "empty_army_overwhelmed" in first.threat_evidence
    assert persisted.threat_level is ThreatLevel.CRITICAL
    assert "hysteresis:critical" in persisted.threat_evidence
    assert temporarily_unseen.threat_level is ThreatLevel.CRITICAL
    assert "enemy_temporarily_unseen" in temporarily_unseen.threat_evidence
    assert cleared.threat_level is ThreatLevel.NONE
    threat_fact = next(fact for fact in first.facts if fact.name == "threat_level")
    assert threat_fact.source == "stateful_threat_rules"
    assert any(item.startswith("score:") for item in threat_fact.evidence)


def test_zero_townhalls_with_living_enemy_is_terminal_combat_crisis() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": base.state.economy.model_copy(
                        update={"army_supply": 0, "workers": 4}
                    ),
                    "own_units": [
                        UnitState(
                            unit_id="0xprobe",
                            unit_type="Probe",
                            alliance="self",
                        )
                    ],
                    "own_structures": [
                        UnitState(
                            unit_id="0xcore",
                            unit_type="CyberneticsCore",
                            alliance="self",
                        )
                    ],
                }
            )
        }
    )

    assessment = DeterministicSituationAnalyzer().assess(observation)

    assert assessment.phase is GamePhase.COMBAT
    assert assessment.threat_level is ThreatLevel.CRITICAL
    phase_fact = next(fact for fact in assessment.facts if fact.name == "game_phase")
    assert "terminal_collapse:no_surviving_townhall" in phase_fact.evidence


def test_situation_threat_records_missing_anti_air_capability() -> None:
    base = _observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "own_structures": [
                        UnitState(
                            unit_id="0xnexus",
                            unit_type="Nexus",
                            alliance="self",
                            position=(10.0, 10.0),
                        )
                    ],
                    "own_units": [
                        UnitState(
                            unit_id=f"0xzealot-{index}",
                            unit_type="Zealot",
                            alliance="self",
                        )
                        for index in range(2)
                    ],
                    "visible_enemies": [
                        UnitState(
                            unit_id=f"0xmutalisk-{index}",
                            unit_type="Mutalisk",
                            alliance="enemy",
                            position=(20.0, 10.0),
                        )
                        for index in range(2)
                    ],
                }
            )
        }
    )

    assessment = DeterministicSituationAnalyzer().assess(observation)

    assert assessment.threat_level is ThreatLevel.CRITICAL
    assert "capability_mismatch:no_anti_air" in assessment.threat_evidence
