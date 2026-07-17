from __future__ import annotations

import pytest

from rtscortex.contracts import (
    ActionArgumentType,
    ActionSource,
    AvailableAction,
    EconomyState,
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
    assert len(third) == 1


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

    assert assessment.phase is GamePhase.COMBAT
    assert assessment.threat_level is ThreatLevel.LOW
    assert assessment.economy_status is EconomyStatus.STABLE
    assert assessment.threats == ["Zergling"]
    assert assessment.valid_until_game_loop == 68
    assert assessment.source_kind == "deterministic"
