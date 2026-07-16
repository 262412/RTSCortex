from __future__ import annotations

import pytest
from pydantic import ValidationError
from rtscortex_llm_pysc2.production import PRODUCTION_SPECS

from rtscortex.contracts import (
    ActionArgumentType,
    ActivePlanSnapshot,
    AvailableAction,
    CommandLifecycleSnapshot,
    EconomyState,
    ObservationEnvelope,
    ProductionItem,
    SC2State,
    UnitState,
)
from rtscortex.progress import (
    GoalBlockerKind,
    GoalProgressStatus,
    GoalProgressVerifier,
    GoalRequirement,
    GoalRequirementKind,
    GoalSpec,
)
from rtscortex.progress.verifier import PROTOSS_SIMPLE64_ACTION_SPECS


def _structure(name: str, *, status: str = "idle", index: int = 1) -> UnitState:
    return UnitState(
        unit_id=f"structure-{index}",
        unit_type=name,
        alliance="self",
        status=status,
    )


def _unit(name: str, *, status: str = "active", index: int = 1) -> UnitState:
    return UnitState(
        unit_id=f"unit-{index}",
        unit_type=name,
        alliance="self",
        status=status,
    )


def _action(name: str) -> AvailableAction:
    if name.endswith("_Screen"):
        return AvailableAction(
            name=name,
            argument_names=["screen"],
            argument_types=[ActionArgumentType.POSITION],
            actor_scopes=["Builder/Builder-Probe-1"],
            argument_candidates=[[[64, 64]]],
        )
    if name.endswith("_Near"):
        return AvailableAction(
            name=name,
            argument_names=["tag"],
            argument_types=[ActionArgumentType.TAG],
            actor_scopes=["Builder/Builder-Probe-1"],
            argument_candidates=[["0x100"]],
        )
    return AvailableAction(name=name, actor_scopes=["Developer/Empty"])


def _observation(
    *,
    minerals: int = 500,
    vespene: int = 500,
    supply_used: int = 12,
    supply_cap: int = 30,
    structures: list[UnitState] | None = None,
    units: list[UnitState] | None = None,
    production_queue: list[ProductionItem] | None = None,
    upgrades: list[str] | None = None,
    actions: list[str] | None = None,
    alerts: list[str] | None = None,
) -> ObservationEnvelope:
    return ObservationEnvelope(
        run_id="run-1",
        episode_id="episode-1",
        step_id=4,
        game_loop=96,
        state=SC2State(
            economy=EconomyState(
                minerals=minerals,
                vespene=vespene,
                supply_used=supply_used,
                supply_cap=supply_cap,
            ),
            own_structures=structures or [_structure("Nexus")],
            own_units=units or [],
            production_queue=production_queue or [],
            upgrades=upgrades or [],
        ),
        available_actions=[_action(name) for name in actions or []],
        alerts=alerts or [],
    )


def test_verifier_identifies_unique_first_action_in_ordered_opening() -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal="Establish production and field the first Zealot",
        action_names=[
            "Build_Pylon_Screen",
            "Build_Gateway_Screen",
            "Train_Zealot",
        ],
    )

    report = verifier.verify(
        _observation(minerals=100, actions=["Build_Pylon_Screen"]),
        goal,
    )

    assert report.status == GoalProgressStatus.ACTIONABLE
    assert report.achieved == []
    assert [item.target for item in report.missing] == ["Pylon", "Gateway", "Zealot"]
    assert report.advancing_actions == ["Build_Pylon_Screen"]
    assert report.unique_next_action == "Build_Pylon_Screen"
    assert report.run_id == "run-1"
    assert report.step_id == 4


def test_warp_gate_goal_accepts_the_canonical_live_upgrade_name() -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal="Unlock Warp Gate",
        action_names=["Research_WarpGate"],
    )

    report = verifier.verify(
        _observation(upgrades=["WarpGateResearch"]),
        goal,
    )

    assert report.status == GoalProgressStatus.ACHIEVED
    assert report.achieved[0].target == "WarpGate"


def test_verifier_waits_for_an_in_progress_requirement_without_reissuing_it() -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal="Establish production",
        action_names=["Build_Pylon_Screen", "Build_Gateway_Screen"],
    )

    report = verifier.verify(
        _observation(
            structures=[
                _structure("Nexus"),
                _structure("Pylon", index=2),
                _structure("Gateway", status="constructing", index=3),
            ],
            actions=["Build_Gateway_Screen"],
        ),
        goal,
    )

    assert report.status == GoalProgressStatus.IN_PROGRESS
    assert [item.target for item in report.achieved] == ["Pylon"]
    assert report.missing[0].target == "Gateway"
    assert report.missing[0].in_progress_count == 1
    assert report.advancing_actions == []
    assert report.unique_next_action is None
    assert GoalBlockerKind.EFFECT_IN_PROGRESS in {
        blocker.kind for blocker in report.blockers
    }


def test_production_queue_counts_as_progress_but_not_completion() -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal="Field a Zealot",
        action_names=["Train_Zealot"],
    )

    report = verifier.verify(
        _observation(
            structures=[_structure("Nexus"), _structure("Gateway", index=2)],
            production_queue=[ProductionItem(name="Train_Zealot", progress=0.5)],
            actions=["Train_Zealot"],
        ),
        goal,
    )

    assert report.status == GoalProgressStatus.IN_PROGRESS
    assert report.missing[0].current_count == 0
    assert report.missing[0].in_progress_count == 1
    assert report.advancing_actions == []


def test_goal_from_actions_adds_plan_delta_to_existing_unit_baseline() -> None:
    verifier = GoalProgressVerifier()
    observation = _observation(
        structures=[_structure("Nexus")],
        units=[_unit("Probe", index=index) for index in range(1, 13)],
        production_queue=[ProductionItem(name="Train_Probe", progress=0.4)],
        actions=["Train_Probe"],
    )

    goal = verifier.goal_from_action_names(
        strategic_goal="Add one more worker",
        action_names=["Train_Probe"],
        observation=observation,
    )

    assert goal.requirements[0].count == 14
    report = verifier.verify(observation, goal)
    assert report.status == GoalProgressStatus.ACTIONABLE
    assert report.missing[0].current_count == 12
    assert report.missing[0].in_progress_count == 1
    assert report.unique_next_action == "Train_Probe"


def test_goal_from_actions_adds_plan_delta_to_existing_structure_baseline() -> None:
    verifier = GoalProgressVerifier()
    observation = _observation(
        structures=[_structure("Nexus"), _structure("Pylon", index=2)],
        actions=["Build_Pylon_Screen"],
    )

    goal = verifier.goal_from_action_names(
        strategic_goal="Add another Pylon",
        action_names=["Build_Pylon_Screen"],
        observation=observation,
    )

    assert goal.requirements[0].count == 2
    report = verifier.verify(observation, goal)
    assert report.status == GoalProgressStatus.ACTIONABLE
    assert report.unique_next_action == "Build_Pylon_Screen"


def test_verifier_marks_goal_achieved_only_from_observed_completed_state() -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal="Establish production and field the first Zealot",
        action_names=[
            "Build_Pylon_Screen",
            "Build_Gateway_Screen",
            "Train_Zealot",
        ],
    )

    report = verifier.verify(
        _observation(
            structures=[
                _structure("Nexus"),
                _structure("Pylon", index=2),
                _structure("Gateway", index=3),
            ],
            units=[_unit("Zealot")],
        ),
        goal,
    )

    assert report.status == GoalProgressStatus.ACHIEVED
    assert [item.target for item in report.achieved] == ["Pylon", "Gateway", "Zealot"]
    assert report.missing == []
    assert report.blockers == []
    assert report.advancing_actions == []


def test_verifier_resolves_implicit_tech_tree_prerequisites() -> None:
    verifier = GoalProgressVerifier()
    goal = GoalSpec(
        strategic_goal="Field a Stalker",
        requirements=[
            GoalRequirement(
                requirement_id="first-stalker",
                kind=GoalRequirementKind.UNIT,
                target="Stalker",
            )
        ],
    )

    report = verifier.verify(
        _observation(minerals=100, actions=["Build_Pylon_Screen"]),
        goal,
    )

    assert report.advancing_actions == ["Build_Pylon_Screen"]
    assert report.unique_next_action == "Build_Pylon_Screen"
    assert GoalBlockerKind.MISSING_PREREQUISITE in {
        blocker.kind for blocker in report.blockers
    }


def test_verifier_reports_resource_blocker_instead_of_an_illegal_action() -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal="Build the first Pylon",
        action_names=["Build_Pylon_Screen"],
    )

    report = verifier.verify(
        _observation(minerals=99, actions=["Build_Pylon_Screen"]),
        goal,
    )

    assert report.status == GoalProgressStatus.BLOCKED
    assert report.advancing_actions == []
    assert report.blockers[0].kind == GoalBlockerKind.INSUFFICIENT_MINERALS


def test_multiple_independent_actions_do_not_claim_a_unique_next_action() -> None:
    verifier = GoalProgressVerifier()
    goal = GoalSpec(
        strategic_goal="Add supply and gas",
        requirements=[
            GoalRequirement(
                requirement_id="supply",
                kind=GoalRequirementKind.STRUCTURE,
                target="Pylon",
            ),
            GoalRequirement(
                requirement_id="gas",
                kind=GoalRequirementKind.STRUCTURE,
                target="Assimilator",
            ),
        ],
    )

    report = verifier.verify(
        _observation(
            actions=["Build_Pylon_Screen", "Build_Assimilator_Near"],
        ),
        goal,
    )

    assert report.advancing_actions == [
        "Build_Pylon_Screen",
        "Build_Assimilator_Near",
    ]
    assert report.unique_next_action is None


def test_defensive_hold_is_derived_from_the_same_alerts_as_reflex() -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal="Build the first Pylon",
        action_names=["Build_Pylon_Screen"],
    )

    report = verifier.verify(
        _observation(
            minerals=100,
            actions=["Build_Pylon_Screen"],
            alerts=["building_under_attack"],
        ),
        goal,
    )

    assert report.defensive_hold_required is True


def test_active_plan_projection_ignores_control_commands() -> None:
    verifier = GoalProgressVerifier()
    active_plan = ActivePlanSnapshot(
        strategic_goal="Establish production",
        summary="Build before waiting",
        commands=(
            CommandLifecycleSnapshot(
                command_id="hold",
                actor="Builder/Builder-Probe-1",
                name="Hold_Position",
                arguments=(),
                source="planner",
                status="pending",
                reason=None,
                created_game_loop=0,
                ttl_game_loops=112,
            ),
            CommandLifecycleSnapshot(
                command_id="pylon",
                actor="Builder/Builder-Probe-1",
                name="Build_Pylon_Screen",
                arguments=([64, 64],),
                source="planner",
                status="pending",
                reason=None,
                created_game_loop=0,
                ttl_game_loops=112,
            ),
        ),
    )

    goal = verifier.goal_from_active_plan(active_plan)

    assert [requirement.action_name for requirement in goal.requirements] == [
        "Build_Pylon_Screen"
    ]


def test_goal_graph_rejects_cycles_and_unknown_actions() -> None:
    with pytest.raises(ValidationError, match="acyclic"):
        GoalSpec(
            strategic_goal="Invalid cycle",
            requirements=[
                GoalRequirement(
                    requirement_id="a",
                    kind=GoalRequirementKind.STRUCTURE,
                    target="Pylon",
                    depends_on=["b"],
                ),
                GoalRequirement(
                    requirement_id="b",
                    kind=GoalRequirementKind.STRUCTURE,
                    target="Gateway",
                    depends_on=["a"],
                ),
            ],
        )

    verifier = GoalProgressVerifier()
    with pytest.raises(ValueError, match="unsupported goal action: Stop"):
        verifier.goal_from_action_names(
            strategic_goal="Do nothing",
            action_names=["Stop"],
        )


@pytest.mark.parametrize(
    ("action_name", "structures", "mineral_cost", "vespene_cost", "supply_cost"),
    [
        (
            "Build_Stargate_Screen",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore"],
            150,
            150,
            0,
        ),
        (
            "Train_Adept",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore"],
            100,
            25,
            2,
        ),
        (
            "Train_VoidRay",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore", "Stargate"],
            250,
            150,
            4,
        ),
        (
            "Build_ShieldBattery_Screen",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore"],
            100,
            0,
            0,
        ),
        (
            "Train_Oracle",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore", "Stargate"],
            150,
            150,
            3,
        ),
        (
            "Train_Phoenix",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore", "Stargate"],
            150,
            100,
            2,
        ),
    ],
)
def test_goal_progress_registers_extended_protoss_action_costs(
    action_name: str,
    structures: list[str],
    mineral_cost: int,
    vespene_cost: int,
    supply_cost: int,
) -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal=f"Complete {action_name}",
        action_names=[action_name],
    )
    unit_states = [_structure(name, index=index + 1) for index, name in enumerate(structures)]

    exact = verifier.verify(
        _observation(
            minerals=mineral_cost,
            vespene=vespene_cost,
            supply_used=20,
            supply_cap=20 + supply_cost,
            structures=unit_states,
            actions=[action_name],
        ),
        goal,
    )
    assert exact.status is GoalProgressStatus.ACTIONABLE
    assert exact.unique_next_action == action_name

    mineral_blocked = verifier.verify(
        _observation(
            minerals=mineral_cost - 1,
            vespene=vespene_cost,
            supply_used=20,
            supply_cap=20 + supply_cost,
            structures=unit_states,
            actions=[action_name],
        ),
        goal,
    )
    assert mineral_blocked.status is GoalProgressStatus.BLOCKED
    assert GoalBlockerKind.INSUFFICIENT_MINERALS in {
        blocker.kind for blocker in mineral_blocked.blockers
    }

    if vespene_cost:
        vespene_blocked = verifier.verify(
            _observation(
                minerals=mineral_cost,
                vespene=vespene_cost - 1,
                supply_used=20,
                supply_cap=20 + supply_cost,
                structures=unit_states,
                actions=[action_name],
            ),
            goal,
        )
        assert vespene_blocked.status is GoalProgressStatus.BLOCKED
        assert GoalBlockerKind.INSUFFICIENT_VESPENE in {
            blocker.kind for blocker in vespene_blocked.blockers
        }

    if supply_cost:
        supply_blocked = verifier.verify(
            _observation(
                minerals=mineral_cost,
                vespene=vespene_cost,
                supply_used=20,
                supply_cap=20 + supply_cost - 1,
                structures=unit_states,
                actions=[action_name],
            ),
            goal,
        )
        assert supply_blocked.status is GoalProgressStatus.BLOCKED
        assert GoalBlockerKind.INSUFFICIENT_SUPPLY in {
            blocker.kind for blocker in supply_blocked.blockers
        }


@pytest.mark.parametrize(
    ("goal_action", "structures", "available_actions", "expected_next_action"),
    [
        (
            "Build_Stargate_Screen",
            ["Nexus"],
            ["Build_Pylon_Screen"],
            "Build_Pylon_Screen",
        ),
        (
            "Train_Adept",
            ["Nexus", "Pylon", "Gateway"],
            ["Build_CyberneticsCore_Screen"],
            "Build_CyberneticsCore_Screen",
        ),
        (
            "Train_VoidRay",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore"],
            ["Build_Stargate_Screen"],
            "Build_Stargate_Screen",
        ),
        (
            "Build_ShieldBattery_Screen",
            ["Nexus", "Pylon", "Gateway"],
            ["Build_CyberneticsCore_Screen"],
            "Build_CyberneticsCore_Screen",
        ),
        (
            "Train_Oracle",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore"],
            ["Build_Stargate_Screen"],
            "Build_Stargate_Screen",
        ),
        (
            "Train_Phoenix",
            ["Nexus", "Pylon", "Gateway", "CyberneticsCore"],
            ["Build_Stargate_Screen"],
            "Build_Stargate_Screen",
        ),
    ],
)
def test_goal_progress_resolves_extended_protoss_prerequisite_chain(
    goal_action: str,
    structures: list[str],
    available_actions: list[str],
    expected_next_action: str,
) -> None:
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal=f"Complete {goal_action}",
        action_names=[goal_action],
    )
    report = verifier.verify(
        _observation(
            minerals=1000,
            vespene=1000,
            supply_used=10,
            supply_cap=30,
            structures=[_structure(name, index=index + 1) for index, name in enumerate(structures)],
            actions=available_actions,
        ),
        goal,
    )

    assert report.status is GoalProgressStatus.ACTIONABLE
    assert report.unique_next_action == expected_next_action


def test_worker_and_runtime_production_specs_remain_contract_equivalent() -> None:
    """Keep the Python 3.9 Worker registry aligned with the Python 3.11 Runtime."""

    runtime_specs = {
        spec.name: spec
        for spec in PROTOSS_SIMPLE64_ACTION_SPECS
        if spec.name in PRODUCTION_SPECS
    }

    assert set(runtime_specs) == set(PRODUCTION_SPECS)
    for action_name, worker_spec in PRODUCTION_SPECS.items():
        runtime_spec = runtime_specs[action_name]
        assert runtime_spec.effect_target == worker_spec.unit_type
        assert (runtime_spec.minerals, runtime_spec.vespene, runtime_spec.supply) == (
            worker_spec.minerals,
            worker_spec.vespene,
            worker_spec.supply,
        )
        assert {item.target for item in runtime_spec.prerequisites} == set(
            worker_spec.prerequisites
        )
