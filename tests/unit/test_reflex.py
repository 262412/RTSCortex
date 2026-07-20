from __future__ import annotations

from rtscortex.contracts import (
    ActionArgumentType,
    AvailableAction,
    EconomyState,
    SC2State,
    UnitState,
)
from rtscortex.reflex import ReflexEngine
from rtscortex.runtime.validation import ActionValidator
from tests.helpers import make_observation


def test_reflex_emits_retreat_and_attack_for_emergency() -> None:
    engine = ReflexEngine(enabled=True, low_health_threshold=0.25)
    commands = engine.evaluate(make_observation(alerts=["under_attack"], health=0.2))
    assert [(command.actor, command.name) for command in commands] == [
        ("unit-1", "Retreat"),
        ("army", "Attack_Unit"),
    ]
    assert all(command.priority >= 90 for command in commands)


def test_reflex_targets_all_live_actor_scopes_for_sc2_alert() -> None:
    observation = make_observation(alerts=["unit_under_attack"]).model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=[
                        "CombatGroupSmac/Zealot-1",
                        "CombatGroupSmac/Zealot-2",
                        "CombatGroupSmac/Stalker-1",
                    ],
                    argument_candidates=[["0x1"]],
                ),
                AvailableAction(name="No_Operation", actor_scopes=["global"]),
            ]
        }
    )
    commands = ReflexEngine(enabled=True, low_health_threshold=0.25).evaluate(observation)

    assert [command.actor for command in commands] == [
        "CombatGroupSmac/Zealot-1",
        "CombatGroupSmac/Zealot-2",
        "CombatGroupSmac/Stalker-1",
    ]
    assert ActionValidator(max_actions=5).validate(commands, observation).rejected == []


def test_reflex_ignores_builder_and_respects_actor_specific_enemy_candidates() -> None:
    observation = make_observation(alerts=["under_attack"]).model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["Builder/Builder-Probe-1"],
                    argument_candidates=[["0x1"]],
                ),
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup0/Zealot-1"],
                    argument_candidates=[["0x1"]],
                ),
            ]
        }
    )

    commands = ReflexEngine(enabled=True, low_health_threshold=0.25).evaluate(observation)

    assert [(command.actor, command.arguments) for command in commands] == [
        ("CombatGroup0/Zealot-1", ["0x1"])
    ]


def test_zerg_queen_controller_prioritizes_inject_over_creep_when_safe() -> None:
    observation = make_observation().model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Effect_InjectLarva",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup1/Queen-1"],
                    argument_candidates=[["0xb00"]],
                ),
                AvailableAction(
                    name="Build_CreepTumor_Queen_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["CombatGroup1/Queen-1"],
                    argument_candidates=[[[65, 65]]],
                ),
            ]
        }
    )

    commands = ReflexEngine(enabled=True, low_health_threshold=0.25).evaluate(observation)

    assert [(command.actor, command.name, command.arguments) for command in commands] == [
        ("CombatGroup1/Queen-1", "Effect_InjectLarva", ["0xb00"])
    ]


def test_terran_economy_controller_automatically_trains_scv_below_saturation() -> None:
    observation = make_observation().model_copy(
        update={
            "state": SC2State(
                economy=EconomyState(minerals=500, supply_used=14, supply_cap=23, workers=12),
                own_structures=[
                    UnitState(
                        unit_id="0xc00",
                        unit_type="CommandCenter",
                        alliance="self",
                        status="idle",
                    )
                ],
            ),
            "available_actions": [
                AvailableAction(name="Train_SCV", actor_scopes=["Developer/Empty"])
            ],
        }
    )

    commands = ReflexEngine(enabled=True, low_health_threshold=0.25).evaluate(observation)

    assert [(command.actor, command.name, command.priority) for command in commands] == [
        ("Developer/Empty", "Train_SCV", 65)
    ]


def test_chained_creep_is_used_after_queen_controllers_are_unavailable() -> None:
    observation = make_observation().model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Build_CreepTumor_Tumor_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["CombatGroup4/CreepTumor-1"],
                    argument_candidates=[[[78, 64]]],
                )
            ]
        }
    )

    commands = ReflexEngine(enabled=True, low_health_threshold=0.25).evaluate(observation)

    assert [(command.actor, command.name, command.arguments) for command in commands] == [
        ("CombatGroup4/CreepTumor-1", "Build_CreepTumor_Tumor_Screen", [[78, 64]])
    ]


def test_zerg_queen_controller_defers_economy_abilities_during_attack_alert() -> None:
    observation = make_observation(alerts=["under_attack"]).model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Effect_InjectLarva",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup1/Queen-1"],
                    argument_candidates=[["0xb00"]],
                )
            ]
        }
    )

    commands = ReflexEngine(enabled=True, low_health_threshold=0.25).evaluate(observation)

    assert commands == []
