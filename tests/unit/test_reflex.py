from __future__ import annotations

from rtscortex.contracts import AvailableAction
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
                    argument_names=["target"],
                    actor_scopes=[
                        "CombatGroupSmac/Zealot-1",
                        "CombatGroupSmac/Zealot-2",
                        "CombatGroupSmac/Stalker-1",
                    ],
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
