from __future__ import annotations

from rtscortex.reflex import ReflexEngine
from tests.helpers import make_observation


def test_reflex_emits_retreat_and_attack_for_emergency() -> None:
    engine = ReflexEngine(enabled=True, low_health_threshold=0.25)
    commands = engine.evaluate(make_observation(alerts=["under_attack"], health=0.2))
    assert [(command.actor, command.name) for command in commands] == [
        ("unit-1", "Retreat"),
        ("army", "Attack_Unit"),
    ]
    assert all(command.priority >= 90 for command in commands)
