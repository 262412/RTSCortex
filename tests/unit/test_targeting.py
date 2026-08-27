from __future__ import annotations

from rtscortex.contracts import (
    ActionArgumentType,
    AvailableAction,
    ObservationEnvelope,
    SC2State,
    UnitState,
)
from rtscortex.targeting import attackable_enemies_for_actor


def _observation(actor: str, own_type: str) -> ObservationEnvelope:
    enemies = [
        UnitState(
            unit_id="0x10",
            unit_type="Zergling",
            alliance="enemy",
        ),
        UnitState(
            unit_id="0x20",
            unit_type="Mutalisk",
            alliance="enemy",
        ),
    ]
    return ObservationEnvelope(
        run_id="targeting-run",
        episode_id="targeting-episode",
        step_id=1,
        game_loop=16,
        state=SC2State(
            own_units=[
                UnitState(
                    unit_id="0x1",
                    unit_type=own_type,
                    alliance="self",
                )
            ],
            visible_enemies=enemies,
        ),
        available_actions=[
            AvailableAction(
                name="Attack_Unit",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=[actor],
                argument_candidates=[["0x10"], ["0x20"]],
            )
        ],
    )


def test_ground_actor_cannot_target_flying_enemy() -> None:
    actor = "CombatGroup3/Adept-1"
    targets = attackable_enemies_for_actor(_observation(actor, "Adept"), actor)

    assert [target.unit_id for target in targets] == ["0x10"]


def test_anti_air_actor_can_target_flying_enemy() -> None:
    actor = "CombatGroup8/Phoenix-1"
    targets = attackable_enemies_for_actor(_observation(actor, "Phoenix"), actor)

    assert [target.unit_id for target in targets] == ["0x20"]


def test_both_domain_actor_can_target_ground_and_air() -> None:
    actor = "CombatGroup1/Stalker-1"
    targets = attackable_enemies_for_actor(_observation(actor, "Stalker"), actor)

    assert [target.unit_id for target in targets] == ["0x10", "0x20"]
