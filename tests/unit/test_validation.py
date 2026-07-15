from __future__ import annotations

from rtscortex.contracts import (
    ActionArgumentType,
    ActionCommand,
    ActionSource,
    AvailableAction,
    EconomyState,
    UnitState,
)
from rtscortex.runtime.validation import (
    ActionArbiter,
    ActionValidator,
    ValidationDisposition,
)
from tests.helpers import make_observation


def command(
    command_id: str,
    *,
    actor: str = "army",
    name: str = "Attack_Unit",
    arguments: list[object] | None = None,
    priority: int = 50,
    source: ActionSource = ActionSource.PLANNER,
    created_game_loop: int = 0,
    ttl: int = 16,
    preconditions: dict[str, object] | None = None,
) -> ActionCommand:
    return ActionCommand(
        command_id=command_id,
        actor=actor,
        name=name,
        arguments=["0x1"] if arguments is None else arguments,
        priority=priority,
        source=source,
        created_game_loop=created_game_loop,
        ttl_game_loops=ttl,
        preconditions={} if preconditions is None else preconditions,
    )


def test_reflex_preempts_only_same_actor() -> None:
    planner = [command("planner"), command("scout", actor="scout")]
    reflex = [command("reflex", priority=90, source=ActionSource.REFLEX)]
    outcome = ActionArbiter().arbitrate(planner, reflex, game_loop=1)
    assert {item.command_id for item in outcome.selected} == {"reflex", "scout"}
    assert len(outcome.preemptions) == 1
    assert outcome.preemptions[0].winner_command_id == "reflex"
    assert outcome.preemptions[0].loser_command_id == "planner"


def test_equal_priority_commands_for_same_actor_keep_plan_order() -> None:
    first = command("plan:0")
    second = command("plan:1")

    outcome = ActionArbiter().arbitrate([first, second], [], game_loop=1)

    assert outcome.selected == [first]


def test_expired_command_is_removed() -> None:
    merged = ActionArbiter().merge([command("old", ttl=4)], [], game_loop=4)
    assert merged == []


def test_validator_rejects_unknown_and_bad_arguments() -> None:
    observation = make_observation()
    outcome = ActionValidator(max_actions=5).validate(
        [
            command("unknown", name="Build_Mothership", arguments=[]),
            command("bad-args", arguments=[]),
        ],
        observation,
    )
    assert outcome.accepted == []
    assert len(outcome.rejected) == 2


def test_validator_applies_action_budget() -> None:
    observation = make_observation()
    outcome = ActionValidator(max_actions=1).validate([command("one"), command("two")], observation)
    assert [item.command_id for item in outcome.accepted] == ["one"]
    assert outcome.rejected == ["two: action budget exceeded"]


def test_validator_rejects_invalid_typed_argument() -> None:
    observation = make_observation().model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Move_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["army"],
                    argument_candidates=[[[40, 40]]],
                )
            ]
        }
    )

    outcome = ActionValidator(max_actions=1).validate(
        [command("bad-position", name="Move_Screen", arguments=["bad"])],
        observation,
    )

    assert outcome.accepted == []
    assert outcome.rejected == ["bad-position: argument 'screen' must be position"]


def test_validator_enforces_preconditions_and_creation_loop() -> None:
    observation = make_observation(game_loop=5)
    outcome = ActionValidator(max_actions=3).validate(
        [
            command("too-expensive", preconditions={"min_minerals": 100}),
            command("missing-unit", preconditions={"unit_exists": "missing"}),
            command("future", created_game_loop=6),
        ],
        observation,
    )

    assert outcome.accepted == []
    assert outcome.rejected == [
        "too-expensive: precondition 'min_minerals' is not satisfied",
        "missing-unit: precondition 'unit_exists' is not satisfied",
        "future: command creation loop is in the future",
    ]


def test_validator_enforces_supply_ceiling_and_pending_structure() -> None:
    base = make_observation()
    tight_supply = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=50,
                        supply_used=11,
                        supply_cap=15,
                    )
                }
            )
        }
    )
    guarded = command(
        "guarded",
        preconditions={
            "max_supply_free": 4,
            "no_pending_structure": "Pylon",
        },
    )

    assert ActionValidator(max_actions=1).validate([guarded], tight_supply).accepted == [guarded]

    high_supply = tight_supply.model_copy(
        update={
            "state": tight_supply.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=50,
                        supply_used=10,
                        supply_cap=15,
                    )
                }
            )
        }
    )
    assert ActionValidator(max_actions=1).validate([guarded], high_supply).rejected == [
        "guarded: precondition 'max_supply_free' is not satisfied"
    ]

    pending_pylon = tight_supply.model_copy(
        update={
            "state": tight_supply.state.model_copy(
                update={
                    "own_structures": [
                        UnitState(
                            unit_id="pylon-1",
                            unit_type="Pylon",
                            alliance="self",
                            status="constructing",
                        )
                    ]
                }
            )
        }
    )
    assert ActionValidator(max_actions=1).validate([guarded], pending_pylon).rejected == [
        "guarded: precondition 'no_pending_structure' is not satisfied"
    ]


def test_validator_enforces_structure_absent() -> None:
    base = make_observation()
    guarded = command(
        "gateway",
        preconditions={"structure_absent": "Gateway"},
    )

    assert ActionValidator(max_actions=1).validate([guarded], base).accepted == [guarded]

    with_gateway = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "own_structures": [
                        UnitState(
                            unit_id="gateway-1",
                            unit_type="Gateway",
                            alliance="self",
                            status="constructing",
                        )
                    ]
                }
            )
        }
    )
    assert ActionValidator(max_actions=1).validate([guarded], with_gateway).rejected == [
        "gateway: precondition 'structure_absent' is not satisfied"
    ]


def test_validator_rejects_friendly_and_non_visible_attack_targets() -> None:
    observation = make_observation()
    outcome = ActionValidator(max_actions=2).validate(
        [
            command("friendly", arguments=["unit-1"]),
            command("missing", arguments=["enemy-missing"]),
        ],
        observation,
    )

    assert outcome.accepted == []
    assert outcome.rejected == [
        "friendly: friendly_target",
        "missing: target_not_visible",
    ]
    assert all(
        failure.disposition is ValidationDisposition.REJECTED for failure in outcome.failures
    )


def test_validator_uses_actor_specific_argument_candidates_for_same_action() -> None:
    base = make_observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "visible_enemies": [
                        UnitState(
                            unit_id="0x1",
                            unit_type="Zergling",
                            alliance="enemy",
                        ),
                        UnitState(
                            unit_id="0x2",
                            unit_type="Roach",
                            alliance="enemy",
                        ),
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["target"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup0/Zealot-1"],
                    argument_candidates=[["0x1"]],
                ),
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["target"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["CombatGroup1/Stalker-1"],
                    argument_candidates=[["0x2"]],
                ),
            ],
        }
    )
    valid = command(
        "valid",
        actor="CombatGroup1/Stalker-1",
        arguments=["0x2"],
    )
    wrong_domain = command(
        "wrong-domain",
        actor="CombatGroup0/Zealot-1",
        arguments=["0x2"],
    )

    outcome = ActionValidator(max_actions=2).validate([valid, wrong_domain], observation)

    assert outcome.accepted == [valid]
    assert outcome.rejected == ["wrong-domain: arguments are outside the available candidate set"]


def test_validator_rejects_candidate_external_screen_coordinate_within_relocation_radius() -> None:
    base = make_observation()
    actor = "Builder/Probe-1"
    observation = base.model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=[actor],
                    argument_candidates=[[[50, 50]], [[55, 50]]],
                )
            ]
        }
    )
    relocated = command(
        "relocated",
        actor=actor,
        name="Build_Pylon_Screen",
        arguments=[[64, 50]],
    )

    outcome = ActionValidator(max_actions=1).validate([relocated], observation)

    assert outcome.accepted == []
    assert outcome.rejected == ["relocated: arguments are outside the available candidate set"]


def test_validator_rejects_screen_relocation_outside_two_sample_strides() -> None:
    base = make_observation()
    actor = "Builder/Probe-1"
    observation = base.model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=[actor],
                    argument_candidates=[[[50, 50]], [[55, 50]]],
                )
            ]
        }
    )
    stale = command(
        "stale",
        actor=actor,
        name="Build_Pylon_Screen",
        arguments=[[66, 50]],
    )

    outcome = ActionValidator(max_actions=1).validate([stale], observation)

    assert outcome.accepted == []
    assert outcome.rejected == ["stale: arguments are outside the available candidate set"]
