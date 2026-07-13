from __future__ import annotations

from rtscortex.contracts import ActionArgumentType, ActionCommand, ActionSource, AvailableAction
from rtscortex.runtime.validation import ActionArbiter, ActionValidator
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
        arguments=["enemy-1"] if arguments is None else arguments,
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
                    name="Attack_Unit",
                    argument_names=["target"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["army"],
                )
            ]
        }
    )

    outcome = ActionValidator(max_actions=1).validate(
        [command("bad-tag", arguments=["enemy-1"])],
        observation,
    )

    assert outcome.accepted == []
    assert outcome.rejected == ["bad-tag: argument 'target' must be tag"]


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
