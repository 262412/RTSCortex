from __future__ import annotations

from rtscortex.contracts import ActionCommand, ActionSource
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
    )


def test_reflex_preempts_only_same_actor() -> None:
    planner = [command("planner"), command("scout", actor="scout")]
    reflex = [command("reflex", priority=90, source=ActionSource.REFLEX)]
    merged = ActionArbiter().merge(planner, reflex, game_loop=1)
    assert {item.command_id for item in merged} == {"reflex", "scout"}


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
