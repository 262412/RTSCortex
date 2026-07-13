"""Action expiry, arbitration, and validation."""

from __future__ import annotations

from dataclasses import dataclass

from rtscortex.contracts import (
    ActionArgumentType,
    ActionCommand,
    ActionSource,
    ObservationEnvelope,
)


class ActionArbiter:
    """Keep the highest-priority command per actor without cancelling other actors."""

    def arbitrate(
        self,
        planner_commands: list[ActionCommand],
        reflex_commands: list[ActionCommand],
        *,
        game_loop: int,
    ) -> ArbitrationOutcome:
        active = [
            command
            for command in [*planner_commands, *reflex_commands]
            if game_loop - command.created_game_loop < command.ttl_game_loops
        ]
        by_actor: dict[str, ActionCommand] = {}
        preemptions: list[PreemptionRecord] = []
        for command in active:
            current = by_actor.get(command.actor)
            if current is None:
                by_actor[command.actor] = command
                continue
            winner, loser = (
                (command, current)
                if self._rank(command) > self._rank(current)
                else (current, command)
            )
            by_actor[command.actor] = winner
            if winner.source is ActionSource.REFLEX and loser.source is ActionSource.PLANNER:
                preemptions.append(
                    PreemptionRecord(
                        actor=command.actor,
                        winner_command_id=winner.command_id,
                        loser_command_id=loser.command_id,
                    )
                )
        return ArbitrationOutcome(
            selected=sorted(by_actor.values(), key=self._rank, reverse=True),
            preemptions=preemptions,
        )

    def merge(
        self,
        planner_commands: list[ActionCommand],
        reflex_commands: list[ActionCommand],
        *,
        game_loop: int,
    ) -> list[ActionCommand]:
        return self.arbitrate(
            planner_commands,
            reflex_commands,
            game_loop=game_loop,
        ).selected

    @staticmethod
    def _rank(command: ActionCommand) -> tuple[int, int, str]:
        source_rank = 1 if command.source is ActionSource.REFLEX else 0
        return command.priority, source_rank, command.command_id


@dataclass(frozen=True)
class ValidationOutcome:
    accepted: list[ActionCommand]
    rejected: list[str]


@dataclass(frozen=True)
class PreemptionRecord:
    actor: str
    winner_command_id: str
    loser_command_id: str


@dataclass(frozen=True)
class ArbitrationOutcome:
    selected: list[ActionCommand]
    preemptions: list[PreemptionRecord]


class ActionValidator:
    def __init__(self, max_actions: int) -> None:
        self.max_actions = max_actions

    def validate(
        self, commands: list[ActionCommand], observation: ObservationEnvelope
    ) -> ValidationOutcome:
        available = {action.name: action for action in observation.available_actions}
        accepted: list[ActionCommand] = []
        rejected: list[str] = []
        seen_ids: set[str] = set()

        for command in commands:
            action = available.get(command.name)
            reason: str | None = None
            if command.command_id in seen_ids:
                reason = "duplicate command_id"
            elif action is None:
                reason = "action is not available"
            elif command.created_game_loop > observation.game_loop:
                reason = "command creation loop is in the future"
            elif len(command.arguments) != len(action.argument_names):
                reason = (
                    f"expected {len(action.argument_names)} arguments, "
                    f"received {len(command.arguments)}"
                )
            elif action.argument_types:
                invalid_index = next(
                    (
                        index
                        for index, (value, argument_type) in enumerate(
                            zip(command.arguments, action.argument_types, strict=True)
                        )
                        if not _argument_matches(value, argument_type)
                    ),
                    None,
                )
                if invalid_index is not None:
                    reason = (
                        f"argument {action.argument_names[invalid_index]!r} must be "
                        f"{action.argument_types[invalid_index].value}"
                    )
            if (
                reason is None
                and action is not None
                and action.actor_scopes
                and command.actor not in action.actor_scopes
            ):
                reason = f"actor {command.actor!r} is outside the action scope"
            if reason is None:
                reason = _precondition_failure(command, observation)

            if reason is not None:
                rejected.append(f"{command.command_id}: {reason}")
                continue
            if len(accepted) >= self.max_actions:
                rejected.append(f"{command.command_id}: action budget exceeded")
                continue
            seen_ids.add(command.command_id)
            accepted.append(command)
        return ValidationOutcome(accepted=accepted, rejected=rejected)


def _argument_matches(value: object, argument_type: ActionArgumentType) -> bool:
    if argument_type is ActionArgumentType.ANY:
        return True
    if argument_type is ActionArgumentType.STRING:
        return isinstance(value, str)
    if argument_type is ActionArgumentType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if argument_type is ActionArgumentType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if argument_type is ActionArgumentType.BOOLEAN:
        return isinstance(value, bool)
    if argument_type is ActionArgumentType.POSITION:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
                for coordinate in value
            )
        )
    if argument_type is ActionArgumentType.TAG:
        if isinstance(value, int) and not isinstance(value, bool):
            return value >= 0
        if isinstance(value, str) and value.startswith("0x"):
            try:
                return int(value, 16) >= 0
            except ValueError:
                return False
    return False


def _precondition_failure(
    command: ActionCommand,
    observation: ObservationEnvelope,
) -> str | None:
    state = observation.state
    unit_ids = {
        unit.unit_id for unit in [*state.own_units, *state.own_structures, *state.visible_enemies]
    }
    checks = {
        "min_minerals": state.economy.minerals,
        "min_vespene": state.economy.vespene,
        "min_supply_free": state.economy.supply_cap - state.economy.supply_used,
    }
    for name, expected in command.preconditions.items():
        if name in checks:
            if not isinstance(expected, (int, float)) or isinstance(expected, bool):
                return f"precondition {name!r} must be numeric"
            if checks[name] < expected:
                return f"precondition {name!r} is not satisfied"
        elif name == "unit_exists":
            if not isinstance(expected, str):
                return "precondition 'unit_exists' must be a unit ID"
            if expected not in unit_ids:
                return "precondition 'unit_exists' is not satisfied"
        elif name == "enemy_visible":
            if not isinstance(expected, bool):
                return "precondition 'enemy_visible' must be boolean"
            if bool(state.visible_enemies) is not expected:
                return "precondition 'enemy_visible' is not satisfied"
        else:
            return f"unsupported precondition {name!r}"
    return None
