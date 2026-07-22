"""Action expiry, arbitration, and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rtscortex.contracts import (
    ActionArgumentType,
    ActionCommand,
    ActionSource,
    AvailableAction,
    ObservationEnvelope,
)
from rtscortex.targeting import living_targetable_enemies


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
                if self._precedence(command) > self._precedence(current)
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
    def _precedence(command: ActionCommand) -> tuple[int, int]:
        source_rank = 1 if command.source is ActionSource.REFLEX else 0
        return command.priority, source_rank

    @staticmethod
    def _rank(command: ActionCommand) -> tuple[int, int, str]:
        source_rank = 1 if command.source is ActionSource.REFLEX else 0
        return command.priority, source_rank, command.command_id


@dataclass(frozen=True)
class ValidationOutcome:
    accepted: list[ActionCommand]
    rejected: list[str]
    failures: list[ValidationFailure] = field(default_factory=list)


class ValidationDisposition(StrEnum):
    DEFERRED = "deferred"
    REJECTED = "rejected"
    OBSOLETE = "obsolete"


@dataclass(frozen=True)
class ValidationFailure:
    command: ActionCommand
    reason: str
    disposition: ValidationDisposition


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
        return self._validate(commands, observation, max_actions=self.max_actions)

    def validate_candidates(
        self, commands: list[ActionCommand], observation: ObservationEnvelope
    ) -> ValidationOutcome:
        """Reject invalid candidates without applying the final action budget."""

        return self._validate(commands, observation, max_actions=None)

    def _validate(
        self,
        commands: list[ActionCommand],
        observation: ObservationEnvelope,
        *,
        max_actions: int | None,
    ) -> ValidationOutcome:
        available_by_name: dict[str, list[AvailableAction]] = {}
        for action in observation.available_actions:
            available_by_name.setdefault(action.name, []).append(action)
        accepted: list[ActionCommand] = []
        rejected: list[str] = []
        failures: list[ValidationFailure] = []
        seen_ids: set[str] = set()

        for command in commands:
            reason: str | None = None
            disposition = ValidationDisposition.REJECTED
            if command.command_id in seen_ids:
                reason = "duplicate command_id"
            elif command.created_game_loop > observation.game_loop:
                reason = "command creation loop is in the future"
            named_actions = available_by_name.get(command.name, [])
            if reason is None and command.name == "Attack_Unit":
                attack_failure = _attack_invariant_failure(command, observation)
                if attack_failure is not None:
                    reason = attack_failure.reason
                    disposition = attack_failure.disposition
                elif not named_actions:
                    reason = "target_not_visible"
            scoped_actions = [
                candidate
                for candidate in named_actions
                if not candidate.actor_scopes or command.actor in candidate.actor_scopes
            ]
            if reason is None and not named_actions:
                reason = "action is not available"
                disposition = ValidationDisposition.DEFERRED
            elif reason is None and not scoped_actions:
                reason = f"actor {command.actor!r} is outside the action scope"
            elif reason is None:
                candidate_failures = [
                    _action_failure(command, candidate, observation) for candidate in scoped_actions
                ]
                accepted_index = next(
                    (index for index, failure in enumerate(candidate_failures) if failure is None),
                    None,
                )
                if accepted_index is None:
                    failure = candidate_failures[0]
                    assert failure is not None
                    reason = failure.reason
                    disposition = failure.disposition
            if reason is None:
                failure = _precondition_failure(command, observation)
                if failure is not None:
                    reason = failure.reason
                    disposition = failure.disposition

            if reason is not None:
                rejected.append(f"{command.command_id}: {reason}")
                failures.append(
                    ValidationFailure(
                        command=command,
                        reason=reason,
                        disposition=disposition,
                    )
                )
                continue
            if max_actions is not None and len(accepted) >= max_actions:
                reason = "action budget exceeded"
                rejected.append(f"{command.command_id}: {reason}")
                failures.append(
                    ValidationFailure(
                        command=command,
                        reason=reason,
                        disposition=ValidationDisposition.DEFERRED,
                    )
                )
                continue
            seen_ids.add(command.command_id)
            accepted.append(command)
        return ValidationOutcome(accepted=accepted, rejected=rejected, failures=failures)


@dataclass(frozen=True)
class _ActionFailure:
    reason: str
    disposition: ValidationDisposition = ValidationDisposition.REJECTED


def _action_failure(
    command: ActionCommand,
    action: AvailableAction,
    observation: ObservationEnvelope,
) -> _ActionFailure | None:
    if len(command.arguments) != len(action.argument_names):
        return _ActionFailure(
            f"expected {len(action.argument_names)} arguments, received {len(command.arguments)}"
        )
    if action.argument_types:
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
            return _ActionFailure(
                f"argument {action.argument_names[invalid_index]!r} must be "
                f"{action.argument_types[invalid_index].value}"
            )
    if command.name == "Attack_Unit":
        failure = _attack_invariant_failure(command, observation)
        if failure is not None:
            return failure
    if action.argument_candidates is not None and not any(
        _arguments_match_candidate(command.arguments, candidate, action.argument_types)
        for candidate in action.argument_candidates
    ):
        return _ActionFailure("arguments are outside the available candidate set")
    return None


def _attack_invariant_failure(
    command: ActionCommand,
    observation: ObservationEnvelope,
) -> _ActionFailure | None:
    if command.actor != "army" and not command.actor.startswith("CombatGroup"):
        return _ActionFailure("Attack_Unit requires a combat actor")
    if not command.arguments:
        return _ActionFailure("Attack_Unit requires an enemy target")
    target = _normalize_tag(command.arguments[0])
    enemy_ids = {
        _normalize_tag(enemy.unit_id)
        for enemy in living_targetable_enemies(observation.state.visible_enemies)
    }
    if target in enemy_ids:
        return None
    own_ids = {
        _normalize_tag(unit.unit_id)
        for unit in [*observation.state.own_units, *observation.state.own_structures]
    }
    reason = "friendly_target" if target in own_ids else "target_not_visible"
    return _ActionFailure(reason)


def _arguments_match_candidate(
    arguments: list[Any],
    candidate: list[Any],
    argument_types: list[ActionArgumentType],
) -> bool:
    if len(arguments) != len(candidate):
        return False
    return all(
        _normalize_argument(argument, argument_types[index] if argument_types else None)
        == _normalize_argument(candidate_value, argument_types[index] if argument_types else None)
        for index, (argument, candidate_value) in enumerate(zip(arguments, candidate, strict=True))
    )


def _normalize_argument(value: Any, argument_type: ActionArgumentType | None) -> Any:
    if argument_type is ActionArgumentType.TAG:
        return _normalize_tag(value)
    if argument_type is ActionArgumentType.POSITION and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()


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
) -> _ActionFailure | None:
    state = observation.state
    supply_free = state.economy.supply_cap - state.economy.supply_used
    unit_ids = {
        unit.unit_id
        for unit in [
            *state.own_units,
            *state.own_structures,
            *living_targetable_enemies(state.visible_enemies),
        ]
    }
    checks = {
        "min_minerals": state.economy.minerals,
        "min_vespene": state.economy.vespene,
        "min_supply_free": supply_free,
    }
    for name, expected in command.preconditions.items():
        if name in checks:
            if not isinstance(expected, (int, float)) or isinstance(expected, bool):
                return _ActionFailure(f"precondition {name!r} must be numeric")
            if checks[name] < expected:
                return _ActionFailure(
                    f"precondition {name!r} is not satisfied",
                    ValidationDisposition.DEFERRED,
                )
        elif name == "max_supply_free":
            if not isinstance(expected, (int, float)) or isinstance(expected, bool):
                return _ActionFailure("precondition 'max_supply_free' must be numeric")
            if supply_free > expected:
                return _ActionFailure(
                    "precondition 'max_supply_free' is not satisfied",
                    ValidationDisposition.OBSOLETE,
                )
        elif name == "no_pending_structure":
            if not isinstance(expected, str):
                return _ActionFailure(
                    "precondition 'no_pending_structure' must be a structure type"
                )
            if any(
                structure.unit_type == expected and structure.status == "constructing"
                for structure in state.own_structures
            ):
                return _ActionFailure(
                    "precondition 'no_pending_structure' is not satisfied",
                    ValidationDisposition.OBSOLETE,
                )
        elif name == "structure_absent":
            if not isinstance(expected, str):
                return _ActionFailure("precondition 'structure_absent' must be a structure type")
            if any(structure.unit_type == expected for structure in state.own_structures):
                return _ActionFailure(
                    "precondition 'structure_absent' is not satisfied",
                    ValidationDisposition.OBSOLETE,
                )
        elif name == "unit_exists":
            if not isinstance(expected, str):
                return _ActionFailure("precondition 'unit_exists' must be a unit ID")
            if expected not in unit_ids:
                return _ActionFailure("precondition 'unit_exists' is not satisfied")
        elif name == "enemy_target_exists":
            if not isinstance(expected, str):
                return _ActionFailure("precondition 'enemy_target_exists' must be a unit ID")
            enemy_ids = {
                _normalize_tag(enemy.unit_id)
                for enemy in living_targetable_enemies(state.visible_enemies)
            }
            if _normalize_tag(expected) not in enemy_ids:
                return _ActionFailure("precondition 'enemy_target_exists' is not satisfied")
        elif name == "enemy_visible":
            if not isinstance(expected, bool):
                return _ActionFailure("precondition 'enemy_visible' must be boolean")
            if bool(living_targetable_enemies(state.visible_enemies)) is not expected:
                return _ActionFailure("precondition 'enemy_visible' is not satisfied")
        else:
            return _ActionFailure(f"unsupported precondition {name!r}")
    return None
