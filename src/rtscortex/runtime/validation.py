"""Action expiry, arbitration, and validation."""

from __future__ import annotations

from dataclasses import dataclass

from rtscortex.contracts import ActionCommand, ActionSource, ObservationEnvelope


class ActionArbiter:
    """Keep the highest-priority command per actor without cancelling other actors."""

    def merge(
        self,
        planner_commands: list[ActionCommand],
        reflex_commands: list[ActionCommand],
        *,
        game_loop: int,
    ) -> list[ActionCommand]:
        active = [
            command
            for command in [*planner_commands, *reflex_commands]
            if game_loop - command.created_game_loop < command.ttl_game_loops
        ]
        by_actor: dict[str, ActionCommand] = {}
        for command in active:
            current = by_actor.get(command.actor)
            if current is None or self._rank(command) > self._rank(current):
                by_actor[command.actor] = command
        return sorted(by_actor.values(), key=self._rank, reverse=True)

    @staticmethod
    def _rank(command: ActionCommand) -> tuple[int, int, str]:
        source_rank = 1 if command.source is ActionSource.REFLEX else 0
        return command.priority, source_rank, command.command_id


@dataclass(frozen=True)
class ValidationOutcome:
    accepted: list[ActionCommand]
    rejected: list[str]


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
            elif len(command.arguments) != len(action.argument_names):
                reason = (
                    f"expected {len(action.argument_names)} arguments, "
                    f"received {len(command.arguments)}"
                )
            elif action.actor_scopes and command.actor not in action.actor_scopes:
                reason = f"actor {command.actor!r} is outside the action scope"

            if reason is not None:
                rejected.append(f"{command.command_id}: {reason}")
                continue
            if len(accepted) >= self.max_actions:
                rejected.append(f"{command.command_id}: action budget exceeded")
                continue
            seen_ids.add(command.command_id)
            accepted.append(command)
        return ValidationOutcome(accepted=accepted, rejected=rejected)
