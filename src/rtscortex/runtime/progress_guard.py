"""Hard action invariants derived from deterministic goal progress."""

from __future__ import annotations

from dataclasses import dataclass

from rtscortex.contracts import ActionCommand, ObservationEnvelope
from rtscortex.progress import GoalProgressReport, GoalProgressStatus
from rtscortex.runtime.validation import ValidationDisposition, ValidationFailure

CONTROL_ACTIONS = frozenset({"No_Operation", "Stop", "Hold_Position"})
CONTROL_ACTION_BLOCKED_REASON = "control_action_blocks_goal_progress"


@dataclass(frozen=True)
class ProgressGuardOutcome:
    accepted: list[ActionCommand]
    failures: list[ValidationFailure]


class ProgressGuard:
    """Reject control actions when a measurable goal can advance now."""

    def filter_commands(
        self,
        commands: list[ActionCommand],
        report: GoalProgressReport | None,
    ) -> ProgressGuardOutcome:
        if not self._blocks_control(report):
            return ProgressGuardOutcome(accepted=list(commands), failures=[])

        accepted: list[ActionCommand] = []
        failures: list[ValidationFailure] = []
        for command in commands:
            if command.name not in CONTROL_ACTIONS:
                accepted.append(command)
                continue
            failures.append(
                ValidationFailure(
                    command=command,
                    reason=CONTROL_ACTION_BLOCKED_REASON,
                    disposition=ValidationDisposition.REJECTED,
                )
            )
        return ProgressGuardOutcome(accepted=accepted, failures=failures)

    def project_observation(
        self,
        observation: ObservationEnvelope,
        report: GoalProgressReport | None,
    ) -> ObservationEnvelope:
        """Hide forbidden controls from the LLM's dynamic output schema."""

        if not self._blocks_control(report):
            return observation
        return observation.model_copy(
            update={
                "available_actions": [
                    action
                    for action in observation.available_actions
                    if action.name not in CONTROL_ACTIONS
                ]
            }
        )

    @staticmethod
    def _blocks_control(report: GoalProgressReport | None) -> bool:
        if report is None or report.defensive_hold_required:
            return False
        return bool(report.advancing_actions) or report.status is GoalProgressStatus.IN_PROGRESS
