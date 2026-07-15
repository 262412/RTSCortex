from __future__ import annotations

from rtscortex.contracts import ActionCommand, ActionSource, AvailableAction
from rtscortex.progress import (
    GoalProgressItem,
    GoalProgressReport,
    GoalProgressStatus,
    GoalRequirementKind,
)
from rtscortex.runtime.progress_guard import (
    CONTROL_ACTION_BLOCKED_REASON,
    ProgressGuard,
)
from tests.helpers import make_observation


def _command(command_id: str, name: str) -> ActionCommand:
    return ActionCommand(
        command_id=command_id,
        actor="Builder/Probe-1",
        name=name,
        source=ActionSource.PLANNER,
        created_game_loop=10,
        ttl_game_loops=16,
    )


def _report(*, defensive_hold_required: bool = False) -> GoalProgressReport:
    return GoalProgressReport(
        run_id="run-1",
        episode_id="episode-1",
        step_id=1,
        game_loop=10,
        goal_id="opening",
        strategic_goal="Build a Gateway",
        status=GoalProgressStatus.ACTIONABLE,
        missing=[
            GoalProgressItem(
                requirement_id="gateway",
                kind=GoalRequirementKind.STRUCTURE,
                target="Gateway",
                required_count=1,
                current_count=0,
            )
        ],
        advancing_actions=["Build_Gateway_Screen"],
        unique_next_action="Build_Gateway_Screen",
        defensive_hold_required=defensive_hold_required,
    )


def test_progress_guard_rejects_controls_when_goal_can_advance() -> None:
    build = _command("build", "Build_Gateway_Screen")
    stop = _command("stop", "Stop")
    hold = _command("hold", "Hold_Position")
    no_op = _command("no-op", "No_Operation")

    outcome = ProgressGuard().filter_commands([stop, build, hold, no_op], _report())

    assert outcome.accepted == [build]
    assert [failure.command for failure in outcome.failures] == [stop, hold, no_op]
    assert {failure.reason for failure in outcome.failures} == {
        CONTROL_ACTION_BLOCKED_REASON
    }


def test_progress_guard_allows_control_when_defence_requires_it() -> None:
    hold = _command("hold", "Hold_Position")

    outcome = ProgressGuard().filter_commands(
        [hold],
        _report(defensive_hold_required=True),
    )

    assert outcome.accepted == [hold]
    assert outcome.failures == []


def test_progress_guard_does_not_cancel_an_effect_already_in_progress() -> None:
    hold = _command("hold", "Hold_Position")
    report = _report().model_copy(
        update={
            "status": GoalProgressStatus.IN_PROGRESS,
            "advancing_actions": [],
            "unique_next_action": None,
        }
    )

    outcome = ProgressGuard().filter_commands([hold], report)

    assert outcome.accepted == []
    assert outcome.failures[0].reason == CONTROL_ACTION_BLOCKED_REASON


def test_progress_guard_projects_forbidden_controls_out_of_planner_schema() -> None:
    observation = make_observation().model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Build_Gateway_Screen",
                    actor_scopes=["Builder/Probe-1"],
                ),
                AvailableAction(name="Stop", actor_scopes=["Builder/Probe-1"]),
                AvailableAction(
                    name="Hold_Position",
                    actor_scopes=["Builder/Probe-1"],
                ),
            ]
        }
    )

    projected = ProgressGuard().project_observation(observation, _report())

    assert [action.name for action in projected.available_actions] == [
        "Build_Gateway_Screen"
    ]
    assert [action.name for action in observation.available_actions] == [
        "Build_Gateway_Screen",
        "Stop",
        "Hold_Position",
    ]
