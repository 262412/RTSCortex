"""Deterministically re-run observation events and compare recorded decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rtscortex.config import ExperimentConfig
from rtscortex.contracts import (
    ActionBatch,
    ActionCommand,
    ExecutionReport,
    IdleReason,
    ObservationEnvelope,
)
from rtscortex.memory import read_event_log
from rtscortex.runtime.factory import build_runtime


@dataclass(frozen=True)
class ReplayResult:
    observations: int
    matched_decisions: int
    mismatched_steps: list[int]
    backfilled_legacy_executions: int = 0
    skipped_legacy_executions: int = 0


async def replay_event_log(
    journal_path: Path, *, config: ExperimentConfig, output_dir: Path
) -> ReplayResult:
    events = list(read_event_log(journal_path))
    expected: dict[tuple[str, str, int], ActionBatch] = {}
    historical_commands: dict[tuple[str, str, str], ActionCommand] = {}
    for event in events:
        if event.event_type != "decision":
            continue
        recorded_batch = ActionBatch.model_validate(event.payload["batch"])
        expected[(event.run_id, event.episode_id, event.step_id)] = _normalize_legacy_batch(
            recorded_batch, config=config
        )
        for command in recorded_batch.commands:
            historical_commands[(event.run_id, event.episode_id, command.command_id)] = command
    executions: dict[tuple[str, str, int], list[ExecutionReport]] = {}
    for event in events:
        if event.event_type == "execution":
            executions.setdefault((event.run_id, event.episode_id, event.step_id), []).append(
                ExecutionReport.model_validate(event.payload)
            )

    runtime = build_runtime(config, output_dir)
    observed = 0
    matched = 0
    mismatched: list[int] = []
    backfilled_legacy_executions = 0
    skipped_legacy_executions = 0
    dispatched: dict[tuple[str, str, str], ActionCommand] = {}
    try:
        for event in events:
            if event.event_type != "observation":
                continue
            observation = ObservationEnvelope.model_validate(event.payload)
            batch = await runtime.tick(observation)
            for command in batch.commands:
                key = (batch.run_id, batch.episode_id, command.command_id)
                previous = dispatched.get(key)
                if previous is not None and previous != command:
                    raise RuntimeError(
                        "replay produced conflicting commands for command_id "
                        f"{command.command_id!r}"
                    )
                dispatched[key] = command
            observed += 1
            event_key = (observation.run_id, observation.episode_id, observation.step_id)
            expected_batch = expected.get(event_key)
            if expected_batch is not None and _same_decision(batch, expected_batch):
                matched += 1
            else:
                mismatched.append(observation.step_id)
            for report in executions.get(event_key, []):
                if report.protocol_version == "1.0":
                    historical_command = historical_commands.get(
                        (report.run_id, report.episode_id, report.command_id)
                    )
                    if historical_command is None:
                        skipped_legacy_executions += 1
                        continue
                    report = _backfill_legacy_execution(report, historical_command)
                    backfilled_legacy_executions += 1
                    dispatched_command = dispatched.get(
                        (report.run_id, report.episode_id, report.command_id)
                    )
                    if dispatched_command is None or not _same_command_semantics(
                        dispatched_command, historical_command
                    ):
                        runtime.store.append_event(
                            run_id=report.run_id,
                            episode_id=report.episode_id,
                            step_id=report.step_id,
                            event_type="legacy_execution_reference",
                            payload={
                                "report": report.model_dump(mode="json"),
                                "reason": (
                                    "replayed runtime did not dispatch the historical command"
                                ),
                            },
                        )
                        skipped_legacy_executions += 1
                        continue
                runtime.record_execution(report)
    finally:
        await runtime.close()
    return ReplayResult(
        observations=observed,
        matched_decisions=matched,
        mismatched_steps=mismatched,
        backfilled_legacy_executions=backfilled_legacy_executions,
        skipped_legacy_executions=skipped_legacy_executions,
    )


def _normalize_legacy_batch(
    batch: ActionBatch,
    *,
    config: ExperimentConfig,
) -> ActionBatch:
    """Turn v1.0 semantic control NoOps into the v1.1 empty-idle representation."""

    if batch.protocol_version != "1.0":
        return batch
    commands = [
        command
        for command in batch.commands
        if not (command.name == "No_Operation" and command.source.value in {"fallback", "planner"})
    ]
    if len(commands) == len(batch.commands):
        return batch
    idle_reason: IdleReason | None = None
    if not commands:
        if config.agent.variant == "noop":
            idle_reason = IdleReason.NOOP_BASELINE
        elif batch.planner_pending:
            idle_reason = IdleReason.WAITING_FOR_PLANNER
        elif batch.strategic_goal or batch.summary:
            idle_reason = IdleReason.PLAN_EXHAUSTED
        else:
            idle_reason = IdleReason.NO_LEGAL_ACTION
    return batch.model_copy(update={"commands": commands, "idle_reason": idle_reason})


def _backfill_legacy_execution(
    report: ExecutionReport,
    command: ActionCommand,
) -> ExecutionReport:
    """Restore semantics only after an exact match to a command dispatched in replay."""

    conflicts = [
        field
        for field, recorded, actual in (
            ("action_name", report.action_name, command.name),
            ("actor", report.actor, command.actor),
            ("source", report.source, command.source),
        )
        if recorded is not None and recorded != actual
    ]
    if report.requested_arguments and report.requested_arguments != command.arguments:
        conflicts.append("requested_arguments")
    if conflicts:
        raise RuntimeError(
            f"legacy execution {report.command_id!r} conflicts with replayed command: "
            + ", ".join(conflicts)
        )
    return report.model_copy(
        update={
            "action_name": command.name,
            "actor": command.actor,
            "source": command.source,
            "requested_arguments": command.arguments,
            "resolved_arguments": report.resolved_arguments or command.arguments,
        }
    )


def _same_command_semantics(left: ActionCommand, right: ActionCommand) -> bool:
    return all(
        (
            left.actor == right.actor,
            left.name == right.name,
            left.arguments == right.arguments,
            left.source is right.source,
            left.preconditions == right.preconditions,
        )
    )


def _same_decision(actual: ActionBatch, expected: ActionBatch) -> bool:
    """Compare behavior while ignoring a historical journal's wire version."""

    if expected.protocol_version != "1.0":
        return actual == expected
    return _legacy_decision_projection(actual) == _legacy_decision_projection(expected)


def _legacy_decision_projection(batch: ActionBatch) -> dict[str, object]:
    return {
        "run_id": batch.run_id,
        "episode_id": batch.episode_id,
        "step_id": batch.step_id,
        "decision_id": batch.decision_id,
        "strategic_goal": batch.strategic_goal,
        "summary": batch.summary,
        "planner_pending": batch.planner_pending,
        "commands": [
            {
                "command_id": command.command_id,
                "actor": command.actor,
                "name": command.name,
                "arguments": command.arguments,
                "priority": command.priority,
                "source": command.source.value,
                "preconditions": command.preconditions,
            }
            for command in batch.commands
        ],
        "rejected_commands": batch.rejected_commands,
    }
