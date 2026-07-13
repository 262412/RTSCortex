"""Deterministically re-run observation events and compare recorded decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rtscortex.config import ExperimentConfig
from rtscortex.contracts import ActionBatch, ExecutionReport, ObservationEnvelope
from rtscortex.memory import read_event_log
from rtscortex.runtime.factory import build_runtime


@dataclass(frozen=True)
class ReplayResult:
    observations: int
    matched_decisions: int
    mismatched_steps: list[int]


async def replay_event_log(
    journal_path: Path, *, config: ExperimentConfig, output_dir: Path
) -> ReplayResult:
    events = list(read_event_log(journal_path))
    expected = {
        event.step_id: ActionBatch.model_validate(event.payload["batch"])
        for event in events
        if event.event_type == "decision"
    }
    executions: dict[int, list[ExecutionReport]] = {}
    for event in events:
        if event.event_type == "execution":
            executions.setdefault(event.step_id, []).append(
                ExecutionReport.model_validate(event.payload)
            )

    runtime = build_runtime(config, output_dir)
    observed = 0
    matched = 0
    mismatched: list[int] = []
    try:
        for event in events:
            if event.event_type != "observation":
                continue
            observation = ObservationEnvelope.model_validate(event.payload)
            batch = await runtime.tick(observation)
            observed += 1
            expected_batch = expected.get(observation.step_id)
            if expected_batch is not None and batch == expected_batch:
                matched += 1
            else:
                mismatched.append(observation.step_id)
            for report in executions.get(observation.step_id, []):
                runtime.record_execution(report)
    finally:
        await runtime.close()
    return ReplayResult(observed, matched, mismatched)
