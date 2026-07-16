"""Extension interfaces for specialist Cortex components."""

from __future__ import annotations

from typing import Protocol

from rtscortex.contracts import ActionCommand, ObservationEnvelope
from rtscortex.cortex.models import (
    CandidateSelection,
    CortexIntent,
    FastExecutorContext,
    SituationAssessment,
)
from rtscortex.progress import GoalProgressReport


class SituationAnalyzer(Protocol):
    def assess(self, observation: ObservationEnvelope) -> SituationAssessment: ...


class IntentCandidateCompiler(Protocol):
    def compile(
        self,
        observation: ObservationEnvelope,
        intent: CortexIntent,
        *,
        goal_progress: GoalProgressReport | None = None,
        busy_actors: tuple[str, ...] = (),
    ) -> FastExecutorContext: ...

    def materialize(
        self,
        context: FastExecutorContext,
        selection: CandidateSelection,
        *,
        command_id: str,
    ) -> ActionCommand: ...


class FastExecutor(Protocol):
    executor_id: str
    executor_version: str

    def select(self, context: FastExecutorContext) -> CandidateSelection: ...
