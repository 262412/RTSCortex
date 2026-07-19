"""Extension interfaces for specialist Cortex components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rtscortex.contracts import ActionCommand, ObservationEnvelope
from rtscortex.cortex.models import (
    CandidateSelection,
    CortexIntent,
    FastExecutorContext,
    SituationAssessment,
    TacticalIntent,
)
from rtscortex.progress import GoalProgressReport


class SituationAnalyzer(Protocol):
    def assess(
        self,
        observation: ObservationEnvelope,
        history: Sequence[ObservationEnvelope] = (),
    ) -> SituationAssessment: ...


SituationProvider = SituationAnalyzer


class TacticalPolicyProvider(Protocol):
    """Fast tactical policy that can run in deterministic, shadow, or active mode."""

    provider_id: str
    provider_version: str

    def evaluate(
        self,
        observation: ObservationEnvelope,
        situation: SituationAssessment,
    ) -> list[TacticalIntent]: ...


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
