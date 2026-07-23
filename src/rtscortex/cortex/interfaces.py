"""Extension interfaces for specialist Cortex components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rtscortex.contracts import ActionCommand, ExecutionReport, ObservationEnvelope
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


@runtime_checkable
class ExecutionAwareTacticalPolicyProvider(Protocol):
    """Tactical policy that consumes terminal command evidence."""

    def record_execution(
        self,
        report: ExecutionReport,
        *,
        game_loop: int,
    ) -> dict[str, object] | None: ...


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
