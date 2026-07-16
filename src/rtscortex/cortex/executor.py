"""Low-latency deterministic baseline for candidate selection."""

from __future__ import annotations

import hashlib
import time

from rtscortex.cortex.models import (
    CandidateSelection,
    CandidateSelectionStatus,
    ExecutableCandidate,
    FastExecutorContext,
)


class DeterministicCandidateExecutor:
    """Select a legal candidate using stable goal-aware ordering."""

    executor_id = "deterministic-candidate-executor"
    executor_version = "0.1.0"

    def select(self, context: FastExecutorContext) -> CandidateSelection:
        started = time.perf_counter()
        candidate = min(context.candidates, key=self._rank, default=None)
        latency_ms = (time.perf_counter() - started) * 1_000
        if candidate is None:
            return CandidateSelection(
                selection_id=self._selection_id(context, None),
                intent_id=context.intent.intent_id,
                status=CandidateSelectionStatus.ABSTAINED,
                executor_id=self.executor_id,
                executor_version=self.executor_version,
                confidence=1.0,
                latency_ms=latency_ms,
                fallback_reason="no_legal_candidate",
            )
        return CandidateSelection(
            selection_id=self._selection_id(context, candidate),
            intent_id=context.intent.intent_id,
            status=CandidateSelectionStatus.SELECTED,
            candidate_id=candidate.candidate_id,
            executor_id=self.executor_id,
            executor_version=self.executor_version,
            confidence=1.0,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _rank(candidate: ExecutableCandidate) -> tuple[int, int, int, int, int]:
        features = candidate.features
        return (
            0 if features.advances_goal else 1,
            features.action_rank,
            features.actor_rank,
            features.argument_rank,
            features.compile_ordinal,
        )

    def _selection_id(
        self,
        context: FastExecutorContext,
        candidate: ExecutableCandidate | None,
    ) -> str:
        identity = "|".join(
            (
                context.observation.run_id,
                context.observation.episode_id,
                str(context.observation.step_id),
                str(context.observation.game_loop),
                context.intent.intent_id,
                candidate.candidate_id if candidate is not None else "abstain",
                self.executor_id,
                self.executor_version,
            )
        )
        return f"selection:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
