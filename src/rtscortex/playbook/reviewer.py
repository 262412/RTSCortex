"""Conservative post-game extraction for CortexPlaybook."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rtscortex.contracts import EpisodeOutcome, EpisodeResult, ExecutionReport
from rtscortex.cortex.models import GamePhase
from rtscortex.memory import StoredEvent
from rtscortex.playbook.models import (
    DecisionCase,
    DecisionQuality,
    FailureOwner,
    LessonStatus,
    PlaybookContext,
    PlaybookLesson,
)
from rtscortex.playbook.store import PlaybookStore


class CortexPlaybookReviewer:
    """Record evidence first and promote only repeated, outcome-backed lessons."""

    def __init__(self, store: PlaybookStore, *, promotion_support: int = 2) -> None:
        self.store = store
        self.promotion_support = promotion_support

    def review_episode(
        self,
        events: Sequence[StoredEvent],
        result: EpisodeResult,
        *,
        agent_race: str,
        opponent_race: str,
    ) -> tuple[list[DecisionCase], list[PlaybookLesson]]:
        lineages = {
            str(event.payload.get("command_id")): event
            for event in events
            if event.event_type == "command_lineage" and _source_role(event.payload) == "macro"
        }
        phases = _phase_timeline(events)
        cases: list[DecisionCase] = []
        lessons: list[PlaybookLesson] = []
        for event in events:
            if event.event_type != "macro_plan_rejected":
                continue
            rejected = _rejected_proposal_case(
                event,
                result,
                phase=_phase_at(phases, event.event_id),
                agent_race=agent_race,
                opponent_race=opponent_race,
            )
            if self.store.add_case(rejected):
                cases.append(rejected)
        for event in events:
            if event.event_type != "execution":
                continue
            report = ExecutionReport.model_validate(event.payload)
            lineage_event = lineages.get(report.command_id)
            if lineage_event is None:
                continue
            phase = _phase_at(phases, event.event_id)
            semantic_action = str(
                lineage_event.payload.get("semantic_action") or report.action_name or "unknown"
            )
            quality, owner, confidence, consequence = _assess(report, result)
            context = PlaybookContext(
                agent_race=agent_race,
                opponent_race=opponent_race,
                phase=phase,
                map_name=result.scenario,
                tags=(semantic_action.lower().replace(" ", "_"),),
            )
            case = DecisionCase(
                case_id=_stable_id("case", result.run_id, result.episode_id, report.command_id),
                run_id=result.run_id,
                episode_id=result.episode_id,
                source_event_id=event.event_id,
                source_step_id=event.step_id,
                command_id=report.command_id,
                macro_plan_id=_macro_plan_id(lineage_event.payload),
                semantic_action=semantic_action,
                objective=None,
                context=context,
                quality=quality,
                failure_owner=owner,
                consequence=consequence,
                evidence={
                    "execution_status": report.status.value,
                    "execution_stage": (
                        None if report.execution_stage is None else report.execution_stage.value
                    ),
                    "failure_code": report.failure_code,
                    "effect_evidence": (
                        None
                        if report.effect_evidence is None
                        else report.effect_evidence.model_dump(mode="json")
                    ),
                },
                episode_outcome=result.outcome.value,
                confidence=confidence,
            )
            if not self.store.add_case(case):
                continue
            cases.append(case)
            lesson = self._consolidate(case)
            if lesson is not None:
                lessons.append(lesson)
        return cases, lessons

    def _consolidate(self, case: DecisionCase) -> PlaybookLesson | None:
        # Execution failures are diagnostic evidence, not tactical truth. A successful
        # action in a lost game is also insufficient to claim strategic value.
        if case.quality is not DecisionQuality.ADVANTAGE_GAINED:
            return None
        signature = "|".join(
            (
                case.context.agent_race,
                case.context.opponent_race,
                case.context.phase.value,
                case.semantic_action,
                "positive",
            )
        )
        previous = self.store.lesson_by_signature(signature)
        source_ids = tuple(
            dict.fromkeys((*(previous.source_case_ids if previous else ()), case.case_id))
        )
        episode_identity = f"{case.run_id}/{case.episode_id}"
        source_episode_ids = tuple(
            dict.fromkeys((*(previous.source_episode_ids if previous else ()), episode_identity))
        )
        support = len(source_episode_ids)
        status = (
            LessonStatus.PROMOTED if support >= self.promotion_support else LessonStatus.CANDIDATE
        )
        confidence = min(0.95, 0.65 + support * 0.1)
        lesson = PlaybookLesson(
            lesson_id=_stable_id("lesson", signature),
            signature=signature,
            context=case.context,
            statement=(
                f"In {case.context.opponent_race} {case.context.phase.value}, "
                f"{case.semantic_action} had a verified effect in a winning episode."
            ),
            recommended_action=case.semantic_action,
            status=status,
            confidence=confidence,
            support_count=support,
            contradiction_count=0,
            source_case_ids=source_ids,
            source_episode_ids=source_episode_ids,
        )
        self.store.upsert_lesson(lesson)
        return lesson


def _assess(
    report: ExecutionReport,
    result: EpisodeResult,
) -> tuple[DecisionQuality, FailureOwner, float, str]:
    if report.success and result.outcome is EpisodeOutcome.VICTORY:
        return (
            DecisionQuality.ADVANTAGE_GAINED,
            FailureOwner.NONE,
            0.85,
            "The action produced a verified effect and the episode ended in victory.",
        )
    if report.success:
        return (
            DecisionQuality.CORRECT_EXECUTION,
            FailureOwner.NONE,
            0.65,
            "The action produced its expected effect; strategic causality is unresolved.",
        )
    stage = None if report.execution_stage is None else report.execution_stage.value
    owner = (
        FailureOwner.BRIDGE
        if stage in {"translation", "pysc2_acceptance", "effect_verification"}
        else FailureOwner.EXECUTOR
        if stage == "pre_dispatch"
        else FailureOwner.ENVIRONMENT
        if stage == "episode_end"
        else FailureOwner.UNKNOWN
    )
    quality = (
        DecisionQuality.INCONCLUSIVE if stage == "episode_end" else DecisionQuality.EXECUTION_ERROR
    )
    return (
        quality,
        owner,
        0.9 if quality is DecisionQuality.EXECUTION_ERROR else 0.4,
        f"The action ended as {report.status.value} at {stage or 'unknown'}: "
        f"{report.failure_code or report.failure_reason or 'no detail'}.",
    )


def _rejected_proposal_case(
    event: StoredEvent,
    result: EpisodeResult,
    *,
    phase: GamePhase,
    agent_race: str,
    opponent_race: str,
) -> DecisionCase:
    proposal = event.payload.get("proposal")
    steps = proposal.get("steps") if isinstance(proposal, dict) else None
    first = steps[0] if isinstance(steps, list) and steps else None
    raw_action = first.get("canonical_action") if isinstance(first, dict) else None
    semantic_action = raw_action if isinstance(raw_action, str) else "UNUSABLE MACRO PROPOSAL"
    reason = str(event.payload.get("reason") or "unusable_runtime_frontier")
    classification = str(event.payload.get("classification") or "unknown")
    return DecisionCase(
        case_id=_stable_id("case", result.run_id, result.episode_id, str(event.event_id)),
        run_id=result.run_id,
        episode_id=result.episode_id,
        source_event_id=event.event_id,
        source_step_id=event.step_id,
        command_id=f"proposal:{event.event_id}",
        semantic_action=semantic_action,
        context=PlaybookContext(
            agent_race=agent_race,
            opponent_race=opponent_race,
            phase=phase,
            map_name=result.scenario,
            tags=("rejected_proposal", classification),
        ),
        quality=DecisionQuality.STRATEGIC_ERROR,
        failure_owner=FailureOwner.CORTEX,
        consequence=f"The proposal was blocked before dispatch: {reason}.",
        evidence={"classification": classification, "reason": reason},
        episode_outcome=result.outcome.value,
        confidence=0.95,
    )


def _phase_timeline(events: Sequence[StoredEvent]) -> list[tuple[int, GamePhase]]:
    timeline: list[tuple[int, GamePhase]] = []
    for event in events:
        if event.event_type != "situation_assessed":
            continue
        phase = event.payload.get("phase")
        if isinstance(phase, str):
            timeline.append((event.event_id, GamePhase(phase)))
    return timeline


def _phase_at(timeline: list[tuple[int, GamePhase]], event_id: int) -> GamePhase:
    return next(
        (phase for source_id, phase in reversed(timeline) if source_id <= event_id),
        GamePhase.EARLY,
    )


def _source_role(payload: dict[str, object]) -> str | None:
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        return None
    role = lineage.get("source_role")
    return role if isinstance(role, str) else None


def _macro_plan_id(payload: dict[str, object]) -> str | None:
    value = payload.get("macro_plan_id")
    return value if isinstance(value, str) else None


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return f"{prefix}:{digest}"
