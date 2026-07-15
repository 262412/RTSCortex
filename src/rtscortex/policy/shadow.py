"""Historical same-fixture runner for policy subagents."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from rtscortex.contracts import ActionCommand, ActionSource, ObservationEnvelope
from rtscortex.memory import read_event_log
from rtscortex.policy.capabilities import DEFAULT_RUNTIME_CAPABILITIES
from rtscortex.policy.hima.mapping import HIMAMacroActionMapper
from rtscortex.policy.models import (
    MacroPolicyProposal,
    PolicyActionAssessment,
    PolicyActionClassification,
    PolicyActionClassificationCounts,
    PolicyAvailabilityStatus,
    PolicyObservationFixture,
    PolicyProposal,
    PolicyShadowComparison,
    PolicyShadowRecord,
    PolicyShadowStatus,
    PolicyShadowSummary,
)
from rtscortex.policy.subagents import PolicySubagentRegistration
from rtscortex.progress import GoalProgressStatus, GoalProgressVerifier, GoalSpec
from rtscortex.runtime.progress_guard import CONTROL_ACTIONS
from rtscortex.runtime.validation import (
    ActionValidator,
    ValidationDisposition,
    ValidationFailure,
)


def build_protoss_opening_goal() -> GoalSpec:
    """Return the fixed measurable goal shared by policy shadow fixtures."""

    return GoalProgressVerifier().goal_from_action_names(
        goal_id="protoss-opening-v1",
        strategic_goal="Build a Pylon, a Gateway, and the first Zealot",
        action_names=[
            "Build_Pylon_Screen",
            "Build_Gateway_Screen",
            "Train_Zealot",
        ],
    )


def load_historical_observations(
    journal_path: Path,
    *,
    stride: int = 1,
    limit: int | None = None,
) -> list[PolicyObservationFixture]:
    """Load deterministic observation fixtures from an RTSCortex event journal."""

    if stride < 1:
        raise ValueError("stride must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    fixtures: list[PolicyObservationFixture] = []
    observation_index = 0
    for event in read_event_log(journal_path):
        if event.event_type != "observation":
            continue
        include = observation_index % stride == 0
        observation_index += 1
        if not include:
            continue
        fixtures.append(
            PolicyObservationFixture(
                fixture_id=(
                    f"{event.run_id}:{event.episode_id}:step-{event.step_id}:event-{event.event_id}"
                ),
                observation=ObservationEnvelope.model_validate(event.payload),
            )
        )
        if limit is not None and len(fixtures) >= limit:
            break
    return fixtures


def attach_goal_progress(
    fixtures: Sequence[PolicyObservationFixture],
    goal_spec: GoalSpec,
    *,
    verifier: GoalProgressVerifier | None = None,
) -> list[PolicyObservationFixture]:
    """Measure one goal against every fixture without invoking a model."""

    progress_verifier = verifier or GoalProgressVerifier()
    return [
        fixture.model_copy(
            update={
                "goal_spec": goal_spec,
                "goal_progress": progress_verifier.verify(
                    fixture.observation,
                    goal_spec,
                ),
            }
        )
        for fixture in fixtures
    ]


class PolicyShadowRunner:
    """Evaluate advisory policies without exposing Runtime or Bridge objects."""

    async def compare(
        self,
        fixtures: Sequence[PolicyObservationFixture],
        registrations: Sequence[PolicySubagentRegistration],
    ) -> PolicyShadowComparison:
        _ensure_unique("fixture", (fixture.fixture_id for fixture in fixtures))
        _ensure_unique("candidate", (item.spec.subagent_id for item in registrations))

        records: list[PolicyShadowRecord] = []
        for fixture in fixtures:
            for registration in registrations:
                records.append(await self._evaluate(fixture, registration))
        return PolicyShadowComparison(
            fixture_ids=[fixture.fixture_id for fixture in fixtures],
            fixtures=list(fixtures),
            candidate_ids=[item.spec.subagent_id for item in registrations],
            records=records,
            summaries=_summarize(records, registrations),
        )

    async def _evaluate(
        self,
        fixture: PolicyObservationFixture,
        registration: PolicySubagentRegistration,
    ) -> PolicyShadowRecord:
        availability = registration.availability
        if availability.status is PolicyAvailabilityStatus.UNAVAILABLE:
            return _record(
                fixture,
                registration,
                status=PolicyShadowStatus.UNAVAILABLE,
            )
        if availability.status is PolicyAvailabilityStatus.SKIPPED:
            return _record(
                fixture,
                registration,
                status=PolicyShadowStatus.SKIPPED,
            )

        assert registration.subagent is not None
        started = time.perf_counter()
        try:
            proposal = await registration.subagent.propose(fixture)
        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000
            return _record(
                fixture,
                registration,
                status=PolicyShadowStatus.FAILED,
                latency_ms=latency_ms,
                error=f"{type(error).__name__}: {error}",
            )
        latency_ms = (time.perf_counter() - started) * 1000
        return _record(
            fixture,
            registration,
            status=PolicyShadowStatus.COMPLETED,
            proposal=proposal,
            latency_ms=latency_ms,
        )


def _record(
    fixture: PolicyObservationFixture,
    registration: PolicySubagentRegistration,
    *,
    status: PolicyShadowStatus,
    proposal: PolicyProposal | MacroPolicyProposal | None = None,
    latency_ms: float = 0.0,
    error: str | None = None,
) -> PolicyShadowRecord:
    scores = _score_proposal(fixture, proposal)
    return PolicyShadowRecord(
        fixture_id=fixture.fixture_id,
        run_id=fixture.observation.run_id,
        episode_id=fixture.observation.episode_id,
        step_id=fixture.observation.step_id,
        game_loop=fixture.observation.game_loop,
        spec=registration.spec,
        availability=registration.availability,
        status=status,
        proposal=proposal,
        goal_id=(
            fixture.goal_progress.goal_id
            if fixture.goal_progress is not None
            else None
        ),
        latency_ms=latency_ms,
        error=error,
        proposed_action_count=scores.proposed_action_count,
        legal_action_count=scores.legal_action_count,
        goal_advancing_action_count=scores.goal_advancing_action_count,
        control_action_violation_count=scores.control_action_violation_count,
        legal_action_rate=scores.legal_action_rate,
        goal_advancing_action_rate=scores.goal_advancing_action_rate,
        action_assessments=list(scores.action_assessments),
        logical_classification_counts=scores.logical_classification_counts,
        effective_classification_counts=scores.effective_classification_counts,
        discovered_macro_step_count=scores.discovered_macro_step_count,
        parsed_known_action_count=scores.parsed_known_action_count,
        effective_action_count=scores.effective_action_count,
        parse_error_count=scores.parse_error_count,
        unsupported_by_runtime_count=scores.unsupported_by_runtime_count,
        mapped_future_count=scores.mapped_future_count,
        mapped_legal_now_count=scores.mapped_legal_now_count,
        mapped_deferred_count=scores.mapped_deferred_count,
        illegal_action_count=scores.illegal_action_count,
        obsolete_count=scores.obsolete_count,
        parse_validity=scores.parse_validity,
        mapping_coverage=scores.mapping_coverage,
        frontier_illegal_rate=scores.frontier_illegal_rate,
    )


@dataclass(frozen=True)
class _ProposalScores:
    proposed_action_count: int = 0
    legal_action_count: int = 0
    goal_advancing_action_count: int = 0
    control_action_violation_count: int = 0
    legal_action_rate: float | None = None
    goal_advancing_action_rate: float | None = None
    action_assessments: tuple[PolicyActionAssessment, ...] = ()
    logical_classification_counts: PolicyActionClassificationCounts = field(
        default_factory=PolicyActionClassificationCounts
    )
    effective_classification_counts: PolicyActionClassificationCounts = field(
        default_factory=PolicyActionClassificationCounts
    )
    discovered_macro_step_count: int = 0
    parsed_known_action_count: int = 0
    effective_action_count: int = 0
    parse_error_count: int = 0
    unsupported_by_runtime_count: int = 0
    mapped_future_count: int = 0
    mapped_legal_now_count: int = 0
    mapped_deferred_count: int = 0
    illegal_action_count: int = 0
    obsolete_count: int = 0
    parse_validity: float | None = None
    mapping_coverage: float | None = None
    frontier_illegal_rate: float | None = None


def _score_proposal(
    fixture: PolicyObservationFixture,
    proposal: PolicyProposal | MacroPolicyProposal | None,
) -> _ProposalScores:
    if proposal is None:
        return _ProposalScores()
    if isinstance(proposal, MacroPolicyProposal):
        assessments = HIMAMacroActionMapper().assess(proposal, fixture)
        scores = _scores_from_assessments(
            assessments,
            discovered_macro_step_count=len(assessments),
            parsed_known_action_count=sum(
                item.classification is not PolicyActionClassification.PARSE_ERROR
                for item in assessments
            ),
            effective_action_count=sum(item.repeat for item in assessments),
        )
        current_proposals = [
            item
            for item in assessments
            if item.is_runtime_frontier
            and item.classification
            in {
                PolicyActionClassification.MAPPED_LEGAL_NOW,
                PolicyActionClassification.MAPPED_DEFERRED,
                PolicyActionClassification.ILLEGAL_ACTION,
                PolicyActionClassification.OBSOLETE,
            }
        ]
        proposed_count = len(current_proposals)
        legal_count = sum(
            item.classification is PolicyActionClassification.MAPPED_LEGAL_NOW
            for item in current_proposals
        )
        report = fixture.goal_progress
        advancing_names = set(report.advancing_actions) if report is not None else set()
        goal_advancing_count = sum(
            item.classification is PolicyActionClassification.MAPPED_LEGAL_NOW
            and item.runtime_action in advancing_names
            for item in current_proposals
        )
        has_progress_opportunity = bool(report is not None and report.advancing_actions)
        return replace(
            scores,
            proposed_action_count=proposed_count,
            legal_action_count=legal_count,
            goal_advancing_action_count=goal_advancing_count,
            legal_action_rate=(
                legal_count / proposed_count if proposed_count else None
            ),
            goal_advancing_action_rate=(
                goal_advancing_count / proposed_count
                if proposed_count and has_progress_opportunity
                else None
            ),
        )

    commands = [
        ActionCommand(
            command_id=f"shadow:{fixture.fixture_id}:{index}",
            actor=action.actor,
            name=action.name,
            arguments=list(action.arguments),
            priority=action.priority,
            ttl_game_loops=1,
            created_game_loop=fixture.observation.game_loop,
            source=ActionSource.PLANNER,
        )
        for index, action in enumerate(proposal.proposed_actions)
    ]
    validated = ActionValidator(max_actions=max(1, len(commands))).validate_candidates(
        commands,
        fixture.observation,
    )
    legal_ids = {command.command_id for command in validated.accepted}
    legal_count = len(legal_ids)
    report = fixture.goal_progress
    advancing_names = set(report.advancing_actions) if report is not None else set()
    goal_advancing_count = sum(
        command.command_id in legal_ids and command.name in advancing_names
        for command in commands
    )
    controls_forbidden = bool(
        report is not None
        and not report.defensive_hold_required
        and (
            report.advancing_actions
            or report.status is GoalProgressStatus.IN_PROGRESS
        )
    )
    control_violations = (
        sum(command.name in CONTROL_ACTIONS for command in commands)
        if controls_forbidden
        else 0
    )
    proposed_count = len(commands)
    has_progress_opportunity = bool(report is not None and report.advancing_actions)
    failures_by_id = {
        failure.command.command_id: failure for failure in validated.failures
    }
    assessments = [
        _native_action_assessment(
            action_name=command.name,
            ordinal=index,
            command_id=command.command_id,
            legal_ids=legal_ids,
            failures_by_id=failures_by_id,
        )
        for index, command in enumerate(commands)
    ]
    v02_scores = _scores_from_assessments(
        assessments,
        effective_action_count=len(assessments),
    )
    return _ProposalScores(
        proposed_action_count=proposed_count,
        legal_action_count=legal_count,
        goal_advancing_action_count=goal_advancing_count,
        control_action_violation_count=control_violations,
        legal_action_rate=legal_count / proposed_count if proposed_count else None,
        goal_advancing_action_rate=(
            goal_advancing_count / proposed_count
            if proposed_count and has_progress_opportunity
            else None
        ),
        action_assessments=tuple(assessments),
        logical_classification_counts=v02_scores.logical_classification_counts,
        effective_classification_counts=v02_scores.effective_classification_counts,
        effective_action_count=v02_scores.effective_action_count,
        mapped_legal_now_count=v02_scores.mapped_legal_now_count,
        mapped_deferred_count=v02_scores.mapped_deferred_count,
        illegal_action_count=v02_scores.illegal_action_count,
        obsolete_count=v02_scores.obsolete_count,
        frontier_illegal_rate=v02_scores.frontier_illegal_rate,
    )


def _native_action_assessment(
    *,
    action_name: str,
    ordinal: int,
    command_id: str,
    legal_ids: set[str],
    failures_by_id: dict[str, ValidationFailure],
) -> PolicyActionAssessment:
    if not DEFAULT_RUNTIME_CAPABILITIES.is_globally_supported(action_name):
        return PolicyActionAssessment(
            ordinal=ordinal,
            source_action=action_name,
            runtime_action=action_name,
            classification=PolicyActionClassification.ILLEGAL_ACTION,
            reason_code="unknown_runtime_action",
            is_logical_frontier=ordinal == 0,
            is_runtime_frontier=ordinal == 0,
            is_frontier=ordinal == 0,
        )
    if command_id in legal_ids:
        return PolicyActionAssessment(
            ordinal=ordinal,
            source_action=action_name,
            runtime_action=action_name,
            classification=PolicyActionClassification.MAPPED_LEGAL_NOW,
            reason_code="validated",
            is_logical_frontier=ordinal == 0,
            is_runtime_frontier=ordinal == 0,
            is_frontier=ordinal == 0,
        )
    failure = failures_by_id.get(command_id)
    disposition = (
        failure.disposition if failure is not None else ValidationDisposition.REJECTED
    )
    reason = failure.reason if failure is not None else "validator_rejected"
    classification = {
        ValidationDisposition.DEFERRED: PolicyActionClassification.MAPPED_DEFERRED,
        ValidationDisposition.OBSOLETE: PolicyActionClassification.OBSOLETE,
        ValidationDisposition.REJECTED: PolicyActionClassification.ILLEGAL_ACTION,
    }[disposition]
    return PolicyActionAssessment(
        ordinal=ordinal,
        source_action=action_name,
        runtime_action=action_name,
        classification=classification,
        reason_code=reason.replace(" ", "_").casefold(),
        is_logical_frontier=ordinal == 0,
        is_runtime_frontier=ordinal == 0,
        is_frontier=ordinal == 0,
    )


def _scores_from_assessments(
    assessments: Sequence[PolicyActionAssessment],
    *,
    discovered_macro_step_count: int = 0,
    parsed_known_action_count: int = 0,
    effective_action_count: int = 0,
) -> _ProposalScores:
    logical_counts = {
        classification: sum(
            item.classification is classification for item in assessments
        )
        for classification in PolicyActionClassification
    }
    effective_counts = {
        classification: sum(
            item.repeat
            for item in assessments
            if item.classification is classification
        )
        for classification in PolicyActionClassification
    }
    logical_vector = _classification_vector(logical_counts)
    effective_vector = _classification_vector(effective_counts)
    classified_effective_count = effective_vector.total
    if effective_action_count not in {0, classified_effective_count}:
        raise ValueError("effective action count must conserve assessment repeats")
    mapped_count = sum(
        logical_counts[classification]
        for classification in (
            PolicyActionClassification.MAPPED_FUTURE,
            PolicyActionClassification.MAPPED_LEGAL_NOW,
            PolicyActionClassification.MAPPED_DEFERRED,
            PolicyActionClassification.ILLEGAL_ACTION,
            PolicyActionClassification.OBSOLETE,
        )
    )
    frontier_evaluated = [
        item
        for item in assessments
        if item.is_runtime_frontier
        and item.classification
        in {
            PolicyActionClassification.MAPPED_LEGAL_NOW,
            PolicyActionClassification.MAPPED_DEFERRED,
            PolicyActionClassification.ILLEGAL_ACTION,
            PolicyActionClassification.OBSOLETE,
        }
    ]
    frontier_illegal = sum(
        item.classification is PolicyActionClassification.ILLEGAL_ACTION
        for item in frontier_evaluated
    )
    return _ProposalScores(
        action_assessments=tuple(assessments),
        logical_classification_counts=logical_vector,
        effective_classification_counts=effective_vector,
        discovered_macro_step_count=discovered_macro_step_count,
        parsed_known_action_count=parsed_known_action_count,
        effective_action_count=classified_effective_count,
        parse_error_count=logical_counts[PolicyActionClassification.PARSE_ERROR],
        unsupported_by_runtime_count=logical_counts[
            PolicyActionClassification.UNSUPPORTED_BY_RUNTIME
        ],
        mapped_future_count=logical_counts[PolicyActionClassification.MAPPED_FUTURE],
        mapped_legal_now_count=logical_counts[
            PolicyActionClassification.MAPPED_LEGAL_NOW
        ],
        mapped_deferred_count=logical_counts[
            PolicyActionClassification.MAPPED_DEFERRED
        ],
        illegal_action_count=logical_counts[PolicyActionClassification.ILLEGAL_ACTION],
        obsolete_count=logical_counts[PolicyActionClassification.OBSOLETE],
        parse_validity=(
            parsed_known_action_count / discovered_macro_step_count
            if discovered_macro_step_count
            else None
        ),
        mapping_coverage=(
            mapped_count / parsed_known_action_count
            if parsed_known_action_count
            else None
        ),
        frontier_illegal_rate=(
            frontier_illegal / len(frontier_evaluated)
            if frontier_evaluated
            else None
        ),
    )


def _classification_vector(
    counts: dict[PolicyActionClassification, int],
) -> PolicyActionClassificationCounts:
    return PolicyActionClassificationCounts(
        parse_error=counts[PolicyActionClassification.PARSE_ERROR],
        unsupported_by_runtime=counts[
            PolicyActionClassification.UNSUPPORTED_BY_RUNTIME
        ],
        mapped_future=counts[PolicyActionClassification.MAPPED_FUTURE],
        mapped_legal_now=counts[PolicyActionClassification.MAPPED_LEGAL_NOW],
        mapped_deferred=counts[PolicyActionClassification.MAPPED_DEFERRED],
        illegal_action=counts[PolicyActionClassification.ILLEGAL_ACTION],
        obsolete=counts[PolicyActionClassification.OBSOLETE],
    )


def _summarize(
    records: Sequence[PolicyShadowRecord],
    registrations: Sequence[PolicySubagentRegistration],
) -> list[PolicyShadowSummary]:
    summaries: list[PolicyShadowSummary] = []
    for registration in registrations:
        candidate_records = [
            record
            for record in records
            if record.spec.subagent_id == registration.spec.subagent_id
        ]
        summaries.append(
            _summarize_candidate(
                registration.spec.subagent_id,
                candidate_records,
            )
        )
    return summaries


def _summarize_candidate(
    subagent_id: str,
    records: Sequence[PolicyShadowRecord],
) -> PolicyShadowSummary:
    proposal_count = sum(record.proposed_action_count for record in records)
    legal_count = sum(record.legal_action_count for record in records)
    goal_advancing_count = sum(record.goal_advancing_action_count for record in records)
    goal_opportunity_proposals = sum(
        record.proposed_action_count
        for record in records
        if record.goal_advancing_action_rate is not None
    )
    goal_opportunity_fixtures = sum(
        record.goal_advancing_action_rate is not None for record in records
    )
    discovered_macro_steps = sum(
        record.discovered_macro_step_count for record in records
    )
    parsed_known_actions = sum(record.parsed_known_action_count for record in records)
    mapped_future = sum(record.mapped_future_count for record in records)
    mapped_legal = sum(record.mapped_legal_now_count for record in records)
    mapped_deferred = sum(record.mapped_deferred_count for record in records)
    illegal_actions = sum(record.illegal_action_count for record in records)
    obsolete = sum(record.obsolete_count for record in records)
    mapped_actions = (
        mapped_future + mapped_legal + mapped_deferred + illegal_actions + obsolete
    )
    frontier_assessments = [
        assessment
        for record in records
        for assessment in record.action_assessments
        if assessment.is_runtime_frontier
        and assessment.classification
        in {
            PolicyActionClassification.MAPPED_LEGAL_NOW,
            PolicyActionClassification.MAPPED_DEFERRED,
            PolicyActionClassification.ILLEGAL_ACTION,
            PolicyActionClassification.OBSOLETE,
        }
    ]
    frontier_illegal = sum(
        assessment.classification is PolicyActionClassification.ILLEGAL_ACTION
        for assessment in frontier_assessments
    )
    effective_actions = sum(record.effective_action_count for record in records)
    parse_errors = sum(record.parse_error_count for record in records)
    unsupported = sum(record.unsupported_by_runtime_count for record in records)
    logical_vector = PolicyActionClassificationCounts(
        parse_error=parse_errors,
        unsupported_by_runtime=unsupported,
        mapped_future=mapped_future,
        mapped_legal_now=mapped_legal,
        mapped_deferred=mapped_deferred,
        illegal_action=illegal_actions,
        obsolete=obsolete,
    )
    effective_vector = PolicyActionClassificationCounts(
        parse_error=sum(
            record.effective_classification_counts.parse_error for record in records
        ),
        unsupported_by_runtime=sum(
            record.effective_classification_counts.unsupported_by_runtime
            for record in records
        ),
        mapped_future=sum(
            record.effective_classification_counts.mapped_future for record in records
        ),
        mapped_legal_now=sum(
            record.effective_classification_counts.mapped_legal_now
            for record in records
        ),
        mapped_deferred=sum(
            record.effective_classification_counts.mapped_deferred for record in records
        ),
        illegal_action=sum(
            record.effective_classification_counts.illegal_action for record in records
        ),
        obsolete=sum(
            record.effective_classification_counts.obsolete for record in records
        ),
    )
    return PolicyShadowSummary(
        subagent_id=subagent_id,
        fixtures=len(records),
        completed=sum(
            record.status is PolicyShadowStatus.COMPLETED for record in records
        ),
        unavailable=sum(
            record.status is PolicyShadowStatus.UNAVAILABLE for record in records
        ),
        skipped=sum(record.status is PolicyShadowStatus.SKIPPED for record in records),
        failed=sum(record.status is PolicyShadowStatus.FAILED for record in records),
        proposals=proposal_count,
        legal_actions=legal_count,
        goal_advancing_actions=goal_advancing_count,
        goal_opportunity_fixtures=goal_opportunity_fixtures,
        goal_opportunity_proposals=goal_opportunity_proposals,
        control_action_violation_count=sum(
            record.control_action_violation_count for record in records
        ),
        legal_action_rate=(legal_count / proposal_count if proposal_count else None),
        goal_advancing_action_rate=(
            goal_advancing_count / goal_opportunity_proposals
            if goal_opportunity_proposals
            else None
        ),
        logical_classification_counts=logical_vector,
        effective_classification_counts=effective_vector,
        discovered_macro_step_count=discovered_macro_steps,
        parsed_known_action_count=parsed_known_actions,
        effective_action_count=effective_actions,
        parse_error_count=parse_errors,
        unsupported_by_runtime_count=unsupported,
        mapped_future_count=mapped_future,
        mapped_legal_now_count=mapped_legal,
        mapped_deferred_count=mapped_deferred,
        illegal_action_count=illegal_actions,
        obsolete_count=obsolete,
        parse_validity=(
            parsed_known_actions / discovered_macro_steps
            if discovered_macro_steps
            else None
        ),
        mapping_coverage=(
            mapped_actions / parsed_known_actions if parsed_known_actions else None
        ),
        frontier_illegal_rate=(
            frontier_illegal / len(frontier_assessments)
            if frontier_assessments
            else None
        ),
    )


def _ensure_unique(kind: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        rendered = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate {kind} ids: {rendered}")
