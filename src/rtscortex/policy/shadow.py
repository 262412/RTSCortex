"""Historical same-fixture runner for policy subagents."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rtscortex.contracts import ActionCommand, ActionSource, ObservationEnvelope
from rtscortex.memory import read_event_log
from rtscortex.policy.models import (
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
from rtscortex.runtime.validation import ActionValidator


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
        PolicyObservationFixture(
            fixture_id=fixture.fixture_id,
            observation=fixture.observation,
            goal_spec=goal_spec,
            goal_progress=progress_verifier.verify(
                fixture.observation,
                goal_spec,
            ),
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
    proposal: PolicyProposal | None = None,
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
    )


@dataclass(frozen=True)
class _ProposalScores:
    proposed_action_count: int = 0
    legal_action_count: int = 0
    goal_advancing_action_count: int = 0
    control_action_violation_count: int = 0
    legal_action_rate: float | None = None
    goal_advancing_action_rate: float | None = None


def _score_proposal(
    fixture: PolicyObservationFixture,
    proposal: PolicyProposal | None,
) -> _ProposalScores:
    if proposal is None:
        return _ProposalScores()

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
