from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import NoReturn

import pytest

from rtscortex.agents.models import ActionProposal, PlanningOutput
from rtscortex.contracts import ActionArgumentType, AvailableAction, UnitState
from rtscortex.contracts.interfaces import ResponseT
from rtscortex.policy import (
    HIERNET_SC2_SPEC,
    HIMA_PROTOSS_SPECS,
    QWEN3_8B_SPEC,
    LLMPlanningPolicySubagent,
    PolicyAvailability,
    PolicyAvailabilityStatus,
    PolicyObservationFixture,
    PolicyProposal,
    PolicyShadowRunner,
    PolicyShadowStatus,
    PolicySubagentRegistration,
    attach_goal_progress,
    built_in_policy_specs,
    default_shadow_registrations,
    load_historical_observations,
)
from rtscortex.progress import (
    GoalProgressItem,
    GoalProgressReport,
    GoalProgressStatus,
    GoalProgressVerifier,
    GoalRequirement,
    GoalRequirementKind,
    GoalSpec,
)
from tests.helpers import make_observation


class RecordingSubagent:
    def __init__(self, *, label: str = "recorded") -> None:
        self.spec = QWEN3_8B_SPEC
        self.label = label
        self.received: list[PolicyObservationFixture] = []

    async def propose(self, fixture: PolicyObservationFixture) -> PolicyProposal:
        self.received.append(fixture)
        return PolicyProposal(strategic_goal=self.label)


class FailingSubagent(RecordingSubagent):
    async def propose(self, fixture: PolicyObservationFixture) -> NoReturn:
        self.received.append(fixture)
        raise RuntimeError("shadow failure")


class FixedProposalSubagent(RecordingSubagent):
    def __init__(self, proposal: PolicyProposal) -> None:
        super().__init__()
        self.proposal = proposal

    async def propose(self, fixture: PolicyObservationFixture) -> PolicyProposal:
        self.received.append(fixture)
        return self.proposal


def _available(subagent: RecordingSubagent) -> PolicySubagentRegistration:
    return PolicySubagentRegistration(
        spec=QWEN3_8B_SPEC,
        availability=PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
        subagent=subagent,
    )


def test_catalog_covers_requested_candidates_without_loading_models() -> None:
    specs = built_in_policy_specs()

    assert specs == (QWEN3_8B_SPEC, *HIMA_PROTOSS_SPECS, HIERNET_SC2_SPEC)
    assert [spec.subagent_id for spec in specs] == [
        "qwen3-8b-current",
        "hima-protoss-a",
        "hima-protoss-b",
        "hima-protoss-c",
        "hiernet-sc2-protoss",
    ]
    assert all(spec.shadow_only for spec in specs)


def test_default_catalog_reports_skipped_and_unavailable_explicitly() -> None:
    registrations = default_shadow_registrations()

    assert registrations[0].availability.status is PolicyAvailabilityStatus.SKIPPED
    assert "not configured" in (registrations[0].availability.reason or "")
    assert all(
        item.availability.status is PolicyAvailabilityStatus.UNAVAILABLE
        for item in registrations[1:]
    )
    assert all(
        "no download attempted" in (item.availability.reason or "")
        for item in registrations[1:]
    )
    assert all(item.subagent is None for item in registrations)


def test_available_registration_requires_matching_implementation() -> None:
    with pytest.raises(ValueError, match="require a subagent"):
        PolicySubagentRegistration(
            spec=QWEN3_8B_SPEC,
            availability=PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
        )


def test_runner_uses_identical_fixtures_and_never_creates_commands() -> None:
    observation = make_observation()
    fixtures = [
        PolicyObservationFixture(fixture_id="opening", observation=observation),
        PolicyObservationFixture(
            fixture_id="later",
            observation=observation.model_copy(update={"step_id": 4, "game_loop": 96}),
        ),
    ]
    first = RecordingSubagent(label="first")
    second = RecordingSubagent(label="second")
    second.spec = QWEN3_8B_SPEC.model_copy(update={"subagent_id": "second"})
    comparison = asyncio.run(
        PolicyShadowRunner().compare(
            fixtures,
            [
                _available(first),
                PolicySubagentRegistration(
                    spec=second.spec,
                    availability=PolicyAvailability(
                        status=PolicyAvailabilityStatus.AVAILABLE
                    ),
                    subagent=second,
                ),
            ],
        )
    )

    assert first.received == fixtures
    assert second.received == fixtures
    assert comparison.fixture_ids == ["opening", "later"]
    assert len(comparison.records) == 4
    assert all(record.shadow_only for record in comparison.records)
    serialized = comparison.model_dump(mode="json")
    assert "command_id" not in json.dumps(serialized)
    assert "ttl_game_loops" not in json.dumps(serialized)


def test_runner_isolates_failures_and_does_not_call_unavailable_policies() -> None:
    fixture = PolicyObservationFixture(fixture_id="opening", observation=make_observation())
    failing = FailingSubagent()
    unavailable = RecordingSubagent()
    unavailable.spec = QWEN3_8B_SPEC.model_copy(update={"subagent_id": "unavailable"})

    comparison = asyncio.run(
        PolicyShadowRunner().compare(
            [fixture],
            [
                _available(failing),
                PolicySubagentRegistration(
                    spec=unavailable.spec,
                    availability=PolicyAvailability(
                        status=PolicyAvailabilityStatus.UNAVAILABLE,
                        reason="not installed",
                    ),
                    subagent=unavailable,
                ),
            ],
        )
    )

    assert [record.status for record in comparison.records] == [
        PolicyShadowStatus.FAILED,
        PolicyShadowStatus.UNAVAILABLE,
    ]
    assert comparison.records[0].error == "RuntimeError: shadow failure"
    assert unavailable.received == []


class FixedPlanningProvider:
    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        assert "shadow-only" in system_prompt
        payload = json.loads(user_prompt)
        assert payload["observation"]["run_id"] == "run-1"
        assert payload["goal_spec"] is None
        assert payload["goal_progress"] is None
        output = PlanningOutput(
            strategic_goal="Advance the opening",
            steps=["Use one legal action"],
            proposed_actions=[],
        )
        return response_type.model_validate(output.model_dump())


class CapturingPlanningProvider:
    def __init__(self) -> None:
        self.user_prompt = ""

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt
        self.user_prompt = user_prompt
        return response_type.model_validate(
            PlanningOutput(strategic_goal="Observe compact state").model_dump()
        )


def test_llm_adapter_returns_advisory_proposal_only() -> None:
    subagent = LLMPlanningPolicySubagent(FixedPlanningProvider())
    result = asyncio.run(
        subagent.propose(
            PolicyObservationFixture(fixture_id="opening", observation=make_observation())
        )
    )

    assert result == PolicyProposal(
        strategic_goal="Advance the opening",
        steps=["Use one legal action"],
    )
    assert not hasattr(result, "commands")


def test_llm_adapter_compacts_historical_observation_before_model_call() -> None:
    provider = CapturingPlanningProvider()
    base = make_observation()
    units = [
        UnitState(
            unit_id=f"probe-{index}",
            unit_type="Probe",
            alliance="self",
        )
        for index in range(100)
    ]
    observation = base.model_copy(
        update={
            "text_observation": "UNBOUNDED_HISTORY " * 2_000,
            "state": base.state.model_copy(update={"own_units": units}),
        }
    )

    asyncio.run(
        LLMPlanningPolicySubagent(provider).propose(
            PolicyObservationFixture(fixture_id="large", observation=observation)
        )
    )

    payload = json.loads(provider.user_prompt)
    assert "text_observation" not in payload["observation"]
    assert "UNBOUNDED_HISTORY" not in provider.user_prompt
    assert len(payload["observation"]["state"]["own_units"]) < len(units)


def test_historical_loader_reuses_observations_in_event_order(tmp_path: Path) -> None:
    first = make_observation()
    second = first.model_copy(update={"step_id": 2, "game_loop": 48})
    journal = tmp_path / "events.jsonl"
    entries = [
        {
            "event_id": 1,
            "run_id": "run-1",
            "episode_id": "episode-1",
            "step_id": 0,
            "event_type": "observation",
            "created_at": "2026-01-01T00:00:00+00:00",
            "payload": first.model_dump(mode="json"),
        },
        {
            "event_id": 2,
            "run_id": "run-1",
            "episode_id": "episode-1",
            "step_id": 1,
            "event_type": "decision",
            "created_at": "2026-01-01T00:00:01+00:00",
            "payload": {},
        },
        {
            "event_id": 3,
            "run_id": "run-1",
            "episode_id": "episode-1",
            "step_id": 2,
            "event_type": "observation",
            "created_at": "2026-01-01T00:00:02+00:00",
            "payload": second.model_dump(mode="json"),
        },
    ]
    journal.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )

    fixtures = load_historical_observations(journal)

    assert [fixture.observation for fixture in fixtures] == [first, second]
    assert [fixture.fixture_id for fixture in fixtures] == [
        "run-1:episode-1:step-0:event-1",
        "run-1:episode-1:step-2:event-3",
    ]
    assert load_historical_observations(journal, stride=2, limit=1) == [fixtures[0]]


def _gateway_goal_context() -> tuple[GoalSpec, GoalProgressReport]:
    goal = GoalSpec(
        goal_id="gateway-opening",
        strategic_goal="Build a Gateway",
        requirements=[
            GoalRequirement(
                requirement_id="gateway",
                kind=GoalRequirementKind.STRUCTURE,
                target="Gateway",
                action_name="Build_Gateway_Screen",
            )
        ],
    )
    report = GoalProgressReport(
        run_id="run-1",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        goal_id=goal.goal_id,
        strategic_goal=goal.strategic_goal,
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
    )
    return goal, report


def test_shadow_metrics_distinguish_legality_progress_and_control_violations() -> None:
    goal, report = _gateway_goal_context()
    base = make_observation(include_enemy=False)
    observation = base.model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Build_Gateway_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[60, 40]]],
                ),
                AvailableAction(name="Stop", actor_scopes=["Builder/Probe-1"]),
            ]
        }
    )
    fixture = PolicyObservationFixture(
        fixture_id="gateway",
        observation=observation,
        goal_spec=goal,
        goal_progress=report,
    )
    subagent = FixedProposalSubagent(
        PolicyProposal(
            strategic_goal="Build a Gateway",
            proposed_actions=[
                ActionProposal(
                    actor="Builder/Probe-1",
                    name="Build_Gateway_Screen",
                    arguments=[[60, 40]],
                ),
                ActionProposal(actor="Builder/Probe-1", name="Stop"),
                ActionProposal(actor="Builder/Probe-1", name="Unknown_Action"),
            ],
        )
    )

    comparison = asyncio.run(
        PolicyShadowRunner().compare([fixture], [_available(subagent)])
    )

    record = comparison.records[0]
    assert record.goal_id == "gateway-opening"
    assert record.proposed_action_count == 3
    assert record.legal_action_count == 2
    assert record.goal_advancing_action_count == 1
    assert record.control_action_violation_count == 1
    assert record.legal_action_rate == pytest.approx(2 / 3)
    assert record.goal_advancing_action_rate == pytest.approx(1 / 3)
    summary = comparison.summaries[0]
    assert summary.legal_action_rate == pytest.approx(2 / 3)
    assert summary.goal_advancing_action_rate == pytest.approx(1 / 3)
    assert summary.goal_opportunity_fixtures == 1
    assert summary.goal_opportunity_proposals == 3
    assert summary.control_action_violation_count == 1


def test_shadow_metrics_forbid_control_while_goal_effect_is_in_progress() -> None:
    goal, actionable_report = _gateway_goal_context()
    report = actionable_report.model_copy(
        update={
            "status": GoalProgressStatus.IN_PROGRESS,
            "advancing_actions": [],
            "unique_next_action": None,
        }
    )
    observation = make_observation(include_enemy=False).model_copy(
        update={
            "available_actions": [
                AvailableAction(name="Hold_Position", actor_scopes=["CombatGroup/Army-1"])
            ]
        }
    )
    subagent = FixedProposalSubagent(
        PolicyProposal(
            strategic_goal="Wait for the Gateway",
            proposed_actions=[
                ActionProposal(actor="CombatGroup/Army-1", name="Hold_Position")
            ],
        )
    )

    comparison = asyncio.run(
        PolicyShadowRunner().compare(
            [
                PolicyObservationFixture(
                    fixture_id="gateway-building",
                    observation=observation,
                    goal_spec=goal,
                    goal_progress=report,
                )
            ],
            [_available(subagent)],
        )
    )

    assert comparison.records[0].control_action_violation_count == 1
    assert comparison.records[0].goal_advancing_action_rate is None
    assert comparison.summaries[0].control_action_violation_count == 1
    assert comparison.summaries[0].goal_advancing_action_rate is None
    assert comparison.summaries[0].goal_opportunity_fixtures == 0
    assert comparison.summaries[0].goal_opportunity_proposals == 0


def test_attach_goal_progress_is_deterministic_and_preserves_observation() -> None:
    fixture = PolicyObservationFixture(fixture_id="opening", observation=make_observation())
    verifier = GoalProgressVerifier()
    goal = verifier.goal_from_action_names(
        strategic_goal="Build one Pylon",
        action_names=["Build_Pylon_Screen"],
        observation=fixture.observation,
        goal_id="pylon-opening",
    )

    enriched = attach_goal_progress([fixture], goal, verifier=verifier)

    assert enriched[0].observation == fixture.observation
    assert enriched[0].goal_spec == goal
    assert enriched[0].goal_progress == verifier.verify(fixture.observation, goal)


def test_fixture_rejects_goal_progress_for_another_observation() -> None:
    goal, report = _gateway_goal_context()

    with pytest.raises(ValueError, match="must describe the fixture observation"):
        PolicyObservationFixture(
            fixture_id="wrong-step",
            observation=make_observation(step_id=1),
            goal_spec=goal,
            goal_progress=report,
        )
