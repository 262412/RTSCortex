from __future__ import annotations

import asyncio

from rtscortex.contracts import ActionArgumentType, AvailableAction
from rtscortex.policy.hima.models import HIMA_PARSER_VERSION, HIMA_VOCABULARY_VERSION
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    PolicyAvailability,
    PolicyAvailabilityStatus,
    PolicyObservationFixture,
)
from rtscortex.policy.shadow import PolicyShadowRunner
from rtscortex.policy.subagents import HIMA_PROTOSS_SPECS, PolicySubagentRegistration
from rtscortex.progress import (
    GoalProgressItem,
    GoalProgressReport,
    GoalProgressStatus,
    GoalRequirementKind,
)
from tests.helpers import make_observation


class GatewayMacroSubagent:
    spec = HIMA_PROTOSS_SPECS[0]

    async def propose(self, fixture: PolicyObservationFixture) -> MacroPolicyProposal:
        del fixture
        return MacroPolicyProposal(
            strategic_objective="Advance Gateway production",
            steps=[
                MacroActionStep(
                    ordinal=0,
                    canonical_action="BUILD GATEWAY",
                    category="build",
                    raw_token="Gateway",
                )
            ],
            vocabulary_version=HIMA_VOCABULARY_VERSION,
            parser_version=HIMA_PARSER_VERSION,
        )


def test_macro_runtime_frontier_contributes_legality_and_goal_progress_rates() -> None:
    observation = make_observation(include_enemy=False).model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Build_Gateway_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[65, 90]]],
                )
            ]
        }
    )
    progress = GoalProgressReport(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        game_loop=observation.game_loop,
        goal_id="gateway",
        strategic_goal="Build a Gateway",
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
    fixture = PolicyObservationFixture(
        fixture_id="gateway",
        observation=observation,
        goal_progress=progress,
    )
    subagent = GatewayMacroSubagent()
    registration = PolicySubagentRegistration(
        spec=subagent.spec,
        availability=PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
        subagent=subagent,
    )

    comparison = asyncio.run(PolicyShadowRunner().compare([fixture], [registration]))
    record = comparison.records[0]

    assert record.proposed_action_count == 1
    assert record.legal_action_count == 1
    assert record.goal_advancing_action_count == 1
    assert record.legal_action_rate == 1.0
    assert record.goal_advancing_action_rate == 1.0
    assert record.control_action_violation_count == 0
    assert record.logical_classification_counts.total == 1
    assert record.discovered_macro_step_count == 1
    assert record.effective_classification_counts.total == 1
    assert record.effective_action_count == 1
