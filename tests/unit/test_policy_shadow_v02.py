from __future__ import annotations

import asyncio

from rtscortex.agents.models import ActionProposal
from rtscortex.contracts import ActionArgumentType, AvailableAction
from rtscortex.policy import PolicyShadowRunner
from rtscortex.policy.hima.models import HIMA_PARSER_VERSION, HIMA_VOCABULARY_VERSION
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    PolicyActionClassification,
    PolicyAvailability,
    PolicyAvailabilityStatus,
    PolicyObservationFixture,
    PolicyProposal,
)
from rtscortex.policy.subagents import (
    HIMA_PROTOSS_SPECS,
    QWEN3_8B_SPEC,
    PolicySubagentRegistration,
)
from tests.helpers import make_observation


class MacroSubagent:
    spec = HIMA_PROTOSS_SPECS[0]

    async def propose(self, fixture: PolicyObservationFixture) -> MacroPolicyProposal:
        del fixture
        return MacroPolicyProposal(
            strategic_objective="Build supply before future carrier technology",
            steps=[
                MacroActionStep(
                    ordinal=0,
                    canonical_action="BUILD PYLON",
                    category="build",
                    raw_token="<BUILD PYLON>",
                ),
                MacroActionStep(
                    ordinal=1,
                    canonical_action="TRAIN CARRIER",
                    category="train",
                    repeat=2,
                    raw_token="<TRAIN CARRIER> x 2",
                ),
            ],
            raw_output="<BUILD PYLON> <TRAIN CARRIER> x 2",
            vocabulary_version=HIMA_VOCABULARY_VERSION,
            parser_version=HIMA_PARSER_VERSION,
        )


class NativeUnknownSubagent:
    spec = QWEN3_8B_SPEC

    async def propose(self, fixture: PolicyObservationFixture) -> PolicyProposal:
        del fixture
        return PolicyProposal(
            strategic_goal="Invent an unavailable action",
            proposed_actions=[
                ActionProposal(actor="Builder/Probe-1", name="Build_FleetBeacon_Screen")
            ],
        )


def _registration(subagent: object, spec: object) -> PolicySubagentRegistration:
    return PolicySubagentRegistration(
        spec=spec,  # type: ignore[arg-type]
        availability=PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
        subagent=subagent,  # type: ignore[arg-type]
    )


def test_runner_scores_macro_mapping_without_creating_runtime_commands() -> None:
    observation = make_observation(include_enemy=False).model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[65, 90]]],
                )
            ]
        }
    )
    fixture = PolicyObservationFixture(fixture_id="opening", observation=observation)
    subagent = MacroSubagent()

    comparison = asyncio.run(
        PolicyShadowRunner().compare(
            [fixture],
            [_registration(subagent, subagent.spec)],
        )
    )

    record = comparison.records[0]
    assert record.discovered_macro_step_count == 2
    assert record.parsed_known_action_count == 2
    assert record.effective_action_count == 3
    assert record.mapped_legal_now_count == 1
    assert record.unsupported_by_runtime_count == 1
    assert record.illegal_action_count == 0
    assert record.parse_validity == 1.0
    assert record.mapping_coverage == 0.5
    assert record.frontier_illegal_rate == 0.0
    assert [item.classification for item in record.action_assessments] == [
        PolicyActionClassification.MAPPED_LEGAL_NOW,
        PolicyActionClassification.UNSUPPORTED_BY_RUNTIME,
    ]
    assert record.proposed_action_count == 1
    assert record.legal_action_count == 1
    assert record.legal_action_rate == 1.0
    assert record.goal_advancing_action_count == 0
    assert record.goal_advancing_action_rate is None
    assert record.logical_classification_counts.total == 2
    assert record.logical_classification_counts.unsupported_by_runtime == 1
    assert record.effective_classification_counts.total == 3
    assert record.effective_classification_counts.unsupported_by_runtime == 2
    assert comparison.summaries[0].logical_classification_counts.total == 2
    assert comparison.summaries[0].effective_classification_counts.total == 3
    assert "command_id" not in comparison.model_dump_json()


def test_native_unknown_action_is_illegal_not_runtime_unsupported() -> None:
    fixture = PolicyObservationFixture(
        fixture_id="native",
        observation=make_observation(include_enemy=False),
    )
    subagent = NativeUnknownSubagent()

    comparison = asyncio.run(
        PolicyShadowRunner().compare(
            [fixture],
            [_registration(subagent, subagent.spec)],
        )
    )

    record = comparison.records[0]
    assert record.illegal_action_count == 1
    assert record.unsupported_by_runtime_count == 0
    assert record.frontier_illegal_rate == 1.0
    assert record.action_assessments[0].reason_code == "unknown_runtime_action"
