from __future__ import annotations

import pytest
from pydantic import ValidationError

from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    ParseDiagnostic,
    PolicyActionAssessment,
    PolicyActionClassification,
    PolicyActionClassificationCounts,
    PolicyAvailability,
    PolicyAvailabilityStatus,
    PolicyFixtureSource,
    PolicyFixtureStratum,
    PolicyObservationFixture,
    PolicyProposal,
    PolicyShadowComparison,
    PolicyShadowRecord,
    PolicyShadowStatus,
    TacticalRationale,
)
from rtscortex.policy.subagents import QWEN3_8B_SPEC
from tests.helpers import make_observation


def test_v01_models_keep_their_existing_construction_surface() -> None:
    proposal = PolicyProposal(strategic_goal="Advance the opening")
    fixture = PolicyObservationFixture(
        fixture_id="opening",
        observation=make_observation(),
    )
    record = PolicyShadowRecord(
        fixture_id=fixture.fixture_id,
        run_id=fixture.observation.run_id,
        episode_id=fixture.observation.episode_id,
        step_id=fixture.observation.step_id,
        game_loop=fixture.observation.game_loop,
        spec=QWEN3_8B_SPEC,
        availability=PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
        status=PolicyShadowStatus.COMPLETED,
        proposal=proposal,
    )
    comparison = PolicyShadowComparison(
        fixture_ids=[fixture.fixture_id],
        fixtures=[fixture],
        candidate_ids=[QWEN3_8B_SPEC.subagent_id],
        records=[record],
        summaries=[],
    )

    assert proposal.proposal_kind == "native"
    assert comparison.comparison_version == "0.2"
    assert record.action_assessments == []
    assert record.discovered_macro_step_count == 0
    assert record.parse_validity is None


def test_macro_policy_proposal_round_trips_without_executable_commands() -> None:
    proposal = MacroPolicyProposal(
        strategic_objective="Reach a safe two-base gateway economy",
        tactical_rationale=TacticalRationale(
            immediate="Avoid a supply block",
            short_term="Add gateway production",
            long_term="Take a natural expansion",
        ),
        steps=[
            MacroActionStep(
                ordinal=0,
                canonical_action="BUILD PYLON",
                category="build",
                repeat=1,
                raw_token="<BUILD PYLON>",
            ),
            MacroActionStep(
                ordinal=1,
                canonical_action="TRAIN PROBE",
                category="train",
                repeat=4,
                raw_token="<TRAIN PROBE> x 4",
            ),
        ],
        raw_output="So my advice is <BUILD PYLON> <TRAIN PROBE> x 4",
        vocabulary_version="hima-protoss-60-v1",
        parser_version="hima-parser-v1",
        diagnostics=[
            ParseDiagnostic(
                code="alias_normalized",
                message="Normalized an upstream alias",
                raw_token="<TRAIN_PROBE>",
                ordinal=1,
            )
        ],
    )

    restored = MacroPolicyProposal.model_validate(proposal.model_dump(mode="json"))

    assert restored == proposal
    assert restored.proposal_kind == "macro"
    assert restored.horizon_seconds == 180
    assert restored.steps[1].repeat == 4
    assert not hasattr(restored, "commands")


def test_macro_record_rejects_nonconserved_logical_and_effective_counts() -> None:
    fixture = PolicyObservationFixture(
        fixture_id="macro",
        observation=make_observation(),
    )
    proposal = MacroPolicyProposal(
        strategic_objective="Build a Pylon",
        steps=[
            MacroActionStep(
                ordinal=0,
                canonical_action="BUILD PYLON",
                category="build",
                raw_token="Pylon",
            )
        ],
        vocabulary_version="fixture-vocabulary",
        parser_version="fixture-parser",
    )
    base = {
        "fixture_id": fixture.fixture_id,
        "run_id": fixture.observation.run_id,
        "episode_id": fixture.observation.episode_id,
        "step_id": fixture.observation.step_id,
        "game_loop": fixture.observation.game_loop,
        "spec": QWEN3_8B_SPEC,
        "availability": PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
        "status": PolicyShadowStatus.COMPLETED,
        "proposal": proposal,
        "discovered_macro_step_count": 1,
        "parsed_known_action_count": 1,
        "effective_action_count": 1,
        "mapped_legal_now_count": 1,
    }

    with pytest.raises(ValidationError, match="logical classification counts"):
        PolicyShadowRecord.model_validate(base)

    with pytest.raises(ValidationError, match="effective classification counts"):
        PolicyShadowRecord.model_validate(
            {
                **base,
                "logical_classification_counts": PolicyActionClassificationCounts(
                    mapped_legal_now=1
                ),
            }
        )


@pytest.mark.parametrize(
    ("classification", "runtime_action"),
    [
        (PolicyActionClassification.PARSE_ERROR, None),
        (PolicyActionClassification.UNSUPPORTED_BY_RUNTIME, None),
        (PolicyActionClassification.MAPPED_FUTURE, "Build_Pylon_Screen"),
        (PolicyActionClassification.MAPPED_LEGAL_NOW, "Build_Pylon_Screen"),
        (PolicyActionClassification.MAPPED_DEFERRED, "Build_Pylon_Screen"),
        (PolicyActionClassification.ILLEGAL_ACTION, "Build_Pylon_Screen"),
        (PolicyActionClassification.OBSOLETE, "Build_Pylon_Screen"),
    ],
)
def test_action_assessment_represents_each_terminal_classification(
    classification: PolicyActionClassification,
    runtime_action: str | None,
) -> None:
    assessment = PolicyActionAssessment(
        ordinal=0,
        repeat=1,
        source_action="BUILD PYLON",
        runtime_action=runtime_action,
        classification=classification,
        reason_code="fixture",
        is_frontier=True,
    )

    assert assessment.classification is classification


def test_mapped_assessment_requires_a_runtime_action() -> None:
    with pytest.raises(ValidationError, match="runtime_action"):
        PolicyActionAssessment(
            ordinal=0,
            source_action="BUILD PYLON",
            classification=PolicyActionClassification.MAPPED_LEGAL_NOW,
            is_frontier=True,
        )


def test_policy_fixture_accepts_versioned_corpus_provenance() -> None:
    observation = make_observation()
    source = PolicyFixtureSource(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        event_id=42,
        seed=0,
        map_name="Simple64",
        game_loop=observation.game_loop,
        protocol_version=observation.protocol_version,
        journal_sha256="a" * 64,
        observation_sha256="b" * 64,
    )
    fixture = PolicyObservationFixture(
        fixture_id="technology-00",
        observation=observation,
        primary_stratum=PolicyFixtureStratum.TECHNOLOGY,
        phase_tags=["cybernetics-core"],
        condition_tags=["actionable"],
        blocker_tags=[],
        selection_evidence=["completed Cybernetics Core"],
        source=source,
        state_fingerprint="c" * 64,
    )

    assert fixture.primary_stratum is PolicyFixtureStratum.TECHNOLOGY
    assert fixture.source == source
    assert fixture.selection_evidence == ["completed Cybernetics Core"]


def test_fixture_source_must_match_the_observation_identity() -> None:
    observation = make_observation()
    source = PolicyFixtureSource(
        run_id="another-run",
        episode_id=observation.episode_id,
        event_id=42,
        game_loop=observation.game_loop,
        protocol_version=observation.protocol_version,
        journal_sha256="a" * 64,
        observation_sha256="b" * 64,
    )

    with pytest.raises(ValidationError, match="fixture source must describe"):
        PolicyObservationFixture(
            fixture_id="opening",
            observation=observation,
            source=source,
        )
