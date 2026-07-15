from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from rtscortex.policy.hima.models import (
    HIMA_ADAPTER_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_VOCABULARY_VERSION,
)
from rtscortex.policy.hima.subagent import HIMA_PINNED_REVISIONS
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    PolicyActionAssessment,
    PolicyActionClassification,
    PolicyAvailability,
    PolicyAvailabilityStatus,
    PolicyFixtureStratum,
    PolicyGenerationMetadata,
    PolicyObservationFixture,
    PolicyProposal,
    PolicyProviderKind,
    PolicyShadowComparison,
    PolicyShadowRecord,
    PolicyShadowStatus,
    PolicySubagentSpec,
)
from rtscortex.policy.report import (
    build_policy_comparison_summary,
    render_policy_comparison_report,
    write_policy_comparison_reports,
)
from rtscortex.policy.subagents import HIMA_PROTOSS_SPECS, QWEN3_8B_SPEC
from tests.helpers import make_observation


def test_report_uses_global_counts_and_excludes_unsupported_from_illegal_rate() -> None:
    comparison = _comparison()

    summary = build_policy_comparison_summary(comparison)
    candidates = cast(dict[str, object], summary["candidates"])
    qwen = cast(dict[str, object], candidates[QWEN3_8B_SPEC.subagent_id])
    qwen_goal = cast(dict[str, object], qwen["goal_and_control"])
    assert qwen_goal["legal_action_rate"] == pytest.approx(9 / 11)

    hima = cast(dict[str, object], candidates[HIMA_PROTOSS_SPECS[0].subagent_id])
    parser = cast(dict[str, object], hima["parser"])
    mapping = cast(dict[str, object], hima["mapping"])
    frontier = cast(dict[str, object], hima["frontier"])
    classification = cast(dict[str, object], hima["classification"])
    latency = cast(dict[str, object], hima["latency_ms"])
    identity = cast(dict[str, object], hima["identity"])

    logical = cast(dict[str, object], classification["logical"])
    effective = cast(dict[str, object], classification["effective"])
    logical_counts = cast(dict[str, object], logical["counts"])
    effective_counts = cast(dict[str, object], effective["counts"])
    assert logical_counts == {
        "parse_error": 1,
        "unsupported_by_runtime": 2,
        "mapped_future": 2,
        "mapped_legal_now": 1,
        "mapped_deferred": 0,
        "illegal_action": 1,
        "obsolete": 0,
    }
    assert effective_counts == {
        "parse_error": 1,
        "unsupported_by_runtime": 6,
        "mapped_future": 4,
        "mapped_legal_now": 2,
        "mapped_deferred": 0,
        "illegal_action": 3,
        "obsolete": 0,
    }
    assert logical["total"] == 7
    assert effective["total"] == 16
    assert logical["conserved"] is True
    assert effective["conserved"] is True

    parser_logical = cast(dict[str, object], parser["logical"])
    parser_effective = cast(dict[str, object], parser["effective"])
    mapping_logical = cast(dict[str, object], mapping["logical"])
    mapping_effective = cast(dict[str, object], mapping["effective"])
    assert parser_logical["parse_validity"] == pytest.approx(6 / 7)
    assert parser_effective["parse_validity"] == pytest.approx(15 / 16)
    assert mapping_logical["coverage"] == pytest.approx(4 / 6)
    assert mapping_effective["coverage"] == pytest.approx(9 / 15)
    assert mapping_logical["matching_denominator"] == 6
    assert mapping_effective["matching_denominator"] == 15

    sequence = cast(dict[str, object], frontier["sequence"])
    runtime = cast(dict[str, object], frontier["runtime"])
    sequence_logical = cast(dict[str, object], sequence["logical"])
    sequence_effective = cast(dict[str, object], sequence["effective"])
    runtime_logical = cast(dict[str, object], runtime["logical"])
    runtime_effective = cast(dict[str, object], runtime["effective"])
    assert cast(dict[str, object], sequence_logical["counts"])[
        "unsupported_by_runtime"
    ] == 2
    assert sequence_logical["total"] == 2
    assert sequence_effective["total"] == 6
    assert runtime_logical["runtime_outcome_conserved"] is True
    assert runtime_effective["runtime_outcome_conserved"] is True
    assert runtime_logical["illegal_action_rate"] == pytest.approx(1 / 2)
    assert runtime_effective["illegal_action_rate"] == pytest.approx(3 / 5)

    assert latency == {"sample_count": 2, "p50": 20.0, "p95": 29.0}
    assert identity["pinned_revision"] == HIMA_PINNED_REVISIONS["SNUMPR/Protoss-a"]
    assert identity["adapter_versions"] == [HIMA_ADAPTER_VERSION]
    assert identity["parser_versions"] == [HIMA_PARSER_VERSION]
    assert identity["vocabulary_versions"] == [HIMA_VOCABULARY_VERSION]

    by_stratum = cast(dict[str, object], hima["by_stratum"])
    early = cast(dict[str, object], by_stratum["early"])
    early_parser = cast(dict[str, object], early["parser"])
    early_mapping = cast(dict[str, object], early["mapping"])
    assert cast(dict[str, object], early_parser["logical"])[
        "parse_validity"
    ] == pytest.approx(3 / 4)
    assert cast(dict[str, object], early_parser["effective"])[
        "parse_validity"
    ] == pytest.approx(9 / 10)
    assert cast(dict[str, object], early_mapping["logical"])[
        "coverage"
    ] == pytest.approx(2 / 3)
    assert cast(dict[str, object], early_mapping["effective"])[
        "coverage"
    ] == pytest.approx(5 / 9)
    assert cast(dict[str, object], early["latency_ms"])["p95"] == 10.0

    for stratum in (
        "early",
        "technology",
        "production",
        "combat",
        "blocked",
        "in_progress",
    ):
        stratum_metrics = cast(dict[str, object], by_stratum[stratum])
        assert "outcomes" in stratum_metrics
        assert "completion_rate" in stratum_metrics
        assert "latency_ms" in stratum_metrics


def test_markdown_covers_strata_candidate_outcomes_and_metric_definitions() -> None:
    report = render_policy_comparison_report(_comparison())

    assert "## Corpus coverage" in report
    assert "| `early` | 1 |" in report
    assert "| `combat` | 1 |" in report
    assert "## Candidate availability and completion" in report
    assert "## Model and adapter provenance" in report
    assert "## Parser validity and Runtime mapping" in report
    assert "## Logical and repeat-weighted classifications" in report
    assert "## Sequence and Runtime frontiers" in report
    assert "## Goal progress and control safety" in report
    assert "## Availability and completion by corpus stratum" in report
    assert "## Logical Runtime outcomes by corpus stratum" in report
    assert "## Quality by corpus stratum" in report
    assert "Runtime frontier skips actions that RTSCortex does not own" in report
    assert "repeat-weighted" in report
    assert "unsupported_by_runtime`, `parse_error`, and `illegal_action`" in report
    assert "not installed for this comparison" in report
    assert HIMA_PINNED_REVISIONS["SNUMPR/Protoss-a"] in report
    assert "20.0 ms" in report
    assert "| None |" not in report


def test_writer_is_idempotent_and_preserves_lossless_comparison_json(
    tmp_path: Path,
) -> None:
    comparison = _comparison()

    first = write_policy_comparison_reports(comparison, tmp_path / "artifacts")
    first_json = first.comparison_path.read_text(encoding="utf-8")
    first_report = first.report_path.read_text(encoding="utf-8")
    second = write_policy_comparison_reports(comparison, tmp_path / "artifacts")

    assert json.loads(first_json) == comparison.model_dump(mode="json")
    assert first.summary == build_policy_comparison_summary(comparison)
    assert second.comparison_path.read_text(encoding="utf-8") == first_json
    assert second.report_path.read_text(encoding="utf-8") == first_report


def test_native_only_v01_style_records_report_macro_metrics_as_not_applicable() -> None:
    fixture = _fixture("legacy", PolicyFixtureStratum.EARLY, step_id=0)
    record = _native_record(
        fixture,
        proposed=2,
        legal=1,
        goal_advancing=0,
        legal_rate=0.5,
    )
    comparison = PolicyShadowComparison(
        fixture_ids=[fixture.fixture_id],
        fixtures=[fixture.model_copy(update={"primary_stratum": None})],
        candidate_ids=[QWEN3_8B_SPEC.subagent_id],
        records=[record],
        summaries=[],
    )

    summary = build_policy_comparison_summary(comparison)
    candidates = cast(dict[str, object], summary["candidates"])
    candidate = cast(dict[str, object], candidates[QWEN3_8B_SPEC.subagent_id])
    parser = cast(dict[str, object], candidate["parser"])
    assert parser["applicable"] is False
    assert parser["parse_validity"] is None
    assert parser["classification_conserved"] is True
    report = render_policy_comparison_report(comparison)
    assert (
        "| `qwen3-8b-current` | `logical` | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        in report
    )
    assert "Native-only v0.1 records show `N/A`" in report


def _comparison() -> PolicyShadowComparison:
    early = _fixture("early-00", PolicyFixtureStratum.EARLY, step_id=0)
    combat = _fixture("combat-00", PolicyFixtureStratum.COMBAT, step_id=1)
    unavailable = PolicyAvailability(
        status=PolicyAvailabilityStatus.UNAVAILABLE,
        reason="not installed for this comparison",
    )
    hima_proposal = MacroPolicyProposal(
        strategic_objective="Build a gateway army",
        steps=[
            MacroActionStep(
                ordinal=0,
                canonical_action="BUILD PYLON",
                category="build",
                raw_token="<BUILD PYLON>",
            )
        ],
        raw_output="<BUILD PYLON>",
        adapter_version=HIMA_ADAPTER_VERSION,
        vocabulary_version=HIMA_VOCABULARY_VERSION,
        parser_version=HIMA_PARSER_VERSION,
        generation_metadata=PolicyGenerationMetadata(
            provider_kind=PolicyProviderKind.HUGGING_FACE_TRANSFORMERS,
            model_id="SNUMPR/Protoss-a",
            model_revision=HIMA_PINNED_REVISIONS["SNUMPR/Protoss-a"],
            checkpoint_path="/models/protoss-a",
            checkpoint_verified=True,
            license_acknowledged=True,
            deterministic=True,
            max_new_tokens=256,
            prompt_token_count=100,
            completion_token_count=20,
            eos_reached=True,
            truncated=False,
        ),
    )
    records = [
        _native_record(
            early,
            proposed=10,
            legal=9,
            goal_advancing=2,
            legal_rate=0.9,
        ),
        _native_record(
            combat,
            proposed=1,
            legal=0,
            goal_advancing=0,
            legal_rate=0.0,
        ),
        _macro_record(
            early,
            hima_proposal,
            assessments=[
                PolicyActionAssessment(
                    ordinal=0,
                    repeat=4,
                    source_action="TRAIN PROBE",
                    classification=PolicyActionClassification.UNSUPPORTED_BY_RUNTIME,
                    reason_code="managed_automatically",
                    is_logical_frontier=True,
                ),
                PolicyActionAssessment(
                    ordinal=1,
                    repeat=2,
                    source_action="BUILD PYLON",
                    runtime_action="Build_Pylon_Screen",
                    classification=PolicyActionClassification.MAPPED_LEGAL_NOW,
                    is_runtime_frontier=True,
                ),
                PolicyActionAssessment(
                    ordinal=2,
                    repeat=3,
                    source_action="BUILD GATEWAY",
                    runtime_action="Build_Gateway_Screen",
                    classification=PolicyActionClassification.MAPPED_FUTURE,
                ),
                PolicyActionAssessment(
                    ordinal=3,
                    source_action="<BAD>",
                    classification=PolicyActionClassification.PARSE_ERROR,
                    reason_code="unknown_action_token",
                ),
            ],
        ),
        _macro_record(
            combat,
            hima_proposal,
            assessments=[
                PolicyActionAssessment(
                    ordinal=0,
                    repeat=2,
                    source_action="TRAIN PROBE",
                    classification=PolicyActionClassification.UNSUPPORTED_BY_RUNTIME,
                    reason_code="managed_automatically",
                    is_logical_frontier=True,
                ),
                PolicyActionAssessment(
                    ordinal=1,
                    repeat=3,
                    source_action="BUILD PYLON",
                    runtime_action="Build_Pylon_Screen",
                    classification=PolicyActionClassification.ILLEGAL_ACTION,
                    reason_code="candidate_not_available",
                    is_runtime_frontier=True,
                ),
                PolicyActionAssessment(
                    ordinal=2,
                    source_action="BUILD GATEWAY",
                    runtime_action="Build_Gateway_Screen",
                    classification=PolicyActionClassification.MAPPED_FUTURE,
                ),
            ],
        ),
        _unavailable_record(early, HIMA_PROTOSS_SPECS[1], unavailable),
        _unavailable_record(combat, HIMA_PROTOSS_SPECS[1], unavailable),
    ]
    return PolicyShadowComparison(
        fixture_ids=[early.fixture_id, combat.fixture_id],
        fixtures=[early, combat],
        candidate_ids=[
            QWEN3_8B_SPEC.subagent_id,
            HIMA_PROTOSS_SPECS[0].subagent_id,
            HIMA_PROTOSS_SPECS[1].subagent_id,
        ],
        records=records,
        summaries=[],
    )


def _fixture(
    fixture_id: str,
    stratum: PolicyFixtureStratum,
    *,
    step_id: int,
) -> PolicyObservationFixture:
    return PolicyObservationFixture(
        fixture_id=fixture_id,
        observation=make_observation(
            step_id=step_id,
            game_loop=step_id * 224,
        ),
        primary_stratum=stratum,
    )


def _native_record(
    fixture: PolicyObservationFixture,
    *,
    proposed: int,
    legal: int,
    goal_advancing: int,
    legal_rate: float,
) -> PolicyShadowRecord:
    return PolicyShadowRecord(
        fixture_id=fixture.fixture_id,
        run_id=fixture.observation.run_id,
        episode_id=fixture.observation.episode_id,
        step_id=fixture.observation.step_id,
        game_loop=fixture.observation.game_loop,
        spec=QWEN3_8B_SPEC,
        availability=PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
        status=PolicyShadowStatus.COMPLETED,
        proposal=PolicyProposal(strategic_goal="Advance"),
        proposed_action_count=proposed,
        legal_action_count=legal,
        goal_advancing_action_count=goal_advancing,
        control_action_violation_count=1,
        latency_ms=100.0 if fixture.observation.step_id == 0 else 300.0,
        legal_action_rate=legal_rate,
        goal_advancing_action_rate=goal_advancing / proposed,
    )


def _macro_record(
    fixture: PolicyObservationFixture,
    proposal: MacroPolicyProposal,
    *,
    assessments: list[PolicyActionAssessment],
) -> PolicyShadowRecord:
    return PolicyShadowRecord(
        fixture_id=fixture.fixture_id,
        run_id=fixture.observation.run_id,
        episode_id=fixture.observation.episode_id,
        step_id=fixture.observation.step_id,
        game_loop=fixture.observation.game_loop,
        spec=HIMA_PROTOSS_SPECS[0],
        availability=PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
        status=PolicyShadowStatus.COMPLETED,
        proposal=proposal,
        action_assessments=assessments,
        latency_ms=10.0 if fixture.observation.step_id == 0 else 30.0,
    )


def _unavailable_record(
    fixture: PolicyObservationFixture,
    spec: PolicySubagentSpec,
    availability: PolicyAvailability,
) -> PolicyShadowRecord:
    return PolicyShadowRecord(
        fixture_id=fixture.fixture_id,
        run_id=fixture.observation.run_id,
        episode_id=fixture.observation.episode_id,
        step_id=fixture.observation.step_id,
        game_loop=fixture.observation.game_loop,
        spec=spec,
        availability=availability,
        status=PolicyShadowStatus.UNAVAILABLE,
    )
