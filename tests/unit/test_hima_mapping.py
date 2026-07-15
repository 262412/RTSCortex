from __future__ import annotations

from rtscortex.contracts import ActionArgumentType, AvailableAction
from rtscortex.policy.hima.mapping import HIMA_RUNTIME_MAPPINGS, HIMAMacroActionMapper
from rtscortex.policy.hima.models import HIMA_PARSER_VERSION, HIMA_VOCABULARY_VERSION
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    ParseDiagnostic,
    PolicyActionClassification,
    PolicyObservationFixture,
)
from tests.helpers import make_observation


def _proposal(
    *steps: MacroActionStep,
    diagnostics: list[ParseDiagnostic] | None = None,
) -> MacroPolicyProposal:
    return MacroPolicyProposal(
        strategic_objective="Advance the Protoss macro plan",
        steps=list(steps),
        raw_output="fixture",
        vocabulary_version=HIMA_VOCABULARY_VERSION,
        parser_version=HIMA_PARSER_VERSION,
        diagnostics=diagnostics or [],
    )


def _step(
    ordinal: int,
    action: str,
    category: str,
    *,
    repeat: int = 1,
) -> MacroActionStep:
    return MacroActionStep(
        ordinal=ordinal,
        canonical_action=action,
        category=category,  # type: ignore[arg-type]
        repeat=repeat,
        raw_token=f"<{action}>",
    )


def test_mapping_registry_exposes_exactly_eight_hima_semantics() -> None:
    assert len(HIMA_RUNTIME_MAPPINGS) == 8
    assert {item.macro_action for item in HIMA_RUNTIME_MAPPINGS} == {
        "TRAIN ZEALOT",
        "TRAIN STALKER",
        "BUILD PYLON",
        "BUILD GATEWAY",
        "BUILD CYBERNETICSCORE",
        "BUILD ASSIMILATOR",
        "BUILD NEXUS",
        "RESEARCH WARPGATERESEARCH",
    }


def test_mapper_binds_frontier_actor_and_candidate_from_observation() -> None:
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

    assessments = HIMAMacroActionMapper().assess(
        _proposal(_step(0, "BUILD PYLON", "build")),
        fixture,
    )

    assert len(assessments) == 1
    assert assessments[0].classification is PolicyActionClassification.MAPPED_LEGAL_NOW
    assert assessments[0].runtime_action == "Build_Pylon_Screen"
    assert assessments[0].is_frontier is True


def test_supported_action_missing_now_is_deferred_not_unsupported_or_illegal() -> None:
    fixture = PolicyObservationFixture(
        fixture_id="no-pylon-candidate",
        observation=make_observation(include_enemy=False).model_copy(
            update={"available_actions": []}
        ),
    )

    assessment = HIMAMacroActionMapper().assess(
        _proposal(_step(0, "BUILD PYLON", "build")),
        fixture,
    )[0]

    assert assessment.classification is PolicyActionClassification.MAPPED_DEFERRED
    assert assessment.reason_code == "action_unavailable_now"


def test_valid_hima_action_without_runtime_capability_is_unsupported() -> None:
    fixture = PolicyObservationFixture(
        fixture_id="carrier",
        observation=make_observation(include_enemy=False),
    )

    assessment = HIMAMacroActionMapper().assess(
        _proposal(_step(0, "TRAIN CARRIER", "train")),
        fixture,
    )[0]

    assert assessment.classification is PolicyActionClassification.UNSUPPORTED_BY_RUNTIME
    assert assessment.reason_code == "not_implemented"


def test_auto_managed_probe_is_unsupported_without_becoming_noop() -> None:
    fixture = PolicyObservationFixture(
        fixture_id="probe",
        observation=make_observation(include_enemy=False),
    )

    assessment = HIMAMacroActionMapper().assess(
        _proposal(_step(0, "TRAIN PROBE", "train", repeat=4)),
        fixture,
    )[0]

    assert assessment.classification is PolicyActionClassification.UNSUPPORTED_BY_RUNTIME
    assert assessment.reason_code == "managed_automatically"
    assert assessment.repeat == 4
    assert assessment.runtime_action is None
    assert assessment.is_logical_frontier is True
    assert assessment.is_runtime_frontier is False
    assert assessment.is_frontier is False


def test_only_first_runtime_frontier_is_validated_against_current_state() -> None:
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
    fixture = PolicyObservationFixture(fixture_id="future", observation=observation)

    assessments = HIMAMacroActionMapper().assess(
        _proposal(
            _step(0, "BUILD PYLON", "build"),
            _step(1, "BUILD GATEWAY", "build"),
        ),
        fixture,
    )

    assert [item.classification for item in assessments] == [
        PolicyActionClassification.MAPPED_LEGAL_NOW,
        PolicyActionClassification.MAPPED_FUTURE,
    ]
    assert assessments[1].runtime_action == "Build_Gateway_Screen"
    assert assessments[0].is_logical_frontier is True
    assert assessments[0].is_runtime_frontier is True
    assert assessments[1].is_runtime_frontier is False


def test_parse_error_is_logical_frontier_but_next_mapped_step_is_runtime_frontier() -> None:
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
    fixture = PolicyObservationFixture(fixture_id="unknown", observation=observation)
    diagnostic = ParseDiagnostic(
        code="unknown_action",
        message="Unknown HIMA action",
        raw_token="<TRAIN ORTHOTOMIST>",
        ordinal=0,
    )

    assessments = HIMAMacroActionMapper().assess(
        _proposal(
            _step(1, "BUILD PYLON", "build"),
            diagnostics=[diagnostic],
        ),
        fixture,
    )

    assert [item.classification for item in assessments] == [
        PolicyActionClassification.PARSE_ERROR,
        PolicyActionClassification.MAPPED_LEGAL_NOW,
    ]
    assert assessments[0].is_logical_frontier is True
    assert assessments[0].is_runtime_frontier is False
    assert assessments[0].is_frontier is False
    assert assessments[1].is_logical_frontier is False
    assert assessments[1].is_runtime_frontier is True
    assert assessments[1].is_frontier is True


def test_auto_managed_probe_does_not_hide_actionable_runtime_frontier() -> None:
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
    fixture = PolicyObservationFixture(fixture_id="probe-first", observation=observation)

    assessments = HIMAMacroActionMapper().assess(
        _proposal(
            _step(0, "TRAIN PROBE", "train", repeat=4),
            _step(1, "BUILD PYLON", "build"),
            _step(2, "BUILD GATEWAY", "build"),
        ),
        fixture,
    )

    assert [item.classification for item in assessments] == [
        PolicyActionClassification.UNSUPPORTED_BY_RUNTIME,
        PolicyActionClassification.MAPPED_LEGAL_NOW,
        PolicyActionClassification.MAPPED_FUTURE,
    ]
    assert [item.is_logical_frontier for item in assessments] == [True, False, False]
    assert [item.is_runtime_frontier for item in assessments] == [False, True, False]


def test_missing_action_section_becomes_a_proposal_level_parse_error() -> None:
    fixture = PolicyObservationFixture(
        fixture_id="missing",
        observation=make_observation(include_enemy=False),
    )
    diagnostic = ParseDiagnostic(
        code="action_section_missing",
        message="No action section was found",
    )

    assessments = HIMAMacroActionMapper().assess(
        _proposal(diagnostics=[diagnostic]),
        fixture,
    )

    assert len(assessments) == 1
    assert assessments[0].classification is PolicyActionClassification.PARSE_ERROR
    assert assessments[0].source_action == "<action_section>"
    assert assessments[0].is_logical_frontier is True
    assert assessments[0].is_runtime_frontier is False
    assert assessments[0].is_frontier is False
