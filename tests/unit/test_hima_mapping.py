from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rtscortex.contracts import ActionArgumentType, AvailableAction, EconomyState, UnitState
from rtscortex.policy.corpus import load_policy_corpus
from rtscortex.policy.hima.mapping import HIMA_RUNTIME_MAPPINGS, HIMAMacroActionMapper
from rtscortex.policy.hima.models import HIMA_PARSER_VERSION, HIMA_VOCABULARY_VERSION
from rtscortex.policy.hima.parser import HIMAProposalParser
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    ParseDiagnostic,
    PolicyActionClassification,
    PolicyObservationFixture,
)
from tests.helpers import make_observation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINNED_CORPUS_MANIFEST = PROJECT_ROOT / "benchmarks/policy/protoss_v0_2/manifest.yaml"
HIMA_PROTOSS_A_OUTPUTS = PROJECT_ROOT / "tests/fixtures/hima_protoss_a_v02_raw_outputs.jsonl"


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


def _macro_ready_state(
    *,
    structures: list[UnitState] | None = None,
) -> dict[str, object]:
    observation = make_observation(include_enemy=False)
    return {
        "economy": EconomyState(
            minerals=500,
            vespene=500,
            supply_used=2,
            supply_cap=30,
        ),
        "own_structures": structures or observation.state.own_structures,
    }


def test_mapping_registry_exposes_first_expanded_protoss_semantics() -> None:
    assert len(HIMA_RUNTIME_MAPPINGS) == 15
    assert {item.macro_action for item in HIMA_RUNTIME_MAPPINGS} == {
        "TRAIN ZEALOT",
        "TRAIN STALKER",
        "TRAIN ADEPT",
        "TRAIN PHOENIX",
        "TRAIN VOIDRAY",
        "TRAIN ORACLE",
        "BUILD PYLON",
        "BUILD GATEWAY",
        "BUILD FORGE",
        "BUILD CYBERNETICSCORE",
        "BUILD ASSIMILATOR",
        "BUILD NEXUS",
        "BUILD STARGATE",
        "BUILD SHIELDBATTERY",
        "RESEARCH WARPGATERESEARCH",
    }
    mappings = {item.macro_action: item.runtime_actions for item in HIMA_RUNTIME_MAPPINGS}
    assert mappings["TRAIN ADEPT"] == ("Train_Adept",)
    assert mappings["BUILD STARGATE"] == ("Build_Stargate_Screen",)
    assert mappings["TRAIN VOIDRAY"] == ("Train_VoidRay",)
    assert mappings["TRAIN ORACLE"] == ("Train_Oracle",)
    assert mappings["TRAIN PHOENIX"] == ("Train_Phoenix",)
    assert mappings["BUILD SHIELDBATTERY"] == ("Build_ShieldBattery_Screen",)
    assert mappings["BUILD FORGE"] == ("Build_Forge_Screen",)


def test_mapper_uses_exact_oracle_phoenix_and_shield_battery_runtime_actions() -> None:
    base = make_observation(include_enemy=False)
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update=_macro_ready_state(
                    structures=[
                        UnitState(
                            unit_id="core-1",
                            unit_type="CyberneticsCore",
                            alliance="self",
                        ),
                        UnitState(
                            unit_id="stargate-1",
                            unit_type="Stargate",
                            alliance="self",
                        ),
                    ]
                )
            ),
            "available_actions": [
                AvailableAction(
                    name="Train_Oracle",
                    actor_scopes=["Developer/Empty"],
                ),
                AvailableAction(
                    name="Train_Phoenix",
                    actor_scopes=["Developer/Empty"],
                ),
                AvailableAction(
                    name="Build_ShieldBattery_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[65, 90]]],
                ),
            ],
        }
    )
    fixture = PolicyObservationFixture(fixture_id="stargate-options", observation=observation)

    expected = (
        ("TRAIN ORACLE", "train", "Train_Oracle"),
        ("TRAIN PHOENIX", "train", "Train_Phoenix"),
        ("BUILD SHIELDBATTERY", "build", "Build_ShieldBattery_Screen"),
    )
    for macro_action, category, runtime_action in expected:
        assessment = HIMAMacroActionMapper().assess(
            _proposal(_step(0, macro_action, category)),
            fixture,
        )[0]
        assert assessment.classification is PolicyActionClassification.MAPPED_LEGAL_NOW
        assert assessment.runtime_action == runtime_action


def test_hima_protoss_a_fixed_outputs_have_expanded_mapping_golden() -> None:
    fixtures = {
        fixture.fixture_id: fixture for fixture in load_policy_corpus(PINNED_CORPUS_MANIFEST)
    }
    parser = HIMAProposalParser()
    mapper = HIMAMacroActionMapper()
    effective_counts: Counter[str] = Counter()
    evaluated_fixture_ids: list[str] = []

    for line in HIMA_PROTOSS_A_OUTPUTS.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        fixture_id = item["fixture_id"]
        evaluated_fixture_ids.append(fixture_id)
        proposal = parser.parse(item["raw_output"])
        for assessment in mapper.assess(proposal, fixtures[fixture_id]):
            effective_counts[assessment.classification.value] += assessment.repeat

    assert evaluated_fixture_ids == list(fixtures)
    assert effective_counts == {
        "parse_error": 1,
        "unsupported_by_runtime": 981,
        "mapped_future": 587,
        "mapped_legal_now": 14,
        "mapped_deferred": 97,
    }
    assert (
        sum(
            effective_counts[classification]
            for classification in (
                "mapped_future",
                "mapped_legal_now",
                "mapped_deferred",
                "illegal_action",
                "obsolete",
            )
        )
        == 698
    )
    assert sum(effective_counts.values()) == 1_680


def test_mapper_binds_frontier_actor_and_candidate_from_observation() -> None:
    observation = make_observation(include_enemy=False).model_copy(
        update={
            "state": make_observation(include_enemy=False).state.model_copy(
                update=_macro_ready_state()
            ),
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[65, 90]]],
                )
            ],
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
            update={
                "state": make_observation(include_enemy=False).state.model_copy(
                    update=_macro_ready_state()
                ),
                "available_actions": [],
            }
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
            "state": make_observation(include_enemy=False).state.model_copy(
                update=_macro_ready_state()
            ),
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[65, 90]]],
                )
            ],
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


def test_mapper_skips_deferred_step_to_earliest_legal_runtime_frontier() -> None:
    observation = make_observation(include_enemy=False).model_copy(
        update={
            "state": make_observation(include_enemy=False).state.model_copy(
                update=_macro_ready_state()
            ),
            "available_actions": [
                AvailableAction(
                    name="Build_Gateway_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[65, 90]]],
                )
            ],
        }
    )
    fixture = PolicyObservationFixture(
        fixture_id="deferred-before-legal",
        observation=observation,
    )

    assessments = HIMAMacroActionMapper().assess(
        _proposal(
            _step(0, "BUILD PYLON", "build"),
            _step(1, "BUILD GATEWAY", "build"),
            _step(2, "BUILD NEXUS", "build"),
        ),
        fixture,
    )

    assert [item.classification for item in assessments] == [
        PolicyActionClassification.MAPPED_DEFERRED,
        PolicyActionClassification.MAPPED_LEGAL_NOW,
        PolicyActionClassification.MAPPED_FUTURE,
    ]
    assert assessments[0].reason_code == "action_unavailable_now"
    assert [item.is_runtime_frontier for item in assessments] == [False, True, False]


def test_mapper_uses_earliest_deferred_frontier_when_no_step_is_legal() -> None:
    fixture = PolicyObservationFixture(
        fixture_id="all-deferred",
        observation=make_observation(include_enemy=False).model_copy(
            update={
                "state": make_observation(include_enemy=False).state.model_copy(
                    update=_macro_ready_state(
                        structures=[
                            UnitState(
                                unit_id="pylon-1",
                                unit_type="Pylon",
                                alliance="self",
                            )
                        ]
                    )
                ),
                "available_actions": [],
            }
        ),
    )

    assessments = HIMAMacroActionMapper().assess(
        _proposal(
            _step(0, "BUILD PYLON", "build"),
            _step(1, "BUILD GATEWAY", "build"),
        ),
        fixture,
    )

    assert [item.classification for item in assessments] == [
        PolicyActionClassification.MAPPED_DEFERRED,
        PolicyActionClassification.MAPPED_DEFERRED,
    ]
    assert [item.reason_code for item in assessments] == [
        "action_unavailable_now",
        "action_unavailable_now",
    ]
    assert [item.is_runtime_frontier for item in assessments] == [True, False]


def test_hard_resource_blocker_preserves_hima_sequence_order() -> None:
    base = make_observation(include_enemy=False)
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=100,
                        vespene=500,
                        supply_used=2,
                        supply_cap=30,
                    ),
                    "own_structures": [
                        UnitState(
                            unit_id="pylon-1",
                            unit_type="Pylon",
                            alliance="self",
                        ),
                        UnitState(
                            unit_id="gateway-1",
                            unit_type="Gateway",
                            alliance="self",
                        ),
                    ],
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Train_Zealot",
                    actor_scopes=["Developer/Empty"],
                )
            ],
        }
    )

    assessments = HIMAMacroActionMapper().assess(
        _proposal(
            _step(0, "BUILD CYBERNETICSCORE", "build"),
            _step(1, "TRAIN ZEALOT", "train"),
        ),
        PolicyObservationFixture(fixture_id="hard-blocker", observation=observation),
    )

    assert assessments[0].classification is PolicyActionClassification.MAPPED_DEFERRED
    assert assessments[0].reason_code == "insufficient_minerals"
    assert assessments[0].is_runtime_frontier is True
    assert assessments[1].classification is PolicyActionClassification.MAPPED_FUTURE


def test_mapper_selects_available_runtime_alternative() -> None:
    base = make_observation(include_enemy=False)
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=500,
                        vespene=500,
                        supply_used=2,
                        supply_cap=30,
                    ),
                    "own_structures": [
                        UnitState(
                            unit_id="gateway-1",
                            unit_type="Gateway",
                            alliance="self",
                        )
                    ],
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Warp_Zealot_Near",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=["Developer/WarpGate-1"],
                    argument_candidates=[["0x100"]],
                )
            ],
        }
    )

    assessment = HIMAMacroActionMapper().assess(
        _proposal(_step(0, "TRAIN ZEALOT", "train")),
        PolicyObservationFixture(fixture_id="warp-alternative", observation=observation),
    )[0]

    assert assessment.classification is PolicyActionClassification.MAPPED_LEGAL_NOW
    assert assessment.runtime_action == "Warp_Zealot_Near"
    assert assessment.is_runtime_frontier is True


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
