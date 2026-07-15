from __future__ import annotations

from rtscortex.policy.hima import (
    HIMA_PARSER_VERSION,
    HIMA_VOCABULARY_VERSION,
    HIMAProposalParser,
)
from rtscortex.policy.hima.parser import MAX_ACTION_ITEMS


def test_parser_reads_angle_brackets_repeats_and_explicit_rationale() -> None:
    raw = """
Reason: **Immediate Steps:** Stabilize the worker count.
**Short-Term Actions:** Add supply and unlock Warp Gate.
**Long-Term Strategy:** Transition to a sustained Gateway composition.
Final Actions Summary: <TRAIN PROBE> x 4 <BUILD PYLON> <RESEARCH WARPGATE>
"""

    proposal = HIMAProposalParser().parse(raw)

    assert proposal.proposal_kind == "macro"
    assert proposal.vocabulary_version == HIMA_VOCABULARY_VERSION
    assert proposal.parser_version == HIMA_PARSER_VERSION
    assert proposal.strategic_objective == "Transition to a sustained Gateway composition"
    assert proposal.tactical_rationale.model_dump() == {
        "immediate": "Stabilize the worker count",
        "short_term": "Add supply and unlock Warp Gate",
        "long_term": "Transition to a sustained Gateway composition",
    }
    assert [step.model_dump() for step in proposal.steps] == [
        {
            "ordinal": 0,
            "canonical_action": "TRAIN PROBE",
            "category": "train",
            "repeat": 4,
            "raw_token": "TRAIN PROBE",
        },
        {
            "ordinal": 1,
            "canonical_action": "BUILD PYLON",
            "category": "build",
            "repeat": 1,
            "raw_token": "BUILD PYLON",
        },
        {
            "ordinal": 2,
            "canonical_action": "RESEARCH WARPGATERESEARCH",
            "category": "research",
            "repeat": 1,
            "raw_token": "RESEARCH WARPGATE",
        },
    ]
    assert proposal.diagnostics == []
    assert proposal.raw_output == raw


def test_parser_reads_official_python_actions_list_and_short_aliases() -> None:
    raw = (
        "Reason: **Immediate Steps:** Build workers. "
        "**Short-Term Actions:** Add technology. "
        "**Long-Term Strategy:** Unlock flexible production. "
        "Actions: ['Probe', 'CyberneticsCore', 'WarpGateResearch']"
    )

    proposal = HIMAProposalParser().parse(raw)

    assert [step.canonical_action for step in proposal.steps] == [
        "TRAIN PROBE",
        "BUILD CYBERNETICSCORE",
        "RESEARCH WARPGATERESEARCH",
    ]
    assert [step.ordinal for step in proposal.steps] == [0, 1, 2]
    assert proposal.diagnostics == []


def test_parser_reads_official_advice_sequence_without_polluting_rationale() -> None:
    raw = (
        "Reason: **Immediate Steps:** Keep producing workers. "
        "**Short-Term Actions:** Add supply. "
        "**Long-Term Strategy:** Unlock a stable Gateway economy. "
        "So my advice is <Probe> x 4 <Pylon> <Gateway>"
    )

    proposal = HIMAProposalParser().parse(raw)

    assert proposal.strategic_objective == "Unlock a stable Gateway economy"
    assert proposal.tactical_rationale.long_term == "Unlock a stable Gateway economy"
    assert [step.canonical_action for step in proposal.steps] == [
        "TRAIN PROBE",
        "BUILD PYLON",
        "BUILD GATEWAY",
    ]
    assert proposal.steps[0].repeat == 4
    assert proposal.diagnostics == []


def test_parser_format_precedence_is_actions_then_final_summary_then_advice() -> None:
    all_formats = HIMAProposalParser().parse(
        "Final Actions Summary: <Pylon> "
        "So my advice is <Gateway> "
        "Actions: ['Probe']"
    )
    final_over_advice = HIMAProposalParser().parse(
        "So my advice is <Gateway> Final Actions Summary: <Pylon>"
    )

    assert [step.canonical_action for step in all_formats.steps] == ["TRAIN PROBE"]
    assert [step.canonical_action for step in final_over_advice.steps] == [
        "BUILD PYLON"
    ]


def test_parser_reads_json_actions_list() -> None:
    proposal = HIMAProposalParser().parse('Actions: ["Zealot", "Pylon"]')

    assert [step.canonical_action for step in proposal.steps] == [
        "TRAIN ZEALOT",
        "BUILD PYLON",
    ]


def test_parser_retains_unknown_token_diagnostic_and_source_ordinal() -> None:
    proposal = HIMAProposalParser().parse(
        'Actions: ["Probe", "Orthotomist", "Pylon"]'
    )

    assert [step.ordinal for step in proposal.steps] == [0, 2]
    assert [step.canonical_action for step in proposal.steps] == [
        "TRAIN PROBE",
        "BUILD PYLON",
    ]
    assert [diagnostic.model_dump() for diagnostic in proposal.diagnostics] == [
        {
            "code": "unknown_action_token",
            "message": "Token is not in the pinned Protoss macro-action vocabulary.",
                "raw_token": "Orthotomist",
                "ordinal": 1,
                "repeat": 1,
            }
        ]


def test_parser_does_not_fuzzy_repair_unknown_angle_token() -> None:
    proposal = HIMAProposalParser().parse(
        "Final Actions Summary: <BUILD CYBERNETICCORE> <TRAIN PROBE>"
    )

    assert [step.canonical_action for step in proposal.steps] == ["TRAIN PROBE"]
    assert proposal.steps[0].ordinal == 1
    assert proposal.diagnostics[0].code == "unknown_action_token"
    assert proposal.diagnostics[0].raw_token == "BUILD CYBERNETICCORE"


def test_parser_reports_invalid_items_lists_and_repeats() -> None:
    invalid_item = HIMAProposalParser().parse('Actions: ["Probe", 42, "Pylon"]')
    invalid_list = HIMAProposalParser().parse("Actions: [Probe]")
    invalid_repeat = HIMAProposalParser().parse("<TRAIN PROBE> x 0")

    assert [step.ordinal for step in invalid_item.steps] == [0, 2]
    assert invalid_item.diagnostics[0].code == "invalid_action_item"
    assert invalid_item.diagnostics[0].ordinal == 1
    assert invalid_list.steps == []
    assert [item.code for item in invalid_list.diagnostics] == ["invalid_actions_list"]
    assert invalid_repeat.steps == []
    assert [item.code for item in invalid_repeat.diagnostics] == ["invalid_repeat"]


def test_parser_rejects_malformed_or_unbounded_repeat_tokens() -> None:
    malformed = HIMAProposalParser().parse(
        "So my advice is <Probe> x foo <Pylon> x -1 <Gateway> x 33"
    )

    assert malformed.steps == []
    assert [item.code for item in malformed.diagnostics] == [
        "invalid_repeat",
        "invalid_repeat",
        "invalid_repeat",
    ]
    assert [item.ordinal for item in malformed.diagnostics] == [0, 1, 2]


def test_parser_bounds_action_items_and_marks_generation_truncation() -> None:
    rendered = ", ".join("'Probe'" for _ in range(MAX_ACTION_ITEMS + 1))
    bounded = HIMAProposalParser().parse(f"Actions: [{rendered}]")
    truncated = HIMAProposalParser().parse("Actions: ['Probe']", truncated=True)

    assert len(bounded.steps) == MAX_ACTION_ITEMS
    assert bounded.diagnostics[-1].code == "action_limit_exceeded"
    assert [item.code for item in truncated.diagnostics] == ["output_truncated"]
    assert [step.canonical_action for step in truncated.steps] == ["TRAIN PROBE"]


def test_parser_reports_missing_and_empty_action_sections() -> None:
    missing = HIMAProposalParser().parse("Reason: Add more production.")
    empty = HIMAProposalParser().parse("Actions: []")

    assert [item.code for item in missing.diagnostics] == ["action_section_missing"]
    assert [item.code for item in empty.diagnostics] == ["empty_action_sequence"]
