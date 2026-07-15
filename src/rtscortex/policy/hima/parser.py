"""Deterministic parsing of the supported HIMA Protoss output formats."""

from __future__ import annotations

import ast
import re

from rtscortex.policy.hima.models import (
    HIMA_ADAPTER_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_VOCABULARY_VERSION,
)
from rtscortex.policy.hima.vocabulary import resolve_hima_action
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    ParseDiagnostic,
    TacticalRationale,
)

MAX_RAW_OUTPUT_CHARS = 32_768
MAX_ACTION_ITEMS = 128
MAX_ACTION_REPEAT = 32
MAX_EXPANDED_ACTIONS = 256

_FINAL_SUMMARY_RE = re.compile(r"final\s+actions\s+summary\s*:?", re.IGNORECASE)
_ADVICE_RE = re.compile(r"so\s+my\s+advice\s+is\s*:?\s*", re.IGNORECASE)
_ANGLE_ITEM_RE = re.compile(r"\s*<([^>\n]+)>(?:\s*[xX]\s*([^\s<]+))?")
_ACTIONS_LIST_RE = re.compile(
    r"\bactions\s*:\s*(\[[^\]]*\])",
    re.IGNORECASE | re.DOTALL,
)
_RATIONALE_HEADING = {
    "immediate": r"immediate(?:\s+steps?)?",
    "short_term": r"short[\s-]*term(?:\s+(?:actions?|strategy))?",
    "long_term": r"long[\s-]*term(?:\s+(?:actions?|strategy))?",
}
_ACTION_BOUNDARY = (
    r"\bActions\s*:|\bFinal\s+Actions\s+Summary\s*:?|"
    r"\bSo\s+My\s+Advice\s+Is\s*:?"
)


class HIMAProposalParser:
    """Parse pinned HIMA formats while retaining all malformed input diagnostics."""

    def parse(
        self,
        raw_output: str,
        *,
        truncated: bool = False,
    ) -> MacroPolicyProposal:
        bounded_output = raw_output[:MAX_RAW_OUTPUT_CHARS]
        diagnostics: list[ParseDiagnostic] = []
        if len(raw_output) > MAX_RAW_OUTPUT_CHARS:
            diagnostics.append(
                ParseDiagnostic(
                    code="output_too_long",
                    message=(
                        "HIMA output exceeded the parser character limit and was truncated."
                    ),
                )
            )
        if truncated:
            diagnostics.append(
                ParseDiagnostic(
                    code="output_truncated",
                    message="Generation reached its token limit before an EOS token.",
                )
            )

        rationale = _parse_rationale(bounded_output)
        tokens, extraction_diagnostics = _extract_tokens(bounded_output)
        diagnostics.extend(extraction_diagnostics)

        steps: list[MacroActionStep] = []
        expanded_count = 0
        if tokens is None:
            if not extraction_diagnostics:
                diagnostics.append(
                    ParseDiagnostic(
                        code="action_section_missing",
                        message="No supported HIMA action section was found.",
                    )
                )
        elif tokens == [] and not extraction_diagnostics:
            diagnostics.append(
                ParseDiagnostic(
                    code="empty_action_sequence",
                    message="The HIMA action section contains no actions.",
                )
            )
        else:
            for ordinal, raw_token, repeat in tokens:
                action = resolve_hima_action(raw_token)
                if action is None:
                    diagnostics.append(
                        ParseDiagnostic(
                            code="unknown_action_token",
                            message=(
                                "Token is not in the pinned Protoss macro-action vocabulary."
                            ),
                            raw_token=raw_token,
                            ordinal=ordinal,
                            repeat=repeat,
                        )
                    )
                    continue
                if expanded_count + repeat > MAX_EXPANDED_ACTIONS:
                    diagnostics.append(
                        ParseDiagnostic(
                            code="expanded_action_limit_exceeded",
                            message="HIMA macro sequence exceeds the expanded action limit.",
                            raw_token=raw_token,
                            ordinal=ordinal,
                            repeat=repeat,
                        )
                    )
                    continue
                expanded_count += repeat
                steps.append(
                    MacroActionStep(
                        ordinal=ordinal,
                        canonical_action=action.canonical_action,
                        category=action.category,
                        repeat=repeat,
                        raw_token=raw_token,
                    )
                )

        objective = next(
            (
                value
                for value in (
                    rationale.long_term,
                    rationale.short_term,
                    rationale.immediate,
                )
                if value
            ),
            "HIMA macro recommendation",
        )
        return MacroPolicyProposal(
            strategic_objective=objective[:500],
            tactical_rationale=rationale,
            steps=steps,
            raw_output=bounded_output,
            adapter_version=HIMA_ADAPTER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            diagnostics=diagnostics,
        )


def _extract_tokens(
    raw_output: str,
) -> tuple[list[tuple[int, str, int]] | None, list[ParseDiagnostic]]:
    # The current checkpoint list format wins over explanatory angle examples.
    list_tokens, list_diagnostics = _extract_action_list(raw_output)
    if list_tokens is not None or list_diagnostics:
        return list_tokens, list_diagnostics

    final_summary = _FINAL_SUMMARY_RE.search(raw_output)
    if final_summary is not None:
        return _scan_angle_sequence(raw_output[final_summary.end() :])

    advice = _ADVICE_RE.search(raw_output)
    if advice is not None:
        return _scan_angle_sequence(raw_output[advice.end() :])

    stripped = raw_output.strip()
    if stripped.startswith("<"):
        return _scan_angle_sequence(stripped)
    return None, []


def _scan_angle_sequence(
    candidate: str,
) -> tuple[list[tuple[int, str, int]], list[ParseDiagnostic]]:
    tokens: list[tuple[int, str, int]] = []
    diagnostics: list[ParseDiagnostic] = []
    position = 0
    ordinal = 0
    while ordinal < MAX_ACTION_ITEMS:
        match = _ANGLE_ITEM_RE.match(candidate, position)
        if match is None:
            break
        raw_token = match.group(1).strip()
        raw_repeat = match.group(2)
        if raw_repeat is None:
            tokens.append((ordinal, raw_token, 1))
        elif not raw_repeat.isdecimal():
            diagnostics.append(
                ParseDiagnostic(
                    code="invalid_repeat",
                    message="A HIMA macro-action repeat must be a positive integer.",
                    raw_token=raw_token,
                    ordinal=ordinal,
                )
            )
        else:
            repeat = int(raw_repeat)
            if not 1 <= repeat <= MAX_ACTION_REPEAT:
                diagnostics.append(
                    ParseDiagnostic(
                        code="invalid_repeat",
                        message=(
                            f"A HIMA macro-action repeat must be between 1 and "
                            f"{MAX_ACTION_REPEAT}."
                        ),
                        raw_token=raw_token,
                        ordinal=ordinal,
                    )
                )
            else:
                tokens.append((ordinal, raw_token, repeat))
        ordinal += 1
        position = match.end()

    if ordinal == MAX_ACTION_ITEMS and _ANGLE_ITEM_RE.match(candidate, position):
        diagnostics.append(
            ParseDiagnostic(
                code="action_limit_exceeded",
                message=f"HIMA output contains more than {MAX_ACTION_ITEMS} action items.",
                ordinal=ordinal,
            )
        )
    return tokens, diagnostics


def _extract_action_list(
    raw_output: str,
) -> tuple[list[tuple[int, str, int]] | None, list[ParseDiagnostic]]:
    match = _ACTIONS_LIST_RE.search(raw_output)
    if match is None:
        return None, []
    rendered = match.group(1)
    try:
        parsed = ast.literal_eval(rendered)
    except (SyntaxError, ValueError):
        return None, [
            ParseDiagnostic(
                code="invalid_actions_list",
                message="Actions must be a valid JSON or Python string list.",
                raw_token=rendered,
            )
        ]
    if not isinstance(parsed, list):
        return None, [
            ParseDiagnostic(
                code="invalid_actions_list",
                message="Actions must be a list.",
                raw_token=rendered,
            )
        ]

    tokens: list[tuple[int, str, int]] = []
    diagnostics: list[ParseDiagnostic] = []
    for ordinal, item in enumerate(parsed):
        if ordinal >= MAX_ACTION_ITEMS:
            diagnostics.append(
                ParseDiagnostic(
                    code="action_limit_exceeded",
                    message=(
                        f"HIMA output contains more than {MAX_ACTION_ITEMS} action items."
                    ),
                    ordinal=ordinal,
                )
            )
            break
        if not isinstance(item, str):
            diagnostics.append(
                ParseDiagnostic(
                    code="invalid_action_item",
                    message="Every Actions list item must be a string.",
                    raw_token=repr(item),
                    ordinal=ordinal,
                )
            )
            continue
        tokens.append((ordinal, item.strip(), 1))
    return tokens, diagnostics


def _parse_rationale(raw_output: str) -> TacticalRationale:
    sections: dict[str, str] = {}
    all_headings = "|".join(f"(?:{heading})" for heading in _RATIONALE_HEADING.values())
    for field, heading in _RATIONALE_HEADING.items():
        pattern = re.compile(
            rf"(?:\*\*)?{heading}(?:\*\*)?\s*:\s*(.*?)"
            rf"(?=(?:\*\*)?(?:{all_headings})(?:\*\*)?\s*:"
            rf"|{_ACTION_BOUNDARY}|$)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(raw_output)
        sections[field] = _clean_rationale(match.group(1)) if match else ""
    return TacticalRationale(**sections)


def _clean_rationale(value: str) -> str:
    return " ".join(value.replace("**", "").strip(" \n\t,.'\"").split())
