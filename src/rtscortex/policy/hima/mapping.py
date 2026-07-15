"""Map HIMA macro proposals into shadow-only RTSCortex assessments."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rtscortex.contracts import ActionCommand, ActionSource, AvailableAction
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    PolicyActionAssessment,
    PolicyActionClassification,
    PolicyObservationFixture,
)
from rtscortex.runtime.validation import ActionValidator, ValidationDisposition


@dataclass(frozen=True, slots=True)
class HIMAMacroMapping:
    """One HIMA semantic action and its ordered Runtime implementations."""

    macro_action: str
    runtime_actions: tuple[str, ...]


HIMA_RUNTIME_MAPPINGS: tuple[HIMAMacroMapping, ...] = (
    HIMAMacroMapping("TRAIN ZEALOT", ("Train_Zealot", "Warp_Zealot_Near")),
    HIMAMacroMapping("TRAIN STALKER", ("Train_Stalker", "Warp_Stalker_Near")),
    HIMAMacroMapping("BUILD PYLON", ("Build_Pylon_Screen",)),
    HIMAMacroMapping("BUILD GATEWAY", ("Build_Gateway_Screen",)),
    HIMAMacroMapping(
        "BUILD CYBERNETICSCORE",
        ("Build_CyberneticsCore_Screen",),
    ),
    HIMAMacroMapping("BUILD ASSIMILATOR", ("Build_Assimilator_Near",)),
    HIMAMacroMapping("BUILD NEXUS", ("Build_Nexus_Near",)),
    HIMAMacroMapping("RESEARCH WARPGATERESEARCH", ("Research_WarpGate",)),
)

_MAPPINGS_BY_MACRO = {mapping.macro_action: mapping for mapping in HIMA_RUNTIME_MAPPINGS}
_STEP_PARSE_ERROR_CODES = frozenset(
    {
        "action_limit_exceeded",
        "action_section_missing",
        "empty_action_sequence",
        "expanded_action_limit_exceeded",
        "invalid_action_item",
        "invalid_actions_list",
        "invalid_repeat",
        "output_too_long",
        "output_truncated",
        "unknown_action",
        "unknown_action_token",
    }
)


class HIMAMacroActionMapper:
    """Assess logical and Runtime-actionable frontiers without dispatching."""

    def assess(
        self,
        proposal: MacroPolicyProposal,
        fixture: PolicyObservationFixture,
    ) -> list[PolicyActionAssessment]:
        parse_errors = [
            diagnostic
            for diagnostic in proposal.diagnostics
            if diagnostic.code in _STEP_PARSE_ERROR_CODES
        ]
        logical_ordinals = [step.ordinal for step in proposal.steps]
        logical_ordinals.extend(
            diagnostic.ordinal
            for diagnostic in parse_errors
            if diagnostic.ordinal is not None
        )
        logical_frontier_ordinal = min(logical_ordinals, default=None)
        runtime_frontier_ordinal = min(
            (
                step.ordinal
                for step in proposal.steps
                if step.canonical_action in _MAPPINGS_BY_MACRO
            ),
            default=None,
        )
        assessments = [
            PolicyActionAssessment(
                ordinal=diagnostic.ordinal if diagnostic.ordinal is not None else 0,
                repeat=diagnostic.repeat,
                source_action=diagnostic.raw_token or "<action_section>",
                classification=PolicyActionClassification.PARSE_ERROR,
                reason_code=diagnostic.code,
                is_logical_frontier=(
                    diagnostic.ordinal == logical_frontier_ordinal
                    if diagnostic.ordinal is not None
                    else logical_frontier_ordinal is None and index == 0
                ),
            )
            for index, diagnostic in enumerate(parse_errors)
        ]
        assessments.extend(
            self._assess_step(
                step,
                fixture,
                is_logical_frontier=step.ordinal == logical_frontier_ordinal,
                is_runtime_frontier=step.ordinal == runtime_frontier_ordinal,
            )
            for step in proposal.steps
        )
        return sorted(assessments, key=lambda item: (item.ordinal, item.source_action))

    def _assess_step(
        self,
        step: MacroActionStep,
        fixture: PolicyObservationFixture,
        *,
        is_logical_frontier: bool,
        is_runtime_frontier: bool,
    ) -> PolicyActionAssessment:
        mapping = _MAPPINGS_BY_MACRO.get(step.canonical_action)
        if mapping is None:
            reason = (
                "managed_automatically"
                if step.canonical_action == "TRAIN PROBE"
                else "not_implemented"
            )
            return PolicyActionAssessment(
                ordinal=step.ordinal,
                repeat=step.repeat,
                source_action=step.canonical_action,
                classification=PolicyActionClassification.UNSUPPORTED_BY_RUNTIME,
                reason_code=reason,
                is_logical_frontier=is_logical_frontier,
            )

        if not is_runtime_frontier:
            return PolicyActionAssessment(
                ordinal=step.ordinal,
                repeat=step.repeat,
                source_action=step.canonical_action,
                runtime_action=mapping.runtime_actions[0],
                classification=PolicyActionClassification.MAPPED_FUTURE,
                reason_code="future_horizon_not_evaluated",
                is_logical_frontier=is_logical_frontier,
            )

        return self._assess_frontier(
            step,
            mapping,
            fixture,
            is_logical_frontier=is_logical_frontier,
        )

    def _assess_frontier(
        self,
        step: MacroActionStep,
        mapping: HIMAMacroMapping,
        fixture: PolicyObservationFixture,
        *,
        is_logical_frontier: bool,
    ) -> PolicyActionAssessment:
        observation = fixture.observation
        available_by_name: dict[str, list[AvailableAction]] = {}
        for action in observation.available_actions:
            available_by_name.setdefault(action.name, []).append(action)

        saw_available_action = False
        saw_deferred = False
        saw_obsolete = False
        rejected_reasons: list[str] = []
        for runtime_action in mapping.runtime_actions:
            for available in available_by_name.get(runtime_action, []):
                saw_available_action = True
                commands = _candidate_commands(
                    step,
                    runtime_action,
                    available,
                    fixture,
                )
                if not commands:
                    saw_deferred = True
                    continue
                outcome = ActionValidator(max_actions=1).validate_candidates(
                    commands,
                    observation,
                )
                if outcome.accepted:
                    return _assessment(
                        step,
                        runtime_action,
                        PolicyActionClassification.MAPPED_LEGAL_NOW,
                        "validated",
                        is_logical_frontier=is_logical_frontier,
                    )
                for failure in outcome.failures:
                    if failure.disposition is ValidationDisposition.DEFERRED:
                        saw_deferred = True
                    elif failure.disposition is ValidationDisposition.OBSOLETE:
                        saw_obsolete = True
                    else:
                        rejected_reasons.append(_reason_code(failure.reason))

        runtime_action = mapping.runtime_actions[0]
        if not saw_available_action:
            return _assessment(
                step,
                runtime_action,
                PolicyActionClassification.MAPPED_DEFERRED,
                "action_unavailable_now",
                is_logical_frontier=is_logical_frontier,
            )
        if saw_deferred:
            return _assessment(
                step,
                runtime_action,
                PolicyActionClassification.MAPPED_DEFERRED,
                "actor_or_candidate_unavailable",
                is_logical_frontier=is_logical_frontier,
            )
        if saw_obsolete:
            return _assessment(
                step,
                runtime_action,
                PolicyActionClassification.OBSOLETE,
                "goal_already_satisfied",
                is_logical_frontier=is_logical_frontier,
            )
        return _assessment(
            step,
            runtime_action,
            PolicyActionClassification.ILLEGAL_ACTION,
            rejected_reasons[0] if rejected_reasons else "validator_rejected",
            is_logical_frontier=is_logical_frontier,
        )


def _candidate_commands(
    step: MacroActionStep,
    runtime_action: str,
    available: AvailableAction,
    fixture: PolicyObservationFixture,
) -> list[ActionCommand]:
    actors = list(dict.fromkeys(available.actor_scopes))
    if not actors:
        return []
    if not available.argument_names:
        argument_sets: list[list[object]] = [[]]
    elif available.argument_candidates:
        argument_sets = [list(candidate) for candidate in available.argument_candidates]
    else:
        return []
    return [
        ActionCommand(
            command_id=(
                f"shadow:hima:{fixture.fixture_id}:{step.ordinal}:"
                f"{actor_index}:{candidate_index}"
            ),
            actor=actor,
            name=runtime_action,
            arguments=arguments,
            priority=50,
            ttl_game_loops=1,
            created_game_loop=fixture.observation.game_loop,
            source=ActionSource.PLANNER,
        )
        for actor_index, actor in enumerate(actors)
        for candidate_index, arguments in enumerate(argument_sets)
    ]


def _assessment(
    step: MacroActionStep,
    runtime_action: str,
    classification: PolicyActionClassification,
    reason_code: str,
    *,
    is_logical_frontier: bool,
) -> PolicyActionAssessment:
    return PolicyActionAssessment(
        ordinal=step.ordinal,
        repeat=step.repeat,
        source_action=step.canonical_action,
        runtime_action=runtime_action,
        classification=classification,
        reason_code=reason_code,
        is_logical_frontier=is_logical_frontier,
        is_runtime_frontier=True,
        is_frontier=True,
    )


def _reason_code(reason: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", reason.casefold()).strip("_")
