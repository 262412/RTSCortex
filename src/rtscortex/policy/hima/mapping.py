"""Map HIMA macro proposals into shadow-only RTSCortex assessments."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rtscortex.contracts import (
    ActionCommand,
    ActionSource,
    AvailableAction,
    SC2State,
    UnitState,
)
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    PolicyActionAssessment,
    PolicyActionClassification,
    PolicyObservationFixture,
)
from rtscortex.progress.models import GoalRequirementKind
from rtscortex.progress.verifier import ProgressActionSpec
from rtscortex.races import PROTOSS_PROFILE_DATA, RaceProfileData, race_profile
from rtscortex.runtime.validation import ActionValidator, ValidationDisposition


@dataclass(frozen=True, slots=True)
class HIMAMacroMapping:
    """One HIMA semantic action and its ordered Runtime implementations."""

    macro_action: str
    runtime_actions: tuple[str, ...]


HIMA_RUNTIME_MAPPINGS: tuple[HIMAMacroMapping, ...] = tuple(
    HIMAMacroMapping(mapping.semantic_action, mapping.runtime_actions)
    for mapping in PROTOSS_PROFILE_DATA.macro_action_mappings
)


def hima_runtime_mappings(race: str) -> tuple[HIMAMacroMapping, ...]:
    return tuple(
        HIMAMacroMapping(mapping.semantic_action, mapping.runtime_actions)
        for mapping in race_profile(race).data.macro_action_mappings
    )
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


@dataclass(frozen=True, slots=True)
class _FrontierProbe:
    runtime_action: str
    classification: PolicyActionClassification
    reason_code: str
    hard_blocker: bool = False


class HIMAMacroActionMapper:
    """Assess logical and Runtime-actionable frontiers without dispatching."""

    def __init__(self, profile: RaceProfileData = PROTOSS_PROFILE_DATA) -> None:
        self.profile = profile
        self._mappings_by_macro = {
            mapping.macro_action: mapping
            for mapping in (
                HIMAMacroMapping(item.semantic_action, item.runtime_actions)
                for item in profile.macro_action_mappings
            )
        }
        self._progress_specs_by_action = {
            spec.name: spec for spec in profile.progress_action_specs
        }
        self._managed_worker_action = f"TRAIN {profile.worker_type.upper()}"

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
            diagnostic.ordinal for diagnostic in parse_errors if diagnostic.ordinal is not None
        )
        logical_frontier_ordinal = min(logical_ordinals, default=None)
        probes: dict[int, _FrontierProbe] = {}
        runtime_frontier_ordinal: int | None = None
        first_soft_deferred_ordinal: int | None = None
        for step in sorted(proposal.steps, key=lambda item: item.ordinal):
            mapping = self._mappings_by_macro.get(step.canonical_action)
            if mapping is None:
                continue
            probe = self._probe_step(step, mapping, fixture)
            probes[step.ordinal] = probe
            if probe.classification is PolicyActionClassification.MAPPED_DEFERRED:
                if first_soft_deferred_ordinal is None:
                    first_soft_deferred_ordinal = step.ordinal
                if probe.hard_blocker:
                    runtime_frontier_ordinal = step.ordinal
                    break
                continue
            if probe.classification is PolicyActionClassification.OBSOLETE:
                continue
            runtime_frontier_ordinal = step.ordinal
            break
        if runtime_frontier_ordinal is None:
            runtime_frontier_ordinal = first_soft_deferred_ordinal
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
                probe=probes.get(step.ordinal),
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
        probe: _FrontierProbe | None,
        is_logical_frontier: bool,
        is_runtime_frontier: bool,
    ) -> PolicyActionAssessment:
        mapping = self._mappings_by_macro.get(step.canonical_action)
        if mapping is None:
            reason = (
                "managed_automatically"
                if step.canonical_action == self._managed_worker_action
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

        if probe is None:
            return PolicyActionAssessment(
                ordinal=step.ordinal,
                repeat=step.repeat,
                source_action=step.canonical_action,
                runtime_action=mapping.runtime_actions[0],
                classification=PolicyActionClassification.MAPPED_FUTURE,
                reason_code="future_horizon_not_evaluated",
                is_logical_frontier=is_logical_frontier,
            )

        return _assessment(
            step,
            probe.runtime_action,
            probe.classification,
            probe.reason_code,
            is_logical_frontier=is_logical_frontier,
            is_runtime_frontier=is_runtime_frontier,
        )

    def _probe_step(
        self,
        step: MacroActionStep,
        mapping: HIMAMacroMapping,
        fixture: PolicyObservationFixture,
    ) -> _FrontierProbe:
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
                    return _FrontierProbe(
                        runtime_action,
                        PolicyActionClassification.MAPPED_LEGAL_NOW,
                        "validated",
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
            hard_blocker = _hard_state_blocker(
                mapping,
                fixture,
                self._progress_specs_by_action,
            )
            return _FrontierProbe(
                runtime_action,
                PolicyActionClassification.MAPPED_DEFERRED,
                hard_blocker or "action_unavailable_now",
                hard_blocker=hard_blocker is not None,
            )
        if saw_deferred:
            return _FrontierProbe(
                runtime_action,
                PolicyActionClassification.MAPPED_DEFERRED,
                "actor_or_candidate_unavailable",
            )
        if saw_obsolete:
            return _FrontierProbe(
                runtime_action,
                PolicyActionClassification.OBSOLETE,
                "goal_already_satisfied",
            )
        return _FrontierProbe(
            runtime_action,
            PolicyActionClassification.ILLEGAL_ACTION,
            rejected_reasons[0] if rejected_reasons else "validator_rejected",
        )


def _hard_state_blocker(
    mapping: HIMAMacroMapping,
    fixture: PolicyObservationFixture,
    progress_specs_by_action: dict[str, ProgressActionSpec],
) -> str | None:
    spec = next(
        (
            progress_specs_by_action[action_name]
            for action_name in mapping.runtime_actions
            if action_name in progress_specs_by_action
        ),
        None,
    )
    if spec is None:
        return None
    state = fixture.observation.state
    for prerequisite in spec.prerequisites:
        if _completed_count(state, prerequisite.kind, prerequisite.target) >= prerequisite.count:
            continue
        suffix = _reason_code(prerequisite.target)
        if _in_progress_count(state, prerequisite.kind, prerequisite.target) > 0:
            return f"prerequisite_in_progress_{suffix}"
        return f"missing_prerequisite_{suffix}"
    economy = state.economy
    if economy.minerals < spec.minerals:
        return "insufficient_minerals"
    if economy.vespene < spec.vespene:
        return "insufficient_vespene"
    if economy.supply_cap - economy.supply_used < spec.supply:
        return "insufficient_supply"
    return None


def _completed_count(state: SC2State, kind: GoalRequirementKind, target: str) -> int:
    units = _state_units(state, kind)
    canonical_target = _reason_code(target).replace("_", "")
    return sum(
        _reason_code(unit.unit_type).replace("_", "") == canonical_target
        and _status_is_complete(unit.status)
        for unit in units
    )


def _in_progress_count(state: SC2State, kind: GoalRequirementKind, target: str) -> int:
    units = _state_units(state, kind)
    canonical_target = _reason_code(target).replace("_", "")
    return sum(
        _reason_code(unit.unit_type).replace("_", "") == canonical_target
        and not _status_is_complete(unit.status)
        for unit in units
    )


def _state_units(state: SC2State, kind: GoalRequirementKind) -> list[UnitState]:
    if kind is GoalRequirementKind.STRUCTURE:
        return state.own_structures
    if kind is GoalRequirementKind.UNIT:
        return state.own_units
    return []


def _status_is_complete(status: str | None) -> bool:
    return status is None or status.casefold() not in {
        "constructing",
        "in_progress",
        "pending",
        "queued",
        "warping_in",
    }


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
                f"shadow:hima:{fixture.fixture_id}:{step.ordinal}:{actor_index}:{candidate_index}"
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
    is_runtime_frontier: bool,
) -> PolicyActionAssessment:
    return PolicyActionAssessment(
        ordinal=step.ordinal,
        repeat=step.repeat,
        source_action=step.canonical_action,
        runtime_action=runtime_action,
        classification=classification,
        reason_code=reason_code,
        is_logical_frontier=is_logical_frontier,
        is_runtime_frontier=is_runtime_frontier,
        is_frontier=is_runtime_frontier,
    )


def _reason_code(reason: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", reason.casefold()).strip("_")
