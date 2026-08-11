"""Shared structured semantics for terminal-collapse decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rtscortex.cortex.models import ArmyReadiness, SituationAssessment
from rtscortex.progress import GoalRequirementKind
from rtscortex.races.models import RaceProfileData


class TerminalCollapseReason(StrEnum):
    """Typed reasons emitted at terminal-collapse semantic boundaries."""

    DEFENSE_PREREQUISITE_SUPPRESSED = "defense_prerequisite_suppressed_terminal_collapse"
    MACRO_FRONTIER_OBSOLETE = "terminal_collapse_macro_frontier_obsolete"
    NON_RECOVERY_MACRO_DISPATCH = "terminal_collapse_non_recovery_macro_dispatch"


@dataclass(frozen=True, slots=True)
class TerminalCollapseState:
    """Minimal structured state shared by runtime and transport boundaries."""

    army_readiness: ArmyReadiness
    own_base_count: int
    own_production_capacity: int


def is_terminal_collapse_state(state: TerminalCollapseState) -> bool:
    """Return the exact terminal-collapse predicate for a typed state snapshot."""

    return (
        state.army_readiness is ArmyReadiness.EMPTY
        and state.own_base_count == 0
        and state.own_production_capacity == 0
    )


def is_terminal_collapse(situation: SituationAssessment) -> bool:
    """Return the narrow, observation-derived terminal-collapse predicate."""

    return is_terminal_collapse_state(
        TerminalCollapseState(
            army_readiness=situation.army_readiness,
            own_base_count=situation.bases.own_base_count,
            own_production_capacity=situation.bases.own_production_capacity,
        )
    )


def townhall_recovery_runtime_actions(profile: RaceProfileData) -> frozenset[str]:
    """Return race-profile actions that directly create a base townhall.

    Morphs and upgrades are excluded because they require an existing townhall and
    therefore cannot recover a zero-townhall state.
    """

    townhall_types = {unit_type.casefold() for unit_type in profile.townhall_types}
    return frozenset(
        spec.name
        for spec in profile.progress_action_specs
        if spec.effect_kind is GoalRequirementKind.STRUCTURE
        and spec.effect_target.casefold() in townhall_types
        and not spec.prerequisites
    )
