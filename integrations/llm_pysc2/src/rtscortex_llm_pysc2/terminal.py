"""Python 3.9-compatible terminal-collapse transport semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TerminalArmyReadiness(str, Enum):  # noqa: UP042 - Worker supports Python 3.9
    """Worker-local projection of the shared army-readiness contract."""

    EMPTY = "empty"
    FORMING = "forming"
    READY = "ready"


class TerminalCollapseReason(str, Enum):  # noqa: UP042 - Worker supports Python 3.9
    """Typed failure reasons emitted at the final Worker dispatch boundary."""

    NON_RECOVERY_MACRO_DISPATCH = "terminal_collapse_non_recovery_macro_dispatch"


@dataclass(frozen=True)
class TerminalCollapseState:
    """Minimal observation-derived state needed by the exact predicate."""

    army_readiness: TerminalArmyReadiness
    own_base_count: int
    own_production_capacity: int


def is_terminal_collapse_state(state: TerminalCollapseState) -> bool:
    """Mirror the shared core predicate without importing Python 3.11 code."""

    return (
        state.army_readiness is TerminalArmyReadiness.EMPTY
        and state.own_base_count == 0
        and state.own_production_capacity == 0
    )


__all__ = [
    "TerminalArmyReadiness",
    "TerminalCollapseReason",
    "TerminalCollapseState",
    "is_terminal_collapse_state",
]
