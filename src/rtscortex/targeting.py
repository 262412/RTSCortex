"""Shared target eligibility invariants for every decision layer."""

from __future__ import annotations

from collections.abc import Iterable

from rtscortex.contracts import UnitState


def living_targetable_enemies(units: Iterable[UnitState]) -> list[UnitState]:
    """Return visible enemy units that can still be targeted."""

    return [
        unit
        for unit in units
        if unit.alliance == "enemy" and unit.health_fraction > 0.0
    ]
