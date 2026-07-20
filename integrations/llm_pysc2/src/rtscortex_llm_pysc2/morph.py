"""Zerg structure-morph semantics shared by availability and verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional


@dataclass(frozen=True)
class MorphSpec:
    """One exact-source structure morph supported by the live Worker."""

    action_name: str
    producer_type: str
    result_type: str
    feature_function_id: int
    raw_order_id: int
    ability_id: int
    minerals: int
    vespene: int
    supply: int
    prerequisites: tuple[str, ...]
    race: str = "zerg"


_MORPH_SPECS = (
    MorphSpec(
        "Morph_OrbitalCommand",
        "CommandCenter",
        "OrbitalCommand",
        309,
        394,
        1516,
        150,
        0,
        0,
        ("CommandCenter", "Barracks"),
        race="terran",
    ),
    MorphSpec(
        "Morph_Lair",
        "Hatchery",
        "Lair",
        303,
        388,
        1216,
        150,
        100,
        0,
        ("Hatchery", "SpawningPool"),
    ),
)

MORPH_REGISTRY_VERSION = "simple64-multirace-morph-v2"
MORPH_SPECS: Mapping[str, MorphSpec] = MappingProxyType(
    {spec.action_name: spec for spec in _MORPH_SPECS}
)
MORPH_SPECS_BY_ORDER: Mapping[int, MorphSpec] = MappingProxyType(
    {spec.raw_order_id: spec for spec in _MORPH_SPECS}
)
MORPH_SPECS_BY_ABILITY: Mapping[int, MorphSpec] = MappingProxyType(
    {spec.ability_id: spec for spec in _MORPH_SPECS}
)


def morph_spec(action_name: str) -> Optional[MorphSpec]:
    return MORPH_SPECS.get(action_name)


def morph_spec_for_order(order_id: int) -> Optional[MorphSpec]:
    return MORPH_SPECS_BY_ORDER.get(order_id) or MORPH_SPECS_BY_ABILITY.get(order_id)


__all__ = [
    "MORPH_REGISTRY_VERSION",
    "MORPH_SPECS",
    "MORPH_SPECS_BY_ABILITY",
    "MORPH_SPECS_BY_ORDER",
    "MorphSpec",
    "morph_spec",
    "morph_spec_for_order",
]
