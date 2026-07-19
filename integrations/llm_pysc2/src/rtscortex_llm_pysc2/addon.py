"""Terran add-on semantics shared by availability and effect verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional


@dataclass(frozen=True)
class AddonSpec:
    """One producer-bound Terran add-on action."""

    action_name: str
    producer_type: str
    addon_type: str
    feature_function_id: int
    generic_feature_function_id: int
    raw_order_id: int
    ability_id: int
    minerals: int
    vespene: int
    prerequisites: tuple[str, ...]
    supply: int = 0
    race: str = "terran"


_ADDON_SPECS = (
    AddonSpec(
        "Build_BarracksTechLab",
        "Barracks",
        "BarracksTechLab",
        94,
        92,
        225,
        421,
        50,
        25,
        ("Barracks",),
    ),
    AddonSpec(
        "Build_BarracksReactor",
        "Barracks",
        "BarracksReactor",
        73,
        71,
        208,
        422,
        50,
        50,
        ("Barracks",),
    ),
    AddonSpec(
        "Build_FactoryTechLab", "Factory", "FactoryTechLab", 96, 92, 227, 454, 50, 25, ("Factory",)
    ),
    AddonSpec(
        "Build_FactoryReactor", "Factory", "FactoryReactor", 75, 71, 210, 455, 50, 50, ("Factory",)
    ),
    AddonSpec(
        "Build_StarportTechLab",
        "Starport",
        "StarportTechLab",
        98,
        92,
        229,
        487,
        50,
        25,
        ("Starport",),
    ),
    AddonSpec(
        "Build_StarportReactor",
        "Starport",
        "StarportReactor",
        77,
        71,
        212,
        488,
        50,
        50,
        ("Starport",),
    ),
)

ADDON_REGISTRY_VERSION = "simple64-terran-addon-v1"
ADDON_SPECS: Mapping[str, AddonSpec] = MappingProxyType(
    {spec.action_name: spec for spec in _ADDON_SPECS}
)
ADDON_SPECS_BY_ORDER: Mapping[int, AddonSpec] = MappingProxyType(
    {spec.raw_order_id: spec for spec in _ADDON_SPECS}
)
ADDON_SPECS_BY_ABILITY: Mapping[int, AddonSpec] = MappingProxyType(
    {spec.ability_id: spec for spec in _ADDON_SPECS}
)


def addon_spec(action_name: str) -> Optional[AddonSpec]:
    return ADDON_SPECS.get(action_name)


def addon_spec_for_order(order_id: int) -> Optional[AddonSpec]:
    return ADDON_SPECS_BY_ORDER.get(order_id) or ADDON_SPECS_BY_ABILITY.get(order_id)


__all__ = [
    "ADDON_REGISTRY_VERSION",
    "ADDON_SPECS",
    "ADDON_SPECS_BY_ABILITY",
    "ADDON_SPECS_BY_ORDER",
    "AddonSpec",
    "addon_spec",
    "addon_spec_for_order",
]
