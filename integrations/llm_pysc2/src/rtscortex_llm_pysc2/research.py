"""Research semantics shared by availability and effect verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional


@dataclass(frozen=True)
class ResearchSpec:
    """One exact-source upgrade supported by the live Worker."""

    action_name: str
    producer_type: str
    upgrade_name: str
    upgrade_id: int
    feature_function_id: int
    raw_order_id: int
    ability_id: int
    minerals: int
    vespene: int
    supply: int
    prerequisites: tuple[str, ...]
    race: str
    required_addon_type: str | None = None


_RESEARCH_SPECS = (
    ResearchSpec(
        "Research_WarpGate",
        "CyberneticsCore",
        "WarpGateResearch",
        84,
        428,
        82,
        1568,
        50,
        50,
        0,
        ("CyberneticsCore",),
        "protoss",
    ),
    ResearchSpec(
        "Research_Stimpack",
        "Barracks",
        "Stimpack",
        15,
        405,
        451,
        730,
        100,
        100,
        0,
        ("Barracks", "BarracksTechLab"),
        "terran",
        required_addon_type="BarracksTechLab",
    ),
)

RESEARCH_REGISTRY_VERSION = "simple64-multirace-research-v1"
RESEARCH_SPECS: Mapping[str, ResearchSpec] = MappingProxyType(
    {spec.action_name: spec for spec in _RESEARCH_SPECS}
)
RESEARCH_SPECS_BY_ORDER: Mapping[int, ResearchSpec] = MappingProxyType(
    {spec.raw_order_id: spec for spec in _RESEARCH_SPECS}
)
RESEARCH_SPECS_BY_ABILITY: Mapping[int, ResearchSpec] = MappingProxyType(
    {spec.ability_id: spec for spec in _RESEARCH_SPECS}
)


def research_spec(action_name: str) -> Optional[ResearchSpec]:
    return RESEARCH_SPECS.get(action_name)


def research_spec_for_order(order_id: int) -> Optional[ResearchSpec]:
    return RESEARCH_SPECS_BY_ORDER.get(order_id) or RESEARCH_SPECS_BY_ABILITY.get(order_id)


__all__ = [
    "RESEARCH_REGISTRY_VERSION",
    "RESEARCH_SPECS",
    "RESEARCH_SPECS_BY_ABILITY",
    "RESEARCH_SPECS_BY_ORDER",
    "ResearchSpec",
    "research_spec",
    "research_spec_for_order",
]
