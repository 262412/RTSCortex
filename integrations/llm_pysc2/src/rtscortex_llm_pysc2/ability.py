"""Exact-source economy ability semantics owned by RTSCortex."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional


@dataclass(frozen=True)
class AbilitySpec:
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
    race: str
    minimum_energy: float
    requires_idle: bool = False


_ABILITY_SPECS = (
    AbilitySpec(
        "Effect_CalldownMULE_Screen",
        "OrbitalCommand",
        "MULE",
        183,
        297,
        171,
        0,
        0,
        0,
        ("OrbitalCommand",),
        "terran",
        50.0,
    ),
)

ABILITY_REGISTRY_VERSION = "simple64-terran-ability-v1"
ABILITY_SPECS: Mapping[str, AbilitySpec] = MappingProxyType(
    {spec.action_name: spec for spec in _ABILITY_SPECS}
)


def ability_spec(action_name: str) -> Optional[AbilitySpec]:
    return ABILITY_SPECS.get(action_name)


__all__ = [
    "ABILITY_REGISTRY_VERSION",
    "ABILITY_SPECS",
    "AbilitySpec",
    "ability_spec",
]
