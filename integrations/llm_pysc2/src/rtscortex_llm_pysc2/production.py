"""Pinned direct-production semantics used by the live Worker.

The feature and raw function identifiers are from the vendored PySC2 revision.
Keeping availability, observation projection, and effect verification on one
registry prevents those three paths from silently disagreeing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional


@dataclass(frozen=True)
class ProductionSpec:
    """One directly trained unit supported by the Simple64 Worker."""

    action_name: str
    producer_type: str
    unit_type: str
    feature_function_id: int
    raw_order_id: int
    ability_id: int
    minerals: int
    vespene: int
    supply: int
    prerequisites: tuple[str, ...]
    race: str = "protoss"


_PRODUCTION_SPECS = (
    ProductionSpec(
        "Train_Zealot",
        "Gateway",
        "Zealot",
        503,
        49,
        916,
        100,
        0,
        2,
        ("Gateway",),
    ),
    ProductionSpec(
        "Train_Stalker",
        "Gateway",
        "Stalker",
        493,
        50,
        917,
        125,
        50,
        2,
        ("Gateway", "CyberneticsCore"),
    ),
    ProductionSpec(
        "Train_Adept",
        "Gateway",
        "Adept",
        457,
        54,
        922,
        100,
        25,
        2,
        ("Gateway", "CyberneticsCore"),
    ),
    ProductionSpec(
        "Train_Phoenix",
        "Stargate",
        "Phoenix",
        484,
        55,
        946,
        150,
        100,
        2,
        ("Stargate",),
    ),
    ProductionSpec(
        "Train_VoidRay",
        "Stargate",
        "VoidRay",
        500,
        57,
        950,
        250,
        150,
        4,
        ("Stargate",),
    ),
    ProductionSpec(
        "Train_Oracle",
        "Stargate",
        "Oracle",
        482,
        58,
        954,
        150,
        150,
        3,
        ("Stargate",),
    ),
    ProductionSpec(
        "Train_Marine",
        "Barracks",
        "Marine",
        477,
        511,
        560,
        50,
        0,
        1,
        ("Barracks",),
        race="terran",
    ),
    ProductionSpec(
        "Train_Marauder",
        "Barracks",
        "Marauder",
        476,
        510,
        563,
        100,
        25,
        2,
        ("Barracks", "BarracksTechLab"),
        race="terran",
    ),
    ProductionSpec(
        "Train_Hellion",
        "Factory",
        "Hellion",
        470,
        506,
        595,
        100,
        0,
        2,
        ("Factory",),
        race="terran",
    ),
    ProductionSpec(
        "Train_SiegeTank",
        "Factory",
        "SiegeTank",
        492,
        521,
        591,
        150,
        125,
        3,
        ("Factory", "FactoryTechLab"),
        race="terran",
    ),
    ProductionSpec(
        "Train_Medivac",
        "Starport",
        "Medivac",
        478,
        512,
        620,
        100,
        100,
        2,
        ("Starport",),
        race="terran",
    ),
    ProductionSpec(
        "Train_VikingFighter",
        "Starport",
        "VikingFighter",
        498,
        525,
        624,
        150,
        75,
        2,
        ("Starport",),
        race="terran",
    ),
)

PRODUCTION_REGISTRY_VERSION = "simple64-multirace-v2"
PRODUCTION_SPECS: Mapping[str, ProductionSpec] = MappingProxyType(
    {spec.action_name: spec for spec in _PRODUCTION_SPECS}
)
PRODUCTION_SPECS_BY_RACE: Mapping[str, Mapping[str, ProductionSpec]] = MappingProxyType(
    {
        race: MappingProxyType(
            {spec.action_name: spec for spec in _PRODUCTION_SPECS if spec.race == race}
        )
        for race in ("protoss", "terran")
    }
)
PRODUCTION_SPECS_BY_RAW_ORDER: Mapping[int, ProductionSpec] = MappingProxyType(
    {spec.raw_order_id: spec for spec in _PRODUCTION_SPECS}
)
PRODUCTION_SPECS_BY_ABILITY: Mapping[int, ProductionSpec] = MappingProxyType(
    {spec.ability_id: spec for spec in _PRODUCTION_SPECS}
)


def production_spec(action_name: str) -> Optional[ProductionSpec]:
    """Return the supported direct-training spec for an action name."""

    return PRODUCTION_SPECS.get(action_name)


def production_spec_for_order(order_id: int) -> Optional[ProductionSpec]:
    """Resolve either a raw function ID or an SC2 ability ID."""

    return PRODUCTION_SPECS_BY_RAW_ORDER.get(order_id) or PRODUCTION_SPECS_BY_ABILITY.get(order_id)


__all__ = [
    "PRODUCTION_REGISTRY_VERSION",
    "PRODUCTION_SPECS",
    "PRODUCTION_SPECS_BY_ABILITY",
    "PRODUCTION_SPECS_BY_RACE",
    "PRODUCTION_SPECS_BY_RAW_ORDER",
    "ProductionSpec",
    "production_spec",
    "production_spec_for_order",
]
