"""Canonical build-placement geometry used by acceptance auditing."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalPlacementSpec:
    """Structure identity and world-grid footprint for one build action."""

    structure_type: str
    footprint: int
    reserves_addon_space: bool = False


CANONICAL_PLACEMENT_SPECS: dict[str, CanonicalPlacementSpec] = {
    "Build_Pylon_Screen": CanonicalPlacementSpec("Pylon", 2),
    "Build_Gateway_Screen": CanonicalPlacementSpec("Gateway", 3),
    "Build_Forge_Screen": CanonicalPlacementSpec("Forge", 3),
    "Build_CyberneticsCore_Screen": CanonicalPlacementSpec("CyberneticsCore", 3),
    "Build_Assimilator_Near": CanonicalPlacementSpec("Assimilator", 3),
    "Build_Nexus_Near": CanonicalPlacementSpec("Nexus", 5),
    "Build_Stargate_Screen": CanonicalPlacementSpec("Stargate", 3),
    "Build_ShieldBattery_Screen": CanonicalPlacementSpec("ShieldBattery", 2),
    "Build_SupplyDepot_Screen": CanonicalPlacementSpec("SupplyDepot", 2),
    "Build_Barracks_Screen": CanonicalPlacementSpec(
        "Barracks",
        3,
        reserves_addon_space=True,
    ),
    "Build_Refinery_Near": CanonicalPlacementSpec("Refinery", 3),
    "Build_CommandCenter_Near": CanonicalPlacementSpec("CommandCenter", 5),
    "Build_Factory_Screen": CanonicalPlacementSpec(
        "Factory",
        3,
        reserves_addon_space=True,
    ),
    "Build_Starport_Screen": CanonicalPlacementSpec(
        "Starport",
        3,
        reserves_addon_space=True,
    ),
    "Build_EngineeringBay_Screen": CanonicalPlacementSpec("EngineeringBay", 3),
    "Build_Bunker_Screen": CanonicalPlacementSpec("Bunker", 3),
    "Build_MissileTurret_Screen": CanonicalPlacementSpec("MissileTurret", 2),
    "Build_Hatchery_Near": CanonicalPlacementSpec("Hatchery", 5),
    "Build_Extractor_Near": CanonicalPlacementSpec("Extractor", 3),
    "Build_SpawningPool_Screen": CanonicalPlacementSpec("SpawningPool", 3),
    "Build_RoachWarren_Screen": CanonicalPlacementSpec("RoachWarren", 3),
    "Build_EvolutionChamber_Screen": CanonicalPlacementSpec("EvolutionChamber", 3),
    "Build_HydraliskDen_Screen": CanonicalPlacementSpec("HydraliskDen", 3),
    "Build_SpineCrawler_Screen": CanonicalPlacementSpec("SpineCrawler", 2),
    "Build_SporeCrawler_Screen": CanonicalPlacementSpec("SporeCrawler", 2),
    "Build_CreepTumor_Queen_Screen": CanonicalPlacementSpec("CreepTumorQueen", 1),
    "Build_CreepTumor_Tumor_Screen": CanonicalPlacementSpec("CreepTumor", 1),
}


def canonical_footprint_cells(
    world_center: tuple[float, float],
    spec: CanonicalPlacementSpec,
) -> frozenset[tuple[int, int]]:
    """Recompute the exact build-grid reservation from authoritative geometry."""

    start_x = math.floor(float(world_center[0]) - spec.footprint / 2 + 0.5)
    start_y = math.floor(float(world_center[1]) - spec.footprint / 2 + 0.5)
    main_cells = frozenset(
        (x, y)
        for y in range(start_y, start_y + spec.footprint)
        for x in range(start_x, start_x + spec.footprint)
    )
    if not spec.reserves_addon_space:
        return main_cells
    max_x = max(cell[0] for cell in main_cells)
    min_y = min(cell[1] for cell in main_cells)
    max_y = max(cell[1] for cell in main_cells)
    addon_cells = {(x, y) for x in range(max_x + 1, max_x + 3) for y in range(min_y, max_y + 1)}
    return frozenset((*main_cells, *addon_cells))
