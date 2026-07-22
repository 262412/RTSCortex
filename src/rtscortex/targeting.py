"""Shared target eligibility invariants for every decision layer."""

from __future__ import annotations

from collections.abc import Iterable

from rtscortex.contracts import ObservationEnvelope, UnitState

ENEMY_STRUCTURE_TYPES = frozenset(
    {
        "Armory",
        "Assimilator",
        "BanelingNest",
        "Barracks",
        "Bunker",
        "CommandCenter",
        "CyberneticsCore",
        "DarkShrine",
        "EngineeringBay",
        "EvolutionChamber",
        "Extractor",
        "Factory",
        "FleetBeacon",
        "Forge",
        "FusionCore",
        "Gateway",
        "GhostAcademy",
        "GreaterSpire",
        "Hatchery",
        "Hive",
        "HydraliskDen",
        "InfestationPit",
        "Lair",
        "MissileTurret",
        "Nexus",
        "NydusNetwork",
        "OrbitalCommand",
        "PhotonCannon",
        "PlanetaryFortress",
        "Pylon",
        "Reactor",
        "Refinery",
        "RoboticsBay",
        "RoboticsFacility",
        "RoachWarren",
        "ShieldBattery",
        "SpawningPool",
        "SpineCrawler",
        "Spire",
        "SporeCrawler",
        "Stargate",
        "Starport",
        "SupplyDepot",
        "TechLab",
        "TemplarArchive",
        "TwilightCouncil",
        "UltraliskCavern",
        "WarpGate",
    }
)


def living_targetable_enemies(units: Iterable[UnitState]) -> list[UnitState]:
    """Return living enemies retained in the structured world state."""

    return [
        unit
        for unit in units
        if unit.alliance == "enemy" and unit.health_fraction > 0.0
    ]


def current_screen_enemy_targets(observation: ObservationEnvelope) -> list[UnitState]:
    """Return living enemies in the exact current ``Attack_Unit`` candidate domain."""

    candidate_tags = {
        _normalize_tag(arguments[0])
        for action in observation.available_actions
        if action.name == "Attack_Unit"
        for arguments in action.argument_candidates or ()
        if arguments
    }
    if not candidate_tags:
        return []
    return [
        enemy
        for enemy in living_targetable_enemies(observation.state.visible_enemies)
        if _normalize_tag(enemy.unit_id) in candidate_tags
    ]


def last_known_enemy_targets(observation: ObservationEnvelope) -> list[UnitState]:
    """Return living state targets that are not attackable on the current screen."""

    current_tags = {
        _normalize_tag(enemy.unit_id) for enemy in current_screen_enemy_targets(observation)
    }
    return [
        enemy
        for enemy in living_targetable_enemies(observation.state.visible_enemies)
        if _normalize_tag(enemy.unit_id) not in current_tags
    ]


def enemy_structures(units: Iterable[UnitState]) -> list[UnitState]:
    """Return living enemy structures using the cross-race structure vocabulary."""

    return [
        enemy
        for enemy in living_targetable_enemies(units)
        if enemy.unit_type in ENEMY_STRUCTURE_TYPES
    ]


def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()
