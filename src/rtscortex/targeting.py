"""Shared target eligibility invariants for every decision layer."""

from __future__ import annotations

from collections.abc import Iterable

from rtscortex.contracts import ObservationEnvelope, UnitState
from rtscortex.races import CombatTargetDomain, combat_target_domain, is_flying_unit

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


def attackable_enemies_for_actor(
    observation: ObservationEnvelope,
    actor: str,
    *,
    current_screen_only: bool = True,
) -> list[UnitState]:
    """Return living enemies compatible with the actor's actual weapon domain."""

    domains = {
        combat_target_domain(unit.unit_type)
        for unit in _units_for_actor(observation, actor)
    }
    if not domains:
        actor_type = _actor_unit_type(actor)
        if actor_type is not None:
            domains.add(combat_target_domain(actor_type))
    domains.discard(CombatTargetDomain.NONE)
    if not domains:
        return []
    can_attack_ground = bool(
        domains.intersection({CombatTargetDomain.GROUND, CombatTargetDomain.BOTH})
    )
    can_attack_air = bool(
        domains.intersection({CombatTargetDomain.AIR, CombatTargetDomain.BOTH})
    )
    candidate_tags = (
        _attack_candidate_tags(observation, actor)
        if current_screen_only
        else None
    )
    return [
        enemy
        for enemy in living_targetable_enemies(observation.state.visible_enemies)
        if (candidate_tags is None or _normalize_tag(enemy.unit_id) in candidate_tags)
        and (
            (is_flying_unit(enemy.unit_type) and can_attack_air)
            or (not is_flying_unit(enemy.unit_type) and can_attack_ground)
        )
    ]


def current_screen_enemy_targets(
    observation: ObservationEnvelope,
    actor: str | None = None,
) -> list[UnitState]:
    """Return living enemies in the exact current ``Attack_Unit`` candidate domain."""

    if actor is not None:
        return attackable_enemies_for_actor(observation, actor)
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


def _attack_candidate_tags(
    observation: ObservationEnvelope,
    actor: str,
) -> set[str]:
    return {
        _normalize_tag(arguments[0])
        for action in observation.available_actions
        if action.name == "Attack_Unit"
        and (not action.actor_scopes or actor in action.actor_scopes)
        for arguments in action.argument_candidates or ()
        if arguments
    }


def _units_for_actor(
    observation: ObservationEnvelope,
    actor: str,
) -> list[UnitState]:
    own_combat_units = [
        unit
        for unit in observation.state.own_units
        if unit.health_fraction > 0.0
        and combat_target_domain(unit.unit_type) is not CombatTargetDomain.NONE
    ]
    actor_type = _actor_unit_type(actor)
    if actor_type is None or actor_type.casefold() in {"army", "combat", "all"}:
        return own_combat_units
    return [unit for unit in own_combat_units if unit.unit_type == actor_type]


def _actor_unit_type(actor: str) -> str | None:
    if "/" not in actor:
        return actor or None
    token = actor.rsplit("/", 1)[-1]
    prefix, separator, suffix = token.rpartition("-")
    return prefix if separator and suffix.isdigit() else token


def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()
