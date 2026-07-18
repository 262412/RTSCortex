"""Deterministic situation interpretation used until a specialist is promoted."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence

from rtscortex.contracts import ObservationEnvelope, UnitState
from rtscortex.cortex.models import (
    ArmyReadiness,
    BaseAssessment,
    EconomyStatus,
    ForceComposition,
    GamePhase,
    KnowledgeStatus,
    ScoutingAssessment,
    SituationAssessment,
    SituationFact,
    SpatialAssessment,
    ThreatLevel,
)

_TECH_STRUCTURES = frozenset(
    {
        "Armory",
        "BanelingNest",
        "CyberneticsCore",
        "Cybernetics_Core",
        "EngineeringBay",
        "EvolutionChamber",
        "FactoryTechLab",
        "FusionCore",
        "GhostAcademy",
        "GreaterSpire",
        "HydraliskDen",
        "InfestationPit",
        "Lair",
        "LurkerDenMP",
        "RoachWarren",
        "Stargate",
        "StarportTechLab",
        "RoboticsFacility",
        "SpawningPool",
        "Spire",
        "TwilightCouncil",
        "TemplarArchive",
        "DarkShrine",
        "UltraliskCavern",
    }
)

_TOWNHALL_TYPES = frozenset(
    {
        "Nexus",
        "CommandCenter",
        "OrbitalCommand",
        "PlanetaryFortress",
        "Hatchery",
        "Lair",
        "Hive",
    }
)
_PRODUCTION_TYPES = frozenset(
    {
        "Gateway",
        "WarpGate",
        "RoboticsFacility",
        "Stargate",
        "Barracks",
        "Factory",
        "Starport",
        "Hatchery",
        "Lair",
        "Hive",
    }
)
_AIR_TYPES = frozenset(
    {
        "Banshee",
        "Battlecruiser",
        "BroodLord",
        "Carrier",
        "Corruptor",
        "Liberator",
        "Medivac",
        "Mothership",
        "Mutalisk",
        "Observer",
        "Oracle",
        "Overlord",
        "Overseer",
        "Phoenix",
        "Raven",
        "Tempest",
        "VikingFighter",
        "VoidRay",
        "WarpPrism",
    }
)
_UNIT_RESOURCE_VALUES = {
    "Adept": 125,
    "Baneling": 75,
    "Battlecruiser": 700,
    "Carrier": 600,
    "Drone": 50,
    "Hellion": 100,
    "Hydralisk": 150,
    "Marine": 50,
    "Marauder": 125,
    "Medivac": 200,
    "Mutalisk": 200,
    "Oracle": 300,
    "Phoenix": 250,
    "Probe": 50,
    "Queen": 150,
    "Roach": 100,
    "SCV": 50,
    "SiegeTank": 275,
    "Stalker": 175,
    "VikingFighter": 225,
    "VoidRay": 400,
    "Zealot": 100,
    "Zergling": 25,
}
_ENEMY_TECH_SIGNALS = {
    "Banshee": "terran_cloaked_air",
    "Battlecruiser": "terran_capital_air",
    "BroodLord": "zerg_greater_spire",
    "Carrier": "protoss_fleet_beacon",
    "Colossus": "protoss_robotics_bay",
    "DarkTemplar": "protoss_dark_shrine",
    "Hive": "zerg_hive",
    "Lair": "zerg_lair",
    "LurkerMP": "zerg_lurker_den",
    "Mutalisk": "zerg_spire",
    "SiegeTank": "terran_factory_tech_lab",
    "Stargate": "protoss_stargate",
    "Starport": "terran_starport",
    "VoidRay": "protoss_stargate",
}


class DeterministicSituationAnalyzer:
    analyzer_id = "deterministic-situation-analyzer"
    analyzer_version = "2.0.0"

    def __init__(self, *, valid_for_game_loops: int = 1) -> None:
        if valid_for_game_loops < 1:
            raise ValueError("valid_for_game_loops must be positive")
        self.valid_for_game_loops = valid_for_game_loops
        self._episode_key: tuple[str, str] | None = None
        self._last_enemy_seen_game_loop: int | None = None

    def assess(
        self,
        observation: ObservationEnvelope,
        history: Sequence[ObservationEnvelope] = (),
    ) -> SituationAssessment:
        del history
        episode_key = (observation.run_id, observation.episode_id)
        if episode_key != self._episode_key:
            self._episode_key = episode_key
            self._last_enemy_seen_game_loop = None
        enemies = observation.state.visible_enemies
        if enemies:
            self._last_enemy_seen_game_loop = observation.game_loop
        alerts = {alert.casefold() for alert in observation.alerts}
        under_attack = any(
            marker in alert
            for alert in alerts
            for marker in ("under_attack", "under attack", "base_attack")
        )
        army_supply = observation.state.economy.army_supply
        phase = self._phase(observation, under_attack=under_attack)
        threat_level = self._threat_level(len(enemies), under_attack=under_attack)
        economy = observation.state.economy
        if economy.minerals >= 800 or economy.vespene >= 500:
            economy_status = EconomyStatus.FLOATING
        elif economy.minerals < 100 and economy.vespene < 100:
            economy_status = EconomyStatus.CONSTRAINED
        else:
            economy_status = EconomyStatus.STABLE
        if enemies and army_supply > 0:
            readiness = ArmyReadiness.ENGAGED
        elif army_supply >= 10:
            readiness = ArmyReadiness.READY
        elif army_supply > 0:
            readiness = ArmyReadiness.FORMING
        else:
            readiness = ArmyReadiness.EMPTY
        threats = sorted({enemy.unit_type for enemy in enemies})
        information_gaps = [] if enemies else ["enemy_force_not_visible"]
        own_force = _force_composition(observation.state.own_units)
        enemy_force = _force_composition(enemies)
        own_structures = observation.state.own_structures
        enemy_structures = [enemy for enemy in enemies if enemy.unit_type in _PRODUCTION_TYPES]
        bases = BaseAssessment(
            own_base_count=_count_types(own_structures, _TOWNHALL_TYPES),
            visible_enemy_base_count=_count_types(enemies, _TOWNHALL_TYPES),
            own_production_capacity=_count_types(own_structures, _PRODUCTION_TYPES),
            visible_enemy_production_capacity=_count_types(
                enemy_structures,
                _PRODUCTION_TYPES,
            ),
        )
        nearest_distance = _nearest_enemy_distance(own_structures, enemies)
        confirmed_tech = tuple(
            sorted(
                {
                    signal
                    for enemy in enemies
                    if (signal := _ENEMY_TECH_SIGNALS.get(enemy.unit_type)) is not None
                }
            )
        )
        possible_transitions = _possible_transitions(enemy_force, confirmed_tech)
        scouting_confidence = _scouting_confidence(
            observation.game_loop,
            self._last_enemy_seen_game_loop,
        )
        facts = _situation_facts(
            phase=phase,
            threat_level=threat_level,
            economy_status=economy_status,
            army_readiness=readiness,
            own_force=own_force,
            enemy_force=enemy_force,
            bases=bases,
            enemies_visible=bool(enemies),
            confirmed_tech=confirmed_tech,
            nearest_distance=nearest_distance,
            possible_transitions=possible_transitions,
        )
        identity = "|".join(
            (
                observation.run_id,
                observation.episode_id,
                str(observation.step_id),
                str(observation.game_loop),
                self.analyzer_id,
                self.analyzer_version,
            )
        )
        return SituationAssessment(
            assessment_id=f"assessment:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            game_loop=observation.game_loop,
            valid_until_game_loop=observation.game_loop + self.valid_for_game_loops,
            phase=phase,
            threat_level=threat_level,
            economy_status=economy_status,
            army_readiness=readiness,
            threats=threats,
            information_gaps=information_gaps,
            own_force=own_force,
            visible_enemy_force=enemy_force,
            bases=bases,
            spatial=SpatialAssessment(
                nearest_threat_distance=nearest_distance,
                threat_eta_seconds=None,
                visible_enemy_regions=(),
                map_control_fraction=None,
            ),
            scouting=ScoutingAssessment(
                enemy_visible=bool(enemies),
                last_enemy_seen_game_loop=self._last_enemy_seen_game_loop,
                confidence=scouting_confidence,
                confirmed_enemy_tech=confirmed_tech,
                inferred_enemy_tech=(),
                possible_transitions=possible_transitions,
            ),
            facts=facts,
            source_kind="deterministic",
            source_id=self.analyzer_id,
            source_version=self.analyzer_version,
        )

    @staticmethod
    def _phase(observation: ObservationEnvelope, *, under_attack: bool) -> GamePhase:
        state = observation.state
        if under_attack or (state.visible_enemies and state.economy.army_supply > 0):
            return GamePhase.COMBAT
        structure_types = {structure.unit_type for structure in state.own_structures}
        if state.upgrades or structure_types.intersection(_TECH_STRUCTURES):
            return GamePhase.TECHNOLOGY
        if state.production_queue or "Gateway" in structure_types:
            return GamePhase.PRODUCTION
        return GamePhase.EARLY

    @staticmethod
    def _threat_level(enemy_count: int, *, under_attack: bool) -> ThreatLevel:
        if under_attack and enemy_count >= 6:
            return ThreatLevel.CRITICAL
        if under_attack:
            return ThreatLevel.HIGH
        if enemy_count:
            return ThreatLevel.LOW
        return ThreatLevel.NONE


def _force_composition(units: Sequence[UnitState]) -> ForceComposition:
    counts = Counter(unit.unit_type for unit in units)
    unknown = tuple(sorted(name for name in counts if name not in _UNIT_RESOURCE_VALUES))
    air_units = sum(count for name, count in counts.items() if name in _AIR_TYPES)
    return ForceComposition(
        counts=dict(sorted(counts.items())),
        total_units=len(units),
        ground_units=len(units) - air_units,
        air_units=air_units,
        estimated_resource_value=sum(
            _UNIT_RESOURCE_VALUES.get(name, 0) * count for name, count in counts.items()
        ),
        unknown_unit_types=unknown,
    )


def _count_types(units: Sequence[UnitState], unit_types: frozenset[str]) -> int:
    return sum(unit.unit_type in unit_types for unit in units)


def _nearest_enemy_distance(
    own_structures: Sequence[UnitState],
    enemies: Sequence[UnitState],
) -> float | None:
    own_positions = [unit.position for unit in own_structures if unit.position is not None]
    enemy_positions = [unit.position for unit in enemies if unit.position is not None]
    if not own_positions or not enemy_positions:
        return None
    return min(
        math.dist(own_position, enemy_position)
        for own_position in own_positions
        for enemy_position in enemy_positions
    )


def _scouting_confidence(game_loop: int, last_seen: int | None) -> float:
    if last_seen is None:
        return 0.0
    age = max(0, game_loop - last_seen)
    return max(0.0, 1.0 - age / 448.0)


def _possible_transitions(
    force: ForceComposition,
    confirmed_tech: tuple[str, ...],
) -> tuple[str, ...]:
    transitions: set[str] = set()
    if force.air_units:
        transitions.add("continued_air_production")
    if force.ground_units:
        transitions.add("continued_ground_production")
    if any("cloaked" in signal or "dark_shrine" in signal for signal in confirmed_tech):
        transitions.add("detection_required")
    return tuple(sorted(transitions))


def _situation_facts(
    *,
    phase: GamePhase,
    threat_level: ThreatLevel,
    economy_status: EconomyStatus,
    army_readiness: ArmyReadiness,
    own_force: ForceComposition,
    enemy_force: ForceComposition,
    bases: BaseAssessment,
    enemies_visible: bool,
    confirmed_tech: tuple[str, ...],
    nearest_distance: float | None,
    possible_transitions: tuple[str, ...],
) -> list[SituationFact]:
    return [
        SituationFact(
            name="game_phase",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="deterministic_phase_rules",
            evidence=(phase.value,),
        ),
        SituationFact(
            name="threat_level",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="visible_enemies_and_alerts",
            evidence=(threat_level.value,),
        ),
        SituationFact(
            name="economy_status",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="observed_resources",
            evidence=(economy_status.value,),
        ),
        SituationFact(
            name="army_readiness",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="observed_army_supply_and_contact",
            evidence=(army_readiness.value,),
        ),
        SituationFact(
            name="own_force_composition",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="own_units",
            evidence=(f"units:{own_force.total_units}",),
        ),
        SituationFact(
            name="visible_enemy_force_composition",
            status=(
                KnowledgeStatus.CONFIRMED
                if enemies_visible
                else KnowledgeStatus.UNKNOWN
            ),
            confidence=1.0 if enemies_visible else 0.0,
            source="visible_enemies",
            evidence=(f"units:{enemy_force.total_units}",),
        ),
        SituationFact(
            name="base_and_production_capacity",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="observed_structures",
            evidence=(
                f"own_bases:{bases.own_base_count}",
                f"own_production:{bases.own_production_capacity}",
            ),
        ),
        SituationFact(
            name="enemy_force_visible",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="visible_enemies",
            evidence=("visible_enemy_count_nonzero",) if enemies_visible else ("no_enemy_visible",),
        ),
        SituationFact(
            name="enemy_technology",
            status=(KnowledgeStatus.CONFIRMED if confirmed_tech else KnowledgeStatus.UNKNOWN),
            confidence=1.0 if confirmed_tech else 0.0,
            source="visible_enemy_type_signals",
            evidence=confirmed_tech,
        ),
        SituationFact(
            name="enemy_transition",
            status=(
                KnowledgeStatus.INFERRED
                if possible_transitions
                else KnowledgeStatus.UNKNOWN
            ),
            confidence=0.5 if possible_transitions else 0.0,
            source="visible_force_and_technology_rules",
            evidence=possible_transitions,
        ),
        SituationFact(
            name="map_control",
            status=KnowledgeStatus.UNKNOWN,
            confidence=0.0,
            source="unavailable_in_protocol_v1.1",
        ),
        SituationFact(
            name="threat_distance",
            status=(
                KnowledgeStatus.CONFIRMED
                if nearest_distance is not None
                else KnowledgeStatus.UNKNOWN
            ),
            confidence=1.0 if nearest_distance is not None else 0.0,
            source="unit_positions",
            evidence=(() if nearest_distance is None else (f"distance:{nearest_distance:.3f}",)),
        ),
    ]
