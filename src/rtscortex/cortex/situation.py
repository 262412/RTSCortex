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
    ResourcePressure,
    ScoutingAssessment,
    SituationAssessment,
    SituationFact,
    SpatialAssessment,
    ThreatLevel,
)
from rtscortex.targeting import living_targetable_enemies

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
_ANTI_AIR_TYPES = frozenset(
    {
        "Archon",
        "Battlecruiser",
        "Carrier",
        "Corruptor",
        "Cyclone",
        "Hydralisk",
        "Marine",
        "Mothership",
        "Mutalisk",
        "Phoenix",
        "Queen",
        "Stalker",
        "Tempest",
        "Thor",
        "VikingFighter",
        "VoidRay",
    }
)
_AIR_COMBAT_THREAT_TYPES = frozenset(
    {
        "Banshee",
        "Battlecruiser",
        "BroodLord",
        "Carrier",
        "Corruptor",
        "Liberator",
        "Mothership",
        "Mutalisk",
        "Oracle",
        "Phoenix",
        "Tempest",
        "VikingFighter",
        "VoidRay",
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
    analyzer_version = "2.2.0"

    def __init__(
        self,
        *,
        valid_for_game_loops: int = 1,
        threat_hysteresis_game_loops: int = 32,
    ) -> None:
        if valid_for_game_loops < 1:
            raise ValueError("valid_for_game_loops must be positive")
        if threat_hysteresis_game_loops < 1:
            raise ValueError("threat_hysteresis_game_loops must be positive")
        self.valid_for_game_loops = valid_for_game_loops
        self.threat_hysteresis_game_loops = threat_hysteresis_game_loops
        self._episode_key: tuple[str, str] | None = None
        self._last_enemy_seen_game_loop: int | None = None
        self._held_threat_level = ThreatLevel.NONE
        self._threat_hold_until_game_loop: int | None = None
        self._previous_own_health: dict[str, float] = {}
        self._previous_worker_count: int | None = None
        self._previous_army_supply: int | None = None
        self._previous_base_count: int | None = None

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
            self._held_threat_level = ThreatLevel.NONE
            self._threat_hold_until_game_loop = None
            self._previous_own_health.clear()
            self._previous_worker_count = None
            self._previous_army_supply = None
            self._previous_base_count = None
        enemies = living_targetable_enemies(observation.state.visible_enemies)
        if enemies:
            self._last_enemy_seen_game_loop = observation.game_loop
        alerts = {alert.casefold() for alert in observation.alerts}
        building_under_attack = any(
            marker in alert
            for alert in alerts
            for marker in ("building_under_attack", "building under attack", "base_attack")
        )
        unit_under_attack = any(
            marker in alert
            for alert in alerts
            for marker in ("unit_under_attack", "unit under attack", "under_attack", "under attack")
        )
        under_attack = unit_under_attack or building_under_attack
        army_supply = observation.state.economy.army_supply
        economy = observation.state.economy
        mineral_pressure = _resource_pressure(
            economy.minerals,
            starved_below=100,
            floating_at=800,
        )
        gas_pressure = _resource_pressure(
            economy.vespene,
            starved_below=50,
            floating_at=500,
        )
        if mineral_pressure is ResourcePressure.STARVED:
            economy_status = EconomyStatus.CONSTRAINED
        elif mineral_pressure is ResourcePressure.FLOATING or (
            gas_pressure is ResourcePressure.FLOATING and economy.minerals >= 400
        ):
            economy_status = EconomyStatus.FLOATING
        else:
            economy_status = EconomyStatus.STABLE
        if army_supply >= 10:
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
        damage_evidence = self._damage_and_loss_evidence(observation)
        threat_level, threat_score, threat_evidence = self._assess_threat(
            observation,
            enemies=enemies,
            own_force=own_force,
            enemy_force=enemy_force,
            bases=bases,
            nearest_distance=nearest_distance,
            unit_under_attack=unit_under_attack,
            building_under_attack=building_under_attack,
            damage_evidence=damage_evidence,
        )
        if (
            army_supply > 0
            and threat_level in {ThreatLevel.HIGH, ThreatLevel.CRITICAL}
        ):
            readiness = ArmyReadiness.ENGAGED
        crisis = (
            bases.own_base_count == 0
            and bool(enemies)
            and bool(
                observation.state.own_structures
                or observation.state.economy.workers
            )
        )
        phase = self._phase(
            observation,
            under_attack=under_attack,
            crisis=crisis,
            threat_level=threat_level,
        )
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
            mineral_pressure=mineral_pressure,
            gas_pressure=gas_pressure,
            army_readiness=readiness,
            own_force=own_force,
            enemy_force=enemy_force,
            bases=bases,
            enemies_visible=bool(enemies),
            confirmed_tech=confirmed_tech,
            nearest_distance=nearest_distance,
            possible_transitions=possible_transitions,
            threat_score=threat_score,
            threat_evidence=threat_evidence,
            crisis=crisis,
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
            threat_score=threat_score,
            threat_evidence=threat_evidence,
            threat_hysteresis_until_game_loop=self._threat_hold_until_game_loop,
            economy_status=economy_status,
            mineral_pressure=mineral_pressure,
            gas_pressure=gas_pressure,
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
    def _phase(
        observation: ObservationEnvelope,
        *,
        under_attack: bool,
        crisis: bool,
        threat_level: ThreatLevel,
    ) -> GamePhase:
        state = observation.state
        if (
            crisis
            or under_attack
            or threat_level in {ThreatLevel.HIGH, ThreatLevel.CRITICAL}
        ):
            return GamePhase.COMBAT
        if state.economy.army_supply >= 24:
            return GamePhase.COMBAT
        structure_types = {structure.unit_type for structure in state.own_structures}
        if state.production_queue:
            return GamePhase.PRODUCTION
        if (
            state.economy.army_supply >= 8
            and structure_types.intersection(_PRODUCTION_TYPES - _TOWNHALL_TYPES)
        ):
            return GamePhase.PRODUCTION
        if state.upgrades or structure_types.intersection(_TECH_STRUCTURES):
            return GamePhase.TECHNOLOGY
        if structure_types.intersection(_PRODUCTION_TYPES - _TOWNHALL_TYPES):
            return GamePhase.PRODUCTION
        return GamePhase.EARLY

    def _damage_and_loss_evidence(
        self,
        observation: ObservationEnvelope,
    ) -> tuple[str, ...]:
        current_health = {
            unit.unit_id: unit.health_fraction
            for unit in (
                *observation.state.own_units,
                *observation.state.own_structures,
            )
        }
        evidence: list[str] = []
        damaged = sum(
            previous - current_health[tag] >= 0.05
            for tag, previous in self._previous_own_health.items()
            if tag in current_health
        )
        if damaged:
            evidence.append(f"recent_damage:{damaged}")
        economy = observation.state.economy
        base_count = _count_types(observation.state.own_structures, _TOWNHALL_TYPES)
        if (
            self._previous_worker_count is not None
            and economy.workers < self._previous_worker_count
        ):
            evidence.append(f"worker_losses:{self._previous_worker_count - economy.workers}")
        if (
            self._previous_army_supply is not None
            and economy.army_supply < self._previous_army_supply
        ):
            evidence.append(f"army_supply_loss:{self._previous_army_supply - economy.army_supply}")
        if self._previous_base_count is not None and base_count < self._previous_base_count:
            evidence.append(f"base_losses:{self._previous_base_count - base_count}")
        self._previous_own_health = current_health
        self._previous_worker_count = economy.workers
        self._previous_army_supply = economy.army_supply
        self._previous_base_count = base_count
        return tuple(evidence)

    def _assess_threat(
        self,
        observation: ObservationEnvelope,
        *,
        enemies: Sequence[UnitState],
        own_force: ForceComposition,
        enemy_force: ForceComposition,
        bases: BaseAssessment,
        nearest_distance: float | None,
        unit_under_attack: bool,
        building_under_attack: bool,
        damage_evidence: tuple[str, ...],
    ) -> tuple[ThreatLevel, float, tuple[str, ...]]:
        if not enemies:
            if (
                self._held_threat_level in {ThreatLevel.HIGH, ThreatLevel.CRITICAL}
                and self._threat_hold_until_game_loop is not None
                and observation.game_loop <= self._threat_hold_until_game_loop
            ):
                level = self._held_threat_level
                score = 4.0 if level is ThreatLevel.HIGH else 7.0
                return level, score, (f"hysteresis:{level.value}", "enemy_temporarily_unseen")
            self._held_threat_level = ThreatLevel.NONE
            self._threat_hold_until_game_loop = None
            return ThreatLevel.NONE, 0.0, ()

        score = 1.0
        evidence = [f"living_enemies:{len(enemies)}"]
        if unit_under_attack:
            score = max(score, 4.0)
            evidence.append("unit_under_attack")
        if building_under_attack:
            score = max(score, 7.0)
            evidence.append("building_under_attack")
        own_combat_value = _combat_value(own_force)
        enemy_combat_value = _combat_value(enemy_force)
        if nearest_distance is not None:
            evidence.append(f"nearest_distance:{nearest_distance:.3f}")
            if nearest_distance <= 8.0 and enemy_combat_value:
                score += 3.0
                evidence.append("base_proximity:immediate")
            elif nearest_distance <= 16.0 and enemy_combat_value:
                score += 2.0
                evidence.append("base_proximity:near")
        if enemy_combat_value:
            ratio = enemy_combat_value / max(1, own_combat_value)
            evidence.append(f"enemy_own_combat_ratio:{ratio:.3f}")
            if own_combat_value == 0:
                score = max(score, 7.0)
                evidence.append("empty_army_overwhelmed")
            elif ratio >= 2.0:
                score += 4.0
                evidence.append("enemy_force_overwhelming")
            elif ratio >= 0.75:
                score += 3.0
                evidence.append("enemy_force_comparable")
        if any(
            enemy_force.counts.get(unit_type, 0)
            for unit_type in _AIR_COMBAT_THREAT_TYPES
        ) and not any(
            own_force.counts.get(unit_type, 0) for unit_type in _ANTI_AIR_TYPES
        ):
            score += 2.0
            evidence.append("capability_mismatch:no_anti_air")
        if (
            bases.own_base_count == 0
            and (observation.state.own_structures or observation.state.economy.workers)
        ):
            score = max(score, 7.0)
            evidence.append("no_surviving_townhall")
        if damage_evidence:
            score += min(3.0, float(len(damage_evidence)))
            evidence.extend(damage_evidence)

        computed = _threat_level_for_score(score)
        if computed in {ThreatLevel.HIGH, ThreatLevel.CRITICAL}:
            self._held_threat_level = computed
            self._threat_hold_until_game_loop = (
                observation.game_loop + self.threat_hysteresis_game_loops
            )
        elif (
            self._held_threat_level in {ThreatLevel.HIGH, ThreatLevel.CRITICAL}
            and self._threat_hold_until_game_loop is not None
            and observation.game_loop <= self._threat_hold_until_game_loop
        ):
            computed = self._held_threat_level
            score = max(score, 4.0 if computed is ThreatLevel.HIGH else 7.0)
            evidence.append(f"hysteresis:{computed.value}")
        else:
            self._held_threat_level = computed
            self._threat_hold_until_game_loop = None
        return computed, score, tuple(evidence)


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


def _combat_value(force: ForceComposition) -> int:
    worker_value = sum(
        force.counts.get(worker_type, 0) * _UNIT_RESOURCE_VALUES[worker_type]
        for worker_type in ("Drone", "Probe", "SCV")
    )
    return max(0, force.estimated_resource_value - worker_value)


def _threat_level_for_score(score: float) -> ThreatLevel:
    if score >= 7.0:
        return ThreatLevel.CRITICAL
    if score >= 4.0:
        return ThreatLevel.HIGH
    if score > 0.0:
        return ThreatLevel.LOW
    return ThreatLevel.NONE


def _resource_pressure(
    amount: int,
    *,
    starved_below: int,
    floating_at: int,
) -> ResourcePressure:
    if amount < starved_below:
        return ResourcePressure.STARVED
    if amount >= floating_at:
        return ResourcePressure.FLOATING
    return ResourcePressure.BALANCED


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
    mineral_pressure: ResourcePressure,
    gas_pressure: ResourcePressure,
    army_readiness: ArmyReadiness,
    own_force: ForceComposition,
    enemy_force: ForceComposition,
    bases: BaseAssessment,
    enemies_visible: bool,
    confirmed_tech: tuple[str, ...],
    nearest_distance: float | None,
    possible_transitions: tuple[str, ...],
    threat_score: float,
    threat_evidence: tuple[str, ...],
    crisis: bool,
) -> list[SituationFact]:
    return [
        SituationFact(
            name="game_phase",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="deterministic_phase_rules",
            evidence=(
                (phase.value, "terminal_collapse:no_surviving_townhall")
                if crisis
                else (phase.value,)
            ),
        ),
        SituationFact(
            name="threat_level",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="stateful_threat_rules",
            evidence=(
                threat_level.value,
                f"score:{threat_score:.3f}",
                *threat_evidence,
            ),
        ),
        SituationFact(
            name="economy_status",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="observed_resources",
            evidence=(economy_status.value,),
        ),
        SituationFact(
            name="mineral_pressure",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="observed_minerals",
            evidence=(mineral_pressure.value,),
        ),
        SituationFact(
            name="gas_pressure",
            status=KnowledgeStatus.CONFIRMED,
            confidence=1.0,
            source="observed_vespene",
            evidence=(gas_pressure.value,),
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
            status=(KnowledgeStatus.CONFIRMED if enemies_visible else KnowledgeStatus.UNKNOWN),
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
            status=(KnowledgeStatus.INFERRED if possible_transitions else KnowledgeStatus.UNKNOWN),
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
