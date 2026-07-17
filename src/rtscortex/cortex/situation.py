"""Deterministic situation interpretation used until a specialist is promoted."""

from __future__ import annotations

import hashlib

from rtscortex.contracts import ObservationEnvelope
from rtscortex.cortex.models import (
    ArmyReadiness,
    EconomyStatus,
    GamePhase,
    SituationAssessment,
    ThreatLevel,
)

_TECH_STRUCTURES = frozenset(
    {
        "CyberneticsCore",
        "Cybernetics_Core",
        "Stargate",
        "RoboticsFacility",
        "TwilightCouncil",
        "TemplarArchive",
        "DarkShrine",
    }
)


class DeterministicSituationAnalyzer:
    analyzer_id = "deterministic-situation-analyzer"
    analyzer_version = "0.1.0"

    def __init__(self, *, valid_for_game_loops: int = 1) -> None:
        if valid_for_game_loops < 1:
            raise ValueError("valid_for_game_loops must be positive")
        self.valid_for_game_loops = valid_for_game_loops

    def assess(self, observation: ObservationEnvelope) -> SituationAssessment:
        enemies = observation.state.visible_enemies
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
