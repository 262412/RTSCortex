"""Deterministic short-horizon combat policy for the SC2-native Cortex."""

from __future__ import annotations

import hashlib

from rtscortex.contracts import ObservationEnvelope, UnitState
from rtscortex.cortex.models import (
    ArmyReadiness,
    IntentTarget,
    IntentTargetKind,
    SituationAssessment,
    TacticalIntent,
    ThreatLevel,
)

_WORKER_TYPES = frozenset({"Drone", "Probe", "SCV", "MULE"})
_HIGH_VALUE_THREATS = frozenset(
    {
        "Banshee",
        "Battlecruiser",
        "Carrier",
        "Colossus",
        "Disruptor",
        "HighTemplar",
        "Infestor",
        "Lurker",
        "Medivac",
        "Mutalisk",
        "Ravager",
        "SiegeTank",
        "SiegeTankSieged",
        "Thor",
        "Viper",
        "VoidRay",
    }
)


class DeterministicTacticalAgent:
    """Turn a current situation into exact, candidate-bound combat intents."""

    agent_id = "deterministic-tactical-agent"
    agent_version = "0.1.0"

    def __init__(
        self,
        *,
        retreat_health_threshold: float,
        minimum_advance_army_supply: int,
    ) -> None:
        self.retreat_health_threshold = retreat_health_threshold
        self.minimum_advance_army_supply = minimum_advance_army_supply
        self._episode_key: tuple[str, str] | None = None
        self._last_focus_target: str | None = None

    def evaluate(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
    ) -> list[TacticalIntent]:
        self._activate_episode(observation)
        attack_actors = _actors_for(observation, "Attack_Unit")
        move_actors = _actors_for(observation, "Move_Minimap")
        combat_health = [
            unit.health_fraction
            for unit in observation.state.own_units
            if unit.unit_type not in _WORKER_TYPES
        ]
        retreat = bool(combat_health) and (
            min(combat_health) <= self.retreat_health_threshold
            or (
                assessment.threat_level is ThreatLevel.CRITICAL
                and len(observation.state.visible_enemies) > len(combat_health)
            )
        )
        if retreat and move_actors:
            return [
                self._intent(
                    observation,
                    assessment,
                    actor=actor,
                    action_name="Move_Minimap",
                    objective="Retreat low-health combat units to the home defensive region",
                    target=IntentTarget(
                        kind=IntentTargetKind.RETREAT_REGION,
                        region="home",
                    ),
                    priority=85,
                    ttl_game_loops=8,
                )
                for actor in move_actors
            ]

        enemies = observation.state.visible_enemies
        if enemies and attack_actors:
            target, reacquired = self._focus_target(enemies)
            objective = (
                f"Reacquire and focus fire {target.unit_type}"
                if reacquired
                else f"Focus fire visible {target.unit_type}"
            )
            return [
                self._intent(
                    observation,
                    assessment,
                    actor=actor,
                    action_name="Attack_Unit",
                    objective=objective,
                    target=IntentTarget(
                        kind=IntentTargetKind.ENEMY,
                        unit_type=target.unit_type,
                    ),
                    priority=75,
                    ttl_game_loops=8,
                )
                for actor in attack_actors
            ]

        ready_to_advance = (
            assessment.army_readiness is ArmyReadiness.READY
            or observation.state.economy.army_supply
            >= self.minimum_advance_army_supply
        )
        if not enemies and ready_to_advance and move_actors:
            return [
                self._intent(
                    observation,
                    assessment,
                    actor=actor,
                    action_name="Move_Minimap",
                    objective="Advance to reacquire the enemy force",
                    target=IntentTarget(
                        kind=IntentTargetKind.ENEMY,
                        region="unexplored",
                    ),
                    priority=60,
                    ttl_game_loops=16,
                )
                for actor in move_actors
            ]
        return []

    def _activate_episode(self, observation: ObservationEnvelope) -> None:
        episode_key = (observation.run_id, observation.episode_id)
        if episode_key == self._episode_key:
            return
        self._episode_key = episode_key
        self._last_focus_target = None

    def _focus_target(self, enemies: list[UnitState]) -> tuple[UnitState, bool]:
        by_tag = {_normalize_tag(enemy.unit_id): enemy for enemy in enemies}
        previous = self._last_focus_target
        if previous is not None and previous in by_tag:
            return by_tag[previous], False
        target = min(enemies, key=_target_rank)
        self._last_focus_target = _normalize_tag(target.unit_id)
        return target, previous is not None

    def _intent(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
        *,
        actor: str,
        action_name: str,
        objective: str,
        target: IntentTarget,
        priority: int,
        ttl_game_loops: int,
    ) -> TacticalIntent:
        identity = "|".join(
            (
                observation.run_id,
                observation.episode_id,
                str(observation.step_id),
                actor,
                action_name,
                target.unit_type or target.region or "none",
            )
        )
        return TacticalIntent(
            intent_id=f"tactical:{hashlib.sha256(identity.encode()).hexdigest()}",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective=objective,
            action_names=[action_name],
            actor_scopes=[actor],
            target=target,
            priority=priority,
            ttl_game_loops=ttl_game_loops,
            source_id=self.agent_id,
            source_version=self.agent_version,
            situation_assessment_id=assessment.assessment_id,
        )


def _actors_for(observation: ObservationEnvelope, action_name: str) -> list[str]:
    return list(
        dict.fromkeys(
            actor
            for action in observation.available_actions
            if action.name == action_name
            for actor in action.actor_scopes
            if actor == "army" or actor.startswith("CombatGroup")
        )
    )


def _target_rank(enemy: UnitState) -> tuple[int, float, str]:
    if enemy.unit_type in _HIGH_VALUE_THREATS:
        class_rank = 0
    elif enemy.unit_type in _WORKER_TYPES:
        class_rank = 2
    else:
        class_rank = 1
    return class_rank, enemy.health_fraction, _normalize_tag(enemy.unit_id)


def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()
