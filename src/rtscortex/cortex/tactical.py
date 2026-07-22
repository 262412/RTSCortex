"""Deterministic short-horizon combat policy for the SC2-native Cortex."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

from rtscortex.contracts import ObservationEnvelope, UnitState
from rtscortex.cortex.models import (
    ArmyReadiness,
    IntentTarget,
    IntentTargetKind,
    SituationAssessment,
    TacticalIntent,
    ThreatLevel,
)
from rtscortex.targeting import living_targetable_enemies

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
_TOWNHALL_TYPES = frozenset(
    {"CommandCenter", "Hatchery", "Hive", "Lair", "Nexus", "OrbitalCommand"}
)


@dataclass(slots=True)
class _ActorRetreatState:
    phase: Literal["retreating", "arrived"]
    entered_game_loop: int
    last_command_game_loop: int
    cooldown_until_game_loop: int


class DeterministicTacticalAgent:
    """Turn a current situation into exact, candidate-bound combat intents."""

    agent_id = "deterministic-tactical-agent"
    agent_version = "0.2.0"
    provider_id = agent_id
    provider_version = agent_version

    def __init__(
        self,
        *,
        retreat_health_threshold: float,
        minimum_advance_army_supply: int,
        reacquire_cooldown_game_loops: int = 112,
        retreat_cooldown_game_loops: int = 112,
        retreat_hysteresis: float = 0.2,
        retreat_home_radius: float = 12.0,
    ) -> None:
        self.retreat_health_threshold = retreat_health_threshold
        self.minimum_advance_army_supply = minimum_advance_army_supply
        self.reacquire_cooldown_game_loops = reacquire_cooldown_game_loops
        self.retreat_cooldown_game_loops = retreat_cooldown_game_loops
        self.retreat_exit_health_threshold = min(
            1.0,
            retreat_health_threshold + retreat_hysteresis,
        )
        self.retreat_home_radius = retreat_home_radius
        self._episode_key: tuple[str, str] | None = None
        self._last_focus_target: str | None = None
        self._last_reacquire_by_actor: dict[str, int] = {}
        self._retreat_by_actor: dict[str, _ActorRetreatState] = {}

    def evaluate(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
    ) -> list[TacticalIntent]:
        self._activate_episode(observation)
        attack_actors = _actors_for(observation, "Attack_Unit")
        move_actors = _actors_for(observation, "Move_Minimap")
        enemies = living_targetable_enemies(observation.state.visible_enemies)
        retreat_intents, retreating_actors = self._retreat_intents(
            observation,
            assessment,
            enemies,
            move_actors,
        )
        attack_actors = [actor for actor in attack_actors if actor not in retreating_actors]
        move_actors = [actor for actor in move_actors if actor not in retreating_actors]
        if enemies:
            self._last_reacquire_by_actor.clear()
        if enemies and attack_actors:
            target, reacquired = self._focus_target(enemies)
            objective = (
                f"Reacquire and focus fire {target.unit_type}"
                if reacquired
                else f"Focus fire visible {target.unit_type}"
            )
            attack_intents = [
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
            return [*retreat_intents, *attack_intents]

        ready_to_advance = (
            assessment.army_readiness is ArmyReadiness.READY
            or observation.state.economy.army_supply >= self.minimum_advance_army_supply
        )
        if not enemies and ready_to_advance and move_actors:
            eligible_actors = [
                actor
                for actor in move_actors
                if observation.game_loop
                - self._last_reacquire_by_actor.get(
                    actor,
                    observation.game_loop - self.reacquire_cooldown_game_loops,
                )
                >= self.reacquire_cooldown_game_loops
            ]
            intents = [
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
                for actor in eligible_actors
            ]
            self._last_reacquire_by_actor.update(
                dict.fromkeys(eligible_actors, observation.game_loop)
            )
            return [*retreat_intents, *intents]
        return retreat_intents

    def _retreat_intents(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
        enemies: list[UnitState],
        move_actors: list[str],
    ) -> tuple[list[TacticalIntent], set[str]]:
        available = set(move_actors)
        for actor in tuple(self._retreat_by_actor):
            if actor not in available:
                # The actor vanished or its move action is no longer available;
                # its old retreat objective is now obsolete.
                del self._retreat_by_actor[actor]

        intents: list[TacticalIntent] = []
        retreating: set[str] = set()
        for actor in move_actors:
            units = _units_for_actor(observation, actor)
            if not units:
                self._retreat_by_actor.pop(actor, None)
                continue
            minimum_health = min(unit.health_fraction for unit in units)
            state = self._retreat_by_actor.get(actor)
            recovered = minimum_health >= self.retreat_exit_health_threshold
            if state is not None and recovered:
                del self._retreat_by_actor[actor]
                state = None

            at_home = _units_at_home(
                units,
                observation.state.own_structures,
                radius=self.retreat_home_radius,
            )
            if state is not None and at_home:
                state.phase = "arrived"
                state.cooldown_until_game_loop = max(
                    state.cooldown_until_game_loop,
                    observation.game_loop + self.retreat_cooldown_game_loops,
                )

            overwhelmed = (
                assessment.threat_level is ThreatLevel.CRITICAL
                and len(enemies) > len(units)
            )
            should_retreat = minimum_health <= self.retreat_health_threshold or overwhelmed
            if not should_retreat:
                continue
            retreating.add(actor)
            if at_home:
                if state is None:
                    self._retreat_by_actor[actor] = _ActorRetreatState(
                        phase="arrived",
                        entered_game_loop=observation.game_loop,
                        last_command_game_loop=observation.game_loop,
                        cooldown_until_game_loop=(
                            observation.game_loop + self.retreat_cooldown_game_loops
                        ),
                    )
                continue
            if (
                state is not None
                and observation.game_loop < state.cooldown_until_game_loop
            ):
                continue
            if state is None:
                state = _ActorRetreatState(
                    phase="retreating",
                    entered_game_loop=observation.game_loop,
                    last_command_game_loop=observation.game_loop,
                    cooldown_until_game_loop=(
                        observation.game_loop + self.retreat_cooldown_game_loops
                    ),
                )
                self._retreat_by_actor[actor] = state
            else:
                state.phase = "retreating"
                state.last_command_game_loop = observation.game_loop
                state.cooldown_until_game_loop = (
                    observation.game_loop + self.retreat_cooldown_game_loops
                )
            intents.append(
                self._intent(
                    observation,
                    assessment,
                    actor=actor,
                    action_name="Move_Minimap",
                    objective="Retreat this low-health combat group to the home defensive region",
                    target=IntentTarget(
                        kind=IntentTargetKind.RETREAT_REGION,
                        region="home",
                    ),
                    priority=85,
                    ttl_game_loops=8,
                )
            )
        return intents, retreating

    def _activate_episode(self, observation: ObservationEnvelope) -> None:
        episode_key = (observation.run_id, observation.episode_id)
        if episode_key == self._episode_key:
            return
        self._episode_key = episode_key
        self._last_focus_target = None
        self._last_reacquire_by_actor.clear()
        self._retreat_by_actor.clear()

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


def _units_for_actor(
    observation: ObservationEnvelope,
    actor: str,
) -> list[UnitState]:
    combat_units = [
        unit
        for unit in observation.state.own_units
        if unit.unit_type not in _WORKER_TYPES and unit.health_fraction > 0.0
    ]
    if "/" not in actor:
        return combat_units
    actor_token = actor.rsplit("/", 1)[-1]
    unit_type = actor_token.rsplit("-", 1)[0]
    if unit_type.casefold() in {"army", "combat", "all"}:
        return combat_units
    return [unit for unit in combat_units if unit.unit_type == unit_type]


def _units_at_home(
    units: list[UnitState],
    structures: list[UnitState],
    *,
    radius: float,
) -> bool:
    townhall_positions: list[tuple[float, float]] = []
    for structure in structures:
        if structure.unit_type in _TOWNHALL_TYPES and structure.position is not None:
            townhall_positions.append(structure.position)
    unit_positions: list[tuple[float, float]] = []
    for unit in units:
        if unit.position is not None:
            unit_positions.append(unit.position)
    if not townhall_positions or not unit_positions:
        return False
    return all(
        any(
            math.dist(unit_position, townhall_position) <= radius
            for townhall_position in townhall_positions
        )
        for unit_position in unit_positions
    )


def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()
