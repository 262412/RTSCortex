"""Deterministic short-horizon combat policy for the SC2-native Cortex."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Literal

from rtscortex.contracts import ExecutionReport, ExecutionStatus, ObservationEnvelope, UnitState
from rtscortex.cortex.models import (
    ArmyReadiness,
    IntentTarget,
    IntentTargetKind,
    SituationAssessment,
    TacticalIntent,
    ThreatLevel,
)
from rtscortex.targeting import (
    ENEMY_STRUCTURE_TYPES,
    current_screen_enemy_targets,
    last_known_enemy_targets,
    living_targetable_enemies,
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
_TOWNHALL_TYPES = frozenset(
    {"CommandCenter", "Hatchery", "Hive", "Lair", "Nexus", "OrbitalCommand"}
)


@dataclass(slots=True)
class _ActorRetreatState:
    phase: Literal["retreating", "arrived"]
    entered_game_loop: int
    last_command_game_loop: int
    cooldown_until_game_loop: int


@dataclass(slots=True)
class _ActorOffenseState:
    phase: Literal["advancing", "arrived", "searching", "engaged"]
    entered_game_loop: int
    last_command_game_loop: int
    cooldown_until_game_loop: int
    waypoint: tuple[int, int] | None = None
    waypoint_index: int = -1
    target_tag: str | None = None
    best_distance: float | None = None
    last_progress_game_loop: int = 0
    obsolete_waypoints: dict[tuple[int, int], int] = field(default_factory=dict)


@dataclass(slots=True)
class _TargetFailureState:
    failure_count: int
    quarantined_until_game_loop: int
    failure_code: str


class DeterministicTacticalAgent:
    """Turn a current situation into exact, candidate-bound combat intents."""

    agent_id = "deterministic-tactical-agent"
    agent_version = "0.4.0"
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
        offense_arrival_radius: float = 4.0,
        offense_stall_game_loops: int = 112,
        offense_waypoint_retry_game_loops: int = 336,
        target_retry_limit: int = 2,
        target_quarantine_game_loops: int = 112,
    ) -> None:
        if target_retry_limit < 1:
            raise ValueError("target_retry_limit must be positive")
        if target_quarantine_game_loops < 1:
            raise ValueError("target_quarantine_game_loops must be positive")
        self.retreat_health_threshold = retreat_health_threshold
        self.minimum_advance_army_supply = minimum_advance_army_supply
        self.reacquire_cooldown_game_loops = reacquire_cooldown_game_loops
        self.retreat_cooldown_game_loops = retreat_cooldown_game_loops
        self.retreat_exit_health_threshold = min(
            1.0,
            retreat_health_threshold + retreat_hysteresis,
        )
        self.retreat_home_radius = retreat_home_radius
        self.offense_arrival_radius = offense_arrival_radius
        self.offense_stall_game_loops = offense_stall_game_loops
        self.offense_waypoint_retry_game_loops = offense_waypoint_retry_game_loops
        self.target_retry_limit = target_retry_limit
        self.target_quarantine_game_loops = target_quarantine_game_loops
        self._episode_key: tuple[str, str] | None = None
        self._focus_target_by_actor: dict[str, str] = {}
        self._target_failures: dict[tuple[str, str], _TargetFailureState] = {}
        self._retreat_by_actor: dict[str, _ActorRetreatState] = {}
        self._offense_by_actor: dict[str, _ActorOffenseState] = {}
        self._known_enemy_structures: dict[str, tuple[float, float]] = {}

    def evaluate(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
    ) -> list[TacticalIntent]:
        self._activate_episode(observation)
        self._remember_enemy_structures(observation)
        attack_actors = _actors_for(observation, "Attack_Unit")
        move_actors = _actors_for(observation, "Move_Minimap")
        enemies = living_targetable_enemies(observation.state.visible_enemies)
        current_targets = current_screen_enemy_targets(observation)
        last_known_targets = last_known_enemy_targets(observation)
        retreat_intents, retreating_actors = self._retreat_intents(
            observation,
            assessment,
            enemies,
            move_actors,
        )
        attack_actors = [actor for actor in attack_actors if actor not in retreating_actors]
        move_actors = [actor for actor in move_actors if actor not in retreating_actors]
        attack_intents: list[TacticalIntent] = []
        engaged_actors: set[str] = set()
        if current_targets and attack_actors:
            for actor in attack_actors:
                actor_targets = self._attack_targets_for_actor(
                    observation,
                    actor,
                    current_targets,
                )
                if not actor_targets:
                    continue
                target, reacquired = self._focus_target(actor, actor_targets)
                target_tag = _normalize_tag(target.unit_id)
                previous = self._offense_by_actor.get(actor)
                self._offense_by_actor[actor] = _ActorOffenseState(
                    phase="engaged",
                    entered_game_loop=(
                        observation.game_loop
                        if previous is None
                        else previous.entered_game_loop
                    ),
                    last_command_game_loop=observation.game_loop,
                    cooldown_until_game_loop=observation.game_loop,
                    target_tag=target_tag,
                )
                target_kind = (
                    "enemy structure"
                    if target.unit_type in ENEMY_STRUCTURE_TYPES
                    else "enemy unit"
                )
                objective = (
                    f"Reacquire and focus fire {target_kind} {target.unit_type}"
                    if reacquired
                    else f"Focus fire current-screen {target_kind} {target.unit_type}"
                )
                attack_intents.append(
                    self._intent(
                        observation,
                        assessment,
                        actor=actor,
                        action_name="Attack_Unit",
                        objective=objective,
                        target=IntentTarget(
                            kind=IntentTargetKind.ENEMY,
                            unit_tag=target_tag,
                            unit_type=target.unit_type,
                        ),
                        priority=75,
                        ttl_game_loops=8,
                    )
                )
                engaged_actors.add(actor)

        ready_to_advance = (
            assessment.army_readiness is ArmyReadiness.READY
            or observation.state.economy.army_supply >= self.minimum_advance_army_supply
        )
        searching_actors = [actor for actor in move_actors if actor not in engaged_actors]
        if ready_to_advance and searching_actors:
            intents = self._offense_search_intents(
                observation,
                assessment,
                searching_actors,
                last_known_targets=last_known_targets,
            )
            return [*retreat_intents, *attack_intents, *intents]
        return [*retreat_intents, *attack_intents]

    def record_execution(
        self,
        report: ExecutionReport,
        *,
        game_loop: int,
    ) -> dict[str, object] | None:
        if report.action_name != "Attack_Unit" or report.actor is None:
            return None
        target = next(
            (
                _normalize_tag(value)
                for value in report.resolved_arguments or report.requested_arguments
                if isinstance(value, (int, str))
            ),
            None,
        )
        if target is None:
            return None
        key = (report.actor, target)
        if report.status is ExecutionStatus.SUCCEEDED:
            self._target_failures.pop(key, None)
            return {
                "actor": report.actor,
                "target_tag": target,
                "state": "confirmed",
                "failure_count": 0,
            }
        failure_code = report.failure_code or "unknown_failure"
        immediate = failure_code in {
            "combat_target_lost",
            "target_not_visible",
            "friendly_target",
        }
        previous = self._target_failures.get(key)
        failure_count = 1 if previous is None else previous.failure_count + 1
        quarantined = immediate or failure_count >= self.target_retry_limit
        until = (
            game_loop + self.target_quarantine_game_loops
            if quarantined
            else game_loop
        )
        self._target_failures[key] = _TargetFailureState(
            failure_count=failure_count,
            quarantined_until_game_loop=until,
            failure_code=failure_code,
        )
        if quarantined:
            self._focus_target_by_actor.pop(report.actor, None)
        return {
            "actor": report.actor,
            "target_tag": target,
            "state": "quarantined" if quarantined else "retryable",
            "failure_count": failure_count,
            "failure_code": failure_code,
            "quarantined_until_game_loop": until if quarantined else None,
        }

    def _offense_search_intents(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
        actors: list[str],
        *,
        last_known_targets: list[UnitState],
    ) -> list[TacticalIntent]:
        available_actors = set(actors)
        for actor in tuple(self._offense_by_actor):
            if actor not in available_actors:
                del self._offense_by_actor[actor]

        intents: list[TacticalIntent] = []
        for actor in actors:
            candidates = _minimap_candidates_for_actor(observation, actor)
            if not candidates:
                self._offense_by_actor.pop(actor, None)
                continue
            state = self._offense_by_actor.get(actor)
            centroid = _actor_minimap_centroid(observation, actor)
            if state is not None:
                state.obsolete_waypoints = {
                    waypoint: retry_after
                    for waypoint, retry_after in state.obsolete_waypoints.items()
                    if retry_after > observation.game_loop
                }
            if state is not None and state.waypoint not in candidates:
                if state.waypoint is not None:
                    state.obsolete_waypoints[state.waypoint] = (
                        observation.game_loop + self.offense_waypoint_retry_game_loops
                    )
                state.phase = "arrived"
                state.waypoint = None
            if state is not None and state.waypoint is not None:
                should_switch = False
                if centroid is not None:
                    distance = math.dist(centroid, state.waypoint)
                    if distance <= self.offense_arrival_radius:
                        state.phase = "arrived"
                        should_switch = True
                        self._forget_searched_enemy_structures(state.waypoint)
                    elif (
                        state.best_distance is None
                        or distance < state.best_distance - 0.5
                    ):
                        state.best_distance = distance
                        state.last_progress_game_loop = observation.game_loop
                    elif (
                        observation.game_loop - state.last_progress_game_loop
                        >= self.offense_stall_game_loops
                    ):
                        should_switch = True
                elif observation.game_loop >= state.cooldown_until_game_loop:
                    should_switch = True
                if should_switch:
                    state.obsolete_waypoints[state.waypoint] = (
                        observation.game_loop + self.offense_waypoint_retry_game_loops
                    )
                    state.waypoint = None
                    state.best_distance = None
                else:
                    continue
            if (
                state is not None
                and observation.game_loop < state.cooldown_until_game_loop
            ):
                continue

            available = [
                candidate
                for candidate in candidates
                if state is None or candidate not in state.obsolete_waypoints
            ]
            if not available:
                continue
            waypoint = self._select_offense_waypoint(
                available,
            )
            next_index = candidates.index(waypoint)
            waypoint_distance = (
                None if centroid is None else math.dist(centroid, waypoint)
            )
            if state is None:
                state = _ActorOffenseState(
                    phase=(
                        "searching"
                        if last_known_targets or self._known_enemy_structures
                        else "advancing"
                    ),
                    entered_game_loop=observation.game_loop,
                    last_command_game_loop=observation.game_loop,
                    cooldown_until_game_loop=(
                        observation.game_loop + self.reacquire_cooldown_game_loops
                    ),
                    waypoint=waypoint,
                    waypoint_index=next_index,
                    best_distance=waypoint_distance,
                    last_progress_game_loop=observation.game_loop,
                )
                self._offense_by_actor[actor] = state
            else:
                state.phase = (
                    "searching"
                    if last_known_targets or self._known_enemy_structures
                    else "advancing"
                )
                state.last_command_game_loop = observation.game_loop
                state.cooldown_until_game_loop = (
                    observation.game_loop + self.reacquire_cooldown_game_loops
                )
                state.waypoint = waypoint
                state.waypoint_index = next_index
                state.target_tag = None
                state.best_distance = waypoint_distance
                state.last_progress_game_loop = observation.game_loop

            objective = (
                "Search the last-known enemy structure location and reacquire targets"
                if self._known_enemy_structures
                else "Reacquire the last-known enemy and search for surviving structures"
                if last_known_targets
                else "Search unexplored map sectors for enemy units and structures"
            )
            intents.append(
                self._intent(
                    observation,
                    assessment,
                    actor=actor,
                    action_name="Move_Minimap",
                    objective=objective,
                    target=IntentTarget(
                        kind=IntentTargetKind.ENEMY,
                        region="reacquire" if last_known_targets else "unexplored",
                        position=waypoint,
                    ),
                    priority=60,
                    ttl_game_loops=16,
                )
            )
        return intents

    def _select_offense_waypoint(
        self,
        candidates: list[tuple[int, int]],
    ) -> tuple[int, int]:
        structure_positions = tuple(self._known_enemy_structures.values())
        if structure_positions:
            return min(
                candidates,
                key=lambda candidate: (
                    min(math.dist(candidate, position) for position in structure_positions),
                    candidate,
                ),
            )
        return candidates[0]

    def _remember_enemy_structures(self, observation: ObservationEnvelope) -> None:
        for enemy in observation.state.visible_enemies:
            tag = _normalize_tag(enemy.unit_id)
            if enemy.unit_type not in ENEMY_STRUCTURE_TYPES:
                continue
            if enemy.health_fraction <= 0.0:
                self._known_enemy_structures.pop(tag, None)
            elif enemy.minimap_position is not None:
                self._known_enemy_structures[tag] = enemy.minimap_position

    def _forget_searched_enemy_structures(
        self,
        waypoint: tuple[int, int],
    ) -> None:
        self._known_enemy_structures = {
            tag: position
            for tag, position in self._known_enemy_structures.items()
            if math.dist(position, waypoint) > self.offense_arrival_radius * 2.0
        }

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
        self._focus_target_by_actor.clear()
        self._target_failures.clear()
        self._retreat_by_actor.clear()
        self._offense_by_actor.clear()
        self._known_enemy_structures.clear()

    def _focus_target(
        self,
        actor: str,
        enemies: list[UnitState],
    ) -> tuple[UnitState, bool]:
        by_tag = {_normalize_tag(enemy.unit_id): enemy for enemy in enemies}
        previous = self._focus_target_by_actor.get(actor)
        if previous is not None and previous in by_tag:
            return by_tag[previous], False
        target = min(enemies, key=_target_rank)
        self._focus_target_by_actor[actor] = _normalize_tag(target.unit_id)
        return target, previous is not None

    def _attack_targets_for_actor(
        self,
        observation: ObservationEnvelope,
        actor: str,
        targets: list[UnitState],
    ) -> list[UnitState]:
        candidate_tags = {
            _normalize_tag(arguments[0])
            for action in observation.available_actions
            if action.name == "Attack_Unit" and actor in action.actor_scopes
            for arguments in action.argument_candidates or ()
            if arguments and isinstance(arguments[0], (int, str))
        }
        targets_by_tag = {_normalize_tag(target.unit_id): target for target in targets}
        eligible: list[UnitState] = []
        for tag in sorted(candidate_tags):
            target = targets_by_tag.get(tag)
            if target is None:
                continue
            failure = self._target_failures.get((actor, tag))
            if (
                failure is not None
                and failure.quarantined_until_game_loop > observation.game_loop
            ):
                continue
            if failure is not None:
                self._target_failures.pop((actor, tag), None)
            eligible.append(target)
        return eligible

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
                target.unit_tag
                or target.unit_type
                or (
                    ",".join(str(value) for value in target.position)
                    if target.position is not None
                    else target.region
                )
                or "none",
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
    elif enemy.unit_type in _TOWNHALL_TYPES:
        class_rank = 3
    elif enemy.unit_type in ENEMY_STRUCTURE_TYPES:
        class_rank = 4
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


def _actor_minimap_centroid(
    observation: ObservationEnvelope,
    actor: str,
) -> tuple[float, float] | None:
    positions = [
        unit.minimap_position
        for unit in _units_for_actor(observation, actor)
        if unit.minimap_position is not None
    ]
    if not positions:
        return None
    return (
        sum(position[0] for position in positions) / len(positions),
        sum(position[1] for position in positions) / len(positions),
    )


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


def _minimap_candidates_for_actor(
    observation: ObservationEnvelope,
    actor: str,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for action in observation.available_actions:
        if action.name != "Move_Minimap" or actor not in action.actor_scopes:
            continue
        for arguments in action.argument_candidates or ():
            if (
                not arguments
                or not isinstance(arguments[0], (list, tuple))
                or len(arguments[0]) != 2
            ):
                continue
            candidates.append((int(arguments[0][0]), int(arguments[0][1])))
    # The extractor reserves the final candidate for the home retreat region.
    return list(dict.fromkeys(candidates[:-1] if len(candidates) > 1 else candidates))
