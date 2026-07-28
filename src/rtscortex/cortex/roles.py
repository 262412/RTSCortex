"""Seven single-owner role agents for the Strategic Cortex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from rtscortex.contracts import ExecutionReport, ExecutionStatus, ObservationEnvelope
from rtscortex.cortex.models import (
    CortexIntent,
    IntentTarget,
    IntentTargetKind,
    MacroIntent,
    ReflexIntent,
    SituationAssessment,
    TacticalIntent,
    ThreatLevel,
)
from rtscortex.cortex.strategic import RoleId, StrategicIntent, StrategicIntentAdapter
from rtscortex.races import ActionDomain, RaceProfile
from rtscortex.targeting import attackable_enemies_for_actor

_DEFENSE_AGENT_ID = "deterministic-defense-agent"
_DEFENSE_COMMITMENT_GAME_LOOPS = 112
_DEFENSE_RETRY_COOLDOWN_GAME_LOOPS = 32


@dataclass(slots=True)
class _DefenseActorState:
    actor: str
    action_name: str
    signature: str
    phase: Literal["responding", "holding", "cooldown"]
    active_until_game_loop: int
    cooldown_until_game_loop: int = 0


@dataclass(frozen=True, slots=True)
class RoleAgentContext:
    observation: ObservationEnvelope
    situation: SituationAssessment
    source_intents: tuple[CortexIntent, ...]


class RoleAgent(Protocol):
    role_id: RoleId
    provider_mode: str

    def evaluate(self, context: RoleAgentContext) -> list[StrategicIntent]: ...


class _RoutingRoleAgent:
    provider_mode = "deterministic"
    agent_id = "routing-role-agent"
    agent_version = "2.0.0"

    def __init__(
        self,
        role_id: RoleId,
        profile: RaceProfile,
        adapter: StrategicIntentAdapter,
    ) -> None:
        self.role_id = role_id
        self.profile = profile
        self.adapter = adapter

    def evaluate(self, context: RoleAgentContext) -> list[StrategicIntent]:
        return [
            self.adapter.adapt(intent).model_copy(update={"role": self.role_id})
            for intent in context.source_intents
            if self.accepts(intent)
        ]

    def accepts(self, intent: CortexIntent) -> bool:
        if intent.source_id == _DEFENSE_AGENT_ID:
            return self.role_id is RoleId.DEFENSE
        tactical_owner = {
            OffenseAgent.agent_id: RoleId.OFFENSE,
            FocusFireAgent.agent_id: RoleId.FOCUS_FIRE,
            RetreatAgent.agent_id: RoleId.RETREAT,
        }.get(intent.source_id)
        if tactical_owner is not None:
            return self.role_id is tactical_owner
        if isinstance(intent, MacroIntent):
            return self.profile.domain_for_action(intent.action_names[0]) is _domain(self.role_id)
        if isinstance(intent, ReflexIntent):
            if self.role_id is RoleId.RETREAT:
                return "retreat" in intent.objective.casefold()
            if intent.action_names[0] in {
                "Effect_InjectLarva",
                "Train_Probe",
                "Train_SCV",
                "Morph_OrbitalCommand",
                "Effect_CalldownMULE_Screen",
            }:
                return self.role_id is RoleId.ECONOMY
            if intent.action_names[0] in {
                "Build_CreepTumor_Queen_Screen",
                "Build_CreepTumor_Tumor_Screen",
            }:
                return self.role_id is RoleId.DEFENSE
            return self.role_id is RoleId.DEFENSE
        if isinstance(intent, TacticalIntent):
            if "retreat" in intent.objective.casefold():
                return self.role_id is RoleId.RETREAT
            if intent.action_names[0] == "Attack_Unit":
                return self.role_id is RoleId.FOCUS_FIRE
            return self.role_id is RoleId.OFFENSE
        return False


class EconomyAgent(_RoutingRoleAgent):
    agent_id = "race-brain-economy-agent"

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.ECONOMY, profile, adapter)


class TechnologyAgent(_RoutingRoleAgent):
    agent_id = "race-brain-technology-agent"

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.TECHNOLOGY, profile, adapter)


class ProductionAgent(_RoutingRoleAgent):
    agent_id = "race-brain-production-agent"

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.PRODUCTION, profile, adapter)


class DefenseAgent(_RoutingRoleAgent):
    agent_id = _DEFENSE_AGENT_ID
    agent_version = "2.0.0"

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.DEFENSE, profile, adapter)
        self._episode_key: tuple[str, str] | None = None
        self._actor_states: dict[str, _DefenseActorState] = {}

    def evaluate(self, context: RoleAgentContext) -> list[StrategicIntent]:
        return [
            (
                intent.model_copy(
                    update={
                        "role": RoleId.DEFENSE,
                        "emergency": True,
                        "urgency": 1.0,
                        "expected_progress": 1.0,
                        "strategic_alignment": 1.0,
                        "risk": 0.05,
                        "horizon_game_loops": 16,
                        "mutually_exclusive_groups": ("defense-emergency-response",),
                    }
                )
                if intent.source_id == self.agent_id
                else intent
            )
            for intent in super().evaluate(context)
        ]

    def propose_source_intents(
        self,
        context: RoleAgentContext,
        *,
        claimed_actor_scopes: frozenset[str] = frozenset(),
    ) -> tuple[TacticalIntent, ...]:
        observation = context.observation
        episode_key = (observation.run_id, observation.episode_id)
        if episode_key != self._episode_key:
            self._episode_key = episode_key
            self._actor_states.clear()
        threatened = context.situation.threat_level in {
            ThreatLevel.HIGH,
            ThreatLevel.CRITICAL,
        }
        if not threatened:
            self._actor_states.clear()
            return ()

        proposals: list[TacticalIntent] = []
        claimed = set(claimed_actor_scopes)
        attack_actions = [
            action for action in observation.available_actions if action.name == "Attack_Unit"
        ]
        base_positions = [
            structure.position
            for structure in observation.state.own_structures
            if structure.position is not None
        ]
        for action in attack_actions:
            for actor in action.actor_scopes:
                if actor in claimed or not _is_combat_actor(actor):
                    continue
                targets = attackable_enemies_for_actor(observation, actor)
                if not targets:
                    continue
                target = min(
                    targets,
                    key=lambda enemy: (
                        _nearest_position_distance(enemy.position, base_positions),
                        enemy.health_fraction,
                        enemy.unit_id,
                    ),
                )
                signature = f"Attack_Unit:{target.unit_id}"
                if not self._may_emit(
                    actor,
                    signature,
                    action_name="Attack_Unit",
                    game_loop=observation.game_loop,
                ):
                    continue
                self._activate_actor(
                    actor,
                    signature,
                    action_name="Attack_Unit",
                    game_loop=observation.game_loop,
                )
                proposals.append(
                    self._source_intent(
                        context,
                        actor=actor,
                        action_name="Attack_Unit",
                        objective=f"Defend against {target.unit_type} near owned territory",
                        target=IntentTarget(
                            kind=IntentTargetKind.ENEMY,
                            unit_tag=target.unit_id,
                            unit_type=target.unit_type,
                        ),
                    )
                )
                claimed.add(actor)

        for action in observation.available_actions:
            if action.name != "Move_Minimap" or not action.argument_candidates:
                continue
            for actor in action.actor_scopes:
                if actor in claimed or not _is_combat_actor(actor):
                    continue
                position = _defensive_minimap_position(
                    observation,
                    action.argument_candidates,
                )
                if position is None:
                    continue
                signature = f"Move_Minimap:{position[0]},{position[1]}"
                if not self._may_emit(
                    actor,
                    signature,
                    action_name="Move_Minimap",
                    game_loop=observation.game_loop,
                ):
                    continue
                self._activate_actor(
                    actor,
                    signature,
                    action_name="Move_Minimap",
                    game_loop=observation.game_loop,
                )
                proposals.append(
                    self._source_intent(
                        context,
                        actor=actor,
                        action_name="Move_Minimap",
                        objective="Rally a combat group to the threatened base",
                        target=IntentTarget(
                            kind=IntentTargetKind.DEFENSIVE_REGION,
                            region="owned_base",
                            position=position,
                        ),
                    )
                )
                claimed.add(actor)

        proposals.extend(
            self._compile_emergency_strategy(
                context,
                claimed_actor_scopes=frozenset(claimed),
                combat_response_exists=any(
                    intent.action_names[0] in {"Attack_Unit", "Move_Minimap"}
                    and not _is_worker_actor(intent.actor_scopes[0], self.profile)
                    for intent in proposals
                ),
            )
        )
        return tuple(proposals)

    def _compile_emergency_strategy(
        self,
        context: RoleAgentContext,
        *,
        claimed_actor_scopes: frozenset[str],
        combat_response_exists: bool,
    ) -> list[TacticalIntent]:
        observation = context.observation
        doctrine = self.profile.data.defense_doctrine
        available = {action.name: action for action in observation.available_actions}
        air_threat = context.situation.visible_enemy_force.air_units > 0
        production = (
            doctrine.anti_air_production_actions
            if air_threat
            else doctrine.ground_production_actions
        )
        static_defense = list(doctrine.static_defense_actions)
        if air_threat:
            static_defense = list(
                dict.fromkeys((*doctrine.anti_air_defense_actions, *static_defense))
            )
        categories = (
            (
                production,
                "Produce an immediate anti-air response"
                if air_threat
                else "Produce an immediate defensive combat response",
                IntentTargetKind.PRODUCTION,
            ),
            (
                tuple(static_defense),
                "Fortify the threatened owned base",
                IntentTargetKind.DEFENSIVE_REGION,
            ),
        )
        proposals: list[TacticalIntent] = []
        immediate_action_available = False
        for action_names, objective, target_kind in categories:
            for action_name in action_names:
                action = available.get(action_name)
                if action is None:
                    continue
                actor = next(
                    (value for value in action.actor_scopes if value not in claimed_actor_scopes),
                    None,
                )
                if actor is None:
                    continue
                immediate_action_available = True
                proposals.append(
                    self._source_intent(
                        context,
                        actor=actor,
                        action_name=action_name,
                        objective=objective,
                        target=IntentTarget(
                            kind=target_kind,
                            region="owned_base",
                            structure_type=(
                                action_name.removeprefix("Build_").removesuffix("_Screen")
                                if action_name.startswith("Build_")
                                else None
                            ),
                        ),
                    )
                )

        if not immediate_action_available:
            for action_name in doctrine.prerequisite_actions:
                action = available.get(action_name)
                if action is None or not action.actor_scopes:
                    continue
                proposals.append(
                    self._source_intent(
                        context,
                        actor=action.actor_scopes[0],
                        action_name=action_name,
                        objective=(
                            "Emergency prerequisite closure for anti-air defense"
                            if air_threat
                            else "Emergency prerequisite closure for base defense"
                        ),
                        target=IntentTarget(
                            kind=IntentTargetKind.DEFENSIVE_REGION,
                            region="owned_base",
                        ),
                    )
                )
                break

        if context.situation.threat_level is ThreatLevel.CRITICAL and not combat_response_exists:
            for action_name in doctrine.worker_defense_actions:
                action = available.get(action_name)
                if action is None:
                    continue
                for actor in action.actor_scopes:
                    if not _is_worker_actor(actor, self.profile):
                        continue
                    candidate_tags = {
                        str(arguments[0]).casefold()
                        for arguments in action.argument_candidates or ()
                        if arguments
                    }
                    targets = [
                        enemy
                        for enemy in observation.state.visible_enemies
                        if enemy.health_fraction > 0
                        and (
                            enemy.unit_id.casefold() in candidate_tags
                            or _normalized_tag(enemy.unit_id) in candidate_tags
                        )
                    ]
                    if not targets:
                        continue
                    target = min(
                        targets,
                        key=lambda enemy: (
                            _nearest_position_distance(
                                enemy.position,
                                _owned_base_positions(observation),
                            ),
                            enemy.health_fraction,
                            enemy.unit_id,
                        ),
                    )
                    proposals.append(
                        self._source_intent(
                            context,
                            actor=actor,
                            action_name=action_name,
                            objective="Last-resort worker defense at the threatened mineral line",
                            target=IntentTarget(
                                kind=IntentTargetKind.ENEMY,
                                unit_tag=target.unit_id,
                                unit_type=target.unit_type,
                            ),
                        )
                    )
                    return proposals
        return proposals

    def record_execution(
        self,
        report: ExecutionReport,
        *,
        game_loop: int,
    ) -> dict[str, object] | None:
        actor = report.actor
        if actor is None:
            return None
        state_key = next(
            (
                key
                for key, value in self._actor_states.items()
                if value.actor == actor and value.action_name == report.action_name
            ),
            None,
        )
        state = None if state_key is None else self._actor_states[state_key]
        if state is None:
            return None
        if report.status is ExecutionStatus.SUCCEEDED:
            state.phase = "holding"
            state.active_until_game_loop = max(
                state.active_until_game_loop,
                game_loop + _DEFENSE_COMMITMENT_GAME_LOOPS,
            )
            return {
                "actor": actor,
                "state": "defense_response_holding",
                "status": "succeeded",
                "game_loop": game_loop,
                "active_until_game_loop": state.active_until_game_loop,
            }
        if report.status is not ExecutionStatus.FAILED:
            return None
        state.active_until_game_loop = game_loop
        state.cooldown_until_game_loop = game_loop + _DEFENSE_RETRY_COOLDOWN_GAME_LOOPS
        state.phase = "cooldown"
        return {
            "actor": actor,
            "state": "defense_response_cooldown",
            "status": "failed",
            "failure_code": report.failure_code,
            "cooldown_until_game_loop": state.cooldown_until_game_loop,
        }

    def _may_emit(
        self,
        actor: str,
        signature: str,
        *,
        action_name: str,
        game_loop: int,
    ) -> bool:
        state = self._actor_states.get(f"{actor}|{action_name}|{signature}")
        if state is None:
            return True
        if state.phase == "holding":
            return False
        if game_loop < state.cooldown_until_game_loop:
            return False
        if game_loop < state.active_until_game_loop:
            may_replace_rally = state.signature.startswith(
                "Move_Minimap:"
            ) and signature.startswith("Attack_Unit:")
            if not may_replace_rally:
                return False
        return True

    def _activate_actor(
        self,
        actor: str,
        signature: str,
        *,
        action_name: str,
        game_loop: int,
    ) -> None:
        self._actor_states[f"{actor}|{action_name}|{signature}"] = _DefenseActorState(
            actor=actor,
            action_name=action_name,
            signature=signature,
            phase="responding",
            active_until_game_loop=(game_loop + _DEFENSE_COMMITMENT_GAME_LOOPS),
        )

    def _source_intent(
        self,
        context: RoleAgentContext,
        *,
        actor: str,
        action_name: str,
        objective: str,
        target: IntentTarget,
    ) -> TacticalIntent:
        observation = context.observation
        identity = (
            f"{observation.run_id}:{observation.episode_id}:"
            f"{observation.step_id}:{actor}:{action_name}:defense"
        )
        return TacticalIntent(
            intent_id=f"defense:{identity}",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective=objective,
            action_names=[action_name],
            actor_scopes=[actor],
            target=target,
            priority=95,
            ttl_game_loops=8,
            source_id=self.agent_id,
            source_version=self.agent_version,
            situation_assessment_id=context.situation.assessment_id,
        )


class OffenseAgent(_RoutingRoleAgent):
    agent_id = "deterministic-offense-agent"

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.OFFENSE, profile, adapter)

    def owns(self, intent: TacticalIntent) -> bool:
        return (
            "retreat" not in intent.objective.casefold() and intent.action_names[0] != "Attack_Unit"
        )


class FocusFireAgent(_RoutingRoleAgent):
    agent_id = "deterministic-focus-fire-agent"

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.FOCUS_FIRE, profile, adapter)

    def owns(self, intent: TacticalIntent) -> bool:
        return (
            "retreat" not in intent.objective.casefold() and intent.action_names[0] == "Attack_Unit"
        )


class RetreatAgent(_RoutingRoleAgent):
    agent_id = "deterministic-retreat-agent"

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.RETREAT, profile, adapter)

    def owns(self, intent: TacticalIntent) -> bool:
        return "retreat" in intent.objective.casefold()


class RoleAgentCoordinator:
    """Route every source intent to exactly one responsibility owner."""

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        self.defense_agent = DefenseAgent(profile, adapter)
        self.offense_agent = OffenseAgent(profile, adapter)
        self.focus_fire_agent = FocusFireAgent(profile, adapter)
        self.retreat_agent = RetreatAgent(profile, adapter)
        self.agents: tuple[RoleAgent, ...] = (
            EconomyAgent(profile, adapter),
            TechnologyAgent(profile, adapter),
            ProductionAgent(profile, adapter),
            self.defense_agent,
            self.offense_agent,
            self.focus_fire_agent,
            self.retreat_agent,
        )

    def evaluate(self, context: RoleAgentContext) -> dict[str, StrategicIntent]:
        routed: dict[str, StrategicIntent] = {}
        for agent in self.agents:
            for intent in agent.evaluate(context):
                source_id = _source_intent_id(intent)
                if source_id in routed:
                    raise RuntimeError(f"source intent has multiple role owners: {source_id}")
                routed[source_id] = intent
        expected = {intent.intent_id for intent in context.source_intents}
        if set(routed) != expected:
            missing = sorted(expected - set(routed))
            raise RuntimeError(f"source intents have no role owner: {missing}")
        return routed

    def propose_defense_intents(
        self,
        context: RoleAgentContext,
        *,
        claimed_actor_scopes: frozenset[str] = frozenset(),
    ) -> tuple[TacticalIntent, ...]:
        return self.defense_agent.propose_source_intents(
            context,
            claimed_actor_scopes=claimed_actor_scopes,
        )

    def own_tactical_intents(
        self,
        intents: tuple[TacticalIntent, ...],
    ) -> tuple[TacticalIntent, ...]:
        """Give every tactical proposal one concrete agent owner and lineage."""

        owners = (
            self.offense_agent,
            self.focus_fire_agent,
            self.retreat_agent,
        )
        result: list[TacticalIntent] = []
        for intent in intents:
            matched = [agent for agent in owners if agent.owns(intent)]
            if len(matched) != 1:
                raise RuntimeError(
                    f"tactical intent requires exactly one role agent owner: {intent.intent_id}"
                )
            owner = matched[0]
            result.append(
                intent.model_copy(
                    update={
                        "source_id": owner.agent_id,
                        "source_version": owner.agent_version,
                    }
                )
            )
        return tuple(result)

    def record_execution(
        self,
        report: ExecutionReport,
        *,
        responsibility: str | None,
        game_loop: int,
    ) -> dict[str, object] | None:
        if responsibility != RoleId.DEFENSE.value:
            return None
        return self.defense_agent.record_execution(report, game_loop=game_loop)


def _source_intent_id(intent: StrategicIntent) -> str:
    source = intent.source_intent_id
    if source is None:
        raise RuntimeError("strategic intent lost its source intent ID")
    return source


def _domain(role: RoleId) -> ActionDomain | None:
    if role.value in {
        ActionDomain.ECONOMY.value,
        ActionDomain.TECHNOLOGY.value,
        ActionDomain.PRODUCTION.value,
        ActionDomain.DEFENSE.value,
    }:
        return ActionDomain(role.value)
    return None


def _nearest_position_distance(
    position: tuple[float, float] | None,
    targets: list[tuple[float, float]],
) -> float:
    if position is None or not targets:
        return float("inf")
    return min(
        (position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2 for target in targets
    )


def _owned_base_positions(observation: ObservationEnvelope) -> list[tuple[float, float]]:
    return [
        structure.position
        for structure in observation.state.own_structures
        if structure.position is not None
    ]


def _is_worker_actor(actor: str, profile: RaceProfile) -> bool:
    worker = profile.data.worker_type.casefold()
    normalized = actor.casefold()
    return normalized.startswith("builder/") or worker in normalized


def _is_combat_actor(actor: str) -> bool:
    return actor.casefold().startswith("combatgroup")


def _normalized_tag(value: str) -> str:
    try:
        return hex(int(value, 0))
    except ValueError:
        return value.casefold()


def _defensive_minimap_position(
    observation: ObservationEnvelope,
    argument_candidates: list[list[object]],
) -> tuple[int, int] | None:
    available = [
        (int(arguments[0][0]), int(arguments[0][1]))
        for arguments in argument_candidates
        if (arguments and isinstance(arguments[0], (list, tuple)) and len(arguments[0]) == 2)
    ]
    if not available:
        return None
    base_positions = [
        structure.minimap_position
        for structure in observation.state.own_structures
        if structure.minimap_position is not None
    ]
    if not base_positions:
        return available[-1]
    return min(
        available,
        key=lambda candidate: min(
            (candidate[0] - base[0]) ** 2 + (candidate[1] - base[1]) ** 2 for base in base_positions
        ),
    )
