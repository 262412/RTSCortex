"""Seven single-owner role agents for the Strategic Cortex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rtscortex.contracts import ObservationEnvelope
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
    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.ECONOMY, profile, adapter)


class TechnologyAgent(_RoutingRoleAgent):
    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.TECHNOLOGY, profile, adapter)


class ProductionAgent(_RoutingRoleAgent):
    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.PRODUCTION, profile, adapter)


class DefenseAgent(_RoutingRoleAgent):
    agent_id = _DEFENSE_AGENT_ID
    agent_version = "1.0.0"

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.DEFENSE, profile, adapter)
        self._episode_key: tuple[str, str] | None = None
        self._committed_until_game_loop = 0

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
            self._committed_until_game_loop = 0
        threatened = context.situation.threat_level in {
            ThreatLevel.HIGH,
            ThreatLevel.CRITICAL,
        }
        if not threatened and observation.game_loop >= self._committed_until_game_loop:
            return ()

        attack_actions = [
            action
            for action in observation.available_actions
            if action.name == "Attack_Unit"
        ]
        base_positions = [
            structure.position
            for structure in observation.state.own_structures
            if structure.position is not None
        ]
        for action in attack_actions:
            for actor in action.actor_scopes:
                if actor in claimed_actor_scopes:
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
                self._committed_until_game_loop = observation.game_loop + 16
                return (
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
                    ),
                )

        for action in observation.available_actions:
            if action.name != "Move_Minimap" or not action.argument_candidates:
                continue
            for actor in action.actor_scopes:
                if actor in claimed_actor_scopes:
                    continue
                position = _defensive_minimap_position(
                    observation,
                    action.argument_candidates,
                )
                if position is None:
                    continue
                self._committed_until_game_loop = observation.game_loop + 16
                return (
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
                    ),
                )
        return ()

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
    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.OFFENSE, profile, adapter)


class FocusFireAgent(_RoutingRoleAgent):
    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.FOCUS_FIRE, profile, adapter)


class RetreatAgent(_RoutingRoleAgent):
    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.RETREAT, profile, adapter)


class RoleAgentCoordinator:
    """Route every source intent to exactly one responsibility owner."""

    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        self.defense_agent = DefenseAgent(profile, adapter)
        self.agents: tuple[_RoutingRoleAgent, ...] = (
            EconomyAgent(profile, adapter),
            TechnologyAgent(profile, adapter),
            ProductionAgent(profile, adapter),
            self.defense_agent,
            OffenseAgent(profile, adapter),
            FocusFireAgent(profile, adapter),
            RetreatAgent(profile, adapter),
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
        (position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2
        for target in targets
    )


def _defensive_minimap_position(
    observation: ObservationEnvelope,
    argument_candidates: list[list[object]],
) -> tuple[int, int] | None:
    available = [
        (int(arguments[0][0]), int(arguments[0][1]))
        for arguments in argument_candidates
        if (
            arguments
            and isinstance(arguments[0], (list, tuple))
            and len(arguments[0]) == 2
        )
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
            (candidate[0] - base[0]) ** 2 + (candidate[1] - base[1]) ** 2
            for base in base_positions
        ),
    )
