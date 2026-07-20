"""Seven single-owner role agents for the Strategic Cortex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rtscortex.contracts import ObservationEnvelope
from rtscortex.cortex.models import (
    CortexIntent,
    MacroIntent,
    ReflexIntent,
    SituationAssessment,
    TacticalIntent,
)
from rtscortex.cortex.strategic import RoleId, StrategicIntent, StrategicIntentAdapter
from rtscortex.races import ActionDomain, RaceProfile


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
        if isinstance(intent, MacroIntent):
            return self.profile.domain_for_action(intent.action_names[0]) is _domain(self.role_id)
        if isinstance(intent, ReflexIntent):
            if self.role_id is RoleId.RETREAT:
                return "retreat" in intent.objective.casefold()
            if intent.action_names[0] in {
                "Effect_InjectLarva",
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
    def __init__(self, profile: RaceProfile, adapter: StrategicIntentAdapter) -> None:
        super().__init__(RoleId.DEFENSE, profile, adapter)


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
        self.agents: tuple[_RoutingRoleAgent, ...] = (
            EconomyAgent(profile, adapter),
            TechnologyAgent(profile, adapter),
            ProductionAgent(profile, adapter),
            DefenseAgent(profile, adapter),
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
