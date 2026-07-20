"""Race-neutral capability contracts for the Strategic Cortex."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from rtscortex.progress import ProgressActionSpec


class RaceId(StrEnum):
    PROTOSS = "protoss"
    TERRAN = "terran"
    ZERG = "zerg"


class ActionDomain(StrEnum):
    ECONOMY = "economy"
    TECHNOLOGY = "technology"
    PRODUCTION = "production"
    DEFENSE = "defense"
    OFFENSE = "offense"
    FOCUS_FIRE = "focus_fire"
    RETREAT = "retreat"


@dataclass(frozen=True, slots=True)
class MacroActionMapping:
    semantic_action: str
    runtime_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RaceProfileData:
    race: RaceId
    worker_type: str
    supply_provider: str
    townhall_types: tuple[str, ...]
    gas_structure: str
    production_structures: tuple[str, ...]
    progress_action_specs: tuple[ProgressActionSpec, ...]
    macro_action_mappings: tuple[MacroActionMapping, ...]
    action_domains: Mapping[str, ActionDomain]
    action_producers: Mapping[str, tuple[str, ...]]
    hima_vocabulary_version: str
    macro_contract_ready: bool = True
    runtime_mapping_ready: bool = False
    live_worker_ready: bool = False
    effect_verification_kinds: tuple[str, ...] = ()
    controller_capabilities: tuple[str, ...] = ()
    controller_managed_actions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        action_names = [spec.name for spec in self.progress_action_specs]
        if len(action_names) != len(set(action_names)):
            raise ValueError(f"{self.race.value} progress action names must be unique")
        semantic_actions = [mapping.semantic_action for mapping in self.macro_action_mappings]
        if len(semantic_actions) != len(set(semantic_actions)):
            raise ValueError(f"{self.race.value} macro mappings must be unique")
        if len(self.controller_managed_actions) != len(set(self.controller_managed_actions)):
            raise ValueError(f"{self.race.value} controller-managed actions must be unique")
        mapped_runtime_actions = {
            action
            for mapping in self.macro_action_mappings
            for action in mapping.runtime_actions
        }
        unknown_managed_actions = set(self.controller_managed_actions) - mapped_runtime_actions
        if unknown_managed_actions:
            rendered = ", ".join(sorted(unknown_managed_actions))
            raise ValueError(
                f"{self.race.value} controller-managed actions lack macro mappings: {rendered}"
            )
        object.__setattr__(self, "action_domains", MappingProxyType(dict(self.action_domains)))
        object.__setattr__(
            self,
            "action_producers",
            MappingProxyType(dict(self.action_producers)),
        )

    def domain_for_action(self, action_name: str) -> ActionDomain | None:
        return self.action_domains.get(action_name)

    def producers_for_action(self, action_name: str) -> tuple[str, ...]:
        return self.action_producers.get(action_name, ())

    def capability_snapshot(self) -> dict[str, object]:
        return {
            "race": self.race.value,
            "hima_vocabulary_version": self.hima_vocabulary_version,
            "macro_contract_ready": self.macro_contract_ready,
            "runtime_mapping_ready": self.runtime_mapping_ready,
            "live_worker_ready": self.live_worker_ready,
            "effect_verification_kinds": list(self.effect_verification_kinds),
            "controller_capabilities": list(self.controller_capabilities),
            "controller_managed_actions": list(self.controller_managed_actions),
            "limitations": list(self.limitations),
        }


class RaceProfile(Protocol):
    @property
    def data(self) -> RaceProfileData: ...

    @property
    def race(self) -> RaceId: ...

    def domain_for_action(self, action_name: str) -> ActionDomain | None: ...

    def producers_for_action(self, action_name: str) -> tuple[str, ...]: ...
