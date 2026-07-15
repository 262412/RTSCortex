"""RTSCortex-owned LLM-PySC2 configuration for a minimal Protoss melee game."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from llm_pysc2.agents.configs.config import (  # type: ignore[import-not-found]
    ProtossAgentConfig,
)

_MELEE_AGENTS = (
    "Builder",
    "Developer",
    "CombatGroup0",
    "CombatGroup1",
    "CombatGroup3",
    "CombatGroup7",
)
_SINGLE_TEAM_AGENTS = (
    "CombatGroup0",
    "CombatGroup1",
    "CombatGroup3",
    "CombatGroup7",
)
_ACTION_NAMES = {
    "Builder": {
        "No_Operation",
        "Stop",
        "Hold_Position",
        "Move_Minimap",
        "Move_Screen",
        "Build_Pylon_Screen",
        "Build_Gateway_Screen",
        "Build_Assimilator_Near",
        "Build_CyberneticsCore_Screen",
        "Build_Nexus_Near",
        "Build_Stargate_Screen",
    },
    "Developer": {
        "No_Operation",
        "Train_Zealot",
        "Train_Stalker",
        "Train_Adept",
        "Train_VoidRay",
        "Research_WarpGate",
        "Warp_Zealot_Near",
        "Warp_Stalker_Near",
    },
    "CombatGroup0": {
        "No_Operation",
        "Stop",
        "Hold_Position",
        "Move_Minimap",
        "Move_Screen",
        "Attack_Unit",
    },
    "CombatGroup1": {
        "No_Operation",
        "Stop",
        "Hold_Position",
        "Move_Minimap",
        "Move_Screen",
        "Attack_Unit",
        "Ability_Blink_Screen",
        "Select_Unit_Blink_Screen",
    },
    "CombatGroup3": {
        "No_Operation",
        "Stop",
        "Hold_Position",
        "Move_Minimap",
        "Move_Screen",
        "Attack_Unit",
    },
    "CombatGroup7": {
        "No_Operation",
        "Stop",
        "Hold_Position",
        "Move_Minimap",
        "Move_Screen",
        "Attack_Unit",
    },
}


class RTSCortexMeleeConfig(ProtossAgentConfig):  # type: ignore[misc]
    """Retain the supported Protoss macro chain and its combat groups."""

    AGENTS: dict[str, dict[str, Any]]
    AGENTS_ALWAYS_DISABLE: list[str]
    ENABLE_INIT_STEPS: bool
    ENABLE_AUTO_WORKER_MANAGE: bool
    ENABLE_AUTO_WORKER_TRAINING: bool

    def __init__(self) -> None:
        super().__init__()

        upstream_agents: dict[str, dict[str, Any]] = self.AGENTS
        self.AGENTS = {
            agent_name: deepcopy(upstream_agents[agent_name]) for agent_name in _MELEE_AGENTS
        }
        for agent_name in _SINGLE_TEAM_AGENTS:
            self.AGENTS[agent_name]["team"] = self.AGENTS[agent_name]["team"][:1]
        for agent_name, agent in self.AGENTS.items():
            allowed = _ACTION_NAMES[agent_name]
            for unit_type, actions in agent["action"].items():
                agent["action"][unit_type] = [
                    action for action in actions if action["name"] in allowed
                ]

        self.AGENTS_ALWAYS_DISABLE = []
        self.ENABLE_INIT_STEPS = True
        # ``select_idle_worker`` cannot exclude the dedicated Builder probe. Keep
        # automatic training, but reserve that actor from economy reassignment.
        self.ENABLE_AUTO_WORKER_MANAGE = False
        self.ENABLE_AUTO_WORKER_TRAINING = True
        _ensure_no_operation(self.AGENTS)


def _ensure_no_operation(agents: dict[str, dict[str, Any]]) -> None:
    no_operation = _find_no_operation(agents)
    for agent in agents.values():
        for actions in agent["action"].values():
            if not any(action["name"] == "No_Operation" for action in actions):
                actions.insert(0, deepcopy(no_operation))


def _find_no_operation(agents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for agent in agents.values():
        for actions in agent["action"].values():
            for action in actions:
                if action["name"] == "No_Operation":
                    return cast(dict[str, Any], action)
    raise RuntimeError("ProtossAgentConfig does not expose a No_Operation action")
