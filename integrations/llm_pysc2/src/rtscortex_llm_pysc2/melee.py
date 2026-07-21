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
    "CombatGroup8",
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
        "Build_Forge_Screen",
        "Build_Assimilator_Near",
        "Build_CyberneticsCore_Screen",
        "Build_Nexus_Near",
        "Build_Stargate_Screen",
        "Build_ShieldBattery_Screen",
    },
    "Developer": {
        "No_Operation",
        "Train_Zealot",
        "Train_Stalker",
        "Train_Adept",
        "Train_VoidRay",
        "Train_Oracle",
        "Train_Phoenix",
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
    # Oracle and Phoenix have different target and ability semantics. The first
    # live slice keeps both upstream teams but exposes movement only, so neither
    # unit can receive an unsafe generic Attack or an unverified special ability.
    "CombatGroup8": {
        "No_Operation",
        "Move_Minimap",
        "Move_Screen",
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
        # Gas saturation uses upstream worker-management primitives. Patch 0013
        # keeps the dedicated Builder Probe reserved while ordinary workers are
        # stopped and reassigned deterministically by the Bridge.
        self.ENABLE_AUTO_WORKER_MANAGE = True
        self.ENABLE_AUTO_WORKER_TRAINING = True
        _ensure_assimilator_camera_settlement(self.AGENTS)
        _ensure_attack_target_reacquisition(self.AGENTS)
        _ensure_no_operation(self.AGENTS)


def _ensure_no_operation(agents: dict[str, dict[str, Any]]) -> None:
    no_operation = _find_no_operation(agents)
    for agent in agents.values():
        for actions in agent["action"].values():
            if not any(action["name"] == "No_Operation" for action in actions):
                actions.insert(0, deepcopy(no_operation))


def _ensure_assimilator_camera_settlement(agents: dict[str, dict[str, Any]]) -> None:
    """Move the camera to the exact geyser before resolving its screen position."""

    nexus_action = _find_action(agents, "Build_Nexus_Near")
    assimilator_action = _find_action(agents, "Build_Assimilator_Near")
    nexus_functions = list(nexus_action.get("func", ()))
    assimilator_functions = list(assimilator_action.get("func", ()))
    if (
        len(nexus_functions) < 2
        or int(nexus_functions[0][0]) != 573
        or int(nexus_functions[1][0]) != 0
        or not assimilator_functions
        or int(assimilator_functions[-1][0]) != 40
    ):
        raise RuntimeError("pinned Protoss Near-build action contract changed")
    assimilator_action["func"] = [
        deepcopy(nexus_functions[0]),
        deepcopy(nexus_functions[1]),
        assimilator_functions[-1],
    ]


def _ensure_attack_target_reacquisition(agents: dict[str, dict[str, Any]]) -> None:
    """Move to the exact enemy tag after selecting the combat control group."""

    nexus_functions = list(_find_action(agents, "Build_Nexus_Near").get("func", ()))
    if len(nexus_functions) < 2 or [int(item[0]) for item in nexus_functions[:2]] != [573, 0]:
        raise RuntimeError("pinned Protoss camera-settlement contract changed")
    for agent in agents.values():
        for actions in agent["action"].values():
            for action in actions:
                if action["name"] != "Attack_Unit":
                    continue
                functions = list(action.get("func", ()))
                if not functions or int(functions[-1][0]) != 12:
                    raise RuntimeError("pinned Protoss Attack_Unit contract changed")
                action["func"] = [
                    deepcopy(nexus_functions[0]),
                    deepcopy(nexus_functions[1]),
                    functions[-1],
                ]


def _find_action(
    agents: dict[str, dict[str, Any]],
    action_name: str,
) -> dict[str, Any]:
    for agent in agents.values():
        for actions in agent["action"].values():
            for action in actions:
                if action["name"] == action_name:
                    return cast(dict[str, Any], action)
    raise RuntimeError(f"ProtossAgentConfig does not expose {action_name}")


def _find_no_operation(agents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for agent in agents.values():
        for actions in agent["action"].values():
            for action in actions:
                if action["name"] == "No_Operation":
                    return cast(dict[str, Any], action)
    raise RuntimeError("ProtossAgentConfig does not expose a No_Operation action")
