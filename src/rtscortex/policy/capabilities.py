"""Static Runtime action capabilities used by shadow policy comparison."""

from __future__ import annotations

from dataclasses import dataclass

RTSCORTEX_MELEE_NATIVE_ACTIONS = frozenset(
    {
        "Ability_Blink_Screen",
        "Attack_Unit",
        "Build_Assimilator_Near",
        "Build_CyberneticsCore_Screen",
        "Build_Forge_Screen",
        "Build_Gateway_Screen",
        "Build_Nexus_Near",
        "Build_Pylon_Screen",
        "Build_ShieldBattery_Screen",
        "Build_Stargate_Screen",
        "Hold_Position",
        "Move_Minimap",
        "Move_Screen",
        "No_Operation",
        "Research_WarpGate",
        "Select_Unit_Blink_Screen",
        "Stop",
        "Train_Stalker",
        "Train_Zealot",
        "Train_Adept",
        "Train_Oracle",
        "Train_Phoenix",
        "Train_VoidRay",
        "Warp_Stalker_Near",
        "Warp_Zealot_Near",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityRegistry:
    """Global semantic actions implemented by one Runtime/Worker profile."""

    supported_actions: frozenset[str] = RTSCORTEX_MELEE_NATIVE_ACTIONS

    def is_globally_supported(self, action_name: str) -> bool:
        """Return whether the profile implements an action in any game state."""

        return action_name in self.supported_actions


DEFAULT_RUNTIME_CAPABILITIES = RuntimeCapabilityRegistry()
