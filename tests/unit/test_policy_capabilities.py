from __future__ import annotations

import ast
from pathlib import Path

from rtscortex.policy.capabilities import (
    DEFAULT_RUNTIME_CAPABILITIES,
    RTSCORTEX_MELEE_NATIVE_ACTIONS,
    RuntimeCapabilityRegistry,
)
from rtscortex.policy.hima.mapping import HIMA_RUNTIME_MAPPINGS


def test_default_registry_matches_the_melee_integration_action_surface() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "integrations"
        / "llm_pysc2"
        / "src"
        / "rtscortex_llm_pysc2"
        / "melee.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    action_names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_ACTION_NAMES" for target in node.targets
        )
    )
    integration_actions = frozenset(
        action_name for actor_actions in action_names.values() for action_name in actor_actions
    )

    assert RTSCORTEX_MELEE_NATIVE_ACTIONS == integration_actions
    assert DEFAULT_RUNTIME_CAPABILITIES.supported_actions == integration_actions


def test_registry_distinguishes_global_support_from_current_availability() -> None:
    registry = RuntimeCapabilityRegistry(
        supported_actions=frozenset({"Build_Pylon_Screen", "Train_Zealot"})
    )

    assert registry.is_globally_supported("Build_Pylon_Screen")
    assert not registry.is_globally_supported("Build_FleetBeacon_Screen")
    assert not registry.is_globally_supported("")


def test_every_hima_mapping_targets_a_live_runtime_capability() -> None:
    mapped_actions = {
        action_name for mapping in HIMA_RUNTIME_MAPPINGS for action_name in mapping.runtime_actions
    }

    assert mapped_actions <= RTSCORTEX_MELEE_NATIVE_ACTIONS
