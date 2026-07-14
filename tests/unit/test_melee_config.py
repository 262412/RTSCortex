from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def melee_config_type(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[type[Any], list[Any]]]:
    shared_warpgate_actions: list[Any] = [
        {"name": "Warp_Zealot_Near", "arg": ["tag"], "func": []},
        {"name": "Warp_Stalker_Near", "arg": ["tag"], "func": []},
    ]

    class FakeProtossAgentConfig:
        def __init__(self) -> None:
            self.ENABLE_INIT_STEPS = False
            self.ENABLE_AUTO_WORKER_MANAGE = False
            self.ENABLE_AUTO_WORKER_TRAINING = False
            self.AGENTS_ALWAYS_DISABLE = ["Builder"]
            self.AGENTS = {
                "Builder": {
                    "team": [{"name": "Builder-Probe-1"}],
                    "action": {
                        "Probe": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Build_Pylon_Screen", "arg": ["screen"], "func": []},
                            {"name": "Build_Stargate_Screen", "arg": ["screen"], "func": []},
                        ]
                    },
                },
                "Developer": {
                    "team": [{"name": "WarpGate-1"}, {"name": "Empty"}],
                    "action": {
                        "WarpGate": shared_warpgate_actions,
                        "EmptyGroup": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Train_Zealot", "arg": [], "func": []},
                            {"name": "Train_Stalker", "arg": [], "func": []},
                            {"name": "Research_WarpGate", "arg": [], "func": []},
                            {"name": "Train_Carrier", "arg": [], "func": []},
                            {
                                "name": "Research_ProtossGroundWeapons",
                                "arg": [],
                                "func": [],
                            },
                        ],
                    },
                },
                "CombatGroup0": {
                    "team": [{"name": "Zealot-1"}, {"name": "Zealot-2"}],
                    "action": {
                        "Zealot": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Attack_Unit", "arg": ["tag"], "func": []},
                        ]
                    },
                },
                "CombatGroup1": {
                    "team": [{"name": "Stalker-1"}, {"name": "Stalker-2"}],
                    "action": {
                        "Stalker": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Attack_Unit", "arg": ["tag"], "func": []},
                        ]
                    },
                },
                "Defender": {"team": [], "action": {}},
            }

    package_names = (
        "llm_pysc2",
        "llm_pysc2.agents",
        "llm_pysc2.agents.configs",
    )
    for name in package_names:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    config_module = ModuleType("llm_pysc2.agents.configs.config")
    config_module.ProtossAgentConfig = FakeProtossAgentConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llm_pysc2.agents.configs.config", config_module)
    sys.modules.pop("rtscortex_llm_pysc2.melee", None)

    module = importlib.import_module("rtscortex_llm_pysc2.melee")
    try:
        yield module.RTSCortexMeleeConfig, shared_warpgate_actions
    finally:
        sys.modules.pop("rtscortex_llm_pysc2.melee", None)


def test_melee_config_keeps_only_the_minimum_protoss_chain(
    melee_config_type: tuple[type[Any], list[Any]],
) -> None:
    config_type, _ = melee_config_type

    config = config_type()

    assert list(config.AGENTS) == ["Builder", "Developer", "CombatGroup0", "CombatGroup1"]
    assert [team["name"] for team in config.AGENTS["Builder"]["team"]] == [
        "Builder-Probe-1"
    ]
    assert [team["name"] for team in config.AGENTS["Developer"]["team"]] == [
        "WarpGate-1",
        "Empty",
    ]
    assert [team["name"] for team in config.AGENTS["CombatGroup0"]["team"]] == [
        "Zealot-1"
    ]
    assert [team["name"] for team in config.AGENTS["CombatGroup1"]["team"]] == [
        "Stalker-1"
    ]


def test_melee_config_enables_opening_and_worker_automation(
    melee_config_type: tuple[type[Any], list[Any]],
) -> None:
    config_type, _ = melee_config_type

    config = config_type()

    assert config.ENABLE_INIT_STEPS is True
    assert config.ENABLE_AUTO_WORKER_MANAGE is True
    assert config.ENABLE_AUTO_WORKER_TRAINING is True
    assert config.AGENTS_ALWAYS_DISABLE == []


def test_melee_config_gives_every_actor_action_set_a_noop_without_mutating_upstream(
    melee_config_type: tuple[type[Any], list[Any]],
) -> None:
    config_type, shared_warpgate_actions = melee_config_type

    config = config_type()

    for agent in config.AGENTS.values():
        for actions in agent["action"].values():
            assert [action["name"] for action in actions].count("No_Operation") == 1
    assert [action["name"] for action in shared_warpgate_actions] == [
        "Warp_Zealot_Near",
        "Warp_Stalker_Near",
    ]


def test_melee_config_preserves_build_train_research_and_combat_actions(
    melee_config_type: tuple[type[Any], list[Any]],
) -> None:
    config_type, _ = melee_config_type
    config = config_type()

    action_names = {
        agent_name: {
            action["name"]
            for actions in agent["action"].values()
            for action in actions
        }
        for agent_name, agent in config.AGENTS.items()
    }

    assert "Build_Pylon_Screen" in action_names["Builder"]
    assert {"Train_Zealot", "Train_Stalker", "Research_WarpGate"} <= action_names[
        "Developer"
    ]
    assert "Attack_Unit" in action_names["CombatGroup0"]
    assert "Attack_Unit" in action_names["CombatGroup1"]
    assert "Build_Stargate_Screen" not in action_names["Builder"]
    assert "Train_Carrier" not in action_names["Developer"]
    assert "Research_ProtossGroundWeapons" not in action_names["Developer"]
    assert "Stop_Building" not in action_names["Developer"]
