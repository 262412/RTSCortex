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
                            {"name": "Attack_Unit", "arg": ["tag"], "func": []},
                            {"name": "Build_Pylon_Screen", "arg": ["screen"], "func": []},
                            {"name": "Build_Forge_Screen", "arg": ["screen"], "func": []},
                            {
                                "name": "Build_Assimilator_Near",
                                "arg": ["tag"],
                                "func": [(40, "build-assimilator", ("queued", "screen_tag"))],
                            },
                            {"name": "Build_Assimilator_Screen", "arg": ["screen"], "func": []},
                            {
                                "name": "Build_Nexus_Near",
                                "arg": ["tag"],
                                "func": [
                                    (573, "move-camera", ("world_tag",)),
                                    (0, "settlement-noop", ()),
                                    (65, "build-nexus", ("queued", "screen_tag")),
                                ],
                            },
                            {"name": "Build_Nexus_Screen", "arg": ["screen"], "func": []},
                            {"name": "Build_Stargate_Screen", "arg": ["screen"], "func": []},
                            {
                                "name": "Build_ShieldBattery_Screen",
                                "arg": ["screen"],
                                "func": [],
                            },
                        ]
                    },
                },
                "Developer": {
                    "team": [{"name": "WarpGate-1"}, {"name": "Empty"}],
                    "action": {
                        "WarpGate": shared_warpgate_actions,
                        "EmptyGroup": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Train_Adept", "arg": [], "func": []},
                            {"name": "Train_Zealot", "arg": [], "func": []},
                            {"name": "Train_Stalker", "arg": [], "func": []},
                            {"name": "Train_VoidRay", "arg": [], "func": []},
                            {"name": "Train_Oracle", "arg": [], "func": []},
                            {"name": "Train_Phoenix", "arg": [], "func": []},
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
                "CombatGroup3": {
                    "team": [{"name": "VoidRay-1"}, {"name": "Carrier-1"}],
                    "action": {
                        "VoidRay": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Stop", "arg": [], "func": []},
                            {"name": "Hold_Position", "arg": [], "func": []},
                            {"name": "Move_Minimap", "arg": ["minimap"], "func": []},
                            {"name": "Move_Screen", "arg": ["screen"], "func": []},
                            {"name": "Attack_Unit", "arg": ["tag"], "func": []},
                            {"name": "Ability_PrismaticAlignment", "arg": [], "func": []},
                        ]
                    },
                },
                "CombatGroup7": {
                    "team": [{"name": "Adept-1"}, {"name": "AdeptPhase-1"}],
                    "action": {
                        "Adept": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Stop", "arg": [], "func": []},
                            {"name": "Hold_Position", "arg": [], "func": []},
                            {"name": "Move_Minimap", "arg": ["minimap"], "func": []},
                            {"name": "Move_Screen", "arg": ["screen"], "func": []},
                            {"name": "Attack_Unit", "arg": ["tag"], "func": []},
                            {
                                "name": "Ability_AdeptPhaseShift_Screen",
                                "arg": ["screen"],
                                "func": [],
                            },
                        ]
                    },
                },
                "CombatGroup8": {
                    "team": [{"name": "Oracle-1"}, {"name": "Phoenix-1"}],
                    "action": {
                        "Oracle": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Stop", "arg": [], "func": []},
                            {"name": "Hold_Position", "arg": [], "func": []},
                            {"name": "Move_Minimap", "arg": ["minimap"], "func": []},
                            {"name": "Move_Screen", "arg": ["screen"], "func": []},
                            {"name": "Attack_Unit", "arg": ["tag"], "func": []},
                            {"name": "Ability_PulsarBeamOn", "arg": [], "func": []},
                        ],
                        "Phoenix": [
                            {"name": "No_Operation", "arg": [], "func": [(0, None, ())]},
                            {"name": "Stop", "arg": [], "func": []},
                            {"name": "Hold_Position", "arg": [], "func": []},
                            {"name": "Move_Minimap", "arg": ["minimap"], "func": []},
                            {"name": "Move_Screen", "arg": ["screen"], "func": []},
                            {"name": "Attack_Unit", "arg": ["tag"], "func": []},
                            {
                                "name": "Ability_GravitonBeam_Unit",
                                "arg": ["tag"],
                                "func": [],
                            },
                        ],
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

    assert list(config.AGENTS) == [
        "Builder",
        "Developer",
        "CombatGroup0",
        "CombatGroup1",
        "CombatGroup3",
        "CombatGroup7",
        "CombatGroup8",
    ]
    assert [team["name"] for team in config.AGENTS["Builder"]["team"]] == ["Builder-Probe-1"]
    assert [team["name"] for team in config.AGENTS["Developer"]["team"]] == [
        "WarpGate-1",
        "Empty",
    ]
    assert [team["name"] for team in config.AGENTS["CombatGroup0"]["team"]] == ["Zealot-1"]
    assert [team["name"] for team in config.AGENTS["CombatGroup1"]["team"]] == ["Stalker-1"]
    assert [team["name"] for team in config.AGENTS["CombatGroup3"]["team"]] == ["VoidRay-1"]
    assert [team["name"] for team in config.AGENTS["CombatGroup7"]["team"]] == ["Adept-1"]
    assert [team["name"] for team in config.AGENTS["CombatGroup8"]["team"]] == [
        "Oracle-1",
        "Phoenix-1",
    ]


def test_melee_config_enables_gas_management_with_reserved_builder(
    melee_config_type: tuple[type[Any], list[Any]],
) -> None:
    config_type, _ = melee_config_type

    config = config_type()

    assert config.ENABLE_INIT_STEPS is True
    assert config.ENABLE_AUTO_WORKER_MANAGE is True
    assert config.ENABLE_AUTO_WORKER_TRAINING is True


def test_melee_config_moves_camera_before_assimilator_screen_translation(
    melee_config_type: tuple[type[Any], list[Any]],
) -> None:
    config_type, _ = melee_config_type
    config = config_type()
    builder_actions = config.AGENTS["Builder"]["action"]["Probe"]
    assimilator = next(
        action for action in builder_actions if action["name"] == "Build_Assimilator_Near"
    )

    assert [function_id for function_id, _, _ in assimilator["func"]] == [573, 0, 40]
    assert assimilator["func"][0][2] == ("world_tag",)
    assert assimilator["func"][2][2] == ("queued", "screen_tag")
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
        agent_name: {action["name"] for actions in agent["action"].values() for action in actions}
        for agent_name, agent in config.AGENTS.items()
    }

    assert "Build_Pylon_Screen" in action_names["Builder"]
    assert "Build_Forge_Screen" in action_names["Builder"]
    assert "Build_Stargate_Screen" in action_names["Builder"]
    assert "Build_ShieldBattery_Screen" in action_names["Builder"]
    assert {
        "Train_Adept",
        "Train_Zealot",
        "Train_Stalker",
        "Train_VoidRay",
        "Train_Oracle",
        "Train_Phoenix",
        "Research_WarpGate",
    } <= action_names["Developer"]
    assert "Attack_Unit" in action_names["CombatGroup0"]
    assert "Attack_Unit" in action_names["CombatGroup1"]
    basic_combat_actions = {
        "No_Operation",
        "Stop",
        "Hold_Position",
        "Move_Minimap",
        "Move_Screen",
        "Attack_Unit",
    }
    assert action_names["CombatGroup3"] == basic_combat_actions
    assert action_names["CombatGroup7"] == basic_combat_actions
    assert action_names["CombatGroup8"] == {
        "No_Operation",
        "Move_Minimap",
        "Move_Screen",
    }
    assert {
        "Stop",
        "Hold_Position",
        "Attack_Unit",
        "Ability_PulsarBeamOn",
        "Ability_GravitonBeam_Unit",
    }.isdisjoint(action_names["CombatGroup8"])
    assert "Attack_Unit" not in action_names["Builder"]
    assert "Build_Assimilator_Near" in action_names["Builder"]
    assert "Build_Nexus_Near" in action_names["Builder"]
    assert "Build_Assimilator_Screen" not in action_names["Builder"]
    assert "Build_Nexus_Screen" not in action_names["Builder"]
    assert "Train_Carrier" not in action_names["Developer"]
    assert "Research_ProtossGroundWeapons" not in action_names["Developer"]
    assert "Stop_Building" not in action_names["Developer"]
