"""RTSCortex-owned Zerg configuration for the Simple64 live Worker."""

from __future__ import annotations

from typing import Any

from llm_pysc2.agents.configs.config import AgentConfig  # type: ignore[import-not-found]
from pysc2.lib import units  # type: ignore[import-not-found]
from pysc2.lib.actions import FUNCTIONS as F  # type: ignore[import-not-found]


def _register_zerg_building_footprints() -> None:
    from llm_pysc2.lib import utils  # type: ignore[import-not-found]

    for name in ("EvolutionChamber", "Extractor", "HydraliskDen", "RoachWarren", "SpawningPool"):
        if name not in utils.SIZE3_BUILDING_NAMES:
            utils.SIZE3_BUILDING_NAMES.append(name)
    for name in ("SpineCrawler", "SporeCrawler"):
        if name not in utils.SIZE2_BUILDING_NAMES:
            utils.SIZE2_BUILDING_NAMES.append(name)
    if "Hatchery" not in utils.SIZE5_BUILDING_NAMES:
        utils.SIZE5_BUILDING_NAMES.append("Hatchery")


_register_zerg_building_footprints()


def _action(name: str, arguments: list[str], function: Any, *function_args: Any) -> dict[str, Any]:
    return {
        "name": name,
        "arg": arguments,
        "func": [(int(function.id), function, tuple(function_args))],
    }


NO_OPERATION = _action("No_Operation", [], F.no_op)
UNIT_ACTIONS = [
    _action("Stop", [], F.Stop_quick, "now"),
    NO_OPERATION,
    _action("Hold_Position", [], F.HoldPosition_quick, "queued"),
    _action("Move_Minimap", ["minimap"], F.Move_minimap, "queued", "minimap"),
    _action("Move_Screen", ["screen"], F.Move_screen, "queued", "screen"),
    _action("Attack_Unit", ["tag"], F.Attack_screen, "queued", "screen_tag"),
]
BUILD_ACTIONS = [
    _action("Build_Hatchery_Near", ["tag"], F.llm_pysc2_move_camera, "world_tag"),
    _action("Build_Extractor_Near", ["tag"], F.llm_pysc2_move_camera, "world_tag"),
    _action(
        "Build_SpawningPool_Screen",
        ["screen"],
        F.Build_SpawningPool_screen,
        "queued",
        "screen",
    ),
    _action(
        "Build_RoachWarren_Screen",
        ["screen"],
        F.Build_RoachWarren_screen,
        "queued",
        "screen",
    ),
    _action(
        "Build_EvolutionChamber_Screen",
        ["screen"],
        F.Build_EvolutionChamber_screen,
        "queued",
        "screen",
    ),
    _action(
        "Build_HydraliskDen_Screen",
        ["screen"],
        F.Build_HydraliskDen_screen,
        "queued",
        "screen",
    ),
    _action(
        "Build_SpineCrawler_Screen",
        ["screen"],
        F.Build_SpineCrawler_screen,
        "queued",
        "screen",
    ),
    _action(
        "Build_SporeCrawler_Screen",
        ["screen"],
        F.Build_SporeCrawler_screen,
        "queued",
        "screen",
    ),
]
BUILD_ACTIONS[0]["func"] = [
    (573, F.llm_pysc2_move_camera, ("world_tag",)),
    (0, F.no_op, ()),
    (int(F.Build_Hatchery_screen.id), F.Build_Hatchery_screen, ("queued", "screen_tag")),
]
BUILD_ACTIONS[1]["func"] = [
    (573, F.llm_pysc2_move_camera, ("world_tag",)),
    (0, F.no_op, ()),
    (int(F.Build_Extractor_screen.id), F.Build_Extractor_screen, ("queued", "screen_tag")),
]
PRODUCTION_ACTIONS = [
    _action("Train_Drone", [], F.Train_Drone_quick, "queued"),
    _action("Train_Overlord", [], F.Train_Overlord_quick, "queued"),
    _action("Train_Queen", [], F.Train_Queen_quick, "queued"),
    _action("Train_Zergling", [], F.Train_Zergling_quick, "queued"),
    _action("Train_Roach", [], F.Train_Roach_quick, "queued"),
    _action("Train_Hydralisk", [], F.Train_Hydralisk_quick, "queued"),
    _action("Morph_Lair", [], F.Morph_Lair_quick, "queued"),
]


def _llm_settings(config: AgentConfig, *, translator_o: str = "default") -> dict[str, Any]:
    return {
        "basic_prompt": config.basic_prompt,
        "translator_o": translator_o,
        "translator_a": config.translator_a,
        "img_fea": config.ENABLE_IMAGE_FEATURE,
        "img_rgb": config.ENABLE_IMAGE_RGB,
        "model_name": config.model_name,
        "api_base": config.api_base,
        "api_key": config.api_key,
    }


class RTSCortexZergMeleeConfig(AgentConfig):  # type: ignore[misc]
    """Initial Zerg macro and combat surface owned by RTSCortex."""

    AGENTS: dict[str, dict[str, Any]]
    AGENTS_ALWAYS_DISABLE: list[str]

    def __init__(self) -> None:
        super().__init__()
        # Upstream Zerg prompt/translator/observation factories are empty. The
        # reviewed Protoss implementations remain an unused compatibility shell;
        # RTSCortex supplies every decision and the actual SC2 player race.
        self.race = "protoss"
        self.rtscortex_player_race = "zerg"
        self.AGENTS_ALWAYS_DISABLE = []
        self.ENABLE_INIT_STEPS = True
        self.ENABLE_AUTO_WORKER_MANAGE = True
        self.ENABLE_AUTO_WORKER_TRAINING = False
        self.AGENTS = {
            "Builder": {
                "describe": "Zerg Drone construction controller.",
                "llm": _llm_settings(self),
                "team": [
                    {
                        "name": "Builder-Drone-1",
                        "unit_type": [units.Zerg.Drone],
                        "game_group": -1,
                        "select_type": "select",
                    }
                ],
                "action": {units.Zerg.Drone: [*UNIT_ACTIONS, *BUILD_ACTIONS]},
            },
            "Developer": {
                "describe": "Zerg larva and hatchery production controller.",
                "llm": _llm_settings(self, translator_o="developer"),
                "team": [
                    {
                        "name": "Empty",
                        "unit_type": [],
                        "game_group": -1,
                        "select_type": "select",
                    }
                ],
                "action": {"EmptyGroup": [NO_OPERATION, *PRODUCTION_ACTIONS]},
            },
            "CombatGroup0": _combat_agent(self, "Zergling-1", units.Zerg.Zergling),
            "CombatGroup1": _combat_agent(self, "Queen-1", units.Zerg.Queen),
            "CombatGroup2": _combat_agent(self, "Roach-1", units.Zerg.Roach),
            "CombatGroup3": _combat_agent(self, "Hydralisk-1", units.Zerg.Hydralisk),
        }


def _combat_agent(config: AgentConfig, team_name: str, unit_type: Any) -> dict[str, Any]:
    return {
        "describe": f"Zerg combat controller for {team_name}.",
        "llm": _llm_settings(config),
        "team": [
            {
                "name": team_name,
                "unit_type": [unit_type],
                "game_group": -1,
                "select_type": "select_all_type",
            }
        ],
        "action": {unit_type: list(UNIT_ACTIONS)},
    }


__all__ = ["RTSCortexZergMeleeConfig"]
