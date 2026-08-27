"""RTSCortex-owned Terran configuration for the Simple64 live Worker.

The pinned LLM-PySC2 checkout leaves its Terran factories empty.  This module
therefore defines the supported action surface locally while retaining the
reviewed upstream MainAgent orchestration and feature-action translator.
"""

from __future__ import annotations

from typing import Any

from llm_pysc2.agents.configs.config import AgentConfig  # type: ignore[import-not-found]
from pysc2.lib import units  # type: ignore[import-not-found]
from pysc2.lib.actions import FUNCTIONS as F  # type: ignore[import-not-found]


def _register_terran_building_footprints() -> None:
    """Complete the pinned translator's intentionally empty Terran size tables."""

    from llm_pysc2.lib import utils  # type: ignore[import-not-found]

    for name in (
        "Barracks",
        "Bunker",
        "EngineeringBay",
        "Factory",
        "Refinery",
        "Starport",
    ):
        if name not in utils.SIZE3_BUILDING_NAMES:
            utils.SIZE3_BUILDING_NAMES.append(name)
    for name in ("MissileTurret", "SupplyDepot"):
        if name not in utils.SIZE2_BUILDING_NAMES:
            utils.SIZE2_BUILDING_NAMES.append(name)


_register_terran_building_footprints()


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
UNIT_ACTIONS[-1]["func"] = [
    (573, F.llm_pysc2_move_camera, ("world_tag",)),
    (0, F.no_op, ()),
    (int(F.Attack_screen.id), F.Attack_screen, ("queued", "screen_tag")),
]
NON_ATTACKING_UNIT_ACTIONS = [
    _action("Stop", [], F.Stop_quick, "now"),
    NO_OPERATION,
    _action("Hold_Position", [], F.HoldPosition_quick, "queued"),
    _action("Move_Minimap", ["minimap"], F.Move_minimap, "queued", "minimap"),
    _action("Move_Screen", ["screen"], F.Move_screen, "queued", "screen"),
]
BUILD_ACTIONS = [
    _action("Build_SupplyDepot_Screen", ["screen"], F.Build_SupplyDepot_screen, "queued", "screen"),
    _action("Build_Barracks_Screen", ["screen"], F.Build_Barracks_screen, "queued", "screen"),
    _action("Build_Refinery_Near", ["tag"], F.llm_pysc2_move_camera, "world_tag"),
    _action("Build_CommandCenter_Near", ["tag"], F.llm_pysc2_move_camera, "world_tag"),
    _action("Build_Factory_Screen", ["screen"], F.Build_Factory_screen, "queued", "screen"),
    _action("Build_Starport_Screen", ["screen"], F.Build_Starport_screen, "queued", "screen"),
    _action(
        "Build_EngineeringBay_Screen",
        ["screen"],
        F.Build_EngineeringBay_screen,
        "queued",
        "screen",
    ),
    _action("Build_Bunker_Screen", ["screen"], F.Build_Bunker_screen, "queued", "screen"),
    _action(
        "Build_MissileTurret_Screen",
        ["screen"],
        F.Build_MissileTurret_screen,
        "queued",
        "screen",
    ),
]

# Near builds settle the camera before resolving the exact feature-layer target.
BUILD_ACTIONS[2]["func"] = [
    (573, F.llm_pysc2_move_camera, ("world_tag",)),
    (0, F.no_op, ()),
    (int(F.Build_Refinery_screen.id), F.Build_Refinery_screen, ("queued", "screen_tag")),
]
BUILD_ACTIONS[3]["func"] = [
    (573, F.llm_pysc2_move_camera, ("world_tag",)),
    (0, F.no_op, ()),
    (
        int(F.Build_CommandCenter_screen.id),
        F.Build_CommandCenter_screen,
        ("queued", "screen_tag"),
    ),
]

PRODUCTION_ACTIONS = [
    _action("Train_SCV", [], F.Train_SCV_quick, "queued"),
    _action("Train_Marine", [], F.Train_Marine_quick, "queued"),
    _action("Train_Marauder", [], F.Train_Marauder_quick, "queued"),
    _action("Train_Hellion", [], F.Train_Hellion_quick, "queued"),
    _action("Train_SiegeTank", [], F.Train_SiegeTank_quick, "queued"),
    _action("Train_Medivac", [], F.Train_Medivac_quick, "queued"),
    _action("Train_VikingFighter", [], F.Train_VikingFighter_quick, "queued"),
]
ECONOMY_ACTIONS = [
    _action("Morph_OrbitalCommand", [], F.Morph_OrbitalCommand_quick, "queued"),
    _action(
        "Effect_CalldownMULE_Screen",
        ["screen"],
        F.Effect_CalldownMULE_screen,
        "queued",
        "screen",
    ),
]
RESEARCH_ACTIONS = [
    _action("Research_Stimpack", [], F.Research_Stimpack_quick, "queued"),
]
ADDON_ACTIONS = [
    _action("Build_BarracksTechLab", [], F.Build_TechLab_Barracks_quick, "queued"),
    _action("Build_BarracksReactor", [], F.Build_Reactor_Barracks_quick, "queued"),
    _action("Build_FactoryTechLab", [], F.Build_TechLab_Factory_quick, "queued"),
    _action("Build_FactoryReactor", [], F.Build_Reactor_Factory_quick, "queued"),
    _action("Build_StarportTechLab", [], F.Build_TechLab_Starport_quick, "queued"),
    _action("Build_StarportReactor", [], F.Build_Reactor_Starport_quick, "queued"),
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


class RTSCortexTerranMeleeConfig(AgentConfig):  # type: ignore[misc]
    """Minimal Terran live surface owned by RTSCortex."""

    AGENTS: dict[str, dict[str, Any]]
    AGENTS_ALWAYS_DISABLE: list[str]

    def __init__(self) -> None:
        super().__init__()
        # The upstream Terran prompt/translator/observation factories are empty.
        # The feature action surface below is Terran, while the compatibility
        # factory remains Protoss until those unused upstream text modules are
        # replaced by RTSCortex-owned implementations.
        self.race = "protoss"
        self.rtscortex_player_race = "terran"
        self.AGENTS_ALWAYS_DISABLE = []
        self.ENABLE_INIT_STEPS = True
        # Upstream's worker workplace tracker is race-neutral (all townhall,
        # worker, mineral, and gas types). RTSCortex adds exact gas saturation
        # and reserved-Builder guards around it.
        self.ENABLE_AUTO_WORKER_MANAGE = True
        self.ENABLE_AUTO_WORKER_TRAINING = False
        self.AGENTS = {
            "Builder": {
                "describe": "Terran SCV construction controller.",
                "llm": _llm_settings(self),
                "team": [
                    {
                        "name": "Builder-SCV-1",
                        "unit_type": [units.Terran.SCV],
                        "game_group": -1,
                        "select_type": "select",
                    }
                ],
                "action": {units.Terran.SCV: [*UNIT_ACTIONS, *BUILD_ACTIONS]},
            },
            "Developer": {
                "describe": "Terran production controller.",
                "llm": _llm_settings(self, translator_o="developer"),
                "team": [
                    {
                        "name": "Empty",
                        "unit_type": [],
                        "game_group": -1,
                        "select_type": "select",
                    }
                ],
                "action": {
                    "EmptyGroup": [
                        NO_OPERATION,
                        *ECONOMY_ACTIONS,
                        *PRODUCTION_ACTIONS,
                        *ADDON_ACTIONS,
                        *RESEARCH_ACTIONS,
                    ]
                },
            },
            "CombatGroup0": _combat_agent(self, "Marine-1", units.Terran.Marine),
            "CombatGroup1": _combat_agent(self, "Marauder-1", units.Terran.Marauder),
            "CombatGroup2": _combat_agent(self, "Hellion-1", units.Terran.Hellion),
            "CombatGroup3": _combat_agent(
                self,
                "SiegeTank-1",
                units.Terran.SiegeTank,
                alternate_types=(units.Terran.SiegeTankSieged,),
            ),
            "CombatGroup4": _combat_agent(
                self,
                "Medivac-1",
                units.Terran.Medivac,
                actions=NON_ATTACKING_UNIT_ACTIONS,
            ),
            "CombatGroup5": _combat_agent(
                self,
                "Viking-1",
                units.Terran.VikingFighter,
                alternate_types=(units.Terran.VikingAssault,),
            ),
        }


def _combat_agent(
    config: AgentConfig,
    team_name: str,
    unit_type: Any,
    *,
    alternate_types: tuple[Any, ...] = (),
    actions: list[dict[str, Any]] = UNIT_ACTIONS,
) -> dict[str, Any]:
    unit_types = [unit_type, *alternate_types]
    return {
        "describe": f"Terran combat controller for {team_name}.",
        "llm": _llm_settings(config),
        "team": [
            {
                "name": team_name,
                "unit_type": unit_types,
                "game_group": -1,
                "select_type": "select_all_type",
            }
        ],
        "action": {value: list(actions) for value in unit_types},
    }


__all__ = ["RTSCortexTerranMeleeConfig"]
