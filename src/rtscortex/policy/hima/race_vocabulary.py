"""Pinned official HIMA macro vocabularies for all three SC2 races."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import Literal

from rtscortex.policy.hima.models import HIMAMacroAction
from rtscortex.policy.hima.vocabulary import HIMA_PROTOSS_ACTIONS

HIMARace = Literal["protoss", "terran", "zerg"]

_TERRAN_ACTIONS: tuple[tuple[int, str, str], ...] = (
    *(
        (action_id, name, "train")
        for action_id, name in enumerate(
            (
                "SCV",
                "MULE",
                "Marine",
                "Reaper",
                "Marauder",
                "Ghost",
                "Hellion",
                "WidowMine",
                "Cyclone",
                "SiegeTank",
                "Thor",
                "VikingFighter",
                "Medivac",
                "Liberator",
                "Banshee",
                "Raven",
                "Battlecruiser",
            ),
            start=100,
        )
    ),
    *(
        (action_id, name, "build")
        for action_id, name in enumerate(
            (
                "CommandCenter",
                "Refinery",
                "OrbitalCommand",
                "PlanetaryFortress",
                "Barracks",
                "Factory",
                "Starport",
                "BarracksReactor",
                "BarracksTechLab",
                "FactoryReactor",
                "FactoryTechLab",
                "StarportReactor",
                "StarportTechLab",
                "SupplyDepot",
                "EngineeringBay",
                "Bunker",
                "MissileTurret",
                "SensorTower",
                "GhostAcademy",
                "Armory",
                "FusionCore",
            ),
            start=200,
        )
    ),
    *(
        (action_id, name, "research")
        for action_id, name in enumerate(
            (
                "TerranInfantryWeaponsLevel1",
                "TerranInfantryWeaponsLevel2",
                "TerranInfantryWeaponsLevel3",
                "TerranInfantryArmorsLevel1",
                "TerranInfantryArmorsLevel2",
                "TerranInfantryArmorsLevel3",
                "TerranVehicleWeaponsLevel1",
                "TerranVehicleWeaponsLevel2",
                "TerranVehicleWeaponsLevel3",
                "TerranShipWeaponsLevel1",
                "TerranShipWeaponsLevel2",
                "TerranShipWeaponsLevel3",
                "TerranVehicleAndShipArmorsLevel1",
                "TerranVehicleAndShipArmorsLevel2",
                "TerranVehicleAndShipArmorsLevel3",
                "TerranBuildingArmor",
                "HiSecAutoTracking",
                "Stimpack",
                "ShieldWall",
                "PunisherGrenades",
                "PersonalCloaking",
                "SmartServos",
                "HighCapacityBarrels",
                "DrillClaws",
                "CycloneLockOnDamageUpgrade",
                "MedivacIncreaseSpeedBoost",
                "LiberatorAGRangeUpgrade",
                "BansheeCloak",
                "BansheeSpeed",
                "InterferenceMatrix",
                "BattlecruiserEnableSpecializations",
            ),
            start=300,
        )
    ),
)

_ZERG_ACTIONS: tuple[tuple[int, str, str], ...] = (
    *(
        (action_id, name, "train")
        for action_id, name in enumerate(
            (
                "Drone",
                "Overlord",
                "Zergling",
                "Queen",
                "Roach",
                "Baneling",
                "Ravager",
                "Overseer",
                "Hydralisk",
                "Mutalisk",
                "Corruptor",
                "Infestor",
                "SwarmHostMP",
                "LurkerMP",
                "Viper",
                "BroodLord",
                "Ultralisk",
            ),
            start=100,
        )
    ),
    *(
        (action_id, name, "build")
        for action_id, name in enumerate(
            (
                "Hatchery",
                "Extractor",
                "Lair",
                "Hive",
                "SpawningPool",
                "EvolutionChamber",
                "RoachWarren",
                "BanelingNest",
                "SpineCrawler",
                "SporeCrawler",
                "HydraliskDen",
                "InfestationPit",
                "LurkerDenMP",
                "Spire",
                "NydusNetwork",
                "UltraliskCavern",
                "GreaterSpire",
            ),
            start=200,
        )
    ),
    *(
        (action_id, name, "research")
        for action_id, name in enumerate(
            (
                "ZergMeleeWeaponsLevel1",
                "ZergMeleeWeaponsLevel2",
                "ZergMeleeWeaponsLevel3",
                "ZergMissileWeaponsLevel1",
                "ZergMissileWeaponsLevel2",
                "ZergMissileWeaponsLevel3",
                "ZergGroundArmorsLevel1",
                "ZergGroundArmorsLevel2",
                "ZergGroundArmorsLevel3",
                "ZergFlyerWeaponsLevel1",
                "ZergFlyerWeaponsLevel2",
                "ZergFlyerWeaponsLevel3",
                "ZergFlyerArmorsLevel1",
                "ZergFlyerArmorsLevel2",
                "ZergFlyerArmorsLevel3",
                "Burrow",
                "overlordspeed",
                "zerglingmovementspeed",
                "zerglingattackspeed",
                "GlialReconstitution",
                "TunnelingClaws",
                "CentrificalHooks",
                "EvolveMuscularAugments",
                "EvolveGroovedSpines",
                "NeuralParasite",
                "DiggingClaws",
                "LurkerRange",
                "ChitinousPlating",
                "AnabolicSynthesis",
            ),
            start=300,
        )
    ),
)


def _make_actions(entries: Iterable[tuple[int, str, str]]) -> tuple[HIMAMacroAction, ...]:
    verbs = {"train": "TRAIN", "build": "BUILD", "research": "RESEARCH"}
    return tuple(
        HIMAMacroAction(
            upstream_action_id=action_id,
            upstream_name=name,
            canonical_action=f"{verbs[category]} {name.upper()}",
            category=category,  # type: ignore[arg-type]
            aliases=(name,),
        )
        for action_id, name, category in entries
    )


HIMA_TERRAN_ACTIONS = _make_actions(_TERRAN_ACTIONS)
HIMA_ZERG_ACTIONS = _make_actions(_ZERG_ACTIONS)
HIMA_ACTIONS_BY_RACE = MappingProxyType(
    {
        "protoss": HIMA_PROTOSS_ACTIONS,
        "terran": HIMA_TERRAN_ACTIONS,
        "zerg": HIMA_ZERG_ACTIONS,
    }
)
HIMA_VOCABULARY_VERSIONS = MappingProxyType(
    {
        "protoss": "hima-protoss-60-v2",
        "terran": "hima-terran-69-v1",
        "zerg": "hima-zerg-63-v1",
    }
)
HIMA_PARSER_VERSIONS = MappingProxyType(
    {
        race: f"hima-{race}-parser-v3" if race != "protoss" else "hima-protoss-parser-v6"
        for race in HIMA_ACTIONS_BY_RACE
    }
)


def hima_actions_for_race(race: str) -> tuple[HIMAMacroAction, ...]:
    try:
        return HIMA_ACTIONS_BY_RACE[race.casefold()]
    except KeyError as error:
        raise ValueError(f"unsupported HIMA race: {race}") from error


def resolve_race_hima_action(value: str, *, race: str) -> HIMAMacroAction | None:
    key = " ".join(value.strip().upper().split())
    for action in hima_actions_for_race(race):
        if key in {
            " ".join(token.strip().upper().split())
            for token in (action.canonical_action, *action.aliases)
        }:
            return action
    return None


def hima_race_for_model(model_id: str) -> HIMARace:
    try:
        race = model_id.split("/", 1)[1].rsplit("-", 1)[0].casefold()
    except IndexError as error:
        raise ValueError(f"invalid HIMA model ID: {model_id}") from error
    if race not in HIMA_ACTIONS_BY_RACE:
        raise ValueError(f"invalid HIMA model race: {model_id}")
    return race  # type: ignore[return-value]
