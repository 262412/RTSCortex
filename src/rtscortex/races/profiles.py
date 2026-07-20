"""Built-in race profiles and the single active-race registry."""

from __future__ import annotations

from dataclasses import dataclass

from rtscortex.progress import (
    PROTOSS_SIMPLE64_ACTION_SPECS,
    GoalRequirementKind,
    ProgressActionSpec,
    StatePrerequisite,
)
from rtscortex.races.models import (
    ActionDomain,
    MacroActionMapping,
    RaceId,
    RaceProfileData,
)


def _canonical_hima_action(verb: str, name: str) -> str:
    return f"{verb} {name.upper()}"


_PROTOSS_MAPPINGS = (
    MacroActionMapping("TRAIN ZEALOT", ("Train_Zealot", "Warp_Zealot_Near")),
    MacroActionMapping("TRAIN STALKER", ("Train_Stalker", "Warp_Stalker_Near")),
    MacroActionMapping("TRAIN ADEPT", ("Train_Adept",)),
    MacroActionMapping("TRAIN PHOENIX", ("Train_Phoenix",)),
    MacroActionMapping("TRAIN VOIDRAY", ("Train_VoidRay",)),
    MacroActionMapping("TRAIN ORACLE", ("Train_Oracle",)),
    MacroActionMapping("BUILD PYLON", ("Build_Pylon_Screen",)),
    MacroActionMapping("BUILD GATEWAY", ("Build_Gateway_Screen",)),
    MacroActionMapping("BUILD FORGE", ("Build_Forge_Screen",)),
    MacroActionMapping("BUILD CYBERNETICSCORE", ("Build_CyberneticsCore_Screen",)),
    MacroActionMapping("BUILD ASSIMILATOR", ("Build_Assimilator_Near",)),
    MacroActionMapping("BUILD NEXUS", ("Build_Nexus_Near",)),
    MacroActionMapping("BUILD STARGATE", ("Build_Stargate_Screen",)),
    MacroActionMapping("BUILD SHIELDBATTERY", ("Build_ShieldBattery_Screen",)),
    MacroActionMapping("RESEARCH WARPGATERESEARCH", ("Research_WarpGate",)),
)


def _domains(
    specs: tuple[ProgressActionSpec, ...],
    *,
    economy: set[str],
    technology: set[str],
    defense: set[str],
) -> dict[str, ActionDomain]:
    domains: dict[str, ActionDomain] = {}
    for spec in specs:
        if spec.name in economy:
            domain = ActionDomain.ECONOMY
        elif spec.name in technology:
            domain = ActionDomain.TECHNOLOGY
        elif spec.name in defense:
            domain = ActionDomain.DEFENSE
        else:
            domain = ActionDomain.PRODUCTION
        domains[spec.name] = domain
    return domains


PROTOSS_PROFILE_DATA = RaceProfileData(
    race=RaceId.PROTOSS,
    worker_type="Probe",
    supply_provider="Pylon",
    townhall_types=("Nexus",),
    gas_structure="Assimilator",
    production_structures=("Gateway", "WarpGate", "RoboticsFacility", "Stargate"),
    progress_action_specs=PROTOSS_SIMPLE64_ACTION_SPECS,
    macro_action_mappings=_PROTOSS_MAPPINGS,
    action_domains=_domains(
        PROTOSS_SIMPLE64_ACTION_SPECS,
        economy={
            "Train_Probe",
            "Build_Pylon_Screen",
            "Build_Assimilator_Near",
            "Build_Nexus_Near",
        },
        technology={
            "Build_Forge_Screen",
            "Build_CyberneticsCore_Screen",
            "Research_WarpGate",
        },
        defense={"Build_ShieldBattery_Screen"},
    ),
    action_producers={
        **{
            spec.name: ("Probe",)
            for spec in PROTOSS_SIMPLE64_ACTION_SPECS
            if spec.name.startswith("Build_")
        },
        "Train_Probe": ("Nexus",),
        "Train_Zealot": ("Gateway",),
        "Train_Stalker": ("Gateway",),
        "Train_Adept": ("Gateway",),
        "Train_Phoenix": ("Stargate",),
        "Train_VoidRay": ("Stargate",),
        "Train_Oracle": ("Stargate",),
        "Research_WarpGate": ("CyberneticsCore",),
    },
    hima_vocabulary_version="hima-protoss-60-v2",
    runtime_mapping_ready=True,
    live_worker_ready=True,
    effect_verification_kinds=("build", "production", "research", "move"),
    controller_capabilities=(
        "gas_workers",
        "supply_emergency",
        "resource_fallback",
        "prerequisite_closure",
    ),
    limitations=(),
)


TERRAN_PROGRESS_ACTION_SPECS: tuple[ProgressActionSpec, ...] = (
    ProgressActionSpec(
        "Train_SCV",
        GoalRequirementKind.UNIT,
        "SCV",
        minerals=50,
        supply=1,
    ),
    ProgressActionSpec(
        "Morph_OrbitalCommand",
        GoalRequirementKind.STRUCTURE,
        "OrbitalCommand",
        minerals=150,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Barracks"),),
    ),
    ProgressActionSpec(
        "Build_SupplyDepot_Screen",
        GoalRequirementKind.STRUCTURE,
        "SupplyDepot",
        minerals=100,
    ),
    ProgressActionSpec(
        "Build_Barracks_Screen",
        GoalRequirementKind.STRUCTURE,
        "Barracks",
        minerals=150,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "SupplyDepot"),),
    ),
    ProgressActionSpec(
        "Build_Refinery_Near",
        GoalRequirementKind.STRUCTURE,
        "Refinery",
        minerals=75,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "CommandCenter"),),
    ),
    ProgressActionSpec(
        "Build_CommandCenter_Near",
        GoalRequirementKind.STRUCTURE,
        "CommandCenter",
        minerals=400,
    ),
    ProgressActionSpec(
        "Build_Factory_Screen",
        GoalRequirementKind.STRUCTURE,
        "Factory",
        minerals=150,
        vespene=100,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Barracks"),),
    ),
    ProgressActionSpec(
        "Build_Starport_Screen",
        GoalRequirementKind.STRUCTURE,
        "Starport",
        minerals=150,
        vespene=100,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Factory"),),
    ),
    ProgressActionSpec(
        "Build_EngineeringBay_Screen",
        GoalRequirementKind.STRUCTURE,
        "EngineeringBay",
        minerals=125,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "CommandCenter"),),
    ),
    ProgressActionSpec(
        "Build_BarracksTechLab",
        GoalRequirementKind.STRUCTURE,
        "BarracksTechLab",
        minerals=50,
        vespene=25,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Barracks"),),
    ),
    ProgressActionSpec(
        "Build_BarracksReactor",
        GoalRequirementKind.STRUCTURE,
        "BarracksReactor",
        minerals=50,
        vespene=50,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Barracks"),),
    ),
    ProgressActionSpec(
        "Build_FactoryTechLab",
        GoalRequirementKind.STRUCTURE,
        "FactoryTechLab",
        minerals=50,
        vespene=25,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Factory"),),
    ),
    ProgressActionSpec(
        "Build_FactoryReactor",
        GoalRequirementKind.STRUCTURE,
        "FactoryReactor",
        minerals=50,
        vespene=50,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Factory"),),
    ),
    ProgressActionSpec(
        "Build_StarportTechLab",
        GoalRequirementKind.STRUCTURE,
        "StarportTechLab",
        minerals=50,
        vespene=25,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Starport"),),
    ),
    ProgressActionSpec(
        "Build_StarportReactor",
        GoalRequirementKind.STRUCTURE,
        "StarportReactor",
        minerals=50,
        vespene=50,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Starport"),),
    ),
    ProgressActionSpec(
        "Build_Bunker_Screen",
        GoalRequirementKind.STRUCTURE,
        "Bunker",
        minerals=100,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Barracks"),),
    ),
    ProgressActionSpec(
        "Build_MissileTurret_Screen",
        GoalRequirementKind.STRUCTURE,
        "MissileTurret",
        minerals=100,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "EngineeringBay"),),
    ),
    ProgressActionSpec(
        "Train_Marine",
        GoalRequirementKind.UNIT,
        "Marine",
        minerals=50,
        supply=1,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Barracks"),),
    ),
    ProgressActionSpec(
        "Train_SiegeTank",
        GoalRequirementKind.UNIT,
        "SiegeTank",
        minerals=150,
        vespene=125,
        supply=3,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Factory"),),
    ),
    ProgressActionSpec(
        "Train_Marauder",
        GoalRequirementKind.UNIT,
        "Marauder",
        minerals=100,
        vespene=25,
        supply=2,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "BarracksTechLab"),),
    ),
    ProgressActionSpec(
        "Train_Hellion",
        GoalRequirementKind.UNIT,
        "Hellion",
        minerals=100,
        supply=2,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Factory"),),
    ),
    ProgressActionSpec(
        "Train_Medivac",
        GoalRequirementKind.UNIT,
        "Medivac",
        minerals=100,
        vespene=100,
        supply=2,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Starport"),),
    ),
    ProgressActionSpec(
        "Train_VikingFighter",
        GoalRequirementKind.UNIT,
        "VikingFighter",
        minerals=150,
        vespene=75,
        supply=2,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Starport"),),
    ),
    ProgressActionSpec(
        "Research_Stimpack",
        GoalRequirementKind.UPGRADE,
        "Stimpack",
        minerals=100,
        vespene=100,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "BarracksTechLab"),),
    ),
)

_TERRAN_MAPPINGS = tuple(
    MacroActionMapping(_canonical_hima_action(verb, name), (runtime_action,))
    for verb, name, runtime_action in (
        ("TRAIN", "SCV", "Train_SCV"),
        ("TRAIN", "MULE", "Effect_CalldownMULE_Screen"),
        ("BUILD", "OrbitalCommand", "Morph_OrbitalCommand"),
        ("BUILD", "SupplyDepot", "Build_SupplyDepot_Screen"),
        ("BUILD", "Barracks", "Build_Barracks_Screen"),
        ("BUILD", "Refinery", "Build_Refinery_Near"),
        ("BUILD", "CommandCenter", "Build_CommandCenter_Near"),
        ("BUILD", "Factory", "Build_Factory_Screen"),
        ("BUILD", "Starport", "Build_Starport_Screen"),
        ("BUILD", "EngineeringBay", "Build_EngineeringBay_Screen"),
        ("BUILD", "BarracksTechLab", "Build_BarracksTechLab"),
        ("BUILD", "BarracksReactor", "Build_BarracksReactor"),
        ("BUILD", "FactoryTechLab", "Build_FactoryTechLab"),
        ("BUILD", "FactoryReactor", "Build_FactoryReactor"),
        ("BUILD", "StarportTechLab", "Build_StarportTechLab"),
        ("BUILD", "StarportReactor", "Build_StarportReactor"),
        ("BUILD", "Bunker", "Build_Bunker_Screen"),
        ("BUILD", "MissileTurret", "Build_MissileTurret_Screen"),
        ("TRAIN", "Marine", "Train_Marine"),
        ("TRAIN", "Marauder", "Train_Marauder"),
        ("TRAIN", "Hellion", "Train_Hellion"),
        ("TRAIN", "SiegeTank", "Train_SiegeTank"),
        ("TRAIN", "Medivac", "Train_Medivac"),
        ("TRAIN", "VikingFighter", "Train_VikingFighter"),
        ("RESEARCH", "Stimpack", "Research_Stimpack"),
    )
)

TERRAN_PROFILE_DATA = RaceProfileData(
    race=RaceId.TERRAN,
    worker_type="SCV",
    supply_provider="SupplyDepot",
    townhall_types=("CommandCenter", "OrbitalCommand", "PlanetaryFortress"),
    gas_structure="Refinery",
    production_structures=("Barracks", "Factory", "Starport"),
    progress_action_specs=TERRAN_PROGRESS_ACTION_SPECS,
    macro_action_mappings=_TERRAN_MAPPINGS,
    action_domains=_domains(
        TERRAN_PROGRESS_ACTION_SPECS,
        economy={
            "Train_SCV",
            "Morph_OrbitalCommand",
            "Effect_CalldownMULE_Screen",
            "Build_SupplyDepot_Screen",
            "Build_Refinery_Near",
            "Build_CommandCenter_Near",
        },
        technology={
            "Build_EngineeringBay_Screen",
            "Build_BarracksTechLab",
            "Build_BarracksReactor",
            "Build_FactoryTechLab",
            "Build_FactoryReactor",
            "Build_StarportTechLab",
            "Build_StarportReactor",
            "Research_Stimpack",
        },
        defense={"Build_Bunker_Screen", "Build_MissileTurret_Screen"},
    ),
    action_producers={
        **{
            spec.name: ("SCV",)
            for spec in TERRAN_PROGRESS_ACTION_SPECS
            if spec.name.startswith("Build_")
        },
        "Build_BarracksTechLab": ("Barracks",),
        "Build_BarracksReactor": ("Barracks",),
        "Build_FactoryTechLab": ("Factory",),
        "Build_FactoryReactor": ("Factory",),
        "Build_StarportTechLab": ("Starport",),
        "Build_StarportReactor": ("Starport",),
        "Train_Marine": ("Barracks",),
        "Train_Marauder": ("BarracksTechLab",),
        "Train_Hellion": ("Factory",),
        "Train_SiegeTank": ("Factory",),
        "Train_Medivac": ("Starport",),
        "Train_VikingFighter": ("Starport",),
        "Research_Stimpack": ("BarracksTechLab",),
        "Train_SCV": ("CommandCenter", "OrbitalCommand", "PlanetaryFortress"),
        "Morph_OrbitalCommand": ("CommandCenter",),
        "Effect_CalldownMULE_Screen": ("OrbitalCommand",),
    },
    hima_vocabulary_version="hima-terran-69-v1",
    runtime_mapping_ready=True,
    live_worker_ready=True,
    effect_verification_kinds=(
        "build",
        "production",
        "addon",
        "morph",
        "research",
        "ability",
        "move",
    ),
    controller_capabilities=(
        "gas_workers",
        "supply_emergency",
        "resource_fallback",
        "prerequisite_closure",
        "automatic_scv_training",
        "orbital_command_morph",
        "mule_calldown",
    ),
    controller_managed_actions=("Effect_CalldownMULE_Screen",),
    limitations=(),
)


ZERG_PROGRESS_ACTION_SPECS: tuple[ProgressActionSpec, ...] = (
    ProgressActionSpec(
        "Train_Drone",
        GoalRequirementKind.UNIT,
        "Drone",
        minerals=50,
        supply=1,
    ),
    ProgressActionSpec(
        "Train_Overlord",
        GoalRequirementKind.UNIT,
        "Overlord",
        minerals=100,
    ),
    ProgressActionSpec(
        "Build_Hatchery_Near",
        GoalRequirementKind.STRUCTURE,
        "Hatchery",
        minerals=300,
    ),
    ProgressActionSpec(
        "Build_Extractor_Near",
        GoalRequirementKind.STRUCTURE,
        "Extractor",
        minerals=25,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Hatchery"),),
    ),
    ProgressActionSpec(
        "Build_SpawningPool_Screen",
        GoalRequirementKind.STRUCTURE,
        "SpawningPool",
        minerals=200,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Hatchery"),),
    ),
    ProgressActionSpec(
        "Build_RoachWarren_Screen",
        GoalRequirementKind.STRUCTURE,
        "RoachWarren",
        minerals=150,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "SpawningPool"),),
    ),
    ProgressActionSpec(
        "Morph_Lair",
        GoalRequirementKind.STRUCTURE,
        "Lair",
        minerals=150,
        vespene=100,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Hatchery"),
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "SpawningPool"),
        ),
    ),
    ProgressActionSpec(
        "Build_EvolutionChamber_Screen",
        GoalRequirementKind.STRUCTURE,
        "EvolutionChamber",
        minerals=75,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Hatchery"),),
    ),
    ProgressActionSpec(
        "Build_HydraliskDen_Screen",
        GoalRequirementKind.STRUCTURE,
        "HydraliskDen",
        minerals=100,
        vespene=100,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "Lair"),),
    ),
    ProgressActionSpec(
        "Build_SpineCrawler_Screen",
        GoalRequirementKind.STRUCTURE,
        "SpineCrawler",
        minerals=100,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "SpawningPool"),),
    ),
    ProgressActionSpec(
        "Build_SporeCrawler_Screen",
        GoalRequirementKind.STRUCTURE,
        "SporeCrawler",
        minerals=75,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "EvolutionChamber"),),
    ),
    ProgressActionSpec(
        "Train_Queen",
        GoalRequirementKind.UNIT,
        "Queen",
        minerals=150,
        supply=2,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "SpawningPool"),),
    ),
    ProgressActionSpec(
        "Train_Zergling",
        GoalRequirementKind.UNIT,
        "Zergling",
        minerals=50,
        supply=1,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "SpawningPool"),),
    ),
    ProgressActionSpec(
        "Train_Roach",
        GoalRequirementKind.UNIT,
        "Roach",
        minerals=75,
        vespene=25,
        supply=2,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "RoachWarren"),),
    ),
    ProgressActionSpec(
        "Train_Hydralisk",
        GoalRequirementKind.UNIT,
        "Hydralisk",
        minerals=100,
        vespene=50,
        supply=2,
        prerequisites=(StatePrerequisite(GoalRequirementKind.STRUCTURE, "HydraliskDen"),),
    ),
)

_ZERG_MAPPINGS = tuple(
    MacroActionMapping(_canonical_hima_action(verb, name), (runtime_action,))
    for verb, name, runtime_action in (
        ("TRAIN", "Drone", "Train_Drone"),
        ("TRAIN", "Overlord", "Train_Overlord"),
        ("BUILD", "Hatchery", "Build_Hatchery_Near"),
        ("BUILD", "Extractor", "Build_Extractor_Near"),
        ("BUILD", "SpawningPool", "Build_SpawningPool_Screen"),
        ("BUILD", "RoachWarren", "Build_RoachWarren_Screen"),
        ("BUILD", "Lair", "Morph_Lair"),
        ("BUILD", "EvolutionChamber", "Build_EvolutionChamber_Screen"),
        ("BUILD", "HydraliskDen", "Build_HydraliskDen_Screen"),
        ("BUILD", "SpineCrawler", "Build_SpineCrawler_Screen"),
        ("BUILD", "SporeCrawler", "Build_SporeCrawler_Screen"),
        ("TRAIN", "Queen", "Train_Queen"),
        ("TRAIN", "Zergling", "Train_Zergling"),
        ("TRAIN", "Roach", "Train_Roach"),
        ("TRAIN", "Hydralisk", "Train_Hydralisk"),
    )
)

ZERG_PROFILE_DATA = RaceProfileData(
    race=RaceId.ZERG,
    worker_type="Drone",
    supply_provider="Overlord",
    townhall_types=("Hatchery", "Lair", "Hive"),
    gas_structure="Extractor",
    production_structures=("Hatchery", "Lair", "Hive"),
    progress_action_specs=ZERG_PROGRESS_ACTION_SPECS,
    macro_action_mappings=_ZERG_MAPPINGS,
    action_domains=_domains(
        ZERG_PROGRESS_ACTION_SPECS,
        economy={
            "Train_Drone",
            "Train_Overlord",
            "Build_Hatchery_Near",
            "Build_Extractor_Near",
        },
        technology={"Morph_Lair", "Build_EvolutionChamber_Screen"},
        defense={
            "Build_SpineCrawler_Screen",
            "Build_SporeCrawler_Screen",
            "Build_CreepTumor_Tumor_Screen",
        },
    ),
    action_producers={
        **{
            spec.name: ("Drone",)
            for spec in ZERG_PROGRESS_ACTION_SPECS
            if spec.name.startswith("Build_")
        },
        "Morph_Lair": ("Hatchery",),
        "Train_Drone": ("Larva",),
        "Train_Overlord": ("Larva",),
        "Train_Queen": ("Hatchery", "Lair", "Hive"),
        "Train_Zergling": ("Larva",),
        "Train_Roach": ("Larva",),
        "Train_Hydralisk": ("Larva",),
        "Build_CreepTumor_Tumor_Screen": (
            "CreepTumor",
            "CreepTumorBurrowed",
            "CreepTumorQueen",
        ),
    },
    hima_vocabulary_version="hima-zerg-63-v1",
    runtime_mapping_ready=True,
    live_worker_ready=True,
    effect_verification_kinds=("build", "production", "morph", "inject", "move"),
    controller_capabilities=(
        "gas_workers",
        "supply_emergency",
        "resource_fallback",
        "prerequisite_closure",
        "queen_larva_inject",
        "queen_creep_tumor",
        "chained_creep_tumor",
    ),
    limitations=(),
)


@dataclass(frozen=True, slots=True)
class BuiltinRaceProfile:
    data: RaceProfileData

    @property
    def race(self) -> RaceId:
        return self.data.race

    def domain_for_action(self, action_name: str) -> ActionDomain | None:
        return self.data.domain_for_action(action_name)

    def producers_for_action(self, action_name: str) -> tuple[str, ...]:
        return self.data.producers_for_action(action_name)


_RACE_PROFILES = {
    data.race: BuiltinRaceProfile(data)
    for data in (PROTOSS_PROFILE_DATA, TERRAN_PROFILE_DATA, ZERG_PROFILE_DATA)
}


def race_profile(race: RaceId | str) -> BuiltinRaceProfile:
    try:
        race_id = race if isinstance(race, RaceId) else RaceId(race.casefold())
        return _RACE_PROFILES[race_id]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported Cortex race profile: {race}") from error


def built_in_race_profiles() -> tuple[BuiltinRaceProfile, ...]:
    return tuple(_RACE_PROFILES[race] for race in RaceId)
