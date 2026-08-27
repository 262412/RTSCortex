"""Public race profiles for the Strategic Cortex."""

from rtscortex.races.models import (
    ActionDomain,
    CombatTargetDomain,
    DefenseDoctrine,
    MacroActionMapping,
    RaceId,
    RaceProfile,
    RaceProfileData,
)
from rtscortex.races.profiles import (
    PROTOSS_PROFILE_DATA,
    TERRAN_PROFILE_DATA,
    TERRAN_PROGRESS_ACTION_SPECS,
    ZERG_PROFILE_DATA,
    ZERG_PROGRESS_ACTION_SPECS,
    BuiltinRaceProfile,
    built_in_race_profiles,
    combat_target_domain,
    is_flying_unit,
    race_profile,
)

__all__ = [
    "ActionDomain",
    "BuiltinRaceProfile",
    "CombatTargetDomain",
    "DefenseDoctrine",
    "MacroActionMapping",
    "PROTOSS_PROFILE_DATA",
    "RaceId",
    "RaceProfile",
    "RaceProfileData",
    "TERRAN_PROFILE_DATA",
    "TERRAN_PROGRESS_ACTION_SPECS",
    "ZERG_PROFILE_DATA",
    "ZERG_PROGRESS_ACTION_SPECS",
    "built_in_race_profiles",
    "combat_target_domain",
    "is_flying_unit",
    "race_profile",
]
