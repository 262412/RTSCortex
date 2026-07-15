"""Pinned Protoss macro-action vocabulary from official HIMA constants."""

from __future__ import annotations

from collections.abc import Iterable

from rtscortex.policy.hima.models import HIMA_VOCABULARY_VERSION, HIMAMacroAction

# Exact ``ACTION_DICT[Race.Protoss]`` entries at HIMA revision 6b2a408.  The
# sparse IDs are part of the upstream contract and must not be renumbered.
_UPSTREAM_ACTIONS: tuple[tuple[int, str, str], ...] = (
    (100, "Probe", "train"),
    (101, "Zealot", "train"),
    (102, "Sentry", "train"),
    (103, "Stalker", "train"),
    (104, "Adept", "train"),
    (105, "HighTemplar", "train"),
    (106, "DarkTemplar", "train"),
    (107, "Observer", "train"),
    (108, "WarpPrism", "train"),
    (109, "Immortal", "train"),
    (110, "Colossus", "train"),
    (111, "Disruptor", "train"),
    (112, "Phoenix", "train"),
    (113, "VoidRay", "train"),
    (114, "Oracle", "train"),
    (115, "Carrier", "train"),
    (116, "Tempest", "train"),
    (117, "Mothership", "train"),
    (118, "Archon", "train"),
    (200, "Nexus", "build"),
    (201, "Assimilator", "build"),
    (202, "Gateway", "build"),
    (203, "RoboticsFacility", "build"),
    (204, "Stargate", "build"),
    (205, "Pylon", "build"),
    (206, "Forge", "build"),
    (207, "CyberneticsCore", "build"),
    (208, "PhotonCannon", "build"),
    (209, "ShieldBattery", "build"),
    (210, "TwilightCouncil", "build"),
    (211, "TemplarArchive", "build"),
    (212, "DarkShrine", "build"),
    (213, "FleetBeacon", "build"),
    (214, "RoboticsBay", "build"),
    (300, "ProtossGroundWeaponsLevel1", "research"),
    (301, "ProtossGroundWeaponsLevel2", "research"),
    (302, "ProtossGroundWeaponsLevel3", "research"),
    (303, "ProtossGroundArmorsLevel1", "research"),
    (304, "ProtossGroundArmorsLevel2", "research"),
    (305, "ProtossGroundArmorsLevel3", "research"),
    (306, "ProtossShieldsLevel1", "research"),
    (307, "ProtossShieldsLevel2", "research"),
    (308, "ProtossShieldsLevel3", "research"),
    (309, "ProtossAirWeaponsLevel1", "research"),
    (310, "ProtossAirWeaponsLevel2", "research"),
    (311, "ProtossAirWeaponsLevel3", "research"),
    (312, "ProtossAirArmorsLevel1", "research"),
    (313, "ProtossAirArmorsLevel2", "research"),
    (314, "ProtossAirArmorsLevel3", "research"),
    (315, "WarpGateResearch", "research"),
    (316, "Charge", "research"),
    (317, "BlinkTech", "research"),
    (318, "AdeptPiercingAttack", "research"),
    (319, "PsiStormTech", "research"),
    (320, "DarkTemplarBlinkUpgrade", "research"),
    (321, "ObserverGraviticBooster", "research"),
    (322, "GraviticDrive", "research"),
    (323, "ExtendedThermalLance", "research"),
    (324, "PhoenixRangeUpgrade", "research"),
    (325, "VoidRaySpeedUpgrade", "research"),
)

# Older HIMA prompt examples use these explicit long-form tokens.  They remain
# accepted aliases, but never alter the pinned IDs or official short names.
_PAPER_ALIASES: dict[str, tuple[str, ...]] = {
    "WarpGateResearch": ("RESEARCH WARPGATE", "RESEARCH WARPGATE_RESEARCH"),
    "ProtossAirWeaponsLevel1": ("RESEARCH AIRWEAPONS_LEVEL1",),
    "ProtossAirWeaponsLevel2": ("RESEARCH AIRWEAPONS_LEVEL2",),
    "ProtossAirWeaponsLevel3": ("RESEARCH AIRWEAPONS_LEVEL3",),
    "ProtossAirArmorsLevel1": (
        "RESEARCH AIRARMOR_LEVEL1",
        "RESEARCH AIRARMORS_LEVEL1",
    ),
    "ProtossAirArmorsLevel2": (
        "RESEARCH AIRARMOR_LEVEL2",
        "RESEARCH AIRARMORS_LEVEL2",
    ),
    "ProtossAirArmorsLevel3": (
        "RESEARCH AIRARMOR_LEVEL3",
        "RESEARCH AIRARMORS_LEVEL3",
    ),
    "AdeptPiercingAttack": (
        "RESEARCH ADEPT_GLAIVES",
        "RESEARCH ADEPT_RESONATING_GLAIVES",
    ),
    "BlinkTech": ("RESEARCH STALKER_BLINK",),
    "Charge": ("RESEARCH ZEALOT_CHARGE",),
    "ProtossGroundWeaponsLevel1": ("RESEARCH GROUNDWEAPONS_LEVEL1",),
    "ProtossGroundWeaponsLevel2": ("RESEARCH GROUNDWEAPONS_LEVEL2",),
    "ProtossGroundWeaponsLevel3": ("RESEARCH GROUNDWEAPONS_LEVEL3",),
    "ProtossGroundArmorsLevel1": (
        "RESEARCH GROUNDARMOR_LEVEL1",
        "RESEARCH GROUNDARMORS_LEVEL1",
    ),
    "ProtossGroundArmorsLevel2": (
        "RESEARCH GROUNDARMOR_LEVEL2",
        "RESEARCH GROUNDARMORS_LEVEL2",
    ),
    "ProtossGroundArmorsLevel3": (
        "RESEARCH GROUNDARMOR_LEVEL3",
        "RESEARCH GROUNDARMORS_LEVEL3",
    ),
    "ProtossShieldsLevel1": ("RESEARCH SHIELDS_LEVEL1",),
    "ProtossShieldsLevel2": ("RESEARCH SHIELDS_LEVEL2",),
    "ProtossShieldsLevel3": ("RESEARCH SHIELDS_LEVEL3",),
    "ExtendedThermalLance": (
        "RESEARCH COLOSSUS_RANGE",
        "RESEARCH COLOSSUS_EXTENDED_THERMAL_LANCE",
    ),
    "GraviticDrive": (
        "RESEARCH WARPPRISM_SPEED",
        "RESEARCH WARPPRISM_GRAVITIC_DRIVE",
    ),
    "ObserverGraviticBooster": (
        "RESEARCH OBSERVER_SPEED",
        "RESEARCH OBSERVER_GRAVITIC_BOOSTERS",
    ),
    "PsiStormTech": (
        "RESEARCH PSISTORM",
        "RESEARCH HIGHTEMPLAR_PSISTORM",
    ),
    "DarkTemplarBlinkUpgrade": (
        "RESEARCH DARKTEMPLAR_BLINK",
        "RESEARCH DARKTEMPLAR_BLINK_UPGRADE",
    ),
    "VoidRaySpeedUpgrade": (
        "RESEARCH VOIDRAY_SPEED",
        "RESEARCH VOIDRAY_SPEED_UPGRADE",
    ),
    "PhoenixRangeUpgrade": (
        "RESEARCH PHOENIX_RANGE",
        "RESEARCH PHOENIX_RANGE_UPGRADE",
    ),
}


def _canonical_action(name: str, category: str) -> str:
    verb = {"train": "TRAIN", "build": "BUILD", "research": "RESEARCH"}[category]
    return f"{verb} {name.upper()}"


def _make_actions() -> tuple[HIMAMacroAction, ...]:
    return tuple(
        HIMAMacroAction(
            upstream_action_id=action_id,
            upstream_name=name,
            canonical_action=_canonical_action(name, category),
            category=category,  # type: ignore[arg-type]
            aliases=(name, *_PAPER_ALIASES.get(name, ())),
        )
        for action_id, name, category in _UPSTREAM_ACTIONS
    )


HIMA_PROTOSS_ACTIONS = _make_actions()


def _lookup_key(value: str) -> str:
    return " ".join(value.strip().upper().split())


def _build_lookup(actions: Iterable[HIMAMacroAction]) -> dict[str, HIMAMacroAction]:
    lookup: dict[str, HIMAMacroAction] = {}
    for action in actions:
        for token in (action.canonical_action, *action.aliases):
            key = _lookup_key(token)
            if key in lookup:
                raise RuntimeError(f"duplicate HIMA macro-action alias: {token}")
            lookup[key] = action
    return lookup


_ACTION_LOOKUP = _build_lookup(HIMA_PROTOSS_ACTIONS)


def resolve_hima_action(value: str) -> HIMAMacroAction | None:
    """Resolve one exact official token or explicitly supported prompt alias."""

    return _ACTION_LOOKUP.get(_lookup_key(value))


def hima_vocabulary_version() -> str:
    """Return the pinned vocabulary version for artifact provenance."""

    return HIMA_VOCABULARY_VERSION
