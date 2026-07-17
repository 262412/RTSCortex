from __future__ import annotations

from collections import Counter

from rtscortex.policy.hima import (
    HIMA_PROTOSS_ACTIONS,
    HIMA_UPSTREAM_REVISION,
    HIMA_VOCABULARY_VERSION,
    resolve_hima_action,
)


def test_pinned_protoss_vocabulary_matches_official_sparse_action_ids() -> None:
    expected = [
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
    ]

    assert HIMA_VOCABULARY_VERSION == "hima-protoss-60-v2"
    assert HIMA_UPSTREAM_REVISION == "6b2a4084f9d6c0d7739aedb860685cf9dfd90d35"
    assert [
        (action.upstream_action_id, action.upstream_name, action.category)
        for action in HIMA_PROTOSS_ACTIONS
    ] == expected
    assert [action.upstream_index for action in HIMA_PROTOSS_ACTIONS] == [
        action_id for action_id, _, _ in expected
    ]
    assert Counter(action.category for action in HIMA_PROTOSS_ACTIONS) == {
        "train": 19,
        "build": 15,
        "research": 26,
    }


def test_vocabulary_resolves_canonical_and_official_short_names() -> None:
    expected = {
        "TRAIN PROBE": "TRAIN PROBE",
        "Probe": "TRAIN PROBE",
        "TRAIN ARCHON": "TRAIN ARCHON",
        "Archon": "TRAIN ARCHON",
        "CyberneticsCore": "BUILD CYBERNETICSCORE",
        "WarpGateResearch": "RESEARCH WARPGATERESEARCH",
        "DarkTemplarBlinkUpgrade": "RESEARCH DARKTEMPLARBLINKUPGRADE",
    }

    assert {
        token: resolve_hima_action(token).canonical_action  # type: ignore[union-attr]
        for token in expected
    } == expected
    assert resolve_hima_action("TempestGroundAttackUpgrade") is None


def test_vocabulary_resolves_supported_paper_aliases_explicitly() -> None:
    aliases = {
        "RESEARCH WARPGATE": "RESEARCH WARPGATERESEARCH",
        "RESEARCH AIRWEAPONS_LEVEL1": "RESEARCH PROTOSSAIRWEAPONSLEVEL1",
        "RESEARCH STALKER_BLINK": "RESEARCH BLINKTECH",
        "RESEARCH ZEALOT_CHARGE": "RESEARCH CHARGE",
        "RESEARCH COLOSSUS_RANGE": "RESEARCH EXTENDEDTHERMALLANCE",
        "RESEARCH DARKTEMPLAR_BLINK": "RESEARCH DARKTEMPLARBLINKUPGRADE",
    }

    assert {
        alias: resolve_hima_action(alias).canonical_action  # type: ignore[union-attr]
        for alias in aliases
    } == aliases


def test_vocabulary_only_normalizes_case_and_whitespace() -> None:
    action = resolve_hima_action("  train   probe ")

    assert action is not None
    assert action.canonical_action == "TRAIN PROBE"
    assert resolve_hima_action("BUILD CYBERNETICCORE") is None
    assert resolve_hima_action("BUILD_CYBERNETICSCORE") is None
