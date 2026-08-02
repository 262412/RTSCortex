from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import rtscortex_llm_pysc2.extractor as extractor_module
from rtscortex_llm_pysc2.effect_lifecycle import (
    POST_ORDER_EFFECT_GRACE_GAME_LOOPS,
    build_effect_max_lifetime,
)
from rtscortex_llm_pysc2.raw_placement import (
    RawPlacementFailure,
    RawPlacementService,
)


def _unit(
    tag: int,
    unit_type: int,
    *,
    alliance: int,
    x: float,
    y: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        tag=tag,
        unit_type=unit_type,
        alliance=alliance,
        display_type=1,
        build_progress=100,
        x=x,
        y=y,
    )


def test_raw_placement_service_persists_and_quarantines_expansion_identity() -> None:
    service = RawPlacementService(unit_names={59: "Nexus", 341: "MineralField"})
    resources = [
        _unit(0x101 + index, 341, alliance=3, x=x, y=y)
        for index, (x, y) in enumerate(
            (
                (87, 80),
                (85, 85),
                (80, 87),
                (75, 85),
                (73, 80),
                (75, 75),
                (80, 73),
                (85, 75),
            )
        )
    ]
    observation = SimpleNamespace(
        raw_units=[_unit(0xC1, 59, alliance=1, x=20, y=20), *resources],
        feature_units=[],
    )
    service.observe(observation, require_feature_visibility=False)

    placement = service.resolve(
        command_id="expand",
        action_name="Build_Nexus_Near",
        requested_arguments=(0x101,),
        observation=observation,
        world_target=None,
    )
    assert placement.world_target == (80.0, 80.0)
    assert placement.requested_world_target == (80.0, 80.0)
    assert placement.final_validated_world_target == (80.0, 80.0)
    assert service.command_target("expand") is placement

    service.quarantine_command(
        command_id="expand",
        action_name="Build_Nexus_Near",
        requested_arguments=(0x101,),
        world_target=None,
    )
    assert service.suppressed_anchors == frozenset(0x101 + index for index in range(len(resources)))
    with pytest.raises(RawPlacementFailure, match="permanently suppressed"):
        service.resolve(
            command_id="expand-again",
            action_name="Build_Nexus_Near",
            requested_arguments=(0x101,),
            observation=observation,
            world_target=None,
        )
    with pytest.raises(RawPlacementFailure, match="permanently suppressed"):
        service.resolve(
            command_id="same-cluster-alternate-tag",
            action_name="Build_Nexus_Near",
            requested_arguments=(0x104,),
            observation=observation,
            world_target=None,
        )


def test_tag_targeted_build_records_requested_and_validated_world_positions() -> None:
    service = RawPlacementService(unit_names={342: "VespeneGeyser"})
    observation = SimpleNamespace(
        raw_units=[_unit(0xA1, 342, alliance=3, x=60.25, y=65.75)],
        feature_units=[],
    )

    placement = service.resolve(
        command_id="build-gas",
        action_name="Build_Assimilator_Near",
        requested_arguments=(0xA1,),
        observation=observation,
        world_target=None,
    )

    assert placement.requested_world_target == (60.25, 65.75)
    assert placement.final_validated_world_target == (60.25, 65.75)
    assert placement.world_target == (60.25, 65.75)


def test_raw_placement_service_quarantines_pre_dispatch_world_target() -> None:
    service = RawPlacementService(unit_names={})

    service.quarantine_command(
        command_id="build",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=(22.25, 24.5),
    )

    assert service.is_quarantined(
        "Build_Pylon_Screen",
        (22.25, 24.5),
        radius=1.5,
    )


def test_raw_placement_service_retires_expansion_cluster_after_confirmation() -> None:
    service = RawPlacementService(unit_names={59: "Nexus", 341: "MineralField"})
    resources = [
        _unit(0x201 + index, 341, alliance=3, x=x, y=y)
        for index, (x, y) in enumerate(
            (
                (87, 80),
                (85, 85),
                (80, 87),
                (75, 85),
                (73, 80),
                (75, 75),
                (80, 73),
                (85, 75),
            )
        )
    ]
    observation = SimpleNamespace(
        raw_units=[_unit(0xC1, 59, alliance=1, x=20, y=20), *resources],
        feature_units=[],
    )
    service.observe(observation, require_feature_visibility=False)
    service.resolve(
        command_id="confirmed-expand",
        action_name="Build_Nexus_Near",
        requested_arguments=(0x204,),
        observation=observation,
        world_target=None,
    )

    service.confirm_command("confirmed-expand")

    assert (
        service.candidates(
            observation,
            "Build_Nexus_Near",
        ).argument_candidates
        == []
    )


def test_raw_placement_service_reports_missing_visible_build_space() -> None:
    service = RawPlacementService(unit_names={})
    observation = SimpleNamespace(feature_screen=None)

    candidates = service.candidates(observation, "Build_Pylon_Screen")

    assert candidates.argument_candidates == []
    assert candidates.unavailable_reason == "out_of_view"
    assert service.placement_alerts == ("placement_unavailable:Build_Pylon_Screen:out_of_view",)


def test_raw_placement_service_rejects_dispatch_time_relocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RawPlacementService(unit_names={})
    observation = SimpleNamespace(feature_screen=object())
    monkeypatch.setattr(
        extractor_module,
        "resolve_screen_build_world_target",
        lambda *args, **kwargs: [70, 70],
    )
    monkeypatch.setattr(
        extractor_module,
        "screen_to_world_target",
        lambda *args, **kwargs: SimpleNamespace(world_target=(30.0, 30.0)),
    )

    with pytest.raises(RawPlacementFailure, match="no longer legal"):
        service.resolve(
            command_id="stale-build",
            action_name="Build_Pylon_Screen",
            requested_arguments=([64, 64],),
            observation=observation,
            world_target=(22.25, 24.5),
            preferred_anchor_tag=0xB1,
            builder_tags=(0xB1,),
        )


def test_builder_lease_is_exact_and_released_at_terminal() -> None:
    service = RawPlacementService(unit_names={2: "Probe"})
    observation = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, alliance=1, x=20, y=20)],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )

    first = service.resolve(
        command_id="build-one",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(22.25, 24.5),
        builder_tag=0xB1,
        ability_name="Build_Pylon_pt",
    )

    assert first.world_target == (22.0, 24.0)
    assert service.leased_builder_tags == frozenset({0xB1})
    with pytest.raises(RawPlacementFailure, match="leased"):
        service.resolve(
            command_id="build-two",
            action_name="Build_Pylon_Screen",
            requested_arguments=([66, 66],),
            observation=observation,
            world_target=(24.0, 26.0),
            builder_tag=0xB1,
            ability_name="Build_Pylon_pt",
        )

    service.release_command("build-one")

    assert service.leased_builder_tags == frozenset()


def test_placement_transition_is_durable_before_command_terminal() -> None:
    durable_events: list[dict[str, object]] = []
    service = RawPlacementService(
        unit_names={2: "Probe"},
        transition_sink=durable_events.append,
    )
    service.set_runtime_context(
        run_id="run",
        episode_id="episode",
        step_id=7,
        game_loop=100,
    )
    observation = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, alliance=1, x=20, y=20)],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )

    reservation = service.resolve(
        command_id="build-durable",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(22.0, 24.0),
        builder_tag=0xB1,
    )

    assert len(durable_events) == 1
    assert durable_events[0]["command_id"] == "build-durable"
    assert durable_events[0]["transition"] == {
        "reservation_id": reservation.reservation_id,
        "structure_type": "Pylon",
        "footprint_cells": [(21, 23), (21, 24), (22, 23), (22, 24)],
        "previous_state": "unreserved",
        "next_state": "reserved",
        "failure_class": None,
        "actor_failure": False,
        "game_loop": 100,
        "release_reason": None,
    }
    assert service.drain_transition_history("build-durable") == []


def test_placement_transition_game_loops_are_monotonic() -> None:
    durable_events: list[dict[str, object]] = []
    service = RawPlacementService(
        unit_names={2: "Probe"},
        transition_sink=durable_events.append,
    )
    service.set_runtime_context(
        run_id="run",
        episode_id="episode",
        step_id=7,
        game_loop=100,
    )
    observation = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, alliance=1, x=20, y=20)],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )
    service.resolve(
        command_id="build-monotonic",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(22.0, 24.0),
        builder_tag=0xB1,
    )
    service.confirm_command("build-monotonic", game_loop=112)
    service.release_command("build-monotonic")

    loops = [int(event["transition"]["game_loop"]) for event in durable_events]  # type: ignore[index]
    assert loops == sorted(loops)
    assert loops == [100, 112, 112]


def test_permanent_spatial_exclusion_applies_across_building_types() -> None:
    service = RawPlacementService(unit_names={})
    service.suppress_world_target(
        "Build_Pylon_Screen",
        (22.0, 24.0),
        reason="not_pathable",
    )

    assert service.is_quarantined(
        "Build_Gateway_Screen",
        (22.0, 24.0),
        radius=2.0,
    )


def test_cross_structure_footprints_cannot_overlap() -> None:
    service = RawPlacementService(unit_names={})
    observation = SimpleNamespace(
        raw_units=[],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )
    service.resolve(
        command_id="gateway-a",
        action_name="Build_Gateway_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(22.0, 22.0),
    )

    assert service.is_quarantined(
        "Build_Gateway_Screen",
        (24.0, 24.0),
    )


def test_terran_producer_ledger_reserves_addon_cells() -> None:
    service = RawPlacementService(unit_names={})
    observation = SimpleNamespace(
        raw_units=[],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )

    reservation = service.resolve(
        command_id="barracks-a",
        action_name="Build_Barracks_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(22.0, 22.0),
    )

    assert reservation.footprint_width == 5
    assert reservation.footprint_height == 3
    assert len(reservation.occupied_grid_cells) == 15
    assert {(24, 21), (25, 21), (24, 22), (25, 22), (24, 23), (25, 23)} <= set(
        reservation.occupied_grid_cells
    )
    assert service.is_quarantined(
        "Build_SupplyDepot_Screen",
        (25.0, 22.0),
    )


def test_actor_failure_does_not_quarantine_placement() -> None:
    service = RawPlacementService(unit_names={})
    service.quarantine_command(
        command_id="missing-builder",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=(22.0, 24.0),
        failure_code="builder_unavailable",
        game_loop=100,
    )

    assert not service.is_quarantined(
        "Build_Pylon_Screen",
        (22.0, 24.0),
    )
    transition = service.drain_transition_history("missing-builder")
    assert transition[0]["next_state"] == "released"
    assert transition[0]["failure_class"] == "nonspatial"
    assert transition[0]["actor_failure"] is True


def test_permanent_spatial_failure_emits_exact_footprint_transition() -> None:
    service = RawPlacementService(unit_names={})
    service.quarantine_command(
        command_id="blocked-pylon",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=(22.0, 24.0),
        failure_code="not_pathable",
        game_loop=100,
    )

    transition = service.drain_transition_history("blocked-pylon")

    assert transition == [
        {
            "reservation_id": transition[0]["reservation_id"],
            "structure_type": "Pylon",
            "footprint_cells": [(21, 23), (21, 24), (22, 23), (22, 24)],
            "previous_state": "unreserved",
            "next_state": "permanent_invalid",
            "failure_class": "spatial_permanent",
            "actor_failure": False,
            "game_loop": 100,
            "release_reason": "not_pathable",
        }
    ]


def test_raw_build_eligibility_is_shared_by_candidates_and_revalidation() -> None:
    service = RawPlacementService(unit_names={2: "Probe", 72: "CyberneticsCore"})
    probe = _unit(0xB1, 2, alliance=1, x=20, y=20)
    missing_tech = SimpleNamespace(
        raw_units=[probe],
        feature_units=[],
        feature_screen=None,
        player_common=SimpleNamespace(minerals=500, vespene=500),
        game_loop=[100],
    )

    candidates = service.candidates(missing_tech, "Build_ShieldBattery_Screen")

    assert candidates.argument_candidates == []
    assert candidates.unavailable_reason == "build_missing_prerequisite"
    with pytest.raises(RawPlacementFailure) as missing:
        service.resolve(
            command_id="battery-missing-tech",
            action_name="Build_ShieldBattery_Screen",
            requested_arguments=([64, 64],),
            observation=missing_tech,
            world_target=(22.0, 24.0),
            builder_tag=0xB1,
        )
    assert missing.value.code == "build_missing_prerequisite"

    insufficient = SimpleNamespace(
        **{
            **vars(missing_tech),
            "player_common": SimpleNamespace(minerals=50, vespene=500),
        }
    )
    assert (
        service.candidates(insufficient, "Build_ShieldBattery_Screen").unavailable_reason
        == "build_insufficient_minerals"
    )

    completed_core = _unit(0xC1, 72, alliance=1, x=25, y=25)
    eligible = SimpleNamespace(**{**vars(missing_tech), "raw_units": [probe, completed_core]})
    assert (
        service.candidates(eligible, "Build_ShieldBattery_Screen").unavailable_reason
        == "out_of_view"
    )

    # A candidate may have been selected while the Core was complete; final
    # resolve must reject if that prerequisite disappears before dispatch.
    with pytest.raises(RawPlacementFailure) as stale:
        service.resolve(
            command_id="battery-stale-tech",
            action_name="Build_ShieldBattery_Screen",
            requested_arguments=([64, 64],),
            observation=missing_tech,
            world_target=(22.0, 24.0),
            builder_tag=0xB1,
        )
    assert stale.value.code == "build_missing_prerequisite"
    assert not service.is_quarantined("Build_ShieldBattery_Screen", (22.0, 24.0))


@pytest.mark.parametrize(
    ("action_name", "expected_lifetime"),
    [
        ("Build_Pylon_Screen", 448),
        ("Build_Nexus_Near", 1344),
    ],
)
def test_default_reservation_uses_the_effect_lifecycle_contract(
    action_name: str,
    expected_lifetime: int,
) -> None:
    service = RawPlacementService(unit_names={})
    observation = SimpleNamespace(
        raw_units=[],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )
    if action_name == "Build_Nexus_Near":
        service._known_resources[0x101] = {  # noqa: SLF001 - focused ownership contract
            "tag": 0x101,
            "unit_type": "MineralField",
            "alliance": 3,
            "x": 80.0,
            "y": 80.0,
            "display_type": 1,
        }
        # A compact synthetic cluster is enough for a deterministic expansion target.
        for index, point in enumerate(((87, 80), (85, 85), (80, 87), (75, 85), (73, 80))):
            service._known_resources[0x102 + index] = {  # noqa: SLF001
                "tag": 0x102 + index,
                "unit_type": "MineralField",
                "alliance": 3,
                "x": float(point[0]),
                "y": float(point[1]),
                "display_type": 1,
            }
        arguments: tuple[Any, ...] = (0x101,)
        target = None
    else:
        arguments = ([64, 64],)
        target = (22.0, 24.0)

    reservation = service.resolve(
        command_id=f"lifetime-{action_name}",
        action_name=action_name,
        requested_arguments=arguments,
        observation=observation,
        world_target=target,
    )

    assert build_effect_max_lifetime(112, action_name) == expected_lifetime
    assert reservation.expires_game_loop == (
        100 + expected_lifetime + POST_ORDER_EFFECT_GRACE_GAME_LOOPS
    )
