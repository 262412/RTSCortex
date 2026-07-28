from __future__ import annotations

from types import SimpleNamespace

import pytest
import rtscortex_llm_pysc2.extractor as extractor_module
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
    assert service.command_target("expand") is placement

    service.quarantine_command(
        command_id="expand",
        action_name="Build_Nexus_Near",
        requested_arguments=(0x101,),
        world_target=None,
    )
    assert service.suppressed_anchors == frozenset(
        0x101 + index for index in range(len(resources))
    )
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

    assert service.candidates(
        observation,
        "Build_Nexus_Near",
    ).argument_candidates == []


def test_raw_placement_service_reports_missing_visible_build_space() -> None:
    service = RawPlacementService(unit_names={})
    observation = SimpleNamespace(feature_screen=None)

    candidates = service.candidates(observation, "Build_Pylon_Screen")

    assert candidates.argument_candidates == []
    assert candidates.unavailable_reason == "out_of_view"
    assert service.placement_alerts == (
        "placement_unavailable:Build_Pylon_Screen:out_of_view",
    )


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
