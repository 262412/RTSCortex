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
    assert service.suppressed_anchors == frozenset({0x101})
    with pytest.raises(RawPlacementFailure, match="permanently suppressed"):
        service.resolve(
            command_id="expand-again",
            action_name="Build_Nexus_Near",
            requested_arguments=(0x101,),
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
