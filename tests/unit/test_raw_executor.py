from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rtscortex_llm_pysc2.coordinator import BridgeDecision
from rtscortex_llm_pysc2.raw_executor import RawActionExecutor
from rtscortex_llm_pysc2.raw_placement import (
    RawPlacementService,
    _placement_candidate_id,
    _placement_revision,
)
from rtscortex_llm_pysc2.routing import RoutedActionBatch, RoutedCommand

pytest.importorskip("pysc2.lib.actions")


class _Broker:
    def __init__(self) -> None:
        self.settled: list[tuple[str, bool]] = []
        self.rejected_world_targets: list[tuple[str, tuple[float, float]]] = []
        self.extractor = SimpleNamespace(
            suppress_expansion_anchor=lambda *args, **kwargs: None,
            suppress_build_world_target=lambda action, target: self.rejected_world_targets.append(
                (action, tuple(target))
            ),
        )

    def settle_primitive(self, dispatch: Any, *, success: bool, **_: Any) -> None:
        self.settled.append((dispatch.command_id, success))


def _decision(command: RoutedCommand) -> BridgeDecision:
    route = RoutedActionBatch(
        protocol_version="1.1",
        run_id="run-raw",
        episode_id="episode-raw",
        step_id=1,
        decision_id="decision-raw",
        agent_name=command.actor.split("/", 1)[0],
        team_order=(command.team_name,),
        commands=(command,),
        action_text="Actions:",
    )
    return BridgeDecision(
        observation={},
        action_batch={"commands": [{"command_id": command.command_id}]},
        routes={route.agent_name: route},
    )


def _agent(team_name: str, tags: list[int]) -> Any:
    return SimpleNamespace(teams=[{"name": team_name, "unit_tags": tags}])


def _unit(
    tag: int,
    unit_type: int,
    *,
    alliance: int = 1,
    order_length: int = 0,
    x: float = 20,
    y: float = 20,
    radius: float = 0.5,
    active: int = 0,
    order_id_0: int = 0,
    display_type: int = 1,
) -> Any:
    return SimpleNamespace(
        tag=tag,
        unit_type=unit_type,
        alliance=alliance,
        build_progress=100,
        order_length=order_length,
        order_id_0=order_id_0,
        active=active,
        x=x,
        y=y,
        radius=radius,
        health=100,
        display_type=display_type,
    )


@pytest.mark.parametrize(
    ("command", "agents", "raw_units", "expected_function", "expected_tags"),
    [
        (
            RoutedCommand(
                command_id="move",
                actor="CombatGroup7/Adept-1",
                team_name="Adept-1",
                name="Move_Minimap",
                rendered_action="",
                requested_arguments=([48, 16],),
            ),
            {"CombatGroup7": _agent("Adept-1", [0xA1, 0xA2])},
            [_unit(0xA1, 1), _unit(0xA2, 1)],
            547,
            [0xA1, 0xA2],
        ),
        (
            RoutedCommand(
                command_id="attack",
                actor="CombatGroup7/Adept-1",
                team_name="Adept-1",
                name="Attack_Unit",
                rendered_action="",
                requested_arguments=(0xE1,),
            ),
            {"CombatGroup7": _agent("Adept-1", [0xA1])},
            [_unit(0xA1, 1), _unit(0xE1, 9, alliance=4)],
            5,
            [0xA1],
        ),
        (
            RoutedCommand(
                command_id="build",
                actor="Builder/Builder-Probe-1",
                team_name="Builder-Probe-1",
                name="Build_Pylon_Screen",
                rendered_action="",
                requested_arguments=([65, 65],),
                screen_world_target=(22.0, 24.0),
                screen_anchor_tag=0xB1,
            ),
            {"Builder": _agent("Builder-Probe-1", [0xB1])},
            [_unit(0xB1, 2)],
            35,
            [0xB1],
        ),
        (
            RoutedCommand(
                command_id="train",
                actor="Developer/Empty",
                team_name="Empty",
                name="Train_Adept",
                rendered_action="",
            ),
            {"Developer": _agent("Empty", [])},
            [_unit(0xD1, 3), _unit(0xC1, 4)],
            54,
            [0xD1],
        ),
        (
            RoutedCommand(
                command_id="research",
                actor="Developer/Empty",
                team_name="Empty",
                name="Research_WarpGate",
                rendered_action="",
            ),
            {"Developer": _agent("Empty", [])},
            [_unit(0xC1, 4)],
            82,
            [0xC1],
        ),
    ],
)
def test_raw_executor_binds_exact_tags_and_one_final_primitive(
    command: RoutedCommand,
    agents: dict[str, Any],
    raw_units: list[Any],
    expected_function: int,
    expected_tags: list[int],
) -> None:
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={1: "Adept", 2: "Probe", 3: "Gateway", 4: "CyberneticsCore"},
    )
    if command.name == "Build_Pylon_Screen":
        placement_revision = _placement_revision(
            SimpleNamespace(raw_units=raw_units, game_loop=[100])
        )
        command = replace(
            command,
            placement_candidate_id=_placement_candidate_id(
                command.name,
                (22.0, 24.0),
                0xB1,
                placement_revision,
                2,
                False,
            ),
            placement_revision=placement_revision,
        )
    executor.enqueue(_decision(command))

    dispatch = executor.next_dispatch(
        SimpleNamespace(
            raw_units=raw_units,
            game_loop=[100],
            player_common=SimpleNamespace(
                minerals=500,
                vespene=500,
                food_used=0,
                food_cap=20,
            ),
        ),
        agents,
    )

    assert dispatch is not None
    assert int(dispatch.action.function) == expected_function
    assert dispatch.primitive.final_primitive is True
    assert dispatch.primitive.ordinal == 0
    assert dispatch.primitive.total == 1
    if dispatch.actor_tags:
        assert list(dispatch.actor_tags) == expected_tags
    else:
        assert dispatch.producer_tag == expected_tags[0]
    assert broker.settled == []


def test_raw_executor_rejects_missing_actor_without_emitting_action() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={1: "Adept"})
    command = RoutedCommand(
        command_id="missing-actor",
        actor="CombatGroup7/Adept-1",
        team_name="Adept-1",
        name="Move_Minimap",
        rendered_action="",
        requested_arguments=([48, 16],),
    )
    executor.enqueue(_decision(command))

    assert (
        executor.next_dispatch(
            SimpleNamespace(raw_units=[], game_loop=[100]),
            {"CombatGroup7": _agent("Adept-1", [0xA1])},
        )
        is None
    )
    assert broker.settled == [("missing-actor", False)]


def test_raw_executor_revalidates_completed_adept_prerequisite() -> None:
    command = RoutedCommand(
        command_id="adept-final-contract",
        actor="Developer/Empty",
        team_name="Empty",
        name="Train_Adept",
        rendered_action="",
    )
    player = SimpleNamespace(
        minerals=500,
        vespene=500,
        food_used=10,
        food_cap=20,
    )
    gateway = _unit(0xD1, 3)
    core = _unit(0xC1, 4)
    core.build_progress = 1
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={3: "Gateway", 4: "CyberneticsCore"},
    )
    executor.enqueue(_decision(command))

    assert (
        executor.next_dispatch(
            SimpleNamespace(
                raw_units=[gateway, core],
                player_common=player,
                game_loop=[100],
            ),
            {"Developer": _agent("Empty", [])},
        )
        is None
    )
    assert broker.settled == [("adept-final-contract", False)]

    core.build_progress = 100
    completed_observation = SimpleNamespace(
        raw_units=[gateway, core],
        player_common=player,
        available_actions=[],
        game_loop=[101],
    )
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={3: "Gateway", 4: "CyberneticsCore"},
    )
    executor.enqueue(_decision(command))
    assert (
        executor.next_dispatch(
            completed_observation,
            {"Developer": _agent("Empty", [])},
        )
        is None
    )
    assert broker.settled == [("adept-final-contract", False)]

    completed_observation.available_actions = [54]
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={3: "Gateway", 4: "CyberneticsCore"},
    )
    executor.enqueue(_decision(command))
    dispatch = executor.next_dispatch(
        completed_observation,
        {"Developer": _agent("Empty", [])},
    )

    assert dispatch is not None
    assert dispatch.producer_tag == 0xD1


def test_raw_executor_does_not_quarantine_build_when_builder_is_missing() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={2: "Probe"})
    command = RoutedCommand(
        command_id="missing-builder",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([65, 65],),
        screen_world_target=(22.25, 24.5),
        screen_anchor_tag=0xB1,
    )
    executor.enqueue(_decision(command))

    assert (
        executor.next_dispatch(
            SimpleNamespace(raw_units=[], game_loop=[100]),
            {"Builder": _agent("Builder-Probe-1", [0xB1])},
        )
        is None
    )
    assert broker.settled == []
    assert executor.placement_service.quarantined_targets == {}
    assert executor.placement_service.active_reservation_count == 0


def test_raw_executor_defers_build_until_a_fresh_builder_is_available() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={2: "Probe"})
    observation = SimpleNamespace(raw_units=[], game_loop=[100])
    command = RoutedCommand(
        command_id="stale-builder",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([65, 65],),
        operation_id="operation:stale-builder",
        screen_world_target=(22.25, 24.5),
        screen_anchor_tag=0xB1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            (22.25, 24.5),
            0xB1,
            _placement_revision(observation),
            2,
            False,
        ),
        placement_revision=_placement_revision(observation),
    )
    executor.enqueue(_decision(command))

    assert (
        executor.next_dispatch(
            observation,
            {"Builder": _agent("Builder-Probe-1", [0xB1])},
        )
        is None
    )
    assert executor.queued_count == 1
    assert broker.settled == []
    assert executor.placement_service.quarantined_targets == {}
    assert executor.diagnostic_snapshot["failure_code"] == "builder_not_ready"
    assert executor.diagnostic_snapshot["operation_id"] == "operation:stale-builder"

    fresh_observation = SimpleNamespace(
        raw_units=[_unit(0xB2, 2, x=20, y=20)],
        game_loop=[101],
    )
    dispatch = executor.next_dispatch(
        fresh_observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    assert dispatch is None
    assert executor.queued_count == 0
    assert broker.settled == [("stale-builder", False)]

    fresh_revision = _placement_revision(fresh_observation)
    revalidated = replace(
        command,
        command_id="fresh-builder-revalidated",
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            (22.25, 24.5),
            0xB1,
            fresh_revision,
            2,
            False,
        ),
        placement_revision=fresh_revision,
    )
    executor.enqueue(_decision(revalidated))
    dispatch = executor.next_dispatch(
        fresh_observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    assert dispatch is not None
    assert dispatch.builder_tag == 0xB2
    assert dispatch.actor_tags == (0xB2,)
    assert broker.settled == [("stale-builder", False)]


@pytest.mark.parametrize(("include_ready_probe", "expected_builder"), [(False, None), (True, 0xB2)])
def test_gateway_waits_for_pylon_builder_order_to_clear_or_rebinds(
    include_ready_probe: bool,
    expected_builder: int | None,
) -> None:
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={2: "Probe", 60: "Pylon"},
    )
    raw_units = [
        _unit(0xB1, 2, order_length=1, order_id_0=35, x=20, y=20),
        _unit(0xC1, 60, x=16, y=16),
    ]
    if include_ready_probe:
        raw_units.append(_unit(0xB2, 2, x=21, y=20))
    observation = SimpleNamespace(
        raw_units=raw_units,
        feature_units=[],
        game_loop=[100],
        player_common=SimpleNamespace(minerals=500, vespene=0, food_used=0, food_cap=20),
    )
    placement_revision = _placement_revision(observation)
    target = (25.0, 25.0)
    command = RoutedCommand(
        command_id=f"gateway-after-pylon-{include_ready_probe}",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Gateway_Screen",
        rendered_action="",
        requested_arguments=([65, 65],),
        screen_world_target=target,
        screen_anchor_tag=0xC1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Gateway_Screen",
            target,
            0xC1,
            placement_revision,
            3,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(command))

    dispatch = executor.next_dispatch(
        observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    if expected_builder is None:
        assert dispatch is None
        assert executor.queued_count == 1
        assert executor.effect_inflight_count == 0
        assert executor.diagnostic_snapshot["failure_code"] == "builder_not_ready"
        assert executor.diagnostic_snapshot["builder_state"][0]["orders"] == [35]
        assert broker.settled == []
        assert executor.placement_service.quarantined_targets == {}
    else:
        assert dispatch is not None
        assert dispatch.builder_tag == expected_builder
        assert dispatch.actor_tags == (expected_builder,)


def test_interruptible_harvest_order_does_not_block_builder_dispatch() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={2: "Probe"})
    observation = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, order_length=1, order_id_0=3666, x=20, y=20)],
        feature_units=[],
        game_loop=[100],
        player_common=SimpleNamespace(minerals=500, vespene=0, food_used=0, food_cap=20),
    )
    placement_revision = _placement_revision(observation)
    target = (24.0, 24.0)
    command = RoutedCommand(
        command_id="harvest-probe-build",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([60, 60],),
        screen_world_target=target,
        screen_anchor_tag=0xB1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            target,
            0xB1,
            placement_revision,
            2,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(command))

    dispatch = executor.next_dispatch(
        observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    assert dispatch is not None
    assert dispatch.builder_tag == 0xB1


def test_snapshot_only_builder_is_not_final_dispatch_ready() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={2: "Probe"})
    observation = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, display_type=2)],
        feature_units=[],
        game_loop=[100],
    )
    placement_revision = _placement_revision(observation)
    target = (24.0, 24.0)
    command = RoutedCommand(
        command_id="snapshot-builder",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([60, 60],),
        screen_world_target=target,
        screen_anchor_tag=0xB1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            target,
            0xB1,
            placement_revision,
            2,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(command))

    assert (
        executor.next_dispatch(
            observation,
            {"Builder": _agent("Builder-Probe-1", [0xB1])},
        )
        is None
    )
    assert executor.queued_count == 1
    assert broker.settled == []
    assert executor.diagnostic_snapshot["failure_code"] == "builder_not_ready"
    assert executor.diagnostic_snapshot["builder_state"][0]["reason"] == "not_visible"
    assert executor.diagnostic_snapshot["observation_revision"] == placement_revision


@pytest.mark.parametrize(("include_safe_probe", "expected_builder"), [(False, None), (True, 0xB2)])
def test_builder_under_immediate_enemy_threat_defers_or_rebinds(
    include_safe_probe: bool,
    expected_builder: int | None,
) -> None:
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={2: "Probe", 9: "Zergling"},
    )
    raw_units = [
        _unit(0xB1, 2, x=20, y=20),
        _unit(0xE1, 9, alliance=4, x=22, y=20, radius=0.375),
    ]
    feature_units: list[Any] = []
    if include_safe_probe:
        raw_units.append(_unit(0xB2, 2, x=16, y=28))
        feature_units.append(SimpleNamespace(tag=0xB2, alliance=1, is_on_screen=True))
    observation = SimpleNamespace(
        raw_units=raw_units,
        feature_units=feature_units,
        game_loop=[100],
    )
    placement_revision = _placement_revision(observation)
    target = (30.0, 30.0)
    command = RoutedCommand(
        command_id=f"threatened-builder-{include_safe_probe}",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([60, 60],),
        screen_world_target=target,
        screen_anchor_tag=0xB1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            target,
            0xB1,
            placement_revision,
            2,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(command))

    dispatch = executor.next_dispatch(
        observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    if expected_builder is None:
        assert dispatch is None
        assert executor.queued_count == 1
        assert broker.settled == []
        assert executor.placement_service.quarantined_targets == {}
        assert executor.diagnostic_snapshot["failure_code"] == "builder_not_ready"
        assert executor.diagnostic_snapshot["builder_state"][0]["reason"] == (
            "enemy_threat_near_builder"
        )
    else:
        assert dispatch is not None
        assert dispatch.builder_tag == expected_builder
        assert dispatch.actor_tags == (expected_builder,)


@pytest.mark.parametrize(("include_ready_probe", "expected_builder"), [(False, None), (True, 0xB2)])
def test_same_observation_build_confirmation_requires_fresh_builder_binding(
    include_ready_probe: bool,
    expected_builder: int | None,
) -> None:
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={2: "Probe", 60: "Pylon"},
    )
    raw_units = [
        _unit(0xB1, 2, x=20, y=20),
        _unit(0xC1, 60, x=16, y=16),
    ]
    if include_ready_probe:
        raw_units.append(_unit(0xB2, 2, x=21, y=20))
    observation = SimpleNamespace(
        raw_units=raw_units,
        feature_units=[],
        game_loop=[100],
        player_common=SimpleNamespace(minerals=500, vespene=0, food_used=0, food_cap=20),
    )
    placement_revision = _placement_revision(observation)

    pylon = RoutedCommand(
        command_id=f"confirmed-pylon-{include_ready_probe}",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([60, 60],),
        screen_world_target=(24.0, 24.0),
        screen_anchor_tag=0xC1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            (24.0, 24.0),
            0xC1,
            placement_revision,
            2,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(pylon))
    first_dispatch = executor.next_dispatch(
        observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )
    assert first_dispatch is not None
    assert first_dispatch.builder_tag == 0xB1
    executor.observe_reports(
        [{"command_id": pylon.command_id, "status": "succeeded"}],
        {},
        game_loop=100,
    )

    gateway = RoutedCommand(
        command_id=f"same-observation-gateway-{include_ready_probe}",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Gateway_Screen",
        rendered_action="",
        requested_arguments=([70, 70],),
        screen_world_target=(30.0, 30.0),
        screen_anchor_tag=0xC1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Gateway_Screen",
            (30.0, 30.0),
            0xC1,
            placement_revision,
            3,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(gateway))
    second_dispatch = executor.next_dispatch(
        observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    if expected_builder is None:
        assert second_dispatch is None
        assert executor.queued_count == 1
        assert executor.diagnostic_snapshot["failure_code"] == "builder_not_ready"
    else:
        assert second_dispatch is not None
        assert second_dispatch.builder_tag == expected_builder


def test_raw_executor_defers_dynamic_target_obstruction_without_quarantine() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={1: "Adept", 2: "Probe"})
    observation = SimpleNamespace(
        raw_units=[
            _unit(0xB1, 2, x=20, y=20),
            _unit(0xA1, 1, x=22, y=24),
        ],
        game_loop=[100],
    )
    placement_revision = _placement_revision(observation)
    command = RoutedCommand(
        command_id="dynamic-blocker",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([65, 65],),
        operation_id="operation:dynamic-blocker",
        screen_world_target=(22.25, 24.5),
        screen_anchor_tag=0xB1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            (22.25, 24.5),
            0xB1,
            placement_revision,
            2,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(command))

    assert (
        executor.next_dispatch(
            observation,
            {"Builder": _agent("Builder-Probe-1", [0xB1])},
        )
        is None
    )
    assert executor.queued_count == 1
    assert executor.effect_inflight_count == 0
    assert broker.settled == []
    assert executor.placement_service.active_reservation_count == 0
    assert executor.placement_service.quarantined_targets == {}
    assert executor.diagnostic_snapshot["failure_code"] == "dynamic_target_obstruction"
    assert executor.diagnostic_snapshot["dynamic_blockers"] == [
        {
            "tag": "0xa1",
            "unit_type": "Adept",
            "alliance": 1,
            "position": [22.0, 24.0],
            "radius": 0.5,
        }
    ]


def test_raw_executor_uses_collision_radius_for_dynamic_target_obstruction() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={1: "Adept", 2: "Probe"})
    observation = SimpleNamespace(
        raw_units=[
            _unit(0xB1, 2, x=20, y=20),
            _unit(0xA1, 1, x=23.0, y=24.0, radius=0.75),
        ],
        game_loop=[100],
    )
    placement_revision = _placement_revision(observation)
    command = RoutedCommand(
        command_id="radius-blocker",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([65, 65],),
        screen_world_target=(22.25, 24.5),
        screen_anchor_tag=0xB1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            (22.25, 24.5),
            0xB1,
            placement_revision,
            2,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(command))

    assert (
        executor.next_dispatch(
            observation,
            {"Builder": _agent("Builder-Probe-1", [0xB1])},
        )
        is None
    )
    assert executor.diagnostic_snapshot["failure_code"] == "dynamic_target_obstruction"


def test_raw_executor_does_not_treat_snapshot_unit_as_current_obstruction() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={1: "Adept", 2: "Probe"})
    observation = SimpleNamespace(
        raw_units=[
            _unit(0xB1, 2, x=20, y=20),
            _unit(0xA1, 1, x=22, y=24, display_type=2),
        ],
        feature_units=[],
        game_loop=[100],
    )
    placement_revision = _placement_revision(observation)
    target = (22.25, 24.5)
    command = RoutedCommand(
        command_id="snapshot-target-unit",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([65, 65],),
        screen_world_target=target,
        screen_anchor_tag=0xB1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            target,
            0xB1,
            placement_revision,
            2,
            False,
        ),
        placement_revision=placement_revision,
    )
    executor.enqueue(_decision(command))

    dispatch = executor.next_dispatch(
        observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    assert dispatch is not None
    assert dispatch.builder_tag == 0xB1


def test_raw_executor_does_not_treat_assimilator_geyser_anchor_as_obstruction() -> None:
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={2: "Probe", 342: "VespeneGeyser"},
    )
    observation = SimpleNamespace(
        raw_units=[
            _unit(0xB1, 2, x=20, y=20),
            _unit(0xA1, 342, alliance=3, x=24, y=20, radius=1.5),
        ],
        feature_units=[],
        game_loop=[100],
        player_common=SimpleNamespace(minerals=500, vespene=0, food_used=0, food_cap=20),
    )
    command = RoutedCommand(
        command_id="assimilator-anchor",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Assimilator_Near",
        rendered_action="",
        requested_arguments=(0xA1,),
    )
    executor.enqueue(_decision(command))

    dispatch = executor.next_dispatch(
        observation,
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    assert dispatch is not None
    assert dispatch.builder_tag == 0xB1


def test_raw_executor_keeps_operation_circuit_open_without_new_target_state() -> None:
    broker = _Broker()
    observation = SimpleNamespace(raw_units=[_unit(0xB1, 2)], game_loop=[100])
    placement_revision = _placement_revision(observation)
    placement_service = RawPlacementService(
        unit_names={2: "Probe"},
        no_start_streak_threshold=1,
    )
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={2: "Probe"},
        placement_service=placement_service,
    )
    command_kwargs: dict[str, Any] = {
        "actor": "Builder/Builder-Probe-1",
        "team_name": "Builder-Probe-1",
        "name": "Build_Pylon_Screen",
        "rendered_action": "",
        "requested_arguments": ([65, 65],),
        "operation_id": "operation:circuit-open",
        "attempt_ordinal": 7,
        "screen_world_target": (22.25, 24.5),
        "screen_anchor_tag": 0xB1,
        "placement_candidate_id": _placement_candidate_id(
            "Build_Pylon_Screen",
            (22.25, 24.5),
            0xB1,
            placement_revision,
            2,
            False,
        ),
        "placement_revision": placement_revision,
    }
    first = RoutedCommand(command_id="circuit-first", **command_kwargs)
    agents = {"Builder": _agent("Builder-Probe-1", [0xB1])}
    executor.enqueue(_decision(first))
    assert executor.next_dispatch(observation, agents) is not None
    reservation = placement_service.command_target(first.command_id)
    assert reservation is not None
    assert reservation.attempt_ordinal == 7
    executor.observe_reports(
        [
            {
                "command_id": first.command_id,
                "status": "failed",
                "failure_code": "no_build_start_evidence",
            }
        ],
        agents,
        game_loop=101,
    )
    state = placement_service.operation_no_start_state("operation:circuit-open")
    assert state is not None and state.circuit_open is True

    retry_observation = SimpleNamespace(raw_units=[_unit(0xB1, 2)], game_loop=[102])
    retry_revision = _placement_revision(retry_observation)
    retry_kwargs: dict[str, Any] = {
        **command_kwargs,
        "screen_world_target": (30.0, 30.0),
        "placement_candidate_id": _placement_candidate_id(
            "Build_Pylon_Screen",
            (30.0, 30.0),
            0xB1,
            retry_revision,
            2,
            False,
        ),
        "placement_revision": retry_revision,
    }
    retry = RoutedCommand(command_id="circuit-retry", **retry_kwargs)
    executor.enqueue(_decision(retry))
    assert executor.next_dispatch(retry_observation, agents) is None
    assert executor.queued_count == 0
    assert executor.effect_inflight_count == 0
    assert broker.settled == [("circuit-retry", False)]
    assert placement_service.active_reservation_count == 0
    assert executor.diagnostic_snapshot["failure_code"] == "operation_no_start_circuit_open"
    assert placement_service.quarantined_targets == {}


def test_failed_build_effect_temporarily_suppresses_emitted_world_target() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={2: "Probe"})
    observation = SimpleNamespace(raw_units=[_unit(0xB1, 2)], game_loop=[100])
    placement_revision = _placement_revision(observation)
    command = RoutedCommand(
        command_id="build-failed",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([65, 65],),
        screen_world_target=(22.25, 24.5),
        screen_anchor_tag=0xB1,
        placement_candidate_id=_placement_candidate_id(
            "Build_Pylon_Screen",
            (22.25, 24.5),
            0xB1,
            placement_revision,
            2,
            False,
        ),
        placement_revision=placement_revision,
    )
    agents = {"Builder": _agent("Builder-Probe-1", [0xB1])}
    executor.enqueue(_decision(command))
    assert (
        executor.next_dispatch(
            observation,
            agents,
        )
        is not None
    )

    executor.observe_reports(
        [
            {
                "command_id": "build-failed",
                "status": "failed",
                "failure_code": "no_build_start_evidence",
                "effect_evidence": {
                    "failure_classification": "placement_invalid",
                    "classification_basis": ["placement_invalid"],
                    "target_side_evidence": True,
                },
            }
        ],
        agents,
        game_loop=212,
    )

    assert executor.placement_service.is_quarantined(
        "Build_Pylon_Screen",
        (22.0, 24.0),
        radius=1.5,
        game_loop=212,
    )
    assert executor.placement_service.is_quarantined(
        "Build_Pylon_Screen",
        (22.0, 24.0),
        radius=1.5,
        game_loop=325,
    )


def test_raw_nexus_uses_resource_clearance_position_not_resource_centroid() -> None:
    broker = _Broker()
    executor = RawActionExecutor(
        cast(Any, broker),
        unit_names={2: "Probe", 59: "Nexus", 341: "MineralField"},
    )
    command = RoutedCommand(
        command_id="expand",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Nexus_Near",
        rendered_action="",
        requested_arguments=(0x101,),
    )
    resources = [
        _unit(
            0x101 + index,
            341,
            alliance=3,
            x=x,
            y=y,
        )
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
    executor.enqueue(_decision(command))

    dispatch = executor.next_dispatch(
        SimpleNamespace(
            raw_units=[
                _unit(0xB1, 2),
                _unit(0xC1, 59, x=20, y=20),
                *resources,
            ],
            game_loop=[100],
        ),
        {"Builder": _agent("Builder-Probe-1", [0xB1])},
    )

    assert dispatch is not None
    assert int(dispatch.action.function) == 34
    assert dispatch.resolved_arguments == ([80, 80],)


def test_raw_executor_exposes_queued_inflight_and_outstanding_work_separately() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={1: "Adept"})
    command = RoutedCommand(
        command_id="move-outstanding",
        actor="CombatGroup7/Adept-1",
        team_name="Adept-1",
        name="Move_Minimap",
        rendered_action="",
        requested_arguments=([48, 16],),
    )
    executor.enqueue(_decision(command))

    assert executor.queued_count == 1
    assert executor.effect_inflight_count == 0
    assert executor.outstanding_work is True

    dispatch = executor.next_dispatch(
        SimpleNamespace(raw_units=[_unit(0xA1, 1)], game_loop=[100]),
        {"CombatGroup7": _agent("Adept-1", [0xA1])},
    )

    assert dispatch is not None
    assert executor.queued_count == 0
    assert executor.effect_inflight_count == 1
    assert executor.outstanding_work is True

    executor.observe_reports(
        [{"command_id": command.command_id, "status": "succeeded"}],
        {},
        game_loop=104,
    )

    assert executor.queued_count == 0
    assert executor.effect_inflight_count == 0
    assert executor.outstanding_work is False
