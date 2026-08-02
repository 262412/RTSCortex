from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from rtscortex_llm_pysc2.coordinator import BridgeDecision
from rtscortex_llm_pysc2.raw_executor import RawActionExecutor
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
) -> Any:
    return SimpleNamespace(
        tag=tag,
        unit_type=unit_type,
        alliance=alliance,
        build_progress=100,
        order_length=order_length,
        x=x,
        y=y,
        health=100,
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
            [_unit(0xD1, 3)],
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
    executor.enqueue(_decision(command))

    dispatch = executor.next_dispatch(
        SimpleNamespace(raw_units=raw_units, game_loop=[100]),
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
    assert broker.settled == [("missing-builder", False)]
    assert executor.placement_service.quarantined_targets == {}
    assert executor.placement_service.active_reservation_count == 0


def test_failed_build_effect_temporarily_suppresses_emitted_world_target() -> None:
    broker = _Broker()
    executor = RawActionExecutor(cast(Any, broker), unit_names={2: "Probe"})
    command = RoutedCommand(
        command_id="build-failed",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="",
        requested_arguments=([65, 65],),
        screen_world_target=(22.25, 24.5),
        screen_anchor_tag=0xB1,
    )
    agents = {"Builder": _agent("Builder-Probe-1", [0xB1])}
    executor.enqueue(_decision(command))
    assert (
        executor.next_dispatch(
            SimpleNamespace(raw_units=[_unit(0xB1, 2)], game_loop=[100]),
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
    assert not executor.placement_service.is_quarantined(
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
