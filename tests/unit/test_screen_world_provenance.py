from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
import rtscortex_llm_pysc2.worker as worker_module
from rtscortex_llm_pysc2.broker import PrimitiveDispatch
from rtscortex_llm_pysc2.extractor import (
    build_screen_candidates,
    resolve_screen_build_world_target,
    resolve_screen_point_world_target,
    screen_to_world_target,
)
from rtscortex_llm_pysc2.hook import RuntimeQueryMixin
from rtscortex_llm_pysc2.observation import ObservationMapper
from rtscortex_llm_pysc2.routing import ActionRouter
from rtscortex_llm_pysc2.worker import RTSCortexLLMAgent


class Grid:
    shape = (128, 128)

    def __init__(self, values: list[list[int]]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> list[int]:
        return self.values[index]


def test_screen_target_reprojects_after_camera_translation() -> None:
    observed = _observation(anchor_screen=(64, 64))
    provenance = screen_to_world_target(observed, [80, 80])

    assert provenance is not None
    assert provenance.world_target == (103.0, 53.0)

    dispatched = _observation(anchor_screen=(32, 96))
    resolved = resolve_screen_build_world_target(
        dispatched,
        "Build_Pylon_Screen",
        provenance.world_target,
        preferred_anchor_tag=provenance.anchor_tag,
    )

    assert resolved == [48, 112]


def test_screen_world_projection_preserves_feature_y_axis() -> None:
    observation = _observation(anchor_screen=(64, 64))

    above_anchor = screen_to_world_target(observation, [64, 48])
    below_anchor = screen_to_world_target(observation, [64, 80])

    assert above_anchor is not None and above_anchor.world_target == (100.0, 47.0)
    assert below_anchor is not None and below_anchor.world_target == (100.0, 53.0)


def test_screen_world_projection_requires_a_shared_raw_feature_anchor() -> None:
    observation = _observation(anchor_screen=(64, 64))
    observation.feature_units[0].tag = 2

    assert screen_to_world_target(observation, [64, 64]) is None
    assert (
        resolve_screen_build_world_target(
            observation,
            "Build_Pylon_Screen",
            (100.0, 50.0),
            preferred_anchor_tag=1,
        )
        is None
    )


def test_screen_build_relocation_rejects_candidate_beyond_two_strides() -> None:
    buildable = [[0 for _ in range(128)] for _ in range(128)]
    pathable = [[0 for _ in range(128)] for _ in range(128)]
    for y in range(90, 111):
        for x in range(90, 111):
            buildable[y][x] = 1
            pathable[y][x] = 1
    observation = _observation(
        anchor_screen=(32, 32),
        buildable=buildable,
        pathable=pathable,
    )

    resolved = resolve_screen_build_world_target(
        observation,
        "Build_Pylon_Screen",
        (106.0, 56.0),  # Reprojects to [64, 64], whose footprint is blocked.
        preferred_anchor_tag=1,
    )

    assert resolved is None


def test_screen_movement_target_reprojects_to_current_legal_candidate() -> None:
    observed = _observation(anchor_screen=(64, 64))
    provenance = screen_to_world_target(observed, [80, 80])

    assert provenance is not None
    dispatched = _observation(anchor_screen=(48, 64))
    resolved = resolve_screen_point_world_target(
        dispatched,
        "Move_Screen",
        provenance.world_target,
        preferred_anchor_tag=provenance.anchor_tag,
    )

    assert resolved == [64, 80]


def test_screen_movement_relocation_rejects_world_target_outside_current_window() -> None:
    observation = _observation(anchor_screen=(32, 32))

    resolved = resolve_screen_point_world_target(
        observation,
        "Move_Screen",
        (120.0, 70.0),
        preferred_anchor_tag=1,
    )

    assert resolved is None


def test_screen_world_target_stays_private_while_route_keeps_provenance() -> None:
    snapshot = {
        "run_id": "run",
        "episode_id": "episode",
        "step_id": 7,
        "game_loop": 112,
        "observed_at": "2026-07-14T00:00:00Z",
        "player_common": {
            "minerals": 500,
            "vespene": 0,
            "food_used": 12,
            "food_cap": 15,
            "food_workers": 12,
            "food_army": 0,
        },
        "production_queue": [],
        "units": [],
        "upgrades": [],
        "teams": [
            {
                "agent_name": "Builder",
                "team_name": "Probe-1",
                "available_actions": [
                    {
                        "name": "Build_Pylon_Screen",
                        "argument_names": ["screen"],
                        "argument_types": ["position"],
                        "argument_candidates": [[[80, 80]]],
                        "bridge_screen_provenance": [
                            {
                                "screen_target": [80, 80],
                                "world_target": [103.0, 53.0],
                                "anchor_tag": 1,
                            }
                        ],
                    },
                    {
                        "name": "Move_Screen",
                        "argument_names": ["screen"],
                        "argument_types": ["position"],
                        "argument_candidates": [[[80, 80]]],
                        "bridge_screen_provenance": [
                            {
                                "screen_target": [80, 80],
                                "world_target": [103.0, 53.0],
                                "anchor_tag": 1,
                            }
                        ],
                    },
                ],
            }
        ],
        "text_observation": "",
        "alerts": [],
        "image_uri": None,
    }
    observation = ObservationMapper().map(snapshot)

    serialized = json.dumps(observation)
    assert "bridge_screen_provenance" not in serialized
    assert "world_target" not in serialized
    assert all(
        action["argument_candidates"] == [[[80, 80]]] for action in observation["available_actions"]
    )

    route = ActionRouter().route(
        {
            "protocol_version": "1.1",
            "run_id": "run",
            "episode_id": "episode",
            "step_id": 7,
            "decision_id": "decision",
            "commands": [
                {
                    "command_id": "build",
                    "actor": "Builder/Probe-1",
                    "name": "Build_Pylon_Screen",
                    "arguments": [[80, 80]],
                    "source": "planner",
                },
                {
                    "command_id": "move",
                    "actor": "Builder/Probe-1",
                    "name": "Move_Screen",
                    "arguments": [[80, 80]],
                    "source": "planner",
                },
            ],
        },
        agent_name="Builder",
        team_order=["Probe-1"],
        available_actions=observation["available_actions"],
    )

    assert len(route.commands) == 2
    assert all(command.screen_world_target == (103.0, 53.0) for command in route.commands)
    assert all(command.screen_anchor_tag == 1 for command in route.commands)
    assert all("screen_world_target" not in command.to_dict() for command in route.commands)


def test_agent_reprojects_before_current_candidate_domain_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buildable = [[0 for _ in range(128)] for _ in range(128)]
    pathable = [[0 for _ in range(128)] for _ in range(128)]
    for y in range(58, 70):
        for x in range(58, 70):
            buildable[y][x] = 1
            pathable[y][x] = 1
    observation = _observation(
        anchor_screen=(32, 32),
        buildable=buildable,
        pathable=pathable,
    )
    observation.player_common = SimpleNamespace(minerals=500)
    observation.game_loop = [400]
    observation.available_actions = [35]
    assert build_screen_candidates(observation, "Build_Pylon_Screen") == []

    action: dict[str, Any] = {
        "name": "Build_Pylon_Screen",
        "arg": [[80, 80]],
        "func": [(35, object(), ("now", [80, 80]))],
    }

    class Broker:
        unattributed_primitives = 0

        def __init__(self) -> None:
            self.resolutions: list[list[Any]] = []

        def command_id_for(self, *_args: Any) -> str:
            return "build-command"

        def screen_route_provenance(self, _command_id: str) -> SimpleNamespace:
            return SimpleNamespace(world_target=(106.0, 56.0), anchor_tag=1)

        def resolve_arguments(self, _command_id: str, arguments: list[Any]) -> None:
            self.resolutions.append(arguments)

        def claim_primitive(self, *_args: Any, **_kwargs: Any) -> PrimitiveDispatch:
            return PrimitiveDispatch(
                "build-command",
                "Build_Pylon_screen",
                True,
                ordinal=0,
                total=1,
                requested_function_id=35,
                emitted_function_id=35,
            )

    translated_positions: list[list[int]] = []

    def upstream_get_func(agent: Any, _obs: Any) -> tuple[int, str]:
        executing_action = agent.action_list[0]
        resolved = [int(value) for value in executing_action["arg"][0]]
        translated_positions.append(resolved)
        agent.last_translation_result = {
            "action_name": "Build_Pylon_Screen",
            "requested_function_id": 35,
            "requested_function_name": "Build_Pylon_screen",
            "emitted_function_id": 35,
            "accepted": True,
            "ordinal": 0,
            "total": 1,
            "resolved_arguments": [resolved],
        }
        # The real upstream translator consumes parts of this mutable action.
        executing_action["arg"].clear()
        executing_action["func"].clear()
        return 35, "translated"

    monkeypatch.setattr(RuntimeQueryMixin, "get_func", upstream_get_func, raising=False)
    broker = Broker()
    agent = object.__new__(RTSCortexLLMAgent)
    agent.name = "Builder"
    agent.team_unit_team_curr = "Probe-1"
    agent.team_unit_tag_curr = 1
    agent.team_unit_tag_list = [1]
    agent.flag_enable_empty_unit_group = False
    agent.func_list = []
    agent.action_list = [action]
    agent.unit_names = {}
    agent.curr_action_name = "Build_Pylon_Screen"
    agent.broker = cast(Any, broker)
    agent._rtscortex_translation_attempt = None
    agent._rtscortex_rejected_build_positions = {}
    agent._rtscortex_rejected_build_targets = {}
    agent._rtscortex_active_build_route = None
    agent.size_screen = 128

    result = agent.get_func(SimpleNamespace(observation=observation))

    assert result == (35, "translated")
    assert translated_positions == [[64, 64]]
    assert broker.resolutions == [[[64, 64]], [[64, 64]]]
    assert agent._rtscortex_translation_attempt is not None


def test_agent_reprojects_screen_build_again_after_selection_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buildable = [[0 for _ in range(128)] for _ in range(128)]
    pathable = [[0 for _ in range(128)] for _ in range(128)]
    for y in range(54, 75):
        for x in range(54, 75):
            buildable[y][x] = 1
            pathable[y][x] = 1
    for coordinate in range(32, 55):
        pathable[coordinate][coordinate] = 1
    observation = _observation(
        anchor_screen=(32, 32),
        buildable=buildable,
        pathable=pathable,
    )
    observation.game_loop = [500]
    observation.available_actions = [57]
    action: dict[str, Any] = {
        "name": "Build_Gateway_Screen",
        "arg": [[60, 85]],
        "func": [],
    }
    pending_function = SimpleNamespace(name="Build_Gateway_screen")
    upstream_validation_calls: list[list[int]] = []

    def validate_like_pinned_translator(
        _obs: Any,
        position: list[int],
        _screen_size: int,
        _action_name: str,
    ) -> bool:
        upstream_validation_calls.append(list(position))
        return position != [64, 64]

    monkeypatch.setattr(
        worker_module,
        "_translator_screen_build_is_legal",
        validate_like_pinned_translator,
    )

    class Broker:
        unattributed_primitives = 0

        def __init__(self) -> None:
            self.resolutions: list[list[Any]] = []

        def command_id_for(self, *_args: Any) -> str:
            return "gateway-command"

        def screen_route_provenance(self, _command_id: str) -> SimpleNamespace:
            return SimpleNamespace(world_target=(106.0, 56.0), anchor_tag=1)

        def resolve_arguments(self, _command_id: str, arguments: list[Any]) -> None:
            self.resolutions.append(arguments)

        def claim_primitive(self, *_args: Any, **_kwargs: Any) -> PrimitiveDispatch:
            return PrimitiveDispatch(
                "gateway-command",
                "Build_Gateway_screen",
                True,
                ordinal=3,
                total=4,
                requested_function_id=57,
                emitted_function_id=57,
            )

    translated_positions: list[list[int]] = []

    def upstream_get_func(agent: Any, _obs: Any) -> tuple[int, str]:
        function_id, _function, arguments = agent.func_list.pop(0)
        resolved = next(value for value in arguments if isinstance(value, list))
        translated_positions.append(list(resolved))
        agent.last_translation_result = {
            "action_name": "Build_Gateway_Screen",
            "requested_function_id": function_id,
            "requested_function_name": "Build_Gateway_screen",
            "emitted_function_id": function_id,
            "accepted": True,
            "ordinal": 3,
            "total": 4,
            "resolved_arguments": [resolved],
        }
        return function_id, "translated"

    monkeypatch.setattr(RuntimeQueryMixin, "get_func", upstream_get_func, raising=False)
    broker = Broker()
    agent = object.__new__(RTSCortexLLMAgent)
    agent.name = "Builder"
    agent.team_unit_team_curr = "Builder-Probe-1"
    agent.team_unit_tag_curr = 1
    agent.team_unit_tag_list = [1]
    agent.flag_enable_empty_unit_group = False
    agent.func_list = [(57, pending_function, ("now", [60, 85]))]
    agent.action_list = []
    agent.unit_names = {}
    agent.curr_action_name = "Build_Gateway_Screen"
    agent.broker = cast(Any, broker)
    agent._rtscortex_semantic_action = action
    agent._rtscortex_translation_attempt = None
    agent._rtscortex_rejected_build_positions = {}
    agent._rtscortex_rejected_build_targets = {}
    agent._rtscortex_active_build_route = None
    agent._rtscortex_camera_settlement_noop = False
    agent._rtscortex_build_selection_retries = 0
    agent.size_screen = 128

    result = agent.get_func(SimpleNamespace(observation=observation))

    assert result == (57, "translated")
    assert upstream_validation_calls == [[64, 64], [65, 65]]
    assert translated_positions == [[65, 65]]
    assert broker.resolutions == [[[65, 65]], [[65, 65]]]
    assert agent._rtscortex_translation_attempt is not None


def test_agent_resamples_screen_build_after_translator_rejects_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation(anchor_screen=(32, 32))
    observation.game_loop = [500]
    observation.available_actions = [70]
    action: dict[str, Any] = {
        "name": "Build_Pylon_Screen",
        "arg": [[64, 64]],
        "func": [],
    }
    pending_function = SimpleNamespace(name="Build_Pylon_screen")

    class Broker:
        unattributed_primitives = 0

        def __init__(self) -> None:
            self.resolutions: list[list[Any]] = []
            self.claims = 0

        def command_id_for(self, *_args: Any) -> str:
            return "pylon-command"

        def screen_route_provenance(self, _command_id: str) -> SimpleNamespace:
            return SimpleNamespace(world_target=(106.0, 56.0), anchor_tag=1)

        def resolve_arguments(self, _command_id: str, arguments: list[Any]) -> None:
            self.resolutions.append(arguments)

        def claim_primitive(self, *_args: Any, **_kwargs: Any) -> PrimitiveDispatch:
            self.claims += 1
            return PrimitiveDispatch(
                "pylon-command",
                "Build_Pylon_screen",
                True,
                ordinal=0,
                total=1,
                requested_function_id=70,
                emitted_function_id=70,
            )

    attempted_positions: list[list[int]] = []

    def upstream_get_func(agent: Any, _obs: Any) -> tuple[int, str]:
        function_id, _function, arguments = agent.func_list.pop(0)
        resolved = [int(value) for value in arguments[1]]
        attempted_positions.append(resolved)
        ordinal = agent._rtscortex_translation_ordinal
        agent._rtscortex_translation_ordinal += 1
        accepted = len(attempted_positions) > 1
        agent.last_translation_result = {
            "action_name": "Build_Pylon_Screen",
            "requested_function_id": function_id,
            "requested_function_name": "Build_Pylon_screen",
            "emitted_function_id": function_id if accepted else 0,
            "accepted": accepted,
            "ordinal": ordinal,
            "total": 1,
            "resolved_arguments": [resolved] if accepted else [],
            "reason": None if accepted else f"area near {tuple(resolved)} not pathable",
        }
        return (function_id if accepted else 0), "translated"

    monkeypatch.setattr(RuntimeQueryMixin, "get_func", upstream_get_func, raising=False)
    broker = Broker()
    agent = object.__new__(RTSCortexLLMAgent)
    agent.name = "Builder"
    agent.team_unit_team_curr = "Builder-Probe-1"
    agent.team_unit_tag_curr = 1
    agent.team_unit_tag_list = [1]
    agent.flag_enable_empty_unit_group = False
    agent.func_list = [(70, pending_function, ("now", [64, 64]))]
    agent.action_list = []
    agent.unit_names = {}
    agent.curr_action_name = "Build_Pylon_Screen"
    agent.broker = cast(Any, broker)
    agent._rtscortex_semantic_action = action
    agent._rtscortex_translation_attempt = None
    agent._rtscortex_translation_ordinal = 0
    agent._rtscortex_rejected_build_positions = {}
    agent._rtscortex_rejected_build_targets = {}
    agent._rtscortex_active_build_route = None
    agent._rtscortex_camera_settlement_noop = False
    agent._rtscortex_build_selection_retries = 0
    agent.size_screen = 128

    first = agent.get_func(SimpleNamespace(observation=observation))

    assert first == (0, "translated")
    assert broker.claims == 0
    assert agent._rtscortex_translation_ordinal == 0
    assert agent.func_list
    assert tuple(attempted_positions[0]) in agent._rtscortex_rejected_build_positions[
        "Build_Pylon_Screen"
    ]

    second = agent.get_func(SimpleNamespace(observation=observation))

    assert second == (70, "translated")
    assert attempted_positions[1] != attempted_positions[0]
    assert broker.claims == 1
    assert agent._rtscortex_translation_attempt is not None


def test_agent_reprojects_move_screen_and_records_resolved_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation(anchor_screen=(48, 64))
    observation.game_loop = [400]
    action: dict[str, Any] = {
        "name": "Move_Screen",
        "arg": [[80, 80]],
        "func": [(12, object(), ("now", [80, 80]))],
    }

    class Broker:
        unattributed_primitives = 0

        def __init__(self) -> None:
            self.resolutions: list[list[Any]] = []

        def command_id_for(self, *_args: Any) -> str:
            return "move-command"

        def screen_route_provenance(self, _command_id: str) -> SimpleNamespace:
            return SimpleNamespace(world_target=(103.0, 53.0), anchor_tag=1)

        def resolve_arguments(self, _command_id: str, arguments: list[Any]) -> None:
            self.resolutions.append(arguments)

        def claim_primitive(self, *_args: Any, **_kwargs: Any) -> PrimitiveDispatch:
            return PrimitiveDispatch(
                "move-command",
                "Move_screen",
                True,
                ordinal=0,
                total=1,
                requested_function_id=12,
                emitted_function_id=12,
            )

        def reject_candidate_outside_dispatch(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("reprojected candidate must remain in the current domain")

    translated_positions: list[list[int]] = []

    def upstream_get_func(agent: Any, _obs: Any) -> tuple[int, str]:
        executing_action = agent.action_list[0]
        resolved = [int(value) for value in executing_action["arg"][0]]
        translated_positions.append(resolved)
        agent.last_translation_result = {
            "action_name": "Move_Screen",
            "requested_function_id": 12,
            "requested_function_name": "Move_screen",
            "emitted_function_id": 12,
            "accepted": True,
            "ordinal": 0,
            "total": 1,
            "resolved_arguments": [resolved],
        }
        executing_action["arg"].clear()
        executing_action["func"].clear()
        return 12, "translated"

    monkeypatch.setattr(RuntimeQueryMixin, "get_func", upstream_get_func, raising=False)
    broker = Broker()
    agent = object.__new__(RTSCortexLLMAgent)
    agent.name = "CombatGroup1"
    agent.team_unit_team_curr = "Stalker-1"
    agent.team_unit_tag_curr = 1
    agent.team_unit_tag_list = [1]
    agent.flag_enable_empty_unit_group = False
    agent.func_list = []
    agent.action_list = [action]
    agent.unit_names = {}
    agent.curr_action_name = "Move_Screen"
    agent.broker = cast(Any, broker)
    agent._rtscortex_translation_attempt = None

    result = agent.get_func(SimpleNamespace(observation=observation))

    assert result == (12, "translated")
    assert translated_positions == [[64, 80]]
    assert broker.resolutions == [[[64, 80]]]
    assert agent._rtscortex_translation_attempt is not None


def _observation(
    *,
    anchor_screen: tuple[int, int],
    buildable: list[list[int]] | None = None,
    pathable: list[list[int]] | None = None,
) -> SimpleNamespace:
    buildable_values = buildable or [[1 for _ in range(128)] for _ in range(128)]
    pathable_values = pathable or [[1 for _ in range(128)] for _ in range(128)]
    return SimpleNamespace(
        raw_units=[SimpleNamespace(tag=1, x=100.0, y=50.0, alliance=1)],
        feature_units=[
            SimpleNamespace(
                tag=1,
                x=anchor_screen[0],
                y=anchor_screen[1],
                alliance=1,
                is_on_screen=True,
                radius=0.5,
            )
        ],
        feature_screen=SimpleNamespace(
            buildable=Grid(buildable_values),
            pathable=Grid(pathable_values),
            player_relative=Grid([[0 for _ in range(128)] for _ in range(128)]),
            power=Grid([[1 for _ in range(128)] for _ in range(128)]),
        ),
    )
