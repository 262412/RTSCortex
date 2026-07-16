"""Python 3.9 smoke checks for the reviewed LLM-PySC2 patch chain."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace

from llm_pysc2.agents.llm_pysc2_agent import LLMAgent
from llm_pysc2.agents.llm_pysc2_agent_main import MainAgent
from llm_pysc2.agents.main_agent_funcs import main_agent_func1, main_agent_func2
from llm_pysc2.lib import llm_action
from pysc2.env import run_loop
from pysc2.lib import actions, features
from rtscortex_llm_pysc2.effect_verifier import _BUILD_RAW_FUNCTION_IDS
from rtscortex_llm_pysc2.observation import _map_argument_candidates
from rtscortex_llm_pysc2.production import PRODUCTION_SPECS


def _assert_candidate_mapping() -> None:
    assert _map_argument_candidates([[1], ["0XABC"]], ("tag",)) == [
        ["0x1"],
        ["0xabc"],
    ]
    assert _map_argument_candidates([[[65, 90]]], ("position",)) == [[[65, 90]]]

    nexus = next(
        action for action in llm_action.PROTOSS_ACTION_BUILD if action["name"] == "Build_Nexus_Near"
    )
    assert [function_id for function_id, _, _ in nexus["func"]] == [573, 0, 65]
    assert "return translator settlement no_op" in inspect.getsource(MainAgent.step)


def _assert_reserved_builder_worker_guard() -> None:
    source = inspect.getsource(main_agent_func2)
    assert "_rtscortex_reserved_worker_tags" in source
    assert "Reserved worker" in source
    assert "HoldPosition_quick('now')" in source


def _assert_max_frame_hook() -> None:
    calls = []

    class Agent:
        def setup(self, observation_spec, action_spec) -> None:
            del observation_spec, action_spec

        def reset(self) -> None:
            pass

        def step(self, timestep):
            del timestep
            return "noop"

        def on_episode_truncated(self, total_frames: int) -> None:
            calls.append(total_frames)

    class Environment:
        def observation_spec(self):
            return [object()]

        def action_spec(self):
            return [object()]

        def reset(self):
            return [SimpleNamespace(last=lambda: False)]

        def step(self, actions):
            raise AssertionError(f"unexpected step: {actions}")

    run_loop.run_loop([Agent()], Environment(), max_frames=1, max_episodes=1)
    assert calls == [1]


def _assert_atomic_log_directory_allocation() -> None:
    source = inspect.getsource(MainAgent._initialize_logger)
    assert "except FileExistsError:" in source
    assert "llm_pysc2_global_log_id = max" in source


def _assert_gas_rebalance_uses_worker_management_flag() -> None:
    source = inspect.getsource(main_agent_func2)
    assert "ENABLE_AUTO_WORKER_MANAGE and self.is_all_nexus_full is False" in source


def _assert_build_order_ids_use_raw_function_domain() -> None:
    ability_ids = {
        "Assimilator": 882,
        "CyberneticsCore": 894,
        "Gateway": 883,
        "Nexus": 880,
        "Pylon": 881,
        "ShieldBattery": 895,
        "Stargate": 889,
    }
    assert _BUILD_RAW_FUNCTION_IDS == {
        structure: int(actions.RAW_ABILITY_ID_TO_FUNC_ID[ability_id])
        for structure, ability_id in ability_ids.items()
    }


def _assert_direct_production_contract() -> None:
    expected = {
        "Train_Zealot": (503, 49, "Gateway", "Zealot", 100, 0, 2, ("Gateway",)),
        "Train_Stalker": (
            493,
            50,
            "Gateway",
            "Stalker",
            125,
            50,
            2,
            ("Gateway", "CyberneticsCore"),
        ),
        "Train_Adept": (
            457,
            54,
            "Gateway",
            "Adept",
            100,
            25,
            2,
            ("Gateway", "CyberneticsCore"),
        ),
        "Train_Phoenix": (484, 55, "Stargate", "Phoenix", 150, 100, 2, ("Stargate",)),
        "Train_VoidRay": (500, 57, "Stargate", "VoidRay", 250, 150, 4, ("Stargate",)),
        "Train_Oracle": (482, 58, "Stargate", "Oracle", 150, 150, 3, ("Stargate",)),
    }
    assert {
        action_name: (
            spec.feature_function_id,
            spec.raw_order_id,
            spec.producer_type,
            spec.unit_type,
            spec.minerals,
            spec.vespene,
            spec.supply,
            spec.prerequisites,
        )
        for action_name, spec in PRODUCTION_SPECS.items()
    } == expected


def _agent_with_tracked_team() -> LLMAgent:
    agent = object.__new__(LLMAgent)
    agent.name = "Builder"
    agent.log_id = 0
    agent.config = SimpleNamespace(AGENTS_ALWAYS_DISABLE=["Builder"])
    agent.enable = False
    agent.query_llm_times = 0
    agent.executing_times = 0
    agent.main_loop_step = 0
    agent.unit_tag_list = []
    agent.unit_tag_list_history = []
    agent.unit_raw_list = []
    agent.teams = [
        {
            "name": "Builder-Probe-1",
            "select_type": "select",
            "unit_tags": [0xABC],
        }
    ]
    agent.last_text_c_inp = ""
    agent.last_text_c_tar = ""
    return agent


def _team_update_observation(*raw_tags: int) -> SimpleNamespace:
    feature_layer = SimpleNamespace(height_map=SimpleNamespace(shape=(128, 128)))
    return SimpleNamespace(
        observation=SimpleNamespace(
            feature_screen=feature_layer,
            feature_minimap=feature_layer,
            raw_units=[SimpleNamespace(tag=tag) for tag in raw_tags],
        )
    )


def _assert_raw_unit_presence_controls_team_lifecycle() -> None:
    visible_elsewhere = _agent_with_tracked_team()
    visible_elsewhere.update(_team_update_observation(0xABC))
    assert visible_elsewhere.teams[0]["unit_tags"] == [0xABC]

    absent = _agent_with_tracked_team()
    absent.update(_team_update_observation())
    assert absent.teams[0]["unit_tags"] == [0xABC]


def _main_agent_lifecycle_state(tag: int) -> SimpleNamespace:
    team = {
        "name": "Builder-Probe-1",
        "select_type": "select",
        "unit_tags": [tag],
    }
    agent = SimpleNamespace(unit_tag_list=[tag], teams=[team])
    return SimpleNamespace(
        unit_uid=[tag],
        unit_uid_total={tag},
        unit_uid_disappear=[],
        unit_uid_appear=[],
        unit_disappear_steps={},
        main_loop_lock=False,
        nexus_info_dict={},
        AGENT_NAMES=["Builder"],
        agents={"Builder": agent},
        log_id=0,
    )


def _unit_lifecycle_observation(tag: int | None = None) -> SimpleNamespace:
    raw_units = []
    if tag is not None:
        raw_units.append(
            SimpleNamespace(
                tag=tag,
                alliance=features.PlayerRelative.SELF,
                build_progress=100,
                unit_type=84,
            )
        )
    return SimpleNamespace(
        observation=SimpleNamespace(
            raw_units=raw_units,
            feature_units=[],
            available_actions=[0],
        )
    )


def _assert_transient_disappearance_grace() -> None:
    tag = 0xABC
    state = _main_agent_lifecycle_state(tag)

    main_agent_func1(state, _unit_lifecycle_observation())
    assert state.unit_disappear_steps == {tag: 1}
    assert state.unit_uid_disappear == []
    assert state.agents["Builder"].unit_tag_list == [tag]
    assert state.agents["Builder"].teams[0]["unit_tags"] == [tag]

    main_agent_func1(state, _unit_lifecycle_observation(tag))
    assert state.unit_disappear_steps == {}
    assert state.unit_uid_disappear == []
    assert state.agents["Builder"].unit_tag_list == [tag]
    assert state.agents["Builder"].teams[0]["unit_tags"] == [tag]


def _assert_confirmed_disappearance_removes_actor() -> None:
    tag = 0xABC
    state = _main_agent_lifecycle_state(tag)
    state.main_loop_lock = True

    for _ in range(39):
        main_agent_func1(state, _unit_lifecycle_observation())
    assert state.unit_disappear_steps == {tag: 39}
    assert state.unit_uid_disappear == []
    assert state.agents["Builder"].unit_tag_list == [tag]
    assert state.agents["Builder"].teams[0]["unit_tags"] == [tag]

    main_agent_func1(state, _unit_lifecycle_observation())
    assert state.unit_disappear_steps == {tag: 40}
    assert state.unit_uid_disappear == [tag]
    assert state.agents["Builder"].unit_tag_list == []
    assert state.agents["Builder"].teams[0]["unit_tags"] == []


def _assert_pretranslation_abort_markers() -> None:
    source = inspect.getsource(MainAgent.step)
    assert "if tag not in self.unit_uid_disappear:" in source
    assert "wait for confirmed disappearance" in source
    assert "keep the action pending" in source
    assert "last_execution_abort" in source
    assert "'failure_code': 'actor_not_available'" in source
    assert "agent.curr_action_name != 'No_Operation'" in source
    assert "'failure_code': 'actor_not_visible'" not in source


def _assert_multi_argument_rejection() -> None:
    move_screen = actions.FUNCTIONS.Move_screen
    with tempfile.TemporaryDirectory() as temporary_directory:
        agent = object.__new__(LLMAgent)
        agent.func_list = [(int(move_screen.id), move_screen, ("invalid-queue", [64, 64]))]
        agent.action_list = []
        agent.curr_action_name = "Move_Screen"
        agent.curr_action_args = []
        agent.first_action = False
        agent.history_func_path = str(Path(temporary_directory) / "actions.txt")
        agent.main_loop_step = 10
        agent.num_step = 1
        agent.log_id = 0
        agent.name = "smoke"
        agent.size_screen = 128
        agent.size_minimap = 64
        agent._rtscortex_translation_ordinal = 0
        agent._rtscortex_translation_total = 1
        observation = SimpleNamespace(
            observation=SimpleNamespace(available_actions=[int(move_screen.id)])
        )

        function_id, _ = agent.get_func(observation)

    assert function_id == 0
    assert agent.last_translation_result == {
        "action_name": "Move_Screen",
        "requested_function_id": move_screen.id,
        "requested_function_name": "Move_screen",
        "emitted_function_id": 0,
        "emitted_function_name": "no_op",
        "ordinal": 0,
        "total": 1,
        "accepted": False,
        "resolved_arguments": [],
        "reason": "invalid-queue",
    }


def _resource(
    tag: int,
    x: float,
    y: float,
    *,
    on_screen: bool = True,
    gas: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        tag=tag,
        unit_type=(llm_action.GAS_TYPE[0] if gas else llm_action.MINERAL_TYPE[0]),
        alliance=features.PlayerRelative.NEUTRAL,
        display_type=1,
        x=x,
        y=y,
        is_on_screen=on_screen,
    )


def _townhall(x: int, y: int) -> SimpleNamespace:
    return SimpleNamespace(
        tag=100,
        unit_type=llm_action.BASE_BUILDING_TYPE[0],
        alliance=features.PlayerRelative.ENEMY,
        display_type=1,
        x=x,
        y=y,
        is_on_screen=True,
    )


def _base_observation(resources: list[SimpleNamespace]) -> SimpleNamespace:
    size = 128
    feature_screen = SimpleNamespace(
        buildable=[[1] * size for _ in range(size)],
        pathable=[[1] * size for _ in range(size)],
        player_relative=[[0] * size for _ in range(size)],
        visibility_map=[[features.Visibility.VISIBLE] * size for _ in range(size)],
    )
    return SimpleNamespace(
        observation=SimpleNamespace(
            feature_units=resources,
            feature_screen=feature_screen,
        )
    )


def _assert_world_uses_resource_centroid() -> None:
    resources = [
        _resource(1, 10, 10),
        _resource(2, 12, 10),
        _resource(3, 14, 10),
        _resource(4, 16, 10),
        _resource(5, 18, 10),
    ]
    observation = SimpleNamespace(observation=SimpleNamespace(raw_units=resources))

    position, valid = llm_action.get_arg_world_tag_base_building(
        observation,
        1,
        0,
        0,
        200,
    )

    assert valid is True
    assert position == (14, 190)

    occupied_observation = SimpleNamespace(
        observation=SimpleNamespace(raw_units=[*resources, _townhall(14, 10)])
    )
    occupied_reason, occupied_valid = llm_action.get_arg_world_tag_base_building(
        occupied_observation,
        1,
        0,
        0,
        200,
    )
    assert occupied_valid is False
    assert "already has a townhall" in occupied_reason


def _assert_exact_anchor_and_footprint() -> None:
    resources = [
        _resource(1, 101, 64),
        _resource(2, 90, 90),
        _resource(3, 64, 101),
        _resource(4, 38, 90),
        _resource(5, 27, 64),
        _resource(6, 38, 38),
        _resource(7, 64, 27),
        _resource(8, 90, 38),
    ]
    observation = _base_observation(resources)

    position, valid = llm_action.get_arg_screen_tag_base_building(
        observation,
        1,
        128,
        "Build_Nexus_Near",
    )
    assert valid is True
    assert abs(position[0] - 64) <= 10 and abs(position[1] - 64) <= 10
    assert all(
        6 * (128 / llm_action.SCREEN_WORLD_GRID)
        < ((position[0] - resource.x) ** 2 + (position[1] - resource.y) ** 2) ** 0.5
        < 9 * (128 / llm_action.SCREEN_WORLD_GRID)
        for resource in resources
    )

    mineral_line = [
        _resource(20 + index, 80, y) for index, y in enumerate((49, 53, 57, 61, 67, 71, 75, 79))
    ]
    line_position, line_valid = llm_action.get_arg_screen_tag_base_building(
        _base_observation(mineral_line),
        20,
        128,
        "Build_Nexus_Near",
    )
    assert line_valid is True
    assert line_position[0] <= 50
    assert all(
        6 * (128 / llm_action.SCREEN_WORLD_GRID)
        < ((line_position[0] - resource.x) ** 2 + (line_position[1] - resource.y) ** 2) ** 0.5
        < 9 * (128 / llm_action.SCREEN_WORLD_GRID)
        for resource in mineral_line
    )

    scale_regression_resources = [
        _resource(index, x, y)
        for index, (x, y) in enumerate(
            (
                (88.25, 77.52),
                (75.94, 95.38),
                (56.61, 113.08),
                (24.78, 101.62),
                (20.06, 77.35),
                (30.74, 42.77),
                (54.47, 36.97),
                (85.98, 49.95),
            ),
            start=60,
        )
    ]
    scale_position, scale_valid = llm_action.get_arg_screen_tag_base_building(
        _base_observation(scale_regression_resources),
        60,
        128,
        "Build_Nexus_Near",
    )
    assert scale_valid is True
    assert scale_position != (50, 75)
    assert all(
        6 * (128 / llm_action.SCREEN_WORLD_GRID)
        < ((scale_position[0] - resource.x) ** 2 + (scale_position[1] - resource.y) ** 2) ** 0.5
        < 9 * (128 / llm_action.SCREEN_WORLD_GRID)
        for resource in scale_regression_resources
    )

    resource_ring_with_gas = [
        _resource(40, 101, 64),
        _resource(41, 90, 90),
        _resource(42, 64, 101),
        _resource(43, 38, 90),
        _resource(44, 27, 64),
        _resource(45, 38, 38),
        _resource(46, 64, 27),
        _resource(47, 90, 38),
        _resource(48, 109, 64, gas=True),
        _resource(49, 19, 64, gas=True),
    ]
    gas_position, gas_valid = llm_action.get_arg_screen_tag_base_building(
        _base_observation(resource_ring_with_gas),
        40,
        128,
        "Build_Nexus_Near",
    )
    assert gas_valid is True
    for resource in resource_ring_with_gas:
        distance = (
            (gas_position[0] - resource.x) ** 2 + (gas_position[1] - resource.y) ** 2
        ) ** 0.5
        minimum, maximum = (7, 10) if resource.unit_type in llm_action.GAS_TYPE else (6, 9)
        assert minimum * (128 / llm_action.SCREEN_WORLD_GRID) < distance
        assert distance < maximum * (128 / llm_action.SCREEN_WORLD_GRID)

    expected_without_stale, expected_valid = llm_action.get_arg_screen_tag_base_building(
        _base_observation([resource for resource in resources if resource.tag != 2]),
        1,
        128,
        "Build_Nexus_Near",
    )
    stale_resource_x = resources[1].x
    resources[1].x = -100
    actual_without_stale, actual_valid = llm_action.get_arg_screen_tag_base_building(
        _base_observation(resources),
        1,
        128,
        "Build_Nexus_Near",
    )
    assert expected_valid is True and actual_valid is True
    assert actual_without_stale == expected_without_stale
    resources[1].x = stale_resource_x

    occupied_observation = _base_observation([*resources, _townhall(64, 64)])
    occupied_reason, occupied_valid = llm_action.get_arg_screen_tag_base_building(
        occupied_observation,
        1,
        128,
        "Build_Nexus_Near",
    )
    assert occupied_valid is False
    assert "already has a townhall" in occupied_reason

    missing_reason, missing_valid = llm_action.get_arg_screen_tag_base_building(
        observation,
        999,
        128,
        "Build_Nexus_Near",
    )
    assert missing_valid is False
    assert missing_reason == "cannot find unit 0x3e7 on screen"

    original_x = resources[0].x
    resources[0].x = -151
    stale_reason, stale_valid = llm_action.get_arg_screen_tag_base_building(
        observation,
        1,
        128,
        "Build_Nexus_Near",
    )
    assert stale_valid is False
    assert stale_reason == "cannot find unit 0x1 on screen"
    resources[0].x = original_x

    resources[0].is_on_screen = False
    offscreen_reason, offscreen_valid = llm_action.get_arg_screen_tag_base_building(
        observation,
        1,
        128,
        "Build_Nexus_Near",
    )
    assert offscreen_valid is False
    assert offscreen_reason == "cannot find unit 0x1 on screen"
    resources[0].is_on_screen = True

    resources[0].display_type = 2
    hidden_anchor_reason, hidden_anchor_valid = llm_action.get_arg_screen_tag_base_building(
        observation,
        1,
        128,
        "Build_Nexus_Near",
    )
    assert hidden_anchor_valid is False
    assert hidden_anchor_reason == "cannot find unit 0x1 on screen"
    resources[0].display_type = 1

    observation.observation.feature_screen.visibility_map = [
        [features.Visibility.SEEN] * 128 for _ in range(128)
    ]
    fogged_reason, fogged_valid = llm_action.get_arg_screen_tag_base_building(
        observation,
        1,
        128,
        "Build_Nexus_Near",
    )
    assert fogged_valid is False
    assert "complete footprint" in fogged_reason
    observation.observation.feature_screen.visibility_map = [
        [features.Visibility.VISIBLE] * 128 for _ in range(128)
    ]

    observation.observation.feature_screen.player_relative = [
        [features.PlayerRelative.SELF] * 128 for _ in range(128)
    ]
    blocked_reason, blocked_valid = llm_action.get_arg_screen_tag_base_building(
        observation,
        1,
        128,
        "Build_Nexus_Near",
    )
    assert blocked_valid is False
    assert "complete footprint" in blocked_reason

    observation = _base_observation(resources)
    observation.observation.feature_screen.buildable = [[0] * 128 for _ in range(128)]
    observation.observation.feature_screen.pathable = [[0] * 128 for _ in range(128)]
    for y in range(52, 79):
        for x in range(52, 79):
            observation.observation.feature_screen.buildable[y][x] = 1
            observation.observation.feature_screen.pathable[y][x] = 1

    relocated_position, relocated_valid = llm_action.get_arg_screen_tag_base_building(
        observation,
        1,
        128,
        "Build_Nexus_Near",
    )
    assert relocated_valid is True
    assert relocated_position == (65, 65)


def main() -> None:
    _assert_candidate_mapping()
    _assert_reserved_builder_worker_guard()
    _assert_max_frame_hook()
    _assert_atomic_log_directory_allocation()
    _assert_gas_rebalance_uses_worker_management_flag()
    _assert_build_order_ids_use_raw_function_domain()
    _assert_direct_production_contract()
    _assert_raw_unit_presence_controls_team_lifecycle()
    _assert_transient_disappearance_grace()
    _assert_confirmed_disappearance_removes_actor()
    _assert_pretranslation_abort_markers()
    _assert_multi_argument_rejection()
    _assert_world_uses_resource_centroid()
    _assert_exact_anchor_and_footprint()
    print("Python 3.9 LLM-PySC2 contract smoke passed")


if __name__ == "__main__":
    main()
