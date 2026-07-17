from __future__ import annotations

from typing import Any

from rtscortex_llm_pysc2.effect_verifier import ActionEffectVerifier
from rtscortex_llm_pysc2.production import PRODUCTION_SPECS, ProductionSpec
from rtscortex_llm_pysc2.routing import RoutedCommand


def test_build_effect_is_confirmed_when_target_structure_appears() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(command.command_id, _observation(game_loop=100, minerals=250), 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=104)

    verdicts = verifier.observe(
        _observation(
            game_loop=126,
            minerals=175,
            structures=["Nexus", "Pylon"],
            builder_orders=[35],
        )
    )

    assert len(verdicts) == 1
    assert verdicts[0].command_id == command.command_id
    assert verdicts[0].success is True
    assert verdicts[0].failure_reason is None
    assert verdicts[0].evidence is not None
    assert verdicts[0].evidence["worker_orders"] == ["35"]
    assert verdicts[0].evidence["resource_delta"] == {"minerals": -75}
    assert verdicts[0].evidence["order_seen"] is True
    assert verifier.observe(_observation(game_loop=148, minerals=200)) == []


def test_tracked_build_blocks_auto_worker_management_until_terminal() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(command.command_id, _observation(game_loop=100, minerals=250), 0xABC)

    assert verifier.blocks_auto_worker_management is True

    verifier.accept_primitive(command.command_id, game_loop=104)

    assert verifier.blocks_auto_worker_management is True

    verifier.observe(
        _observation(
            game_loop=126,
            minerals=175,
            structures=["Nexus", "Pylon"],
            builder_orders=[35],
        )
    )

    assert verifier.blocks_auto_worker_management is False


def test_build_effect_uses_world_target_after_camera_moves() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(command.command_id, _observation(game_loop=100, minerals=250), 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=104)

    verdicts = verifier.observe(
        _observation(
            game_loop=126,
            minerals=150,
            structures=["Nexus", "Pylon"],
            pylon_screen=(10, 10),
        )
    )

    assert [verdict.success for verdict in verdicts] == [True]


def test_build_effect_preserves_live_feature_screen_y_direction() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command(position=(45, 30))
    baseline = _observation(game_loop=713, minerals=170)
    baseline["raw_units"][0]["x"] = 56
    baseline["raw_units"][0]["y"] = 60
    baseline["feature_units"][0]["x"] = 64
    baseline["feature_units"][0]["y"] = 64
    verifier.track(command)
    verifier.prepare(command.command_id, baseline, 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=717)

    assert verifier.observe(_observation(game_loop=1058, minerals=170, builder_orders=[35])) == []
    assert verifier.observe(_observation(game_loop=1070, minerals=250, builder_orders=[])) == []

    current = _observation(
        game_loop=1084,
        minerals=250,
        structures=["Nexus", "Pylon"],
        builder_orders=[],
    )
    pylon = next(unit for unit in current["raw_units"] if unit["unit_type"] == "Pylon")
    pylon["x"] = 53
    pylon["y"] = 54

    verdicts = verifier.observe(current)

    assert [verdict.success for verdict in verdicts] == [True]
    assert verdicts[0].evidence is not None
    assert verdicts[0].evidence["target_position"] == (52.4375, 53.625)
    assert verdicts[0].evidence["order_last_seen_game_loop"] == 1058
    assert verdicts[0].evidence["post_order_grace_game_loops"] == 32


def test_near_build_uses_translator_resolved_screen_position() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = RoutedCommand(
        command_id="command-nexus",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Nexus_Near",
        source="planner",
        requested_arguments=("0x100",),
        resolved_arguments=("0x100",),
        rendered_action="<Build_Nexus_Near(0x100)>",
    )
    baseline = _observation(game_loop=100, minerals=500)
    verifier.track(command)
    verifier.resolve_arguments(command.command_id, [[95, 65]])
    verifier.prepare(command.command_id, baseline, 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=101)

    current = _observation(game_loop=110, minerals=100)
    current["raw_units"].append(
        {
            "tag": 0x999,
            "unit_type": "Nexus",
            "alliance": 1,
            "is_structure": True,
            "order_length": 0,
            "is_selected": False,
            "build_progress": 10,
            "x": 37.5,
            "y": 30,
        }
    )

    verdicts = verifier.observe(current)

    assert [verdict.success for verdict in verdicts] == [True]
    assert verdicts[0].evidence is not None
    assert verdicts[0].evidence["target_position"] == (37.5, 30.0)


def test_assimilator_near_keeps_exact_requested_geyser_target() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = RoutedCommand(
        command_id="command-assimilator",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Assimilator_Near",
        source="planner",
        requested_arguments=("0x100",),
        resolved_arguments=("0x100",),
        rendered_action="<Build_Assimilator_Near(0x100)>",
    )
    baseline = _observation(game_loop=100, minerals=200)
    baseline["raw_units"].append(
        {
            "tag": 0x100,
            "unit_type": "VespeneGeyser",
            "alliance": 3,
            "x": 51,
            "y": 75,
        }
    )
    verifier.track(command)
    verifier.resolve_arguments(command.command_id, [[93, 29]])
    verifier.prepare(command.command_id, baseline, 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=101)

    current = _observation(game_loop=110, minerals=125)
    current["raw_units"].append(
        {
            "tag": 0x999,
            "unit_type": "Assimilator",
            "alliance": 1,
            "is_structure": True,
            "order_length": 0,
            "is_selected": False,
            "build_progress": 10,
            "x": 51,
            "y": 75,
        }
    )

    verdicts = verifier.observe(current)

    assert [verdict.success for verdict in verdicts] == [True]
    assert verdicts[0].evidence is not None
    assert verdicts[0].evidence["target_tag"] == "0x100"
    assert verdicts[0].evidence["target_position"] == (51.0, 75.0)


def test_same_structure_at_another_position_does_not_confirm_effect() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(command.command_id, _observation(game_loop=100, minerals=250), 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=101)
    observation = _observation(
        game_loop=111,
        minerals=150,
        structures=["Nexus", "Pylon"],
    )
    pylon = next(unit for unit in observation["raw_units"] if unit["unit_type"] == "Pylon")
    pylon["x"] = 60
    pylon["y"] = 60

    verdict = verifier.observe(observation)[0]

    assert verdict.success is False
    assert verdict.failure_code == "no_build_order_observed"


def test_concurrent_same_type_builds_match_new_tags_one_to_one() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=20)
    first = _build_command(command_id="pylon-a", position=(65, 65))
    second = _build_command(command_id="pylon-b", position=(75, 65))
    baseline = _observation(game_loop=100, minerals=400)
    for command in (first, second):
        verifier.track(command)
        verifier.prepare(command.command_id, baseline, 0xABC)
        verifier.accept_primitive(command.command_id, game_loop=101)

    observation = _observation(
        game_loop=110,
        minerals=200,
        structures=["Nexus", "Pylon"],
    )
    observation["raw_units"].append(
        {
            "tag": 999,
            "unit_type": "Pylon",
            "alliance": 1,
            "is_structure": True,
            "order_length": 0,
            "is_selected": False,
            "build_progress": 50,
            "x": 33.75,
            "y": 30,
        }
    )
    observation["feature_units"].append({"tag": 999, "x": 75, "y": 65, "is_on_screen": True})

    verdicts = verifier.observe(observation)

    assert {verdict.command_id for verdict in verdicts} == {"pylon-a", "pylon-b"}
    tags = {
        verdict.evidence["observed_structure_tag"]
        for verdict in verdicts
        if verdict.evidence is not None
    }
    assert len(tags) == 2


def test_claimed_structure_tag_is_not_reused_across_observations() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=20)
    first = _build_command(command_id="pylon-a")
    second = _build_command(command_id="pylon-b")
    baseline = _observation(game_loop=100, minerals=400)
    for command in (first, second):
        verifier.track(command)
        verifier.prepare(command.command_id, baseline, 0xABC)
        verifier.accept_primitive(command.command_id, game_loop=101)

    one_new_pylon = _observation(
        game_loop=110,
        minerals=300,
        structures=["Nexus", "Pylon"],
    )

    first_verdicts = verifier.observe(one_new_pylon)

    assert len(first_verdicts) == 1
    assert first_verdicts[0].success is True
    assert first_verdicts[0].evidence is not None
    claimed_tag = first_verdicts[0].evidence["observed_structure_tag"]

    assert verifier.observe(one_new_pylon) == []

    timeout_observation = _observation(
        game_loop=121,
        minerals=300,
        structures=["Nexus", "Pylon"],
    )
    timeout_verdicts = verifier.observe(timeout_observation)

    assert len(timeout_verdicts) == 1
    assert timeout_verdicts[0].success is False
    assert timeout_verdicts[0].evidence is not None
    assert timeout_verdicts[0].evidence["observed_structure_tag"] is None
    assert claimed_tag == "0x2"


def test_resource_and_builder_order_are_diagnostic_only() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=300, builder_orders=[358]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=104)

    assert verifier.observe(_observation(game_loop=126, minerals=225, builder_orders=[35])) == []
    assert verifier.observe(_observation(game_loop=216, minerals=225, builder_orders=[35])) == []
    verdicts = verifier.observe(_observation(game_loop=552, minerals=225, builder_orders=[35]))

    assert len(verdicts) == 1
    assert verdicts[0].success is False
    assert verdicts[0].failure_code == "target_not_created"


def test_build_effect_times_out_with_diagnostic_evidence() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=250, builder_orders=[358]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=104)

    assert verifier.observe(_observation(game_loop=215, minerals=275, builder_orders=[358])) == []
    verdicts = verifier.observe(_observation(game_loop=216, minerals=275, builder_orders=[358]))

    assert len(verdicts) == 1
    assert verdicts[0].success is False
    reason = verdicts[0].failure_reason or ""
    assert "primitive accepted by PySC2" in reason
    assert "no gameplay effect confirmed after 112 game loops" in reason
    assert "base timeout 112, maximum 112" in reason
    assert "Pylon tags []->[]" in reason
    assert "minerals 250->275" in reason
    assert "builder 0xabc status active->active" in reason
    assert "orders [358]->[358]" in reason
    assert "primitive did not establish construction" in reason


def test_build_effect_diagnostic_identifies_replaced_worker_order() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=250, builder_orders=[358]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)
    assert verifier.observe(_observation(game_loop=105, minerals=175, builder_orders=[35])) == []

    assert verifier.observe(_observation(game_loop=111, minerals=250, builder_orders=[154])) == []
    verdict = verifier.observe(_observation(game_loop=141, minerals=250, builder_orders=[154]))[0]

    assert verdict.success is False
    assert verdict.failure_code == "worker_order_replaced"
    assert "observed and later changed" in (verdict.failure_reason or "")


def test_post_order_grace_expires_at_32_loops_without_structure() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(command.command_id, _observation(game_loop=100, minerals=250), 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=101)

    assert verifier.observe(_observation(game_loop=105, minerals=150, builder_orders=[35])) == []
    assert verifier.observe(_observation(game_loop=136, minerals=150, builder_orders=[])) == []
    verdict = verifier.observe(_observation(game_loop=137, minerals=150, builder_orders=[]))[0]

    assert verdict.failure_code == "target_not_created"


def test_active_nexus_order_extends_timeout_until_effect_is_visible() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _nexus_command()
    verifier.track(command)
    verifier.prepare(command.command_id, _observation(game_loop=100, minerals=500), 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=101)

    assert verifier.observe(_observation(game_loop=111, minerals=100, builder_orders=[34])) == []
    assert verifier.blocks_auto_worker_management is True

    observation = _observation(game_loop=130, minerals=100, builder_orders=[34])
    observation["raw_units"].append(
        {
            "tag": 0x999,
            "unit_type": "Nexus",
            "alliance": 1,
            "is_structure": True,
            "order_length": 0,
            "is_selected": False,
            "build_progress": 1,
            "x": 31.875,
            "y": 30,
        }
    )

    verdict = verifier.observe(observation)[0]

    assert verdict.success is True
    assert verdict.evidence is not None
    assert verdict.evidence["active_order_extension"] is True
    assert verdict.evidence["effective_timeout_game_loops"] == 120
    assert verifier.blocks_auto_worker_management is False


def test_active_build_order_extension_has_a_hard_limit() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _nexus_command()
    verifier.track(command)
    verifier.prepare(command.command_id, _observation(game_loop=100, minerals=500), 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=101)

    assert verifier.observe(_observation(game_loop=111, minerals=100, builder_orders=[34])) == []
    assert verifier.observe(_observation(game_loop=140, minerals=100, builder_orders=[34])) == []
    assert verifier.observe(_observation(game_loop=210, minerals=100, builder_orders=[34])) == []
    verdict = verifier.observe(_observation(game_loop=221, minerals=100, builder_orders=[34]))[0]

    assert verdict.success is False
    assert verdict.failure_code == "target_not_created"
    assert verdict.evidence is not None
    assert verdict.evidence["elapsed_game_loops"] == 120
    assert verdict.evidence["effective_timeout_game_loops"] == 120


def test_changed_order_without_observed_build_order_is_not_called_replaced() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=250, builder_orders=[358]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    verdict = verifier.observe(_observation(game_loop=111, minerals=250, builder_orders=[154]))[0]

    assert verdict.success is False
    assert verdict.failure_code == "no_build_order_observed"
    assert "automatic worker" not in (verdict.failure_reason or "")


def test_move_minimap_uses_builder_motion_instead_of_global_camera_position() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _move_command()

    assert verifier.track(command) is True
    assert verifier.blocks_auto_worker_management is False
    verifier.prepare(
        command.command_id,
        _move_observation(game_loop=100, center=(8, 8), builder_position=(30, 30)),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    assert (
        verifier.observe(
            _move_observation(game_loop=102, center=(48, 48), builder_position=(30, 30))
        )
        == []
    )
    assert verifier.blocks_auto_worker_management is False
    verdict = verifier.observe(
        _move_observation(game_loop=103, center=(8, 8), builder_position=(31.5, 30))
    )[0]

    assert verdict.success is True
    assert verdict.status == "succeeded"
    assert verdict.evidence is not None
    assert verdict.evidence["target_type"] == "Move_Minimap"
    assert verdict.evidence["target_position"] == (48.0, 48.0)
    assert verdict.evidence["builder_tag"] == "0xabc"
    assert verdict.evidence["dispatched_loop"] == 100
    assert verdict.evidence["accepted_loop"] == 101
    assert verdict.evidence["confirmed_loop"] == 103
    assert verdict.evidence["baseline_builder_position"] == (30.0, 30.0)
    assert verdict.evidence["observed_builder_position"] == (31.5, 30.0)
    assert verdict.evidence["builder_displacement"] == 1.5
    assert verdict.evidence["move_order_seen"] is False
    assert verdict.evidence["effective_timeout_game_loops"] == 10
    assert verifier.is_tracked(command.command_id) is False


def test_move_minimap_accepts_raw_move_order_before_position_changes() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _move_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _move_observation(game_loop=100, center=(8, 8), builder_position=(30, 30)),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    verdict = verifier.observe(
        _move_observation(
            game_loop=102,
            center=(8, 8),
            builder_position=(30, 30),
            builder_orders=[13],
        )
    )[0]

    assert verdict.success is True
    assert verdict.evidence is not None
    assert verdict.evidence["move_order_seen"] is True
    assert verdict.evidence["builder_displacement"] == 0.0


def test_move_minimap_times_out_after_one_base_window_without_unit_effect() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _move_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _move_observation(game_loop=100, center=(8, 8), builder_position=(30, 30)),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    assert (
        verifier.observe(
            _move_observation(game_loop=110, center=(48, 48), builder_position=(30, 30))
        )
        == []
    )
    verdict = verifier.observe(
        _move_observation(game_loop=111, center=(48, 48), builder_position=(30, 30))
    )[0]

    assert verdict.success is False
    assert verdict.status == "failed"
    assert verdict.failure_code == "effect_timeout"
    assert "did not start after 10 game loops" in (verdict.failure_reason or "")
    assert verdict.evidence is not None
    assert verdict.evidence["confirmed_loop"] is None
    assert verdict.evidence["elapsed_game_loops"] == 10
    assert verdict.evidence["effective_timeout_game_loops"] == 10


def test_move_minimap_is_unconfirmed_when_episode_ends_in_transit() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _move_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _move_observation(game_loop=100, center=(8, 8), builder_position=(30, 30)),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)
    assert (
        verifier.observe(
            _move_observation(game_loop=105, center=(20, 20), builder_position=(30, 30))
        )
        == []
    )

    verdict = verifier.fail_pending("episode ended before gameplay effect was confirmed")[0]

    assert verdict.success is False
    assert verdict.status == "unconfirmed"
    assert verdict.failure_code == "episode_ended_unconfirmed"
    assert verdict.evidence is not None
    assert verdict.evidence["confirmed_loop"] is None
    assert verifier.is_tracked(command.command_id) is False


def test_immediate_action_is_not_tracked_for_effect_confirmation() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = RoutedCommand(
        command_id="command-noop",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="No_Operation",
        rendered_action="<No_Operation()>",
    )

    assert verifier.track(command) is False
    assert verifier.is_tracked(command.command_id) is False


def test_stargate_build_effect_is_confirmed_at_resolved_structure_position() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _stargate_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=500, builder_orders=[]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=104)
    current = _observation(
        game_loop=126,
        minerals=350,
        structures=["Nexus", "Stargate"],
        builder_orders=[],
    )
    stargate = next(unit for unit in current["raw_units"] if unit["unit_type"] == "Stargate")
    stargate["x"] = 31.875
    stargate["y"] = 30

    verdicts = verifier.observe(current)

    assert len(verdicts) == 1
    assert verdicts[0].success is True
    assert verdicts[0].evidence is not None
    assert verdicts[0].evidence["target_type"] == "Stargate"
    assert verdicts[0].evidence["target_position"] == (31.875, 30.0)
    assert verdicts[0].evidence["observed_structure_tag"] == "0x2"


def test_stargate_raw_build_order_marks_order_seen_for_diagnostics() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _stargate_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=500, builder_orders=[]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    assert verifier.observe(_observation(game_loop=105, minerals=350, builder_orders=[42])) == []
    assert verifier.observe(_observation(game_loop=111, minerals=500, builder_orders=[154])) == []
    verdict = verifier.observe(_observation(game_loop=141, minerals=500, builder_orders=[154]))[0]

    assert verdict.success is False
    assert verdict.failure_code == "worker_order_replaced"
    assert verdict.evidence is not None
    assert verdict.evidence["order_seen"] is True
    assert verdict.evidence["order_last_seen_game_loop"] == 105


def test_shield_battery_order_48_and_new_tag_confirm_build_effect() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = RoutedCommand(
        command_id="command-shield-battery",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_ShieldBattery_Screen",
        source="planner",
        requested_arguments=([65, 65],),
        resolved_arguments=([65, 65],),
        rendered_action="<Build_ShieldBattery_Screen([65,65])>",
    )
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=200, builder_orders=[]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    assert verifier.observe(_observation(game_loop=105, minerals=100, builder_orders=[48])) == []
    current = _observation(
        game_loop=106,
        minerals=100,
        structures=["Nexus", "ShieldBattery"],
        builder_orders=[48],
    )
    battery = next(unit for unit in current["raw_units"] if unit["unit_type"] == "ShieldBattery")
    battery["x"] = 31.875
    battery["y"] = 30

    verdict = verifier.observe(current)[0]

    assert verdict.success is True
    assert verdict.evidence is not None
    assert verdict.evidence["target_type"] == "ShieldBattery"
    assert verdict.evidence["observed_structure_tag"] == "0x2"
    assert verdict.evidence["order_seen"] is True


def test_forge_order_38_and_new_tag_confirm_build_effect() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = RoutedCommand(
        command_id="command-forge",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Forge_Screen",
        source="planner",
        requested_arguments=([65, 65],),
        resolved_arguments=([65, 65],),
        rendered_action="<Build_Forge_Screen([65,65])>",
    )
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=250, builder_orders=[]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    assert verifier.observe(_observation(game_loop=105, minerals=100, builder_orders=[38])) == []
    current = _observation(
        game_loop=106,
        minerals=100,
        structures=["Nexus", "Forge"],
        builder_orders=[38],
    )
    forge = next(unit for unit in current["raw_units"] if unit["unit_type"] == "Forge")
    forge["x"] = 31.875
    forge["y"] = 30

    verdict = verifier.observe(current)[0]

    assert verdict.success is True
    assert verdict.evidence is not None
    assert verdict.evidence["target_type"] == "Forge"
    assert verdict.evidence["observed_structure_tag"] == "0x2"
    assert verdict.evidence["order_seen"] is True


def test_all_supported_train_actions_confirm_the_exact_producer_order() -> None:
    for index, spec in enumerate(PRODUCTION_SPECS.values()):
        verifier = ActionEffectVerifier(timeout_game_loops=10)
        command = _production_command(spec, command_id=f"train-{index}")
        producer_tag = 0xA00 + index
        verifier.track(command)
        verifier.prepare(
            command.command_id,
            _production_observation(spec, game_loop=100, producer_tag=producer_tag),
            None,
            producer_tag=producer_tag,
        )
        verifier.accept_primitive(command.command_id, game_loop=101)

        verdict = verifier.observe(
            _production_observation(
                spec,
                game_loop=102,
                producer_tag=producer_tag,
                producer_orders=[spec.raw_order_id],
            )
        )[0]

        assert verdict.success is True
        assert verdict.evidence is not None
        assert verdict.evidence["producer_tag"] == hex(producer_tag)
        assert verdict.evidence["producer_type"] == spec.producer_type
        assert verdict.evidence["expected_unit_type"] == spec.unit_type
        assert verdict.evidence["expected_order_id"] == spec.raw_order_id
        assert verdict.evidence["production_order_seen"] is True
        assert verdict.evidence["confirmation_kind"] == "producer_order"


def test_new_unit_near_the_exact_producer_can_confirm_missed_short_order() -> None:
    spec = PRODUCTION_SPECS["Train_Adept"]
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _production_command(spec)
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _production_observation(spec, game_loop=100, producer_position=(20, 20)),
        None,
        producer_tag=0xA00,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    far = _production_observation(
        spec,
        game_loop=102,
        producer_position=(20, 20),
        trained_units=[(0xB00, (40, 40))],
    )
    assert verifier.observe(far) == []

    near = _production_observation(
        spec,
        game_loop=103,
        producer_position=(20, 20),
        trained_units=[(0xB00, (40, 40)), (0xB01, (24, 20))],
    )
    verdict = verifier.observe(near)[0]

    assert verdict.success is True
    assert verdict.evidence is not None
    assert verdict.evidence["confirmation_kind"] == "new_unit"
    assert verdict.evidence["new_unit_tag"] == "0xb01"
    assert verdict.evidence["production_order_seen"] is False


def test_one_producer_order_transition_confirms_only_one_pending_command() -> None:
    spec = PRODUCTION_SPECS["Train_Zealot"]
    verifier = ActionEffectVerifier(timeout_game_loops=20)
    commands = [
        _production_command(spec, command_id="train-a"),
        _production_command(spec, command_id="train-b"),
    ]
    baseline = _production_observation(spec, game_loop=100)
    for command in commands:
        verifier.track(command)
        verifier.prepare(command.command_id, baseline, None, producer_tag=0xA00)
        verifier.accept_primitive(command.command_id, game_loop=101)

    active = _production_observation(
        spec,
        game_loop=102,
        producer_orders=[spec.raw_order_id],
    )
    first = verifier.observe(active)

    assert [verdict.command_id for verdict in first] == ["train-a"]
    assert verifier.observe(active) == []
    assert verifier.observe(_production_observation(spec, game_loop=103)) == []

    second = verifier.observe(
        _production_observation(
            spec,
            game_loop=104,
            producer_orders=[spec.raw_order_id],
        )
    )
    assert [verdict.command_id for verdict in second] == ["train-b"]


def test_earlier_accepted_command_claims_order_before_lexicographically_smaller_id() -> None:
    spec = PRODUCTION_SPECS["Train_Zealot"]
    verifier = ActionEffectVerifier(timeout_game_loops=20)
    older = _production_command(spec, command_id="train-z-older")
    newer = _production_command(spec, command_id="train-a-newer")
    baseline = _production_observation(spec, game_loop=100)
    for command, accepted_loop in ((older, 101), (newer, 102)):
        verifier.track(command)
        verifier.prepare(command.command_id, baseline, None, producer_tag=0xA00)
        verifier.accept_primitive(command.command_id, game_loop=accepted_loop)

    verdicts = verifier.observe(
        _production_observation(
            spec,
            game_loop=103,
            producer_orders=[spec.raw_order_id],
        )
    )

    assert [verdict.command_id for verdict in verdicts] == ["train-z-older"]


def test_concurrent_new_units_match_nearest_producers_one_to_one() -> None:
    spec = PRODUCTION_SPECS["Train_Phoenix"]
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    commands = [
        (_production_command(spec, command_id="phoenix-left"), 0xA00),
        (_production_command(spec, command_id="phoenix-right"), 0xA01),
    ]
    baseline = _production_observation(
        spec,
        game_loop=100,
        producers=[(0xA00, (10, 10), []), (0xA01, (30, 10), [])],
    )
    for command, producer_tag in commands:
        verifier.track(command)
        verifier.prepare(command.command_id, baseline, None, producer_tag=producer_tag)
        verifier.accept_primitive(command.command_id, game_loop=101)

    current = _production_observation(
        spec,
        game_loop=102,
        producers=[(0xA00, (10, 10), []), (0xA01, (30, 10), [])],
        trained_units=[(0xB00, (12, 10)), (0xB01, (29, 10))],
    )
    verdicts = verifier.observe(current)

    evidence = {
        verdict.command_id: verdict.evidence for verdict in verdicts if verdict.evidence is not None
    }
    assert evidence["phoenix-left"]["new_unit_tag"] == "0xb00"
    assert evidence["phoenix-right"]["new_unit_tag"] == "0xb01"


def test_production_timeout_distinguishes_missing_empty_and_replaced_producer() -> None:
    spec = PRODUCTION_SPECS["Train_Oracle"]
    cases = [
        ([], True, "no_production_order_observed"),
        ([13], True, "production_order_replaced"),
        ([], False, "producer_not_observable"),
    ]
    for index, (orders, producer_visible, expected_code) in enumerate(cases):
        verifier = ActionEffectVerifier(timeout_game_loops=10)
        command = _production_command(spec, command_id=f"oracle-{index}")
        verifier.track(command)
        verifier.prepare(
            command.command_id,
            _production_observation(spec, game_loop=100),
            None,
            producer_tag=0xA00,
        )
        verifier.accept_primitive(command.command_id, game_loop=101)
        current = _production_observation(
            spec,
            game_loop=111,
            producer_orders=orders,
            include_default_producer=producer_visible,
        )

        verdict = verifier.observe(current)[0]

        assert verdict.success is False
        assert verdict.failure_code == expected_code


def test_accepted_production_is_unconfirmed_at_episode_end() -> None:
    spec = PRODUCTION_SPECS["Train_VoidRay"]
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _production_command(spec)
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _production_observation(spec, game_loop=100),
        None,
        producer_tag=0xA00,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    verdict = verifier.fail_pending("episode ended before gameplay effect was confirmed")[0]

    assert verdict.status == "unconfirmed"
    assert verdict.failure_code == "episode_ended_unconfirmed"
    assert verdict.evidence is not None
    assert verdict.evidence["effect_kind"] == "production"
    assert verifier.is_tracked(command.command_id) is False


def test_resources_and_top_level_queue_cannot_confirm_production() -> None:
    spec = PRODUCTION_SPECS["Train_Stalker"]
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _production_command(spec)
    baseline = _production_observation(spec, game_loop=100)
    verifier.track(command)
    verifier.prepare(command.command_id, baseline, None, producer_tag=0xA00)
    verifier.accept_primitive(command.command_id, game_loop=101)

    changed = _production_observation(spec, game_loop=102)
    changed["player_common"] = {"minerals": 375, "vespene": 450, "food_used": 22}
    changed["production_queue"] = [{"ability_id": spec.ability_id, "build_progress": 0.2}]
    assert verifier.observe(changed) == []

    changed["game_loop"] = 111
    verdict = verifier.observe(changed)[0]

    assert verdict.success is False
    assert verdict.failure_code == "no_production_order_observed"


def test_pending_production_does_not_block_auto_worker_management() -> None:
    spec = PRODUCTION_SPECS["Train_Zealot"]
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _production_command(spec)

    verifier.track(command)

    assert verifier.blocks_auto_worker_management is False


def _production_command(
    spec: ProductionSpec,
    *,
    command_id: str = "command-train",
) -> RoutedCommand:
    return RoutedCommand(
        command_id=command_id,
        actor="Developer/Empty",
        team_name="Empty",
        name=spec.action_name,
        source="planner",
        requested_arguments=(),
        resolved_arguments=(),
        rendered_action=f"<{spec.action_name}()>",
    )


def _production_observation(
    spec: ProductionSpec,
    *,
    game_loop: int,
    producer_tag: int = 0xA00,
    producer_position: tuple[float, float] = (20, 20),
    producer_orders: list[int] | None = None,
    producers: list[tuple[int, tuple[float, float], list[int]]] | None = None,
    trained_units: list[tuple[int, tuple[float, float]]] | None = None,
    include_default_producer: bool = True,
) -> dict[str, Any]:
    producer_definitions = producers
    if producer_definitions is None:
        producer_definitions = (
            [(producer_tag, producer_position, producer_orders or [])]
            if include_default_producer
            else []
        )
    raw_units: list[dict[str, Any]] = []
    for tag, position, orders in producer_definitions:
        raw_units.append(
            {
                "tag": tag,
                "unit_type": spec.producer_type,
                "alliance": 1,
                "is_structure": True,
                "order_length": len(orders),
                **{f"order_id_{index}": order for index, order in enumerate(orders)},
                "build_progress": 100,
                "x": position[0],
                "y": position[1],
            }
        )
    raw_units.extend(
        {
            "tag": tag,
            "unit_type": spec.unit_type,
            "alliance": 1,
            "is_structure": False,
            "order_length": 0,
            "build_progress": 100,
            "x": position[0],
            "y": position[1],
        }
        for tag, position in trained_units or []
    )
    return {
        "game_loop": game_loop,
        "player_common": {
            "minerals": 500,
            "vespene": 500,
            "food_used": 20,
        },
        "raw_units": raw_units,
    }


def _build_command(
    *,
    command_id: str = "command-pylon",
    position: tuple[int, int] = (65, 65),
) -> RoutedCommand:
    return RoutedCommand(
        command_id=command_id,
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        source="planner",
        requested_arguments=(list(position),),
        resolved_arguments=(list(position),),
        rendered_action=f"<Build_Pylon_Screen([{position[0]},{position[1]}])>",
    )


def _stargate_command() -> RoutedCommand:
    return RoutedCommand(
        command_id="command-stargate",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Stargate_Screen",
        source="planner",
        requested_arguments=([65, 65],),
        resolved_arguments=([65, 65],),
        rendered_action="<Build_Stargate_Screen([65,65])>",
    )


def _nexus_command() -> RoutedCommand:
    return RoutedCommand(
        command_id="command-nexus",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Nexus_Near",
        source="planner",
        requested_arguments=("0x100",),
        resolved_arguments=([65, 65],),
        rendered_action="<Build_Nexus_Near(0x100)>",
    )


def _move_command() -> RoutedCommand:
    return RoutedCommand(
        command_id="command-move",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Move_Minimap",
        source="planner",
        requested_arguments=([48, 48],),
        resolved_arguments=([48, 48],),
        rendered_action="<Move_Minimap([48,48])>",
    )


def _observation(
    *,
    game_loop: int,
    minerals: int,
    structures: list[str] | None = None,
    builder_orders: list[int] | None = None,
    pylon_screen: tuple[int, int] = (65, 65),
) -> dict[str, Any]:
    units: list[dict[str, Any]] = [
        {
            "tag": int("abc", 16),
            "unit_type": "Probe",
            "alliance": 1,
            "is_structure": False,
            "order_length": len([358] if builder_orders is None else builder_orders),
            **{
                f"order_id_{index}": order
                for index, order in enumerate([358] if builder_orders is None else builder_orders)
            },
            "is_selected": True,
            "build_progress": 100,
            "x": 30,
            "y": 30,
        }
    ]
    units.extend(
        {
            "tag": index + 1,
            "unit_type": unit_type,
            "alliance": 1,
            "is_structure": True,
            "order_length": 0,
            "is_selected": False,
            "build_progress": 50 if unit_type == "Pylon" else 100,
            "x": 31.875 if unit_type == "Pylon" else 25,
            "y": 30 if unit_type == "Pylon" else 25,
        }
        for index, unit_type in enumerate(structures or ["Nexus"])
    )
    return {
        "game_loop": game_loop,
        "player_common": {"minerals": minerals},
        "raw_units": units,
        "feature_units": [
            {"tag": 0xABC, "x": 55, "y": 65, "is_on_screen": True},
            *[
                {
                    "tag": unit["tag"],
                    "x": pylon_screen[0] if unit["unit_type"] == "Pylon" else 30,
                    "y": pylon_screen[1] if unit["unit_type"] == "Pylon" else 30,
                    "is_on_screen": True,
                }
                for unit in units
                if unit["unit_type"] != "Probe"
            ],
        ],
    }


def _move_observation(
    *,
    game_loop: int,
    center: tuple[int, int],
    builder_position: tuple[float, float] = (30, 30),
    builder_orders: list[int] | None = None,
) -> dict[str, Any]:
    observation = _observation(
        game_loop=game_loop,
        minerals=250,
        builder_orders=[] if builder_orders is None else builder_orders,
    )
    observation["raw_units"][0]["x"] = builder_position[0]
    observation["raw_units"][0]["y"] = builder_position[1]
    camera = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(center[1] - 1, center[1] + 2):
        for x in range(center[0] - 1, center[0] + 2):
            camera[y][x] = 1
    observation["feature_minimap"] = {"camera": camera}
    return observation
