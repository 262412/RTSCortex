from __future__ import annotations

from typing import Any

from rtscortex_llm_pysc2.effect_verifier import ActionEffectVerifier
from rtscortex_llm_pysc2.routing import RoutedCommand


def test_build_effect_is_confirmed_when_target_structure_appears() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(command.command_id, _observation(game_loop=100, minerals=250), 0xABC)
    verifier.accept_primitive(command.command_id, game_loop=104)

    verdicts = verifier.observe(
        _observation(game_loop=126, minerals=175, structures=["Nexus", "Pylon"])
    )

    assert len(verdicts) == 1
    assert verdicts[0].command_id == command.command_id
    assert verdicts[0].success is True
    assert verdicts[0].failure_reason is None
    assert verifier.observe(_observation(game_loop=148, minerals=200)) == []


def test_build_effect_accepts_combined_resource_and_builder_order_evidence() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=300, builder_orders=[295]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=104)

    verdicts = verifier.observe(_observation(game_loop=126, minerals=225, builder_orders=[881]))

    assert len(verdicts) == 1
    assert verdicts[0].success is True


def test_build_effect_times_out_with_diagnostic_evidence() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=112)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=250, builder_orders=[295]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=104)

    assert verifier.observe(_observation(game_loop=215, minerals=275, builder_orders=[295])) == []
    verdicts = verifier.observe(_observation(game_loop=216, minerals=275, builder_orders=[295]))

    assert len(verdicts) == 1
    assert verdicts[0].success is False
    reason = verdicts[0].failure_reason or ""
    assert "primitive accepted by PySC2" in reason
    assert "no gameplay effect confirmed within 112 game loops" in reason
    assert "Pylon count 0->0" in reason
    assert "minerals 250->275" in reason
    assert "builder 0xabc status active->active" in reason
    assert "orders [295]->[295]" in reason
    assert "feature-action placement likely failed" in reason


def test_build_effect_diagnostic_identifies_replaced_worker_order() -> None:
    verifier = ActionEffectVerifier(timeout_game_loops=10)
    command = _build_command()
    verifier.track(command)
    verifier.prepare(
        command.command_id,
        _observation(game_loop=100, minerals=250, builder_orders=[295]),
        0xABC,
    )
    verifier.accept_primitive(command.command_id, game_loop=101)

    verdict = verifier.observe(_observation(game_loop=111, minerals=250, builder_orders=[16]))[0]

    assert verdict.success is False
    assert "automatic worker management or a later action" in (verdict.failure_reason or "")


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


def _build_command() -> RoutedCommand:
    return RoutedCommand(
        command_id="command-pylon",
        actor="Builder/Builder-Probe-1",
        team_name="Builder-Probe-1",
        name="Build_Pylon_Screen",
        rendered_action="<Build_Pylon_Screen([65,65])>",
    )


def _observation(
    *,
    game_loop: int,
    minerals: int,
    structures: list[str] | None = None,
    builder_orders: list[int] | None = None,
) -> dict[str, Any]:
    units: list[dict[str, Any]] = [
        {
            "tag": int("abc", 16),
            "unit_type": "Probe",
            "alliance": 1,
            "is_structure": False,
            "order_length": len([295] if builder_orders is None else builder_orders),
            **{
                f"order_id_{index}": order
                for index, order in enumerate([295] if builder_orders is None else builder_orders)
            },
            "is_selected": True,
            "build_progress": 100,
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
        }
        for index, unit_type in enumerate(structures or ["Nexus"])
    )
    return {
        "game_loop": game_loop,
        "player_common": {"minerals": minerals},
        "raw_units": units,
    }
