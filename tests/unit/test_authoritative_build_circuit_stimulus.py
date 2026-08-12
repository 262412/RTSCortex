from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rtscortex_llm_pysc2.circuit_canary import AuthoritativeBuildCircuitCanary
from rtscortex_llm_pysc2.routing import RoutedBuildPreflightRequest, RoutedCommand

from rtscortex.contracts import (
    authoritative_build_preflight_authorization_id,
    authoritative_build_preflight_request_id,
)

OPERATION_ID = "operation:" + "a" * 64


def _command(
    ordinal: int,
    *,
    authorization: dict[str, object] | None = None,
) -> RoutedCommand:
    command_id = f"command-{ordinal}"
    attempt_id = (
        "attempt:"
        + hashlib.sha256(
            json.dumps(
                {
                    "operation_id": OPERATION_ID,
                    "command_id": command_id,
                    "attempt_ordinal": ordinal,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    return RoutedCommand(
        command_id=command_id,
        actor="Builder/Probe",
        team_name="Probe",
        name="Build_Pylon_Screen",
        rendered_action="Build_Pylon_Screen('now', [100], [30, 30])",
        operation_id=OPERATION_ID,
        attempt_id=attempt_id,
        attempt_ordinal=ordinal,
        semantic_action="BUILD PYLON",
        placement_candidate_id="candidate:" + "b" * 64,
        placement_revision=f"revision-{ordinal}",
        authoritative_build_preflight=authorization,
    )


def _state(
    streak: int,
    *,
    circuit_open: bool,
    pending_authorization_id: str | None = None,
) -> dict[str, object]:
    return {
        "operation_id": OPERATION_ID,
        "streak": streak,
        "threshold": 3,
        "circuit_open": circuit_open,
        "pending_authorization_id": pending_authorization_id,
    }


def _preflight() -> tuple[RoutedBuildPreflightRequest, dict[str, object]]:
    opened_attempt_id = _command(2).attempt_id
    assert opened_attempt_id is not None
    arguments: list[object] = []
    request_id = authoritative_build_preflight_request_id(
        operation_id=OPERATION_ID,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        actor="Builder/Probe",
        requested_arguments=arguments,
        opened_command_id="command-2",
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity="build-legality:" + "c" * 64,
        observation_revision="revision-preflight",
        observation_game_loop=42,
    )
    request = RoutedBuildPreflightRequest(
        request_id=request_id,
        run_id="run-canary",
        episode_id="episode-0",
        step_id=42,
        operation_id=OPERATION_ID,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        actor="Builder/Probe",
        team_name="Probe",
        requested_arguments=(),
        opened_command_id="command-2",
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity="build-legality:" + "c" * 64,
        observation_revision="revision-preflight",
        observation_game_loop=42,
    )
    target = (31.0, 30.0)
    material = "build-legality:" + "d" * 64
    authorization_id = authoritative_build_preflight_authorization_id(
        request_id=request_id,
        operation_id=OPERATION_ID,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        builder_tag=200,
        ability_id=881,
        world_target=target,
        target_state_revision="target-preflight",
        material_legality_identity=material,
        observation_revision="revision-preflight",
        observation_game_loop=42,
        expires_game_loop=154,
    )
    result: dict[str, object] = {
        "protocol_version": "1.1",
        "request_id": request_id,
        "authorization_id": authorization_id,
        "run_id": "run-canary",
        "episode_id": "episode-0",
        "step_id": 42,
        "operation_id": OPERATION_ID,
        "operation_epoch": 0,
        "action_name": "Build_Pylon_Screen",
        "actor": "Builder/Probe",
        "requested_arguments": [],
        "opened_command_id": "command-2",
        "opened_attempt_id": opened_attempt_id,
        "opened_attempt_ordinal": 2,
        "blocked_material_legality_identity": "build-legality:" + "c" * 64,
        "status": "authorized",
        "authorized": True,
        "reason": "authoritative_material_change",
        "circuit_open": False,
        "builder_tag": 200,
        "ability_id": 881,
        "world_target": [31.0, 30.0],
        "target_state_revision": "target-preflight",
        "material_legality_identity": material,
        "observation_revision": "revision-preflight",
        "observation_game_loop": 42,
        "expires_game_loop": 154,
        "state_transition": "open_to_reset",
        "material_change_reason": "builder_changed",
        "invalid_evidence_reasons": [],
    }
    return request, result


def _failure_diagnostic(command: RoutedCommand, streak: int) -> dict[str, object]:
    opened = streak == 3
    return {
        "command_id": command.command_id,
        "failure_code": "placement_candidate_stale",
        "primitive_constructed": False,
        "authoritative_pre_dispatch": {
            "operation_id": OPERATION_ID,
            "attempt_id": command.attempt_id,
            "attempt_ordinal": command.attempt_ordinal,
            "builder_tag": 100,
            "streak": streak,
            "threshold": 3,
            "circuit_open": opened,
            "duplicate_attempt": False,
            "state_transition": "closed_to_open" if opened else None,
            "material_legality_identity": "build-legality:" + "c" * 64,
            "material_evidence_valid": True,
            "invalid_evidence_reasons": [],
            "ability_id": 881,
            "world_target": [30.0, 30.0],
        },
    }


def _observe_failure(
    controller: AuthoritativeBuildCircuitCanary,
    ordinal: int,
) -> RoutedCommand:
    command = _command(ordinal)
    controller.observe_runtime_decision(
        (command,),
        idle_reason=None,
        planner_pending=False,
        game_loop=ordinal * 10 + 1,
        observation_revision=f"revision-{ordinal}",
        builder_tag=100,
        reservation_count=0,
        leased_builder_tags=(),
        effect_inflight_count=0,
        authoritative_state=None if ordinal == 0 else _state(ordinal, circuit_open=False),
    )
    assert controller.should_hold_raw_dispatch(observation_revision=f"revision-{ordinal}")
    assert not controller.should_hold_raw_dispatch(observation_revision=f"revision-{ordinal}")
    controller.observe_raw_result(
        dispatch=None,
        diagnostic=_failure_diagnostic(command, ordinal + 1),
        game_loop=ordinal * 10 + 2,
        observation_revision=f"revision-{ordinal}-new",
        authoritative_state=_state(ordinal + 1, circuit_open=ordinal == 2),
        reservation_count=0,
        leased_builder_tags=(),
        effect_inflight_count=0,
    )
    return command


def test_canary_requires_three_failures_core_defer_and_real_effect(tmp_path: Path) -> None:
    controller = AuthoritativeBuildCircuitCanary(
        run_id="run-canary",
        episode_id="episode-0",
        seed=7,
        journal_path=tmp_path / "authoritative-build-circuit-canary.jsonl",
    )

    for ordinal in range(3):
        _observe_failure(controller, ordinal)

    assert controller.phase == "awaiting_core_defer"
    assert controller.required_builder_tag == 100
    waiting = controller.observe_runtime_decision(
        (),
        idle_reason="waiting_for_planner",
        planner_pending=True,
        game_loop=39,
        observation_revision="revision-planner-pending",
        builder_tag=100,
        reservation_count=0,
        leased_builder_tags=(),
        effect_inflight_count=0,
        authoritative_state=_state(3, circuit_open=True),
    )
    assert not waiting.core_defer_observed
    assert controller.phase == "awaiting_core_defer"
    result = controller.observe_runtime_decision(
        (),
        idle_reason="plan_commands_deferred",
        planner_pending=False,
        game_loop=40,
        observation_revision="revision-core",
        builder_tag=100,
        reservation_count=0,
        leased_builder_tags=(),
        effect_inflight_count=0,
        authoritative_state=_state(3, circuit_open=True),
    )
    assert result.core_defer_observed

    controller.bind_replacement_builder(
        200,
        game_loop=41,
        observation_revision="revision-rebind",
    )
    assert controller.required_builder_tag == 200
    preflight_request, preflight_result = _preflight()
    authorization_id = str(preflight_result["authorization_id"])
    preflight_decision = controller.observe_runtime_decision(
        (),
        preflight_results=((preflight_request, preflight_result),),
        idle_reason="plan_commands_deferred",
        planner_pending=False,
        game_loop=42,
        observation_revision="revision-preflight",
        builder_tag=200,
        reservation_count=0,
        leased_builder_tags=(),
        effect_inflight_count=0,
        authoritative_state=_state(
            0,
            circuit_open=False,
            pending_authorization_id=authorization_id,
        ),
    )
    assert preflight_decision.core_defer_observed
    reset = _command(3, authorization=preflight_result)
    controller.observe_runtime_decision(
        (reset,),
        idle_reason=None,
        planner_pending=False,
        game_loop=42,
        observation_revision="revision-reset-command",
        builder_tag=200,
        reservation_count=0,
        leased_builder_tags=(),
        effect_inflight_count=0,
        authoritative_state=_state(
            0,
            circuit_open=False,
            pending_authorization_id=authorization_id,
        ),
    )
    dispatch = SimpleNamespace(command=reset, approach_only=False)
    diagnostic = {
        "command_id": reset.command_id,
        "builder_tag": "0xc8",
        "placement_query_result": "Success",
        "available_ability_query": "available",
        "target_legality_fingerprint": "legal:" + "d" * 64,
        "primitive_constructed": True,
        "primitive_submitted": False,
        "observation_revision": "revision-query",
    }
    controller.observe_raw_result(
        dispatch=dispatch,
        diagnostic=diagnostic,
        game_loop=43,
        observation_revision="revision-query",
        authoritative_state=_state(0, circuit_open=False),
        reservation_count=1,
        leased_builder_tags=(200,),
        effect_inflight_count=1,
    )
    submitted = {
        **diagnostic,
        "primitive_submitted": True,
        "primitive_submitted_game_loop": 43,
    }
    controller.observe_primitive_submitted(
        dispatch,
        game_loop=43,
        diagnostic=submitted,
        reservation_count=1,
        leased_builder_tags=(200,),
        effect_inflight_count=1,
    )
    controller.observe_effect_reports(
        (
            {
                "command_id": reset.command_id,
                "status": "succeeded",
                "effect_evidence": {
                    "effect_kind": "build",
                    "build_started": True,
                    "observed_structure_tag": "0x300",
                    "confirmation_kind": "new_structure",
                },
            },
        ),
        game_loop=44,
        observation_revision="revision-effect",
        reservation_count=0,
        leased_builder_tags=(),
        effect_inflight_count=0,
        authoritative_state=_state(0, circuit_open=False),
    )

    assert controller.complete
    journal = [
        json.loads(line)
        for line in controller.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["phase"] for event in journal] == [
        "initialized",
        "failure_command_held",
        "authoritative_failure_observed",
        "failure_command_held",
        "authoritative_failure_observed",
        "failure_command_held",
        "authoritative_failure_observed",
        "circuit_open_observed",
        "core_defer_observed",
        "builder_rebound",
        "preflight_authorized",
        "reset_command_observed",
        "reset_dispatch_observed",
        "reset_dispatch_submitted",
        "effect_confirmed",
        "complete",
    ]
    assert all(event["diagnostic_only"] is True for event in journal)


def test_canary_fails_if_held_command_sees_the_same_revision_twice(tmp_path: Path) -> None:
    controller = AuthoritativeBuildCircuitCanary(
        run_id="run-canary",
        episode_id="episode-0",
        seed=0,
        journal_path=tmp_path / "journal.jsonl",
    )
    command = _command(0)
    controller.observe_runtime_decision(
        (command,),
        idle_reason=None,
        planner_pending=False,
        game_loop=1,
        observation_revision="same",
        builder_tag=100,
        reservation_count=0,
        leased_builder_tags=(),
        effect_inflight_count=0,
        authoritative_state=None,
    )
    assert controller.should_hold_raw_dispatch(observation_revision="same")

    with pytest.raises(RuntimeError, match="held_command_dispatched_in_source_observation"):
        controller.observe_raw_result(
            dispatch=None,
            diagnostic=_failure_diagnostic(command, 1),
            game_loop=2,
            observation_revision="same",
            authoritative_state=_state(1, circuit_open=False),
            reservation_count=0,
            leased_builder_tags=(),
            effect_inflight_count=0,
        )


@pytest.mark.parametrize(
    ("commands", "idle_reason", "planner_pending", "builder_tag"),
    [
        ((_command(3),), "plan_commands_deferred", False, 100),
        ((), "no_legal_action", False, 100),
        ((), "waiting_for_planner", False, 100),
        ((), "plan_commands_deferred", False, 200),
    ],
)
def test_canary_fails_closed_at_the_core_open_boundary(
    tmp_path: Path,
    commands: tuple[RoutedCommand, ...],
    idle_reason: str,
    planner_pending: bool,
    builder_tag: int,
) -> None:
    controller = AuthoritativeBuildCircuitCanary(
        run_id="run-canary",
        episode_id="episode-0",
        seed=0,
        journal_path=tmp_path / "journal.jsonl",
    )
    for ordinal in range(3):
        _observe_failure(controller, ordinal)

    with pytest.raises(RuntimeError, match="canary failed"):
        controller.observe_runtime_decision(
            commands,
            idle_reason=idle_reason,
            planner_pending=planner_pending,
            game_loop=40,
            observation_revision="revision-core",
            builder_tag=builder_tag,
            reservation_count=0,
            leased_builder_tags=(),
            effect_inflight_count=0,
            authoritative_state=_state(3, circuit_open=True),
        )


def test_canary_rejects_effect_success_without_new_structure_evidence(tmp_path: Path) -> None:
    controller = AuthoritativeBuildCircuitCanary(
        run_id="run-canary",
        episode_id="episode-0",
        seed=0,
        journal_path=tmp_path / "journal.jsonl",
    )
    controller.phase = "awaiting_effect"
    controller.operation_id = OPERATION_ID
    controller.semantic_action = "BUILD PYLON"
    controller.final_command_id = "command-3"

    with pytest.raises(RuntimeError, match="reset_build_effect_not_new_structure"):
        controller.observe_effect_reports(
            (
                {
                    "command_id": "command-3",
                    "status": "succeeded",
                    "effect_evidence": {
                        "effect_kind": "build",
                        "build_started": True,
                        "confirmation_kind": "builder_order",
                    },
                },
            ),
            game_loop=50,
            observation_revision="effect",
            reservation_count=0,
            leased_builder_tags=(),
            effect_inflight_count=0,
            authoritative_state=_state(0, circuit_open=False),
        )
