from __future__ import annotations

import hashlib
import json
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
    _placement_candidate_id,
    _placement_revision,
    build_material_legality_identity,
)

from rtscortex.contracts import authoritative_build_preflight_request_id


def _operation_id(marker: str) -> str:
    return f"operation:{marker * 64}"


def _material_id(marker: str) -> str:
    return f"build-legality:{marker * 64}"


def _attempt_id(operation_id: str, command_id: str, ordinal: int) -> str:
    encoded = json.dumps(
        {
            "operation_id": operation_id,
            "command_id": command_id,
            "attempt_ordinal": ordinal,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"attempt:{hashlib.sha256(encoded).hexdigest()}"


def _preflight_request_id(
    operation_id: str,
    *,
    screen_target: tuple[int, int] = (30, 25),
    operation_epoch: int = 0,
) -> str:
    opened_attempt_id = _attempt_id(operation_id, "preflight-open-2", 2)
    return authoritative_build_preflight_request_id(
        operation_id=operation_id,
        operation_epoch=operation_epoch,
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        requested_arguments=[[screen_target[0], screen_target[1]]],
        opened_command_id="preflight-open-2",
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity=_material_id("a"),
        observation_revision="observation-new",
        observation_game_loop=116,
    )


def _open_authoritative_circuit(
    service: RawPlacementService,
    *,
    operation_id: str,
    builder_tag: int = 0xB1,
    ability_id: int = 881,
    world_target: tuple[float, float] = (22.0, 24.0),
    target_state_revision: str = "target-stable",
    material_identity: str | None = None,
) -> tuple[str, str]:
    identity = material_identity or _material_id("a")
    opened_command_id = "preflight-open-2"
    for ordinal in range(3):
        command_id = f"preflight-open-{ordinal}"
        service.record_authoritative_pre_dispatch_failure(
            operation_id=operation_id,
            action_name="Build_Pylon_Screen",
            command_id=command_id,
            failure_code="placement_query_rejected",
            attempt_id=_attempt_id(operation_id, command_id, ordinal),
            attempt_ordinal=ordinal,
            builder_tag=builder_tag,
            ability_id=ability_id,
            world_target=world_target,
            target_state_revision=target_state_revision,
            observation_revision=f"observation-{ordinal}",
            observation_game_loop=100 + ordinal,
            material_legality_identity=identity,
        )
    return opened_command_id, _attempt_id(operation_id, opened_command_id, 2)


def test_authoritative_pre_dispatch_circuit_counts_exact_failures_without_placement_state() -> None:
    transitions: list[dict[str, Any]] = []
    service = RawPlacementService(
        unit_names={},
        transition_sink=transitions.append,
        no_start_streak_threshold=3,
    )
    service.set_runtime_context(run_id="run", episode_id="episode", step_id=4, game_loop=100)

    operation_id = _operation_id("a")
    material_identity = _material_id("b")
    decisions = [
        service.record_authoritative_pre_dispatch_failure(
            operation_id=operation_id,
            action_name="Build_Pylon_Screen",
            command_id=f"command-{ordinal}",
            failure_code=code,
            attempt_id=_attempt_id(operation_id, f"command-{ordinal}", ordinal),
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            ability_id=881,
            world_target=(22.0, 24.0),
            placement_revision=f"candidate-{ordinal}",
            target_state_revision="target-stable",
            observation_revision=f"observation-{ordinal}",
            observation_game_loop=100 + ordinal,
            material_legality_identity=material_identity,
        )
        for ordinal, code in enumerate(
            (
                "placement_query_rejected",
                "placement_candidate_stale",
                "no_legal_placement",
            )
        )
    ]

    assert [decision.streak for decision in decisions] == [1, 2, 3]
    assert decisions[-1].circuit_open is True
    assert decisions[-1].transition == "closed_to_open"
    # A failure before reservation is execution evidence, not a fabricated
    # placement-ledger transition.
    assert transitions == []
    payload = decisions[-1].to_dict()
    assert payload["operation_id"] == operation_id
    assert payload["action_name"] == "Build_Pylon_Screen"
    assert payload["builder_tag"] == 0xB1
    assert payload["ability_id"] == 881
    assert payload["world_target"] == [22.0, 24.0]
    assert payload["observation_game_loop"] == 102
    assert payload["material_legality_identity"] == material_identity
    assert payload["material_evidence_valid"] is True
    assert payload["invalid_evidence_reasons"] == []
    assert payload["reset_reason"] is None
    assert payload["material_change_reason"] is None

    blocked = service.record_authoritative_pre_dispatch_failure(
        operation_id=operation_id,
        action_name="Build_Pylon_Screen",
        command_id="command-4",
        failure_code="no_legal_placement",
        attempt_id=_attempt_id(operation_id, "command-4", 3),
        attempt_ordinal=3,
        builder_tag=0xB1,
        ability_id=881,
        world_target=(22.0, 24.0),
        placement_revision="candidate-4",
        target_state_revision="target-stable",
        observation_revision="observation-4",
        observation_game_loop=103,
        material_legality_identity=material_identity,
    )
    assert blocked.circuit_open is True
    assert blocked.streak == 3
    assert blocked.status == "defer_replan"
    assert blocked.transition is None


def test_authoritative_pre_dispatch_remains_open_until_authorized_preflight() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    old_identity = _material_id("c")
    new_identity = _material_id("d")
    kwargs: dict[str, Any] = {
        "operation_id": _operation_id("e"),
        "action_name": "Build_Pylon_Screen",
        "builder_tag": 0xB1,
        "ability_id": 881,
        "world_target": (22.0, 24.0),
        "target_state_revision": "target-stable",
        "observation_revision": "observation-1",
        "observation_game_loop": 100,
    }
    for ordinal in range(3):
        command_id = f"command-{ordinal}"
        service.record_authoritative_pre_dispatch_failure(
            **kwargs,
            command_id=command_id,
            failure_code="no_legal_placement",
            attempt_id=_attempt_id(str(kwargs["operation_id"]), command_id, ordinal),
            attempt_ordinal=ordinal,
            placement_revision=f"candidate-{ordinal}",
            material_legality_identity=old_identity,
        )

    assert (
        service.authoritative_pre_dispatch_retry_allowed(
            **kwargs,
            material_legality_identity=old_identity,
        )
        is False
    )
    # A different candidate/hash is not proof of a material legality change.
    assert (
        service.authoritative_pre_dispatch_retry_allowed(
            **{
                **kwargs,
                "observation_revision": "observation-new",
                "world_target": (24.0, 24.0),
            },
            placement_revision="candidate-new",
            material_legality_identity=new_identity,
        )
        is False
    )
    # Even real target-side change is only a preflight hint.  The retry helper
    # cannot create the forbidden "retry allowed while Raw is still open"
    # intermediate state; the typed preflight performs the atomic reset.
    assert (
        service.authoritative_pre_dispatch_retry_allowed(
            **{
                **kwargs,
                "observation_revision": "observation-new",
                "target_state_revision": "target-changed",
            },
            placement_revision="candidate-new",
            material_legality_identity=new_identity,
        )
        is False
    )
    still_open = service.record_authoritative_pre_dispatch_failure(
        **{
            **kwargs,
            "observation_revision": "observation-new",
            "target_state_revision": "target-changed",
        },
        command_id="command-new",
        failure_code="placement_candidate_stale",
        attempt_id=_attempt_id(str(kwargs["operation_id"]), "command-new", 3),
        attempt_ordinal=3,
        placement_revision="candidate-new",
        material_legality_identity=new_identity,
    )
    assert still_open.transition is None
    assert still_open.reset_reason is None
    assert still_open.material_change_reason is None
    assert still_open.streak == 3
    assert still_open.circuit_open is True
    assert still_open.next_action == "preflight_required"


def test_authoritative_preflight_keeps_open_circuit_for_coordinate_only_churn() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = _operation_id("9")
    opened_command_id, opened_attempt_id = _open_authoritative_circuit(
        service,
        operation_id=operation_id,
    )

    result = service.authorize_authoritative_pre_dispatch_preflight(
        request_id=_preflight_request_id(operation_id, screen_target=(31, 25)),
        operation_id=operation_id,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        requested_arguments=((31, 25),),
        opened_command_id=opened_command_id,
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity=_material_id("a"),
        request_observation_revision="observation-new",
        request_observation_game_loop=116,
        builder_tag=0xB1,
        ability_id=881,
        world_target=(23.0, 24.0),
        target_state_revision="target-stable",
        observation_revision="observation-new",
        observation_game_loop=116,
    )

    assert result.authorized is False
    assert result.status == "deferred"
    assert result.reason == "material_change_not_authoritative"
    assert result.state_transition is None
    state = service.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is True
    assert state.streak == 3


def test_authoritative_preflight_rejects_noncanonical_request_identity() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = _operation_id("4")
    opened_command_id, opened_attempt_id = _open_authoritative_circuit(
        service,
        operation_id=operation_id,
    )

    result = service.authorize_authoritative_pre_dispatch_preflight(
        request_id="build-preflight:" + "f" * 64,
        operation_id=operation_id,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        requested_arguments=((30, 25),),
        opened_command_id=opened_command_id,
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity=_material_id("a"),
        request_observation_revision="observation-new",
        request_observation_game_loop=116,
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="target-stable",
        observation_revision="observation-new",
        observation_game_loop=116,
    )

    assert result.authorized is False
    assert result.status == "deferred"
    assert "request_id_invalid" in result.invalid_evidence_reasons
    state = service.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is True
    assert state.streak == 3


def test_authoritative_preflight_atomically_resets_and_authorizes_exact_builder_change() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = _operation_id("8")
    opened_command_id, opened_attempt_id = _open_authoritative_circuit(
        service,
        operation_id=operation_id,
    )
    request_id = _preflight_request_id(operation_id)

    result = service.authorize_authoritative_pre_dispatch_preflight(
        request_id=request_id,
        operation_id=operation_id,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        requested_arguments=((30, 25),),
        opened_command_id=opened_command_id,
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity=_material_id("a"),
        request_observation_revision="observation-new",
        request_observation_game_loop=116,
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="target-stable",
        observation_revision="observation-new",
        observation_game_loop=116,
    )

    assert result.authorized is True
    assert result.status == "authorized"
    assert result.state_transition == "open_to_reset"
    assert result.material_change_reason == "builder_changed"
    assert result.authorization_id is not None
    state = service.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is False
    assert state.streak == 0
    assert state.pending_authorization_id == result.authorization_id

    replay = service.authorize_authoritative_pre_dispatch_preflight(
        request_id=request_id,
        operation_id=operation_id,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        requested_arguments=((30, 25),),
        opened_command_id=opened_command_id,
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity=_material_id("a"),
        request_observation_revision="observation-new",
        request_observation_game_loop=116,
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="target-stable",
        observation_revision="observation-new",
        observation_game_loop=116,
    )
    assert replay.authorized is False
    assert replay.reason == "preflight_request_replayed"


@pytest.mark.parametrize(
    ("change", "expected_reason"),
    [
        ("ability", "ability_changed"),
        ("target_state", "target_state_changed"),
        ("operation_epoch", "operation_epoch_changed"),
    ],
)
def test_authoritative_preflight_atomically_authorizes_other_material_changes(
    change: str,
    expected_reason: str,
) -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = _operation_id("5")
    opened_command_id, opened_attempt_id = _open_authoritative_circuit(
        service,
        operation_id=operation_id,
    )
    operation_epoch = 1 if change == "operation_epoch" else 0
    result = service.authorize_authoritative_pre_dispatch_preflight(
        request_id=_preflight_request_id(
            operation_id,
            operation_epoch=operation_epoch,
        ),
        operation_id=operation_id,
        operation_epoch=operation_epoch,
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        requested_arguments=((30, 25),),
        opened_command_id=opened_command_id,
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity=_material_id("a"),
        request_observation_revision="observation-new",
        request_observation_game_loop=116,
        builder_tag=0xB1,
        ability_id=882 if change == "ability" else 881,
        world_target=(22.0, 24.0),
        target_state_revision=("target-changed" if change == "target_state" else "target-stable"),
        observation_revision="observation-new",
        observation_game_loop=116,
    )

    assert result.authorized is True
    assert result.state_transition == "open_to_reset"
    assert result.material_change_reason == expected_reason
    state = service.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is False
    assert state.streak == 0
    assert state.operation_epoch == operation_epoch


@pytest.mark.parametrize(
    "changed_field",
    ["builder", "ability", "target", "material", "epoch", "expired"],
)
def test_authoritative_preflight_consumption_is_exact_single_use_and_fail_closed(
    changed_field: str,
) -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = _operation_id("7")
    opened_command_id, opened_attempt_id = _open_authoritative_circuit(
        service,
        operation_id=operation_id,
    )
    authorization = service.authorize_authoritative_pre_dispatch_preflight(
        request_id=_preflight_request_id(operation_id),
        operation_id=operation_id,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        requested_arguments=((30, 25),),
        opened_command_id=opened_command_id,
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity=_material_id("a"),
        request_observation_revision="observation-new",
        request_observation_game_loop=116,
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="target-stable",
        observation_revision="observation-new",
        observation_game_loop=116,
    )
    assert authorization.authorization_id is not None
    values: dict[str, Any] = {
        "authorization_id": authorization.authorization_id,
        "request_id": authorization.request_id,
        "operation_id": operation_id,
        "operation_epoch": 0,
        "action_name": "Build_Pylon_Screen",
        "builder_tag": 0xB2,
        "ability_id": 881,
        "world_target": (22.0, 24.0),
        "target_state_revision": "target-stable",
        "material_legality_identity": authorization.material_legality_identity,
        "observation_game_loop": 117,
    }
    if changed_field == "builder":
        values["builder_tag"] = 0xB3
    elif changed_field == "ability":
        values["ability_id"] = 883
    elif changed_field == "target":
        values["world_target"] = (23.0, 24.0)
    elif changed_field == "material":
        values["material_legality_identity"] = _material_id("f")
    elif changed_field == "epoch":
        values["operation_epoch"] = 1
    elif changed_field == "expired":
        assert authorization.expires_game_loop is not None
        values["observation_game_loop"] = authorization.expires_game_loop + 1

    consumed = service.consume_authoritative_pre_dispatch_authorization(**values)
    assert consumed.authorized is False
    assert consumed.status == "deferred"
    state = service.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is True
    assert state.streak == 3


def test_authoritative_preflight_checkpoint_drops_pending_authorization_and_reopens() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = _operation_id("6")
    opened_command_id, opened_attempt_id = _open_authoritative_circuit(
        service,
        operation_id=operation_id,
    )
    authorization = service.authorize_authoritative_pre_dispatch_preflight(
        request_id=_preflight_request_id(operation_id),
        operation_id=operation_id,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        actor="Builder/Builder-Probe-1",
        requested_arguments=((30, 25),),
        opened_command_id=opened_command_id,
        opened_attempt_id=opened_attempt_id,
        opened_attempt_ordinal=2,
        blocked_material_legality_identity=_material_id("a"),
        request_observation_revision="observation-new",
        request_observation_game_loop=116,
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="target-stable",
        observation_revision="observation-new",
        observation_game_loop=116,
    )
    assert authorization.authorized is True

    restored = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    restored.restore_checkpoint_state(service.checkpoint_state())
    state = restored.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is True
    assert state.streak == 3
    assert state.pending_authorization_id is None
    consumed = restored.consume_authoritative_pre_dispatch_authorization(
        authorization_id=str(authorization.authorization_id),
        request_id=authorization.request_id,
        operation_id=operation_id,
        operation_epoch=0,
        action_name="Build_Pylon_Screen",
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="target-stable",
        material_legality_identity=authorization.material_legality_identity,
        observation_game_loop=117,
    )
    assert consumed.authorized is False
    assert consumed.reason == "authorization_missing_or_replayed"


def test_authoritative_pre_dispatch_checkpoint_restore_and_success_reset() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = _operation_id("f")
    for ordinal in range(3):
        command_id = f"checkpoint-{ordinal}"
        service.record_authoritative_pre_dispatch_failure(
            operation_id=operation_id,
            action_name="Build_Pylon_Screen",
            command_id=command_id,
            failure_code="no_legal_placement",
            attempt_id=_attempt_id(operation_id, command_id, ordinal),
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            ability_id=881,
            world_target=(22.0, 24.0),
            target_state_revision="target",
            material_legality_identity=_material_id("1"),
            observation_game_loop=ordinal,
        )
    restored = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    restored.restore_checkpoint_state(service.checkpoint_state())
    state = restored.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is True
    assert state.streak == 3

    reset = restored.reset_authoritative_pre_dispatch(
        operation_id,
        reason="build_started",
        command_id="success",
        action_name="Build_Pylon_Screen",
        attempt_id=_attempt_id(operation_id, "success", 3),
        attempt_ordinal=3,
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="target-revalidated",
        material_legality_identity=_material_id("2"),
    )
    assert reset is not None
    assert reset.transition == "open_to_reset"
    assert reset.reset_reason == "build_started"
    assert reset.material_legality_identity == _material_id("2")
    assert reset.material_evidence_valid is True
    assert reset.invalid_evidence_reasons == ()
    state = restored.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is False
    assert state.streak == 0


def test_authoritative_pre_dispatch_success_clears_closed_streak_without_open_transition() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = _operation_id("2")
    service.record_authoritative_pre_dispatch_failure(
        operation_id=operation_id,
        action_name="Build_Gateway_Screen",
        command_id="gateway-failure",
        failure_code="placement_candidate_stale",
        attempt_id=_attempt_id(operation_id, "gateway-failure", 0),
        attempt_ordinal=0,
        builder_tag=0xB1,
        ability_id=883,
        world_target=(24.0, 28.0),
        target_state_revision="target",
        material_legality_identity=_material_id("3"),
    )

    reset = service.reset_authoritative_pre_dispatch(
        operation_id,
        reason="build_started",
        command_id="gateway-success",
        action_name="Build_Gateway_Screen",
        attempt_id=_attempt_id(operation_id, "gateway-success", 1),
        attempt_ordinal=1,
        builder_tag=0xB1,
        ability_id=883,
        world_target=(24.0, 28.0),
        target_state_revision="target-revalidated",
        material_legality_identity=_material_id("4"),
    )

    assert reset is not None
    assert reset.status == "reset"
    assert reset.state_transition is None
    state = service.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.streak == 0
    assert state.circuit_open is False


@pytest.mark.parametrize(
    ("material_identity", "target_state_revision", "expected_reason"),
    (
        ("build-legality:1", "revalidated-target", "material_legality_identity_invalid"),
        (_material_id("9"), None, "target_state_revision_missing"),
    ),
)
def test_authoritative_success_reset_with_invalid_material_evidence_stays_open(
    material_identity: str,
    target_state_revision: str | None,
    expected_reason: str,
) -> None:
    service = RawPlacementService(unit_names={})
    operation_id = _operation_id("5")
    for ordinal in range(3):
        command_id = f"invalid-reset-{ordinal}"
        service.record_authoritative_pre_dispatch_failure(
            operation_id=operation_id,
            action_name="Build_Pylon_Screen",
            command_id=command_id,
            failure_code="placement_candidate_stale",
            attempt_id=_attempt_id(operation_id, command_id, ordinal),
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            ability_id=881,
            world_target=(22.0, 24.0),
            target_state_revision="failed-target",
            material_legality_identity=_material_id("5"),
        )

    reset = service.reset_authoritative_pre_dispatch(
        operation_id,
        reason="build_started",
        command_id="invalid-reset-success",
        action_name="Build_Pylon_Screen",
        attempt_id=_attempt_id(operation_id, "invalid-reset-success", 3),
        attempt_ordinal=3,
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision=target_state_revision,
        material_legality_identity=material_identity,
    )

    assert reset is not None
    assert reset.status == "defer_replan"
    assert reset.circuit_open is True
    assert reset.state_transition is None
    assert reset.material_evidence_valid is False
    assert expected_reason in reset.invalid_evidence_reasons
    state = service.authoritative_pre_dispatch_state(operation_id)
    assert state is not None
    assert state.circuit_open is True
    assert state.streak == 3


def test_mark_build_started_emits_complete_authoritative_success_reset_identity() -> None:
    service = RawPlacementService(unit_names={2: "Probe"})
    operation_id = _operation_id("6")
    for ordinal in range(3):
        command_id = f"start-failure-{ordinal}"
        service.record_authoritative_pre_dispatch_failure(
            operation_id=operation_id,
            action_name="Build_Pylon_Screen",
            command_id=command_id,
            failure_code="placement_candidate_stale",
            attempt_id=_attempt_id(operation_id, command_id, ordinal),
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            ability_id=881,
            world_target=(22.0, 24.0),
            target_state_revision="failed-target",
            material_legality_identity=_material_id("6"),
        )
    observation = SimpleNamespace(
        raw_units=[_unit(0xB2, 2, alliance=1, x=20, y=20)],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )
    command_id = "start-success"
    attempt_id = _attempt_id(operation_id, command_id, 3)
    service.resolve(
        command_id=command_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
        attempt_ordinal=3,
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(22.0, 24.0),
        builder_tag=0xB2,
        ability_name="Build_Pylon_pt",
    )
    service.record_build_authorization(
        command_id,
        ability_id=881,
        authorization=SimpleNamespace(
            placement_query_status="Success",
            available_ability_query="available",
            target_legality_fingerprint=f"sc2:{'7' * 64}",
            details={},
        ),
        game_loop=100,
    )
    reservation = service.command_target(command_id)
    assert reservation is not None
    assert reservation.material_legality_identity is not None

    service.mark_build_started(command_id, game_loop=101, expires_game_loop=549)

    transition = service.drain_transition_history(command_id)[-1]
    nested = transition["authoritative_pre_dispatch"]
    assert transition["builder_tag"] == reservation.builder_tag
    assert transition["ability_id"] == reservation.ability_id
    assert transition["world_target"] == reservation.world_target
    assert transition["material_legality_identity"] == reservation.material_legality_identity
    assert nested["operation_id"] == operation_id
    assert nested["command_id"] == command_id
    assert nested["attempt_id"] == attempt_id
    assert nested["builder_tag"] == reservation.builder_tag
    assert nested["ability_id"] == reservation.ability_id
    assert nested["world_target"] == list(reservation.world_target)
    assert nested["material_legality_identity"] == reservation.material_legality_identity
    assert nested["material_evidence_valid"] is True
    assert nested["invalid_evidence_reasons"] == []


def test_confirm_command_emits_complete_authoritative_success_reset_identity() -> None:
    service = RawPlacementService(unit_names={2: "Probe"})
    operation_id = _operation_id("7")
    failed_command = "confirm-failure"
    service.record_authoritative_pre_dispatch_failure(
        operation_id=operation_id,
        action_name="Build_Pylon_Screen",
        command_id=failed_command,
        failure_code="placement_candidate_stale",
        attempt_id=_attempt_id(operation_id, failed_command, 0),
        attempt_ordinal=0,
        builder_tag=0xB1,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="failed-target",
        material_legality_identity=_material_id("8"),
    )
    observation = SimpleNamespace(
        raw_units=[_unit(0xB2, 2, alliance=1, x=20, y=20)],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )
    command_id = "confirm-success"
    attempt_id = _attempt_id(operation_id, command_id, 1)
    service.resolve(
        command_id=command_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
        attempt_ordinal=1,
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(22.0, 24.0),
        builder_tag=0xB2,
        ability_name="Build_Pylon_pt",
    )
    service.record_build_authorization(
        command_id,
        ability_id=881,
        authorization=SimpleNamespace(
            placement_query_status="Success",
            available_ability_query="available",
            target_legality_fingerprint=f"sc2:{'9' * 64}",
            details={},
        ),
        game_loop=100,
    )
    reservation = service.command_target(command_id)
    assert reservation is not None

    service.confirm_command(command_id, game_loop=102)

    transition = service.drain_transition_history(command_id)[-1]
    nested = transition["authoritative_pre_dispatch"]
    assert transition["next_state"] == "occupied"
    assert nested["reset_reason"] == "effect_confirmed"
    assert nested["builder_tag"] == reservation.builder_tag
    assert nested["ability_id"] == reservation.ability_id
    assert nested["world_target"] == list(reservation.world_target)
    assert nested["material_legality_identity"] == reservation.material_legality_identity
    assert nested["material_evidence_valid"] is True
    assert nested["invalid_evidence_reasons"] == []


def test_authoritative_pre_dispatch_operation_state_is_isolated() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    blocked_operation = _operation_id("4")
    allowed_operation = _operation_id("5")
    material_identity = _material_id("6")
    for ordinal in range(3):
        command_id = f"blocked-{ordinal}"
        service.record_authoritative_pre_dispatch_failure(
            operation_id=blocked_operation,
            action_name="Build_Pylon_Screen",
            command_id=command_id,
            failure_code="no_legal_placement",
            attempt_id=_attempt_id(blocked_operation, command_id, ordinal),
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            ability_id=881,
            world_target=(22.0, 24.0),
            target_state_revision="target",
            material_legality_identity=material_identity,
        )

    assert service.authoritative_pre_dispatch_state(allowed_operation) is None
    assert service.authoritative_pre_dispatch_retry_allowed(
        allowed_operation,
        builder_tag=0xB1,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="target",
        material_legality_identity=material_identity,
    )


def test_cached_authoritative_rejection_is_a_new_command_failure_without_a_new_query() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    common: dict[str, Any] = {
        "operation_id": _operation_id("7"),
        "action_name": "Build_Pylon_Screen",
        "builder_tag": 0xB1,
        "ability_id": 881,
        "world_target": (22.0, 24.0),
        "target_state_revision": "target",
        "material_legality_identity": _material_id("8"),
    }
    first = service.record_authoritative_pre_dispatch_failure(
        **common,
        command_id="query-rejection",
        failure_code="placement_query_rejected",
        attempt_id=_attempt_id(str(common["operation_id"]), "query-rejection", 1),
        attempt_ordinal=1,
    )
    cached = service.record_authoritative_pre_dispatch_failure(
        **common,
        command_id="cached-rejection",
        failure_code="placement_query_rejected_cached",
        attempt_id=_attempt_id(str(common["operation_id"]), "cached-rejection", 2),
        attempt_ordinal=2,
    )
    assert first.streak == 1
    assert cached.material_duplicate is True
    assert cached.duplicate_attempt is False
    assert cached.streak == 2
    assert cached.circuit_open is False


def test_authoritative_threshold_is_fixed_independently_of_no_start_configuration() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=1)
    operation_id = _operation_id("9")
    decisions = [
        service.record_authoritative_pre_dispatch_failure(
            operation_id=operation_id,
            action_name="Build_Pylon_Screen",
            command_id=f"fixed-threshold-{ordinal}",
            failure_code="no_legal_placement",
            attempt_id=_attempt_id(operation_id, f"fixed-threshold-{ordinal}", ordinal),
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            ability_id=881,
            world_target=(22.0, 24.0),
            target_state_revision="stable-target",
            material_legality_identity=_material_id("a"),
        )
        for ordinal in range(3)
    ]

    assert [decision.threshold for decision in decisions] == [3, 3, 3]
    assert [decision.streak for decision in decisions] == [1, 2, 3]
    assert [decision.state_transition for decision in decisions] == [
        None,
        None,
        "closed_to_open",
    ]
    assert decisions[-1].circuit_open is True


def test_missing_material_identity_cannot_reset_an_open_authoritative_circuit() -> None:
    service = RawPlacementService(unit_names={})
    operation_id = _operation_id("b")
    for ordinal in range(3):
        command_id = f"missing-material-{ordinal}"
        opened = service.record_authoritative_pre_dispatch_failure(
            operation_id=operation_id,
            action_name="Build_Pylon_Screen",
            command_id=command_id,
            failure_code="no_legal_placement",
            attempt_id=_attempt_id(operation_id, command_id, ordinal),
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            ability_id=881,
            world_target=(22.0, 24.0),
            target_state_revision="stable-target",
            material_legality_identity="build-legality:1",
        )

    assert opened.circuit_open is True
    assert opened.material_evidence_valid is False
    assert "material_legality_identity_invalid" in opened.invalid_evidence_reasons
    assert (
        service.authoritative_pre_dispatch_retry_allowed(
            operation_id,
            builder_tag=0xB2,
            ability_id=881,
            world_target=(22.0, 24.0),
            target_state_revision="changed-target",
            material_legality_identity=_material_id("c"),
        )
        is False
    )
    blocked = service.record_authoritative_pre_dispatch_failure(
        operation_id=operation_id,
        action_name="Build_Pylon_Screen",
        command_id="missing-material-post-open",
        failure_code="authoritative_pre_dispatch_circuit_open",
        attempt_id=_attempt_id(operation_id, "missing-material-post-open", 4),
        attempt_ordinal=4,
        builder_tag=0xB2,
        ability_id=881,
        world_target=(22.0, 24.0),
        target_state_revision="changed-target",
        material_legality_identity=_material_id("c"),
    )
    assert blocked.circuit_open is True
    assert blocked.streak == 3
    assert blocked.state_transition is None


def test_authoritative_checkpoint_rejects_non_production_threshold() -> None:
    service = RawPlacementService(unit_names={})
    with pytest.raises(
        ValueError,
        match="authoritative pre-dispatch checkpoint threshold must be exactly 3",
    ):
        service.restore_checkpoint_state(
            {
                "authoritative_pre_dispatch_operations": {
                    _operation_id("d"): {
                        "operation_id": _operation_id("d"),
                        "threshold": 1,
                    }
                }
            }
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


def _gas_observation(
    *,
    game_loop: int,
    extra_units: tuple[SimpleNamespace, ...] = (),
    include_geyser: bool = True,
) -> SimpleNamespace:
    units = [
        _unit(0xC1, 59, alliance=1, x=20, y=20),
        *([_unit(0xA1, 342, alliance=3, x=24, y=20)] if include_geyser else []),
        *extra_units,
    ]
    return SimpleNamespace(
        raw_units=units,
        feature_units=[],
        feature_screen=None,
        player_common=SimpleNamespace(minerals=500, vespene=500),
        game_loop=[game_loop],
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


def test_geyser_candidates_exclude_any_current_structure_occupancy() -> None:
    service = RawPlacementService(unit_names={59: "Nexus", 88: "Extractor", 342: "VespeneGeyser"})
    observation = _gas_observation(
        game_loop=100,
        extra_units=(_unit(0xE1, 88, alliance=4, x=24, y=20),),
    )
    service.observe(observation, require_feature_visibility=False)

    candidates = service.candidates(observation, "Build_Assimilator_Near")

    assert candidates.argument_candidates == []
    assert candidates.unavailable_reason == "no_unoccupied_geyser"


def test_geyser_no_start_requires_state_change_before_one_bounded_retry() -> None:
    service = RawPlacementService(unit_names={2: "Probe", 59: "Nexus", 342: "VespeneGeyser"})
    observation = _gas_observation(
        game_loop=100,
        extra_units=(_unit(0xB1, 2, alliance=1, x=21, y=20),),
    )
    service.observe(observation, require_feature_visibility=False)
    assert service.candidates(observation, "Build_Assimilator_Near").argument_candidates == [[0xA1]]
    service.resolve(
        command_id="gas-rebuild-one",
        action_name="Build_Assimilator_Near",
        requested_arguments=(0xA1,),
        observation=observation,
        world_target=None,
        builder_tag=0xB1,
    )
    service.quarantine_command(
        command_id="gas-rebuild-one",
        action_name="Build_Assimilator_Near",
        requested_arguments=(0xA1,),
        world_target=None,
        failure_code="no_build_start_evidence",
        game_loop=212,
        failure_classification="dynamic_target_obstruction",
        classification_basis=("dynamic_unit_inside_footprint",),
    )

    unchanged = _gas_observation(
        game_loop=400,
        extra_units=(_unit(0xB1, 2, alliance=1, x=21, y=20),),
    )
    service.observe(unchanged, require_feature_visibility=False)
    assert service.candidates(unchanged, "Build_Assimilator_Near").argument_candidates == []

    hidden = _gas_observation(
        game_loop=401,
        extra_units=(_unit(0xB1, 2, alliance=1, x=21, y=20),),
        include_geyser=False,
    )
    service.observe(hidden, require_feature_visibility=False)
    revalidated = _gas_observation(
        game_loop=402,
        extra_units=(_unit(0xB1, 2, alliance=1, x=21, y=20),),
    )
    service.observe(revalidated, require_feature_visibility=False)
    assert service.candidates(revalidated, "Build_Assimilator_Near").argument_candidates == [[0xA1]]

    service.resolve(
        command_id="gas-rebuild-two",
        action_name="Build_Assimilator_Near",
        requested_arguments=(0xA1,),
        observation=revalidated,
        world_target=None,
        builder_tag=0xB1,
    )
    service.quarantine_command(
        command_id="gas-rebuild-two",
        action_name="Build_Assimilator_Near",
        requested_arguments=(0xA1,),
        world_target=None,
        failure_code="no_build_start_evidence",
        game_loop=514,
        failure_classification="dynamic_target_obstruction",
        classification_basis=("dynamic_unit_inside_footprint",),
    )
    assert service.candidates(revalidated, "Build_Assimilator_Near").argument_candidates == []


def test_operation_no_start_streak_spans_target_revision_command_and_builder() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = "operation:placement-streak"
    decisions = []
    for ordinal in range(3):
        decision = service.quarantine_command(
            command_id=f"no-start-{ordinal}",
            action_name="Build_Pylon_Screen",
            requested_arguments=([64 + ordinal, 64 + ordinal],),
            world_target=(22.0 + ordinal, 24.0 + ordinal),
            failure_code="no_build_start_evidence",
            game_loop=100 + ordinal,
            operation_id=operation_id,
            attempt_ordinal=ordinal,
            builder_tag=0xB1 + ordinal,
            placement_revision=f"placement-{ordinal}",
            target_state_revision=f"target-{ordinal}",
        )
        assert decision is not None
        decisions.append(decision)

    assert [decision.status for decision in decisions] == ["retry", "retry", "defer_replan"]
    assert decisions[-1].circuit_open is True
    assert decisions[-1].streak == 3
    assert all(decision.suppressed_target is False for decision in decisions)
    state = service.operation_no_start_state(operation_id)
    assert state is not None and state.streak == 3
    assert service.is_quarantined("Build_Pylon_Screen", (22.0, 24.0)) is False


def test_successful_build_start_resets_operation_no_start_streak() -> None:
    service = RawPlacementService(unit_names={})
    operation_id = "operation:placement-reset"
    service.quarantine_command(
        command_id="no-start-before-reset",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=(22.0, 24.0),
        failure_code="no_build_start_evidence",
        game_loop=100,
        operation_id=operation_id,
        attempt_ordinal=0,
        target_state_revision="target-0",
    )

    service.resolve(
        command_id="start-after-reset",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=SimpleNamespace(
            raw_units=[],
            feature_units=[],
            feature_screen=None,
            game_loop=[200],
        ),
        world_target=(22.0, 24.0),
        operation_id=operation_id,
        attempt_ordinal=1,
    )
    service.mark_build_started("start-after-reset", game_loop=210, expires_game_loop=500)

    state = service.operation_no_start_state(operation_id)
    assert state is not None
    assert state.streak == 0
    assert state.circuit_open is False
    assert state.last_status == "reset"


def test_historical_attempt_ordinal_does_not_increment_no_start_streak() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=2)
    operation_id = "operation:placement-history"
    first = service.quarantine_command(
        command_id="attempt-four",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=(22.0, 24.0),
        failure_code="no_build_start_evidence",
        game_loop=100,
        operation_id=operation_id,
        attempt_ordinal=4,
    )
    replay = service.quarantine_command(
        command_id="replayed-attempt-four",
        action_name="Build_Pylon_Screen",
        requested_arguments=([70, 70],),
        world_target=(30.0, 30.0),
        failure_code="no_build_start_evidence",
        game_loop=200,
        operation_id=operation_id,
        attempt_ordinal=4,
    )
    historical = service.quarantine_command(
        command_id="historical-attempt-three",
        action_name="Build_Pylon_Screen",
        requested_arguments=([72, 72],),
        world_target=(32.0, 32.0),
        failure_code="no_build_start_evidence",
        game_loop=300,
        operation_id=operation_id,
        attempt_ordinal=3,
    )

    assert first is not None
    assert replay is not None
    assert historical is not None
    assert first.streak == 1
    assert replay.duplicate_attempt is True
    assert historical.duplicate_attempt is True
    assert replay.streak == historical.streak == 1
    assert replay.circuit_open is historical.circuit_open is False


def test_no_start_circuit_reopens_only_after_real_target_state_change() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=2)
    operation_id = "operation:placement-retry"
    for ordinal in range(2):
        service.quarantine_command(
            command_id=f"blocked-{ordinal}",
            action_name="Build_Pylon_Screen",
            requested_arguments=([64, 64],),
            world_target=(22.0 + ordinal, 24.0),
            failure_code="no_build_start_evidence",
            game_loop=100 + ordinal,
            operation_id=operation_id,
            attempt_ordinal=ordinal,
            target_state_revision="unchanged",
            failure_classification="dynamic_target_obstruction",
            classification_basis=("dynamic_unit_inside_footprint",),
        )

    assert (
        service.operation_retry_allowed(
            operation_id,
            world_target=(23.0, 24.0),
            target_state_revision="unchanged",
        )
        is False
    )
    assert (
        service.operation_retry_allowed(
            operation_id,
            world_target=(40.0, 40.0),
            target_state_revision="changed",
            failure_classification="dynamic_target_obstruction",
            target_side_evidence=True,
        )
        is False
    )
    assert (
        service.operation_retry_allowed(
            operation_id,
            world_target=(23.0, 24.0),
            target_state_revision="changed",
            failure_classification="dynamic_target_obstruction",
            target_side_evidence=True,
        )
        is True
    )
    state = service.operation_no_start_state(operation_id)
    assert state is not None and state.circuit_open is False

    retried = service.quarantine_command(
        command_id="after-change",
        action_name="Build_Pylon_Screen",
        requested_arguments=([80, 80],),
        world_target=(40.0, 40.0),
        failure_code="no_build_start_evidence",
        game_loop=300,
        operation_id=operation_id,
        attempt_ordinal=2,
        target_state_revision="changed",
        failure_classification="dynamic_target_obstruction",
        classification_basis=("dynamic_unit_inside_footprint",),
    )
    assert retried is not None
    assert retried.status == "defer_replan"
    assert retried.circuit_open is True
    assert retried.evidence["operation_id"] == operation_id
    assert retried.evidence["streak"] == 3


def test_no_start_circuit_reopens_for_fresh_ready_builder() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=3)
    operation_id = "operation:builder-refresh"
    for ordinal in range(3):
        service.quarantine_command(
            command_id=f"builder-attempt-{ordinal}",
            action_name="Build_Pylon_Screen",
            requested_arguments=([64, 64],),
            world_target=(22.0, 24.0),
            failure_code="no_build_start_evidence",
            game_loop=100 + ordinal,
            operation_id=operation_id,
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            placement_revision="dispatch-observation",
            target_state_revision="target-0",
            failure_classification="builder_not_ready",
            classification_basis=("builder_unavailable_after_dispatch",),
            observation_revision="failure-observation",
        )

    assert (
        service.operation_retry_allowed(
            operation_id,
            builder_tag=0xB2,
            builder_ready=True,
            observation_revision="failure-observation",
        )
        is False
    )
    assert (
        service.operation_retry_allowed(
            operation_id,
            builder_tag=0xB2,
            builder_ready=True,
            observation_revision="newer-observation",
        )
        is True
    )
    state = service.operation_no_start_state(operation_id)
    assert state is not None
    assert state.circuit_open is False


def test_material_legality_identity_blocks_observation_only_retry_and_checkpoint_restores() -> None:
    service = RawPlacementService(unit_names={}, no_start_streak_threshold=2)
    operation_id = "operation:material-tombstone"
    for ordinal in range(2):
        service.record_no_start_failure(
            operation_id=operation_id,
            command_id=f"material-attempt-{ordinal}",
            attempt_ordinal=ordinal,
            builder_tag=0xB1,
            world_target=(65.0, 65.0),
            placement_revision=f"obs-{ordinal}",
            target_state_revision=f"state-{ordinal}",
            failure_classification="dynamic_target_obstruction",
            target_side_evidence=True,
            material_legality_identity=(
                "build-legality:blocked-material"
                if ordinal == 0
                else "build-legality:second-material"
            ),
        )

    state = service.operation_no_start_state(operation_id)
    assert state is not None and state.circuit_open is True
    assert state.blocked_material_legality_identity == "build-legality:second-material"
    assert (
        service.operation_retry_allowed(
            operation_id,
            world_target=(65.0, 65.0),
            target_state_revision="state-new",
            builder_tag=0xB1,
            builder_ready=True,
            observation_revision="obs-new",
            material_legality_identity="build-legality:second-material",
        )
        is False
    )

    checkpoint = service.checkpoint_state()
    restored = RawPlacementService(unit_names={}, no_start_streak_threshold=2)
    restored.restore_checkpoint_state(checkpoint)
    restored_state = restored.operation_no_start_state(operation_id)
    assert restored_state == state
    assert (
        restored.operation_retry_allowed(
            operation_id,
            world_target=(65.0, 65.0),
            target_state_revision="state-new",
            builder_tag=0xB2,
            builder_ready=True,
            observation_revision="obs-new",
            material_legality_identity="build-legality:changed-material",
        )
        is True
    )


def test_material_identity_includes_target_legality_state_revision() -> None:
    common: dict[str, Any] = {
        "operation_id": "operation:material-state",
        "builder_tag": 0xB1,
        "ability_id": 881,
        "world_target": (30.0, 25.0),
        "target_legality_fingerprint": "sc2:stable",
    }
    state_a = build_material_legality_identity(
        **common,
        target_state_revision="state-a",
    )
    state_b = build_material_legality_identity(
        **common,
        target_state_revision="state-b",
    )
    assert state_a != state_b
    assert state_a == build_material_legality_identity(
        **common,
        target_state_revision="state-a",
    )


def test_target_state_revision_checkpoint_preserves_a_b_a_generation() -> None:
    service = RawPlacementService(unit_names={9: "Zergling"})
    target = (22.0, 24.0)
    action_name = "Build_Pylon_Screen"
    observation_a = SimpleNamespace(raw_units=[], feature_units=[], game_loop=[100])
    observation_b = SimpleNamespace(
        raw_units=[_unit(0xE1, 9, alliance=4, x=22.0, y=24.0)],
        feature_units=[],
        game_loop=[101],
    )

    service.observe(observation_a, require_feature_visibility=False)
    revision_a = service._target_state_revision(  # noqa: SLF001
        observation_a,
        action_name,
        target,
        anchor_tag=None,
    )
    service.observe(observation_b, require_feature_visibility=False)
    revision_b = service._target_state_revision(  # noqa: SLF001
        observation_b,
        action_name,
        target,
        anchor_tag=None,
    )
    service.observe(observation_a, require_feature_visibility=False)
    revision_a_again = service._target_state_revision(  # noqa: SLF001
        observation_a,
        action_name,
        target,
        anchor_tag=None,
    )

    assert revision_a != revision_b
    assert revision_a_again != revision_a

    checkpoint = service.checkpoint_state()
    restored = RawPlacementService(unit_names={9: "Zergling"})
    restored.restore_checkpoint_state(checkpoint)
    restored.observe(observation_a, require_feature_visibility=False)
    assert (
        restored._target_state_revision(  # noqa: SLF001
            observation_a,
            action_name,
            target,
            anchor_tag=None,
        )
        == revision_a_again
    )


def test_dynamic_circuit_reopens_when_same_target_obstruction_state_changes() -> None:
    service = RawPlacementService(
        unit_names={9: "Zergling"},
        no_start_streak_threshold=1,
    )
    operation_id = "operation:dynamic-refresh"
    target = (22.0, 24.0)
    blocked = SimpleNamespace(
        raw_units=[_unit(0xE1, 9, alliance=4, x=22.0, y=24.0)],
        feature_units=[],
        game_loop=[100],
    )
    service.observe(blocked, require_feature_visibility=False)
    blocked_revision = service._target_state_revision(  # noqa: SLF001
        blocked,
        "Build_Pylon_Screen",
        target,
        anchor_tag=None,
    )
    decision = service.record_no_start_failure(
        operation_id=operation_id,
        command_id="dynamic-blocked",
        attempt_ordinal=0,
        builder_tag=0xB1,
        world_target=target,
        placement_revision="observation-100",
        target_state_revision=blocked_revision,
        failure_classification="dynamic_target_obstruction",
        classification_basis=("dynamic_unit_inside_footprint",),
        target_side_evidence=True,
    )
    assert decision.circuit_open is True

    cleared = SimpleNamespace(raw_units=[], feature_units=[], game_loop=[101])
    service.observe(cleared, require_feature_visibility=False)
    cleared_revision = service._target_state_revision(  # noqa: SLF001
        cleared,
        "Build_Pylon_Screen",
        target,
        anchor_tag=None,
    )
    assert cleared_revision != blocked_revision
    assert (
        service.operation_retry_allowed(
            operation_id,
            world_target=target,
            target_state_revision=cleared_revision,
        )
        is True
    )


def test_dynamic_suppression_binds_failure_observation_not_dispatch_snapshot() -> None:
    service = RawPlacementService(unit_names={9: "Zergling"})
    target = (22.0, 24.0)
    blocked = SimpleNamespace(
        raw_units=[_unit(0xE1, 9, alliance=4, x=22.0, y=24.0)],
        feature_units=[],
        game_loop=[212],
    )
    service.observe(blocked, require_feature_visibility=False)
    service.quarantine_command(
        command_id="dynamic-failure-observation",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=target,
        failure_code="no_build_start_evidence",
        game_loop=212,
        operation_id="operation:dynamic-observation",
        attempt_ordinal=0,
        failure_classification="dynamic_target_obstruction",
        classification_basis=("dynamic_unit_inside_footprint",),
        observation=blocked,
    )

    unchanged = SimpleNamespace(
        raw_units=[_unit(0xE1, 9, alliance=4, x=22.0, y=24.0)],
        feature_units=[],
        game_loop=[213],
    )
    assert (
        service.is_quarantined(
            "Build_Pylon_Screen",
            target,
            game_loop=213,
            observation=unchanged,
        )
        is True
    )

    cleared = SimpleNamespace(raw_units=[], feature_units=[], game_loop=[214])
    assert (
        service.is_quarantined(
            "Build_Pylon_Screen",
            target,
            game_loop=214,
            observation=cleared,
        )
        is False
    )


def test_classification_controls_target_suppression() -> None:
    service = RawPlacementService(unit_names={})
    target = (22.0, 24.0)
    unknown = service.quarantine_command(
        command_id="unknown-no-start",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=target,
        failure_code="no_build_start_evidence",
        operation_id="operation:unknown",
        attempt_ordinal=0,
        failure_classification="gameplay_no_start_unknown",
        classification_basis=("no_authoritative_rejection_evidence",),
    )
    assert unknown is not None and unknown.suppressed_target is False
    assert not service.is_quarantined("Build_Pylon_Screen", target)

    dynamic = service.quarantine_command(
        command_id="dynamic-no-start",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=(30.0, 30.0),
        failure_code="no_build_start_evidence",
        operation_id="operation:dynamic",
        attempt_ordinal=0,
        target_state_revision="target-0",
        failure_classification="dynamic_target_obstruction",
        classification_basis=("dynamic_unit_inside_footprint",),
    )
    assert dynamic is not None and dynamic.suppressed_target is True
    assert service.is_quarantined("Build_Pylon_Screen", (30.0, 30.0))


def test_builder_and_freshness_failures_do_not_quarantine_target() -> None:
    service = RawPlacementService(unit_names={})
    target = (22.0, 24.0)
    for code in ("builder_unavailable", "stale_observation"):
        service.quarantine_command(
            command_id=code,
            action_name="Build_Pylon_Screen",
            requested_arguments=([64, 64],),
            world_target=target,
            failure_code=code,
            game_loop=100,
            operation_id="operation:attribution",
            attempt_ordinal=0 if code == "builder_unavailable" else 1,
        )

    assert service.is_quarantined("Build_Pylon_Screen", target) is False


def test_snapshot_structure_does_not_count_as_current_footprint_occupancy() -> None:
    service = RawPlacementService(unit_names={60: "Pylon"})
    target = (22.0, 24.0)
    observation = SimpleNamespace(
        raw_units=[
            SimpleNamespace(
                tag=0xC1,
                unit_type=60,
                alliance=4,
                display_type=2,
                build_progress=100,
                x=22.0,
                y=24.0,
            )
        ],
        feature_units=[],
        game_loop=[100],
    )

    service.observe(observation, require_feature_visibility=False)

    assert (
        service.is_quarantined(
            "Build_Pylon_Screen",
            target,
            observation=observation,
        )
        is False
    )


def test_same_pylon_failure_does_not_expire_without_target_state_change() -> None:
    service = RawPlacementService(unit_names={2: "Probe"})
    observation = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, alliance=1, x=20, y=20)],
        feature_units=[],
        feature_screen=None,
        player_common=SimpleNamespace(minerals=500, vespene=500),
        game_loop=[100],
    )
    service.resolve(
        command_id="pylon-one",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(22.0, 24.0),
        builder_tag=0xB1,
    )
    service.quarantine_command(
        command_id="pylon-one",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        world_target=(22.0, 24.0),
        failure_code="build_started_effect_missing",
        game_loop=212,
    )
    later_same_state = SimpleNamespace(**{**vars(observation), "game_loop": [500]})

    assert not service.is_quarantined(
        "Build_Pylon_Screen",
        (22.0, 24.0),
        game_loop=500,
        observation=later_same_state,
    )
    assert not service.is_quarantined(
        "Build_Pylon_Screen",
        (30.0, 30.0),
        game_loop=500,
        observation=later_same_state,
    )


def test_expansion_rejects_illegal_exact_world_footprint_before_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RawPlacementService(unit_names={2: "Probe", 59: "Nexus", 341: "MineralField"})
    resources = [
        _unit(0x201 + index, 341, alliance=3, x=x, y=y)
        for index, (x, y) in enumerate(((87, 80), (85, 85), (80, 87), (75, 85), (73, 80), (75, 75)))
    ]
    observation = SimpleNamespace(
        raw_units=[
            _unit(0xB1, 2, alliance=1, x=20, y=20),
            _unit(0xC1, 59, alliance=1, x=20, y=20),
            *resources,
        ],
        feature_units=[],
        feature_screen=object(),
        player_common=SimpleNamespace(minerals=500, vespene=500),
        game_loop=[100],
    )
    service.observe(observation, require_feature_visibility=False)
    monkeypatch.setattr(
        extractor_module,
        "world_build_target_is_legal",
        lambda *args, **kwargs: False,
        raising=False,
    )

    with pytest.raises(RawPlacementFailure) as failure:
        service.resolve(
            command_id="illegal-expansion",
            action_name="Build_Nexus_Near",
            requested_arguments=(0x201,),
            observation=observation,
            world_target=None,
            builder_tag=0xB1,
        )

    assert failure.value.code == "no_legal_placement"
    assert service.active_reservation_count == 0


def test_expansion_rejects_unpathable_full_map_footprint() -> None:
    class Grid:
        def __init__(self, value: int) -> None:
            self.shape = (128, 128)
            self.rows = [[value] * 128 for _ in range(128)]

        def __getitem__(self, index: int) -> list[int]:
            return self.rows[index]

    service = RawPlacementService(unit_names={2: "Probe", 59: "Nexus", 341: "MineralField"})
    service.set_world_to_minimap_transform((1.0, 0.0, 0.0, 128.0, 127.0))
    resources = [
        _unit(0x301 + index, 341, alliance=3, x=x, y=y)
        for index, (x, y) in enumerate(((87, 80), (85, 85), (80, 87), (75, 85), (73, 80), (75, 75)))
    ]
    buildable = Grid(1)
    pathable = Grid(1)
    pathable[48][80] = 0
    observation = SimpleNamespace(
        raw_units=[
            _unit(0xB1, 2, alliance=1, x=20, y=20),
            _unit(0xC1, 59, alliance=1, x=20, y=20),
            *resources,
        ],
        feature_units=[],
        feature_screen=None,
        feature_minimap=SimpleNamespace(
            buildable=buildable,
            pathable=pathable,
            visibility_map=Grid(2),
            player_relative=Grid(0),
        ),
        player_common=SimpleNamespace(minerals=500, vespene=500),
        game_loop=[100],
    )

    with pytest.raises(RawPlacementFailure) as failure:
        service.resolve(
            command_id="unpathable-expansion",
            action_name="Build_Nexus_Near",
            requested_arguments=(0x301,),
            observation=observation,
            world_target=None,
            builder_tag=0xB1,
        )

    assert failure.value.code == "no_legal_placement"
    assert service.active_reservation_count == 0


def test_target_state_revision_normalizes_foreign_boolean_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign_bool = type("bool", (), {"__bool__": lambda self: True})
    service = RawPlacementService(unit_names={})
    observation = SimpleNamespace(
        raw_units=[],
        feature_units=[],
        feature_screen=object(),
    )
    monkeypatch.setattr(
        extractor_module,
        "world_build_target_is_legal",
        lambda *args, **kwargs: foreign_bool(),
    )

    revision = service._target_state_revision(
        observation,
        "Build_Pylon_Screen",
        (30.0, 25.0),
        anchor_tag=None,
    )

    assert len(revision) == 64


def test_builder_approach_does_not_route_through_dynamic_structure_occupancy() -> None:
    class Grid:
        def __init__(self, value: int, size: int = 16) -> None:
            self.shape = (size, size)
            self.rows = [[value] * size for _ in range(size)]

        def __getitem__(self, index: int) -> list[int]:
            return self.rows[index]

    pathable = Grid(1)
    player_relative = Grid(0)
    for y in range(16):
        player_relative[y][7] = 1
    builder = SimpleNamespace(
        tag=0xB1,
        unit_type=2,
        alliance=1,
        is_on_screen=True,
        x=3,
        y=8,
        radius=0.375,
    )

    reachable = extractor_module._builder_reachable_cells(
        pathable,
        player_relative,
        [builder],
        {2: "Probe"},
        16,
        builder_tags=(0xB1,),
    )

    assert reachable is not None
    assert (6, 8) in reachable
    assert (8, 8) not in reachable


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


def test_raw_placement_rejects_candidate_from_older_observation_revision() -> None:
    service = RawPlacementService(unit_names={2: "Probe"})
    target = (22.25, 24.5)
    first = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, alliance=1, x=20, y=20)],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )
    candidate_revision = _placement_revision(first)
    candidate_id = _placement_candidate_id(
        "Build_Pylon_Screen",
        target,
        0xB1,
        candidate_revision,
        2,
        False,
    )
    newer = SimpleNamespace(
        raw_units=first.raw_units,
        feature_units=[],
        feature_screen=None,
        game_loop=[101],
    )

    with pytest.raises(RawPlacementFailure) as stale:
        service.resolve(
            command_id="older-placement-candidate",
            action_name="Build_Pylon_Screen",
            requested_arguments=([64, 64],),
            observation=newer,
            world_target=target,
            preferred_anchor_tag=0xB1,
            builder_tags=(0xB1,),
            builder_tag=0xB1,
            placement_candidate_id=candidate_id,
            candidate_placement_revision=candidate_revision,
        )

    assert stale.value.code == "placement_candidate_stale"


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


def test_new_observation_revision_replaces_reservation_with_closed_ledger_chain() -> None:
    service = RawPlacementService(unit_names={2: "Probe"})
    first_observation = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, alliance=1, x=20, y=20)],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )
    newer_observation = SimpleNamespace(
        raw_units=[_unit(0xB1, 2, alliance=1, x=21, y=20)],
        feature_units=[],
        feature_screen=None,
        game_loop=[116],
    )

    first = service.resolve(
        command_id="build-across-observations",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=first_observation,
        world_target=(30.0, 25.0),
        builder_tag=0xB1,
        ability_name="Build_Pylon_pt",
    )
    same_revision = service.resolve(
        command_id="build-across-observations",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=first_observation,
        world_target=(30.0, 25.0),
        builder_tag=0xB1,
        ability_name="Build_Pylon_pt",
    )
    refreshed = service.resolve(
        command_id="build-across-observations",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=newer_observation,
        world_target=(30.0, 25.0),
        builder_tag=0xB1,
        ability_name="Build_Pylon_pt",
    )

    assert same_revision.reservation_id == first.reservation_id
    assert refreshed.reservation_id != first.reservation_id
    transitions = service.drain_transition_history("build-across-observations")
    assert [
        (transition["reservation_id"], transition["previous_state"], transition["next_state"])
        for transition in transitions
    ] == [
        (first.reservation_id, "unreserved", "reserved"),
        (first.reservation_id, "reserved", "released"),
        (refreshed.reservation_id, "unreserved", "reserved"),
    ]
    assert transitions[1]["release_reason"] == "reservation_replaced"
    assert service.leased_builder_tags == frozenset({0xB1})


def test_builder_rebind_closes_old_reservation_before_acquiring_new_lease() -> None:
    service = RawPlacementService(unit_names={2: "Probe"})
    observation = SimpleNamespace(
        raw_units=[
            _unit(0xB1, 2, alliance=1, x=20, y=20),
            _unit(0xB2, 2, alliance=1, x=21, y=20),
        ],
        feature_units=[],
        feature_screen=None,
        game_loop=[100],
    )

    first = service.resolve(
        command_id="build-rebound",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(30.0, 25.0),
        builder_tags=(0xB1, 0xB2),
        builder_tag=0xB1,
        ability_name="Build_Pylon_pt",
    )
    rebound = service.resolve(
        command_id="build-rebound",
        action_name="Build_Pylon_Screen",
        requested_arguments=([64, 64],),
        observation=observation,
        world_target=(30.0, 25.0),
        builder_tags=(0xB1, 0xB2),
        builder_tag=0xB2,
        ability_name="Build_Pylon_pt",
    )

    assert rebound.reservation_id != first.reservation_id
    assert service.leased_builder_tags == frozenset({0xB2})
    assert service.builder_lease_owner(0xB1) is None
    assert service.builder_lease_owner(0xB2) == "build-rebound"
    transitions = service.drain_transition_history("build-rebound")
    assert [
        (transition["reservation_id"], transition["previous_state"], transition["next_state"])
        for transition in transitions
    ] == [
        (first.reservation_id, "unreserved", "reserved"),
        (first.reservation_id, "reserved", "released"),
        (rebound.reservation_id, "unreserved", "reserved"),
    ]
    assert transitions[1]["release_reason"] == "reservation_replaced"

    service.release_command("build-rebound", game_loop=101)
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
        "target_state_revision": reservation.target_state_revision,
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
