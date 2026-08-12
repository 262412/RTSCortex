from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.analyze_authoritative_build_circuit_canary as analyzer
from rtscortex.memory import StoredEvent
from scripts.analyze_authoritative_build_circuit_canary import (
    CanaryArtifactError,
    _expected_attempt_id,
    _replay_phases,
    _replay_runtime_events,
    analyze_canary_run,
)


def _attempt(operation_id: str, command_id: str, ordinal: int) -> str:
    return _expected_attempt_id(operation_id, command_id, ordinal)


def _event(
    index: int,
    phase: str,
    *,
    operation_id: str | None = None,
    action: str | None = "Build_Pylon_Screen",
    command_id: str | None = None,
    attempt_id: str | None = None,
    attempt_ordinal: int | None = None,
    game_loop: int,
    observation_revision: str | None = None,
    builder_tag: int | None = None,
    **values: Any,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_type": "authoritative_build_circuit_canary_phase",
        "schema_version": "1.0",
        "diagnostic_only": True,
        "mode": "stale_candidate_then_builder_rebind",
        "event_index": index,
        "run_id": "run-canary",
        "episode_id": "episode-0",
        "seed": 7,
        "phase": phase,
        "game_loop": game_loop,
        "observation_revision": observation_revision,
        "operation_id": operation_id,
        "semantic_action": None if action is None else "Pylon",
        "reason": "test",
    }
    if action is not None:
        payload["runtime_action"] = "Build_Pylon_Screen"
    if command_id is not None:
        payload["command_id"] = command_id
    if attempt_id is not None:
        payload["attempt_id"] = attempt_id
    if attempt_ordinal is not None:
        payload["attempt_ordinal"] = attempt_ordinal
    if builder_tag is not None:
        payload["builder_tag"] = builder_tag
    payload.update(values)
    return payload


def _zero() -> dict[str, Any]:
    return {
        "reservation_count": 0,
        "leased_builder_tags": [],
        "effect_inflight_count": 0,
    }


def _valid_phases() -> list[dict[str, object]]:
    operation_id = "operation:" + "a" * 64
    action = "Pylon"
    events: list[dict[str, object]] = [
        _event(0, "initialized", game_loop=0, observation_revision="obs-0", action=None),
    ]
    for index in range(3):
        command_id = f"command-{index}"
        attempt_id = _attempt(operation_id, command_id, index)
        events.append(
            _event(
                len(events),
                "failure_command_held",
                operation_id=operation_id,
                action=action,
                command_id=command_id,
                attempt_id=attempt_id,
                attempt_ordinal=index,
                game_loop=index * 2 + 1,
                observation_revision=f"obs-{index + 1}",
                builder_tag=100,
                **_zero(),
            )
        )
        events.append(
            _event(
                len(events),
                "authoritative_failure_observed",
                operation_id=operation_id,
                action=action,
                command_id=command_id,
                attempt_id=attempt_id,
                attempt_ordinal=index,
                game_loop=index * 2 + 2,
                observation_revision=f"obs-{index + 1}",
                builder_tag=100,
                authoritative_streak=index + 1,
                authoritative_threshold=3,
                circuit_open=index == 2,
                transition="closed_to_open" if index == 2 else None,
                duplicate_attempt=False,
                reason="placement_candidate_stale",
                **_zero(),
            )
        )
    reset_command = "command-reset"
    reset_attempt = _attempt(operation_id, reset_command, 3)
    events.extend(
        [
            _event(
                len(events),
                "circuit_open_observed",
                operation_id=operation_id,
                action=action,
                command_id="command-2",
                attempt_id=_attempt(operation_id, "command-2", 2),
                attempt_ordinal=2,
                game_loop=7,
                observation_revision="obs-3",
                builder_tag=100,
                authoritative_streak=3,
                authoritative_threshold=3,
                circuit_open=True,
                transition="closed_to_open",
                **_zero(),
            ),
            _event(
                len(events),
                "core_defer_observed",
                operation_id=operation_id,
                action=action,
                game_loop=8,
                observation_revision="obs-core",
                reason="authoritative_build_pre_dispatch_circuit_defer",
                command_count=0,
                idle_reason="plan_commands_deferred",
                builder_tag=100,
                authoritative_state={"streak": 3, "circuit_open": True},
                **_zero(),
            ),
            _event(
                len(events),
                "builder_rebound",
                operation_id=operation_id,
                action=action,
                game_loop=9,
                observation_revision="obs-rebound",
                builder_tag=100,
                replacement_builder_tag=200,
                reason="fresh_ready_unleased_builder_observed",
            ),
            _event(
                len(events),
                "reset_command_observed",
                operation_id=operation_id,
                action=action,
                command_id=reset_command,
                attempt_id=reset_attempt,
                attempt_ordinal=3,
                game_loop=10,
                observation_revision="obs-reset-command",
                builder_tag=200,
                replacement_builder_tag=200,
                reason="validated_builder_material_change",
                **_zero(),
            ),
            _event(
                len(events),
                "reset_dispatch_observed",
                operation_id=operation_id,
                action=action,
                command_id=reset_command,
                attempt_id=reset_attempt,
                attempt_ordinal=3,
                game_loop=11,
                observation_revision="obs-reset-command",
                builder_tag=200,
                replacement_builder_tag=200,
                query_result="Success",
                primitive_constructed=True,
                primitive_submitted=False,
                raw_diagnostic={
                    "placement_query_result": "Success",
                    "available_ability_query": "available",
                    "target_legality_fingerprint": "sc2:fingerprint-reset",
                    "primitive_constructed": True,
                    "primitive_submitted": False,
                },
            ),
            _event(
                len(events),
                "reset_dispatch_submitted",
                operation_id=operation_id,
                action=action,
                command_id=reset_command,
                attempt_id=reset_attempt,
                attempt_ordinal=3,
                game_loop=12,
                observation_revision="obs-reset-command",
                builder_tag=200,
                replacement_builder_tag=200,
                query_result="Success",
                primitive_constructed=True,
                primitive_submitted=True,
                primitive_submitted_game_loop=12,
                raw_diagnostic={
                    "placement_query_result": "Success",
                    "available_ability_query": "available",
                    "target_legality_fingerprint": "sc2:fingerprint-reset",
                    "primitive_constructed": True,
                    "primitive_submitted": True,
                },
            ),
            _event(
                len(events),
                "effect_confirmed",
                operation_id=operation_id,
                action=action,
                command_id=reset_command,
                game_loop=13,
                observation_revision="obs-effect",
                builder_tag=200,
                primitive_submitted=True,
                effect_status="succeeded",
                effect_evidence={
                    "effect_kind": "build",
                    "build_started": True,
                    "observed_structure_tag": "0xfeed",
                    "confirmation_kind": "new_structure",
                },
                **_zero(),
            ),
            _event(
                len(events),
                "complete",
                operation_id=operation_id,
                action=action,
                command_id=reset_command,
                game_loop=13,
                observation_revision="obs-effect",
                **_zero(),
            ),
        ]
    )
    for index, event in enumerate(events):
        event["event_index"] = index
    return events


def _runtime_event(event_id: int, event_type: str, payload: dict[str, object]) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="run-canary",
        episode_id="episode-0",
        step_id=event_id,
        event_type=event_type,
        created_at=f"2026-08-12T00:00:{event_id:02d}+00:00",
        payload=payload,
    )


def _valid_runtime_events() -> list[StoredEvent]:
    operation_id = "operation:" + "a" * 64
    events: list[StoredEvent] = []
    for ordinal in range(3):
        command_id = f"command-{ordinal}"
        attempt_id = _attempt(operation_id, command_id, ordinal)
        events.append(
            _runtime_event(
                ordinal + 1,
                "execution",
                {
                    "operation_id": operation_id,
                    "command_id": command_id,
                    "attempt_id": attempt_id,
                    "attempt_ordinal": ordinal,
                    "action_name": "Build_Pylon_Screen",
                    "semantic_action": "Pylon",
                    "runtime_action": "Build_Pylon_Screen",
                    "status": "failed",
                    "success": False,
                    "execution_stage": "pre_dispatch",
                    "failure_code": "placement_candidate_stale",
                    "authoritative_pre_dispatch": {
                        "operation_id": operation_id,
                        "command_id": command_id,
                        "attempt_id": attempt_id,
                        "attempt_ordinal": ordinal,
                        "action_name": "Build_Pylon_Screen",
                        "failure_code": "placement_candidate_stale",
                        "status": "defer_replan" if ordinal == 2 else "retry",
                        "streak": ordinal + 1,
                        "threshold": 3,
                        "circuit_open": ordinal == 2,
                        "duplicate_attempt": False,
                        "builder_tag": 100,
                        "ability_id": 881,
                        "world_target": [30.0 + ordinal, 30.0],
                        "material_legality_identity": "build-legality:" + str(ordinal) * 64,
                        "material_evidence_valid": True,
                        "invalid_evidence_reasons": [],
                        "state_transition": "closed_to_open" if ordinal == 2 else None,
                    },
                },
            )
        )
    third_attempt = _attempt(operation_id, "command-2", 2)
    reset_attempt = _attempt(operation_id, "command-reset", 3)
    events.extend(
        [
            _runtime_event(
                4,
                "authoritative_build_pre_dispatch_circuit_defer",
                {
                    "operation_id": operation_id,
                    "action_name": "Build_Pylon_Screen",
                    "streak": 3,
                    "threshold": 3,
                    "failure_count": 3,
                    "opened_command_id": "command-2",
                    "opened_attempt_id": third_attempt,
                    "opened_attempt_ordinal": 2,
                },
            ),
            _runtime_event(
                5,
                "authoritative_build_pre_dispatch_circuit_reset",
                {
                    "operation_id": operation_id,
                    "action_name": "Build_Pylon_Screen",
                    "reason": "semantic_legality_material_change",
                    "previous_semantic_material_identity": "semantic-build-material:" + "1" * 64,
                    "current_semantic_material_identity": "semantic-build-material:" + "2" * 64,
                    "raw_material_legality_identity": "build-legality:" + "3" * 64,
                    "previous_builder_tag": 100,
                    "bound_builder_tag": 200,
                    "reset_from_streak": 3,
                    "operation_epoch_changed": False,
                },
            ),
            _runtime_event(
                6,
                "placement_ledger_transition",
                {
                    "operation_id": operation_id,
                    "command_id": "command-reset",
                    "attempt_id": reset_attempt,
                    "attempt_ordinal": 3,
                    "action_name": "Build_Pylon_Screen",
                    "builder_lease_state": None,
                    "next_state": "build_started",
                    "release_reason": "build_start_observed",
                    "authoritative_pre_dispatch": {
                        "operation_id": operation_id,
                        "command_id": "command-reset",
                        "attempt_id": reset_attempt,
                        "attempt_ordinal": 3,
                        "action_name": "Build_Pylon_Screen",
                        "status": "reset",
                        "state_transition": "open_to_reset",
                        "circuit_open": False,
                    },
                },
            ),
            _runtime_event(
                7,
                "placement_ledger_transition",
                {
                    "operation_id": None,
                    "command_id": "command-reset",
                    "attempt_id": None,
                    "attempt_ordinal": None,
                    "action_name": "Build_Pylon_Screen",
                    "builder_lease_state": "released",
                    "next_state": "occupied",
                    "release_reason": "effect_confirmed",
                },
            ),
            _runtime_event(
                8,
                "execution",
                {
                    "operation_id": operation_id,
                    "command_id": "command-reset",
                    "attempt_id": reset_attempt,
                    "attempt_ordinal": 3,
                    "action_name": "Build_Pylon_Screen",
                    "semantic_action": "Pylon",
                    "runtime_action": "Build_Pylon_Screen",
                    "status": "succeeded",
                    "success": True,
                    "effect_evidence": {
                        "effect_kind": "build",
                        "build_started": True,
                        "observed_structure_tag": "0xfeed",
                        "confirmation_kind": "new_structure",
                    },
                },
            ),
        ]
    )
    return events


def _artifact_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    run_set = tmp_path / "run-set"
    run_dir = run_set / "actual-run"
    run_dir.mkdir(parents=True)
    (run_set / "canonical-run").symlink_to(run_dir, target_is_directory=True)
    phase_path = run_dir / "authoritative-build-circuit-canary.jsonl"
    phase_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in _valid_phases()) + "\n",
        encoding="utf-8",
    )
    events_path = run_dir / "events.jsonl"
    events_path.write_text("runtime-events\n", encoding="utf-8")
    summary_path = run_dir / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    expected_sha = "a" * 40
    source_hash = "c" * 64
    source_commit = "b" * 40
    source = {
        "git_head_before": expected_sha,
        "git_head_after": expected_sha,
        "superproject_dirty_before": "false",
        "superproject_dirty_after": "false",
        "submodule_commit_before": source_commit,
        "submodule_commit_after": source_commit,
        "submodule_dirty_before": "false",
        "submodule_dirty_after": "false",
        "submodule_gitlink_before": source_commit,
        "submodule_gitlink_after": source_commit,
        "submodule_diff_sha256_before": source_hash,
        "submodule_diff_sha256_after": source_hash,
        "reviewed_source_commit_before": source_commit,
        "reviewed_source_commit_after": source_commit,
        "reviewed_source_diff_sha256_before": source_hash,
        "reviewed_source_diff_sha256_after": source_hash,
        "reviewed_source_tree_sha256_before": source_hash,
        "reviewed_source_tree_sha256_after": source_hash,
    }
    attestation = {
        "diagnostic_only": True,
        "expected_git_sha": expected_sha,
        "seed": 7,
        "exit_code": 0,
        "run_dir": str(run_dir.resolve()),
        "canary_journal_path": str(phase_path.resolve()),
        "canary_journal_sha256": hashlib.sha256(phase_path.read_bytes()).hexdigest(),
        "events_path": str(events_path.resolve()),
        "events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "source_attestation": source,
    }
    (run_set / "canary-attestation.json").write_text(
        json.dumps(attestation, sort_keys=True), encoding="utf-8"
    )
    return run_set, run_dir, attestation


def test_replay_accepts_exact_three_failure_real_reset_path() -> None:
    report = _replay_phases(_valid_phases(), expected_seed=7)

    assert report["failure_count"] == 3
    assert report["open_count"] == 1
    assert report["material_reset_count"] == 1
    assert report["success_reset_count"] == 1
    assert report["primitive_submission_count"] == 1
    assert report["post_open_rejection_count"] == 0


def test_runtime_events_are_authoritative_for_the_canary_path() -> None:
    phase_events = _valid_phases()
    phase_replay = _replay_phases(phase_events, expected_seed=7)
    report = _replay_runtime_events(
        _valid_runtime_events(), phase_events=phase_events, phase_replay=phase_replay
    )

    assert report["runtime_failure_count"] == 3
    assert report["runtime_core_defer_count"] == 1
    assert report["runtime_core_reset_count"] == 1
    assert report["runtime_raw_reset_count"] == 1
    assert report["runtime_success_count"] == 1


def test_analyzer_rejects_tampered_hash_and_wrong_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_set, run_dir, attestation = _artifact_fixture(tmp_path)
    monkeypatch.setattr(analyzer, "_summary_artifact_is_canonical", lambda *args, **kwargs: True)
    monkeypatch.setattr(analyzer, "read_event_log", lambda _path: _valid_runtime_events())
    monkeypatch.setattr(
        analyzer,
        "_replay_runtime_events",
        lambda *args, **kwargs: {
            "runtime_raw_reset_count": 1,
        },
    )
    report = analyze_canary_run(run_set, expected_git_sha="a" * 40, seed=7)
    assert report["accepted"] is True

    attestation["events_sha256"] = "0" * 64
    (run_set / "canary-attestation.json").write_text(
        json.dumps(attestation, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(CanaryArtifactError, match="events.jsonl SHA"):
        analyze_canary_run(run_set, expected_git_sha="a" * 40, seed=7)

    attestation["events_sha256"] = hashlib.sha256(
        (run_dir / "events.jsonl").read_bytes()
    ).hexdigest()
    attestation["run_dir"] = str((run_set / "wrong-run").resolve())
    (run_set / "canary-attestation.json").write_text(
        json.dumps(attestation, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(CanaryArtifactError, match="run_dir attestation"):
        analyze_canary_run(run_set, expected_git_sha="a" * 40, seed=7)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "idle", "post_open", "zero", "release", "material", "effect"),
)
def test_runtime_replay_rejects_forged_or_incomplete_evidence(mutation: str) -> None:
    phase_events = _valid_phases()
    phase_replay = _replay_phases(phase_events, expected_seed=7)
    runtime_events = _valid_runtime_events()
    if mutation == "missing":
        runtime_events = runtime_events[1:]
    elif mutation == "idle":
        phase_events[8]["idle_reason"] = "no_legal_action"
        with pytest.raises(CanaryArtifactError):
            _replay_phases(phase_events, expected_seed=7)
        return
    elif mutation == "post_open":
        runtime_events.insert(
            3,
            _runtime_event(
                4,
                "command_lifecycle",
                {
                    "status": "dispatched",
                    "command": {
                        "command_id": "forged-after-open",
                        "operation_id": "operation:" + "a" * 64,
                    },
                },
            ),
        )
        runtime_events = [event if event.event_id < 4 else event for event in runtime_events]
    elif mutation == "zero":
        runtime_events = []
    elif mutation == "release":
        runtime_events = [event for event in runtime_events if event.event_id != 7]
    elif mutation == "material":
        runtime_events[0].payload["authoritative_pre_dispatch"]["material_legality_identity"] = (
            "build-legality:1"
        )
    else:
        runtime_events[-1].payload["effect_evidence"]["observed_structure_tag"] = ""

    with pytest.raises(CanaryArtifactError):
        _replay_runtime_events(runtime_events, phase_events=phase_events, phase_replay=phase_replay)


@pytest.mark.parametrize(
    "mutation",
    ("streak", "open_transition", "query", "submission", "ownership"),
)
def test_replay_rejects_tampered_phase_evidence(mutation: str) -> None:
    events = _valid_phases()
    if mutation == "streak":
        events[2]["authoritative_streak"] = 3
    elif mutation == "open_transition":
        events[7]["transition"] = None
    elif mutation == "query":
        events[11]["query_result"] = "Success (cached)"
    elif mutation == "submission":
        events[12]["primitive_submitted"] = False
    else:
        events[7]["effect_inflight_count"] = 1

    with pytest.raises(CanaryArtifactError):
        _replay_phases(events, expected_seed=7)
