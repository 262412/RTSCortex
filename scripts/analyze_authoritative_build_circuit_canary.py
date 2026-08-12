"""Fail-closed analysis for the diagnostic-only authoritative Build canary.

The worker writes a small phase journal in addition to the normal Runtime
``events.jsonl``/``summary.json`` pair.  This module replays that phase journal
and derives every count from the journal; fields such as ``failure_count`` in a
runner manifest are never used as evidence.  The canary is deliberately not a
qualification or production gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rtscortex.evaluation.report import _build_run_summary
from rtscortex.memory import read_event_log

_SHA40_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA64_RE = re.compile(r"[0-9a-f]{64}\Z")
_OPERATION_RE = re.compile(r"operation:[0-9a-f]{64}\Z")
_ATTEMPT_RE = re.compile(r"attempt:[0-9a-f]{64}\Z")
_BUILD_LEGALITY_RE = re.compile(r"build-legality:[0-9a-f]{64}\Z")
_SEMANTIC_MATERIAL_RE = re.compile(r"semantic-build-material:[0-9a-f]{64}\Z")
_CANARY_JOURNAL_NAME = "authoritative-build-circuit-canary.jsonl"
_PHASE_EVENT_TYPE = "authoritative_build_circuit_canary_phase"
_CANARY_SCHEMA_VERSION = "1.0"
_CANARY_MODE = "stale_candidate_then_builder_rebind"
_SEMANTIC_ACTION = "BUILD PYLON"
_RUNTIME_ACTION = "Build_Pylon_Screen"
_RAW_CIRCUIT_OPEN_CODES = frozenset(
    {"authoritative_pre_dispatch_circuit_open", "operation_no_start_circuit_open"}
)
_PHASES = (
    "initialized",
    "failure_command_held",
    "authoritative_failure_observed",
    "circuit_open_observed",
    "core_defer_observed",
    "builder_rebound",
    "reset_command_observed",
    "reset_dispatch_observed",
    "reset_dispatch_submitted",
    "effect_confirmed",
    "complete",
)


class CanaryArtifactError(ValueError):
    """Raised when a canary artifact is malformed or not fully attested."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_attempt_id(operation_id: str, command_id: str, attempt_ordinal: int) -> str:
    encoded = json.dumps(
        {
            "operation_id": operation_id,
            "command_id": command_id,
            "attempt_ordinal": attempt_ordinal,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "attempt:" + hashlib.sha256(encoded).hexdigest()


def _required_string(payload: Mapping[str, Any], key: str, *, phase: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CanaryArtifactError(f"{phase}: {key} is missing or not a string")
    return value


def _nonnegative_int(payload: Mapping[str, Any], key: str, *, phase: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value < 0:
        raise CanaryArtifactError(f"{phase}: {key} is missing or invalid")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = int(value, 0)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    raise CanaryArtifactError(f"{label} is missing or invalid")


def _tags(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CanaryArtifactError(f"{label} is missing or not a list")
    parsed = tuple(_positive_int(item, label=label) for item in value)
    if len(parsed) != len(set(parsed)):
        raise CanaryArtifactError(f"{label} contains duplicate Builder tags")
    return parsed


def _zero_ownership(payload: Mapping[str, Any], *, phase: str) -> None:
    reservation_count = payload.get("reservation_count")
    if type(reservation_count) is not int or reservation_count != 0:
        raise CanaryArtifactError(f"{phase}: reservation_count is not exactly zero")
    if _tags(payload.get("leased_builder_tags"), label=f"{phase}.leased_builder_tags"):
        raise CanaryArtifactError(f"{phase}: a Builder lease remains")
    inflight = payload.get("effect_inflight_count")
    if type(inflight) is not int or inflight != 0:
        raise CanaryArtifactError(f"{phase}: effect_inflight_count is not exactly zero")


def _operation(payload: Mapping[str, Any], *, phase: str, expected: str | None) -> str:
    operation_id = _required_string(payload, "operation_id", phase=phase)
    if _OPERATION_RE.fullmatch(operation_id) is None:
        raise CanaryArtifactError(f"{phase}: malformed operation_id")
    if expected is not None and operation_id != expected:
        raise CanaryArtifactError(f"{phase}: operation identity changed")
    return operation_id


def _semantic_action(payload: Mapping[str, Any], *, phase: str, expected: str | None) -> str:
    action = payload.get("semantic_action")
    if action != _SEMANTIC_ACTION:
        raise CanaryArtifactError(
            f"{phase}: semantic_action must be {_SEMANTIC_ACTION!r} for the canary profile"
        )
    if expected is not None and action != expected:
        raise CanaryArtifactError(f"{phase}: semantic action changed")
    runtime_action = payload.get("runtime_action")
    if runtime_action is not None and runtime_action != _RUNTIME_ACTION:
        raise CanaryArtifactError(
            f"{phase}: runtime_action must be {_RUNTIME_ACTION!r} for the canary profile"
        )
    return action


def _runtime_action(payload: Mapping[str, Any], *, phase: str) -> str:
    value = payload.get("runtime_action")
    if value != _RUNTIME_ACTION:
        raise CanaryArtifactError(
            f"{phase}: runtime_action must be {_RUNTIME_ACTION!r} for the canary profile"
        )
    return value


def _identity(payload: Mapping[str, Any], *, phase: str) -> tuple[str, str, int]:
    operation_id = _required_string(payload, "operation_id", phase=phase)
    command_id = _required_string(payload, "command_id", phase=phase)
    attempt_id = _required_string(payload, "attempt_id", phase=phase)
    attempt_ordinal = _nonnegative_int(payload, "attempt_ordinal", phase=phase)
    if _OPERATION_RE.fullmatch(operation_id) is None or _ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise CanaryArtifactError(f"{phase}: malformed operation/attempt identity")
    if attempt_id != _expected_attempt_id(operation_id, command_id, attempt_ordinal):
        raise CanaryArtifactError(f"{phase}: attempt identity is not canonical")
    return operation_id, attempt_id, attempt_ordinal


def _query_result(payload: Mapping[str, Any], *, phase: str) -> str:
    values = [payload.get("query_result"), payload.get("placement_query_result")]
    observed = [value for value in values if value is not None]
    if not observed or any(value != "Success" for value in observed):
        raise CanaryArtifactError(f"{phase}: fresh placement query is not Success")
    return "Success"


def _dispatch_authority(payload: Mapping[str, Any], *, phase: str) -> Mapping[str, Any]:
    """Require the fresh SC2 authority snapshot carried by a reset dispatch."""

    diagnostic = payload.get("raw_diagnostic")
    merged: dict[str, Any] = dict(diagnostic) if isinstance(diagnostic, Mapping) else {}
    merged.update({key: value for key, value in payload.items() if key not in {"raw_diagnostic"}})
    _query_result(merged, phase=phase)
    if merged.get("available_ability_query") != "available":
        raise CanaryArtifactError(f"{phase}: available ability query is not available")
    fingerprint = merged.get("target_legality_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise CanaryArtifactError(f"{phase}: target legality fingerprint is missing")
    return merged


def _read_phase_journal(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            raw_events = [json.loads(line) for line in stream if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryArtifactError(f"could not read canary journal {path}: {error}") from error
    if not raw_events:
        raise CanaryArtifactError("canary journal is empty")
    if not all(isinstance(event, dict) for event in raw_events):
        raise CanaryArtifactError("canary journal contains a non-object event")
    events = [event for event in raw_events if isinstance(event, dict)]
    indexes = [event.get("event_index") for event in events]
    if not all(type(index) is int for index in indexes):
        raise CanaryArtifactError("canary event_index values must be integers")
    first_index = indexes[0]
    if not isinstance(first_index, int):
        raise CanaryArtifactError("canary event_index values must be integers")
    if indexes != list(range(first_index, first_index + len(indexes))):
        raise CanaryArtifactError("canary event_index sequence is not contiguous")
    if first_index not in {0, 1}:
        raise CanaryArtifactError("canary event_index must start at zero or one")
    for event in events:
        if event.get("event_type") != _PHASE_EVENT_TYPE:
            raise CanaryArtifactError("canary journal contains an unexpected event_type")
        if event.get("schema_version") != _CANARY_SCHEMA_VERSION:
            raise CanaryArtifactError("canary journal schema_version is not 1.0")
        if event.get("diagnostic_only") is not True:
            raise CanaryArtifactError("canary journal is not diagnostic_only=true")
        if event.get("mode") != _CANARY_MODE:
            raise CanaryArtifactError("canary journal mode is not the dedicated canary profile")
        phase = event.get("phase")
        if phase not in _PHASES:
            raise CanaryArtifactError(f"unknown canary phase: {phase!r}")
        if type(event.get("seed")) is not int or event["seed"] < 0:
            raise CanaryArtifactError(f"{phase}: invalid seed")
        if not isinstance(event.get("run_id"), str) or not event["run_id"]:
            raise CanaryArtifactError(f"{phase}: invalid run_id")
        if not isinstance(event.get("episode_id"), str) or not event["episode_id"]:
            raise CanaryArtifactError(f"{phase}: invalid episode_id")
        if type(event.get("game_loop")) is not int or event["game_loop"] < 0:
            raise CanaryArtifactError(f"{phase}: invalid game_loop")
        if "reason" not in event or not isinstance(event.get("reason"), str) or not event["reason"]:
            raise CanaryArtifactError(f"{phase}: reason is missing")
        if event.get("observation_revision") is not None and not isinstance(
            event.get("observation_revision"), str
        ):
            raise CanaryArtifactError(f"{phase}: invalid observation_revision")
    run_ids = {event["run_id"] for event in events}
    episode_ids = {event["episode_id"] for event in events}
    seeds = {event["seed"] for event in events}
    if len(run_ids) != 1 or len(episode_ids) != 1 or len(seeds) != 1:
        raise CanaryArtifactError("canary journal mixes run, episode, or seed identities")
    singleton_phases = set(_PHASES) - {
        "failure_command_held",
        "authoritative_failure_observed",
    }
    if any(sum(event["phase"] == phase for event in events) != 1 for phase in singleton_phases):
        raise CanaryArtifactError("canary journal repeats or omits a singleton phase")
    for previous, current in zip(events, events[1:], strict=False):
        if current["game_loop"] < previous["game_loop"]:
            raise CanaryArtifactError("canary game_loop values are not monotonic")
    return events


def _phase_pattern(events: Sequence[Mapping[str, Any]]) -> None:
    expected = ["initialized"]
    for _ in range(3):
        expected.extend(("failure_command_held", "authoritative_failure_observed"))
    expected.extend(_PHASES[3:])
    observed = [str(event["phase"]) for event in events]
    if observed != expected:
        raise CanaryArtifactError(
            "canary phase sequence mismatch: expected "
            + ",".join(expected)
            + "; observed "
            + ",".join(observed)
        )


def _replay_phases(events: Sequence[Mapping[str, Any]], *, expected_seed: int) -> dict[str, Any]:
    _phase_pattern(events)
    first = events[0]
    run_id = str(first["run_id"])
    episode_id = str(first["episode_id"])
    if first["seed"] != expected_seed:
        raise CanaryArtifactError("canary seed does not match caller-provided seed")
    if first.get("operation_id") is not None or first.get("attempt_id") is not None:
        raise CanaryArtifactError("initialized phase already claims an operation")
    action: str | None = None
    operation_id: str | None = None
    held_events = [event for event in events if event["phase"] == "failure_command_held"]
    failure_events = [
        event for event in events if event["phase"] == "authoritative_failure_observed"
    ]
    if len(held_events) != 3 or len(failure_events) != 3:
        raise CanaryArtifactError("canary must have exactly three held/failure pairs")
    failure_attempts: list[str] = []
    failure_ordinals: list[int] = []
    initial_builder: int | None = None
    for index, (held, failure) in enumerate(zip(held_events, failure_events, strict=True), start=1):
        held_operation, held_attempt, held_ordinal = _identity(held, phase="failure_command_held")
        failure_operation, failure_attempt, failure_ordinal = _identity(
            failure, phase="authoritative_failure_observed"
        )
        if (held_operation, held_attempt, held_ordinal) != (
            failure_operation,
            failure_attempt,
            failure_ordinal,
        ):
            raise CanaryArtifactError("held command and authoritative failure identities differ")
        if operation_id is None:
            operation_id = held_operation
        if held_operation != operation_id:
            raise CanaryArtifactError("multiple canary operations observed")
        held_action = _semantic_action(held, phase="failure_command_held", expected=action)
        action = held_action if action is None else action
        _semantic_action(failure, phase="authoritative_failure_observed", expected=action)
        _runtime_action(held, phase="failure_command_held")
        _runtime_action(failure, phase="authoritative_failure_observed")
        if failure_ordinal != index - 1 or failure_attempt in failure_attempts:
            raise CanaryArtifactError("failure attempt ordinals/identities are not distinct 0/1/2")
        failure_attempts.append(failure_attempt)
        failure_ordinals.append(failure_ordinal)
        if failure.get("reason") != "placement_candidate_stale":
            raise CanaryArtifactError(
                "authoritative failure reason is not placement_candidate_stale"
            )
        if failure.get("authoritative_streak") != index:
            raise CanaryArtifactError("authoritative streak is not 1,2,3")
        if failure.get("authoritative_threshold") != 3:
            raise CanaryArtifactError("authoritative threshold is not three")
        if failure.get("circuit_open") is not (index == 3):
            raise CanaryArtifactError("authoritative circuit state is inconsistent")
        expected_transition = "closed_to_open" if index == 3 else None
        if failure.get("transition") != expected_transition:
            raise CanaryArtifactError("authoritative transition is inconsistent")
        if failure.get("duplicate_attempt") is not False:
            raise CanaryArtifactError("authoritative failure is marked duplicate")
        held_builder = _positive_int(
            held.get("builder_tag"), label="failure_command_held.builder_tag"
        )
        failure_builder = _positive_int(
            failure.get("builder_tag"), label="authoritative_failure_observed.builder_tag"
        )
        if held_builder != failure_builder:
            raise CanaryArtifactError("failure Builder changed between hold and observation")
        if initial_builder is None:
            initial_builder = failure_builder
        elif failure_builder != initial_builder:
            raise CanaryArtifactError("failure Builder changed before circuit open")
        _zero_ownership(held, phase="failure_command_held")
        _zero_ownership(failure, phase="authoritative_failure_observed")

    open_event = next(event for event in events if event["phase"] == "circuit_open_observed")
    open_operation, open_attempt, open_ordinal = _identity(
        open_event, phase="circuit_open_observed"
    )
    if (open_operation, open_attempt, open_ordinal) != (
        operation_id,
        failure_attempts[-1],
        2,
    ):
        raise CanaryArtifactError("open boundary is not bound to the unique third failure")
    _semantic_action(open_event, phase="circuit_open_observed", expected=action)
    if (
        open_event.get("authoritative_streak") != 3
        or open_event.get("authoritative_threshold") != 3
        or open_event.get("circuit_open") is not True
        or open_event.get("transition") != "closed_to_open"
    ):
        raise CanaryArtifactError("circuit_open_observed does not prove the third failure boundary")
    _zero_ownership(open_event, phase="circuit_open_observed")

    core = next(event for event in events if event["phase"] == "core_defer_observed")
    if _operation(core, phase="core_defer_observed", expected=operation_id) != operation_id:
        raise CanaryArtifactError("Core defer operation mismatch")
    _semantic_action(core, phase="core_defer_observed", expected=action)
    if core.get("reason") != "authoritative_build_pre_dispatch_circuit_defer":
        raise CanaryArtifactError("Core post-open defer reason is not authoritative")
    if any(core.get(key) is not None for key in ("command_id", "attempt_id", "attempt_ordinal")):
        raise CanaryArtifactError("Core defer carries a post-open command identity")
    if core.get("command_count") != 0:
        raise CanaryArtifactError("Core defer did not observe command_count=0")
    if core.get("idle_reason") != "plan_commands_deferred":
        raise CanaryArtifactError("Core defer idle_reason is not plan_commands_deferred")
    if core.get("planner_pending") is not False:
        raise CanaryArtifactError("Core defer was recorded while planner remained pending")
    if (
        _positive_int(core.get("builder_tag"), label="core_defer_observed.builder_tag")
        != initial_builder
    ):
        raise CanaryArtifactError("Core defer Builder is not the pre-open Builder")
    core_state = core.get("authoritative_state")
    if not isinstance(core_state, Mapping):
        raise CanaryArtifactError("Core defer lacks raw authoritative circuit state")
    if core_state.get("streak") != 3 or core_state.get("circuit_open") is not True:
        raise CanaryArtifactError("Core defer raw circuit is not open at streak three")
    _zero_ownership(core, phase="core_defer_observed")

    rebound = next(event for event in events if event["phase"] == "builder_rebound")
    _operation(rebound, phase="builder_rebound", expected=operation_id)
    _semantic_action(rebound, phase="builder_rebound", expected=action)
    if rebound.get("reason") != "fresh_ready_unleased_builder_observed":
        raise CanaryArtifactError("builder rebound is not a fresh ready Builder observation")
    rebound_initial = _positive_int(rebound.get("builder_tag"), label="builder_rebound.builder_tag")
    replacement_builder = _positive_int(
        rebound.get("replacement_builder_tag"), label="builder_rebound.replacement_builder_tag"
    )
    if initial_builder != rebound_initial or replacement_builder == initial_builder:
        raise CanaryArtifactError("builder material reset did not bind one distinct Builder")
    if rebound.get("observation_revision") in {
        open_event.get("observation_revision"),
        *[event.get("observation_revision") for event in failure_events],
    }:
        raise CanaryArtifactError("builder rebound did not observe a fresh material revision")

    reset_command = next(event for event in events if event["phase"] == "reset_command_observed")
    reset_operation, reset_attempt, reset_ordinal = _identity(
        reset_command, phase="reset_command_observed"
    )
    if reset_operation != operation_id or reset_ordinal != 3 or reset_attempt in failure_attempts:
        raise CanaryArtifactError(
            "reset command is not a distinct fourth attempt for the operation"
        )
    _semantic_action(reset_command, phase="reset_command_observed", expected=action)
    _runtime_action(reset_command, phase="reset_command_observed")
    if reset_command.get("reason") != "validated_builder_material_change":
        raise CanaryArtifactError("reset command does not prove material builder change")
    reset_builder = _positive_int(
        reset_command.get("builder_tag"), label="reset_command.builder_tag"
    )
    if reset_builder != replacement_builder:
        raise CanaryArtifactError("reset command is not bound to replacement Builder")
    if reset_command.get("replacement_builder_tag") != replacement_builder:
        raise CanaryArtifactError("reset command replacement Builder identity changed")
    _zero_ownership(reset_command, phase="reset_command_observed")

    dispatch = next(event for event in events if event["phase"] == "reset_dispatch_observed")
    submitted = next(event for event in events if event["phase"] == "reset_dispatch_submitted")
    for phase, event in (
        ("reset_dispatch_observed", dispatch),
        ("reset_dispatch_submitted", submitted),
    ):
        identity = _identity(event, phase=phase)
        if identity != (reset_operation, reset_attempt, reset_ordinal):
            raise CanaryArtifactError(f"{phase}: reset command identity changed")
        _semantic_action(event, phase=phase, expected=action)
        _runtime_action(event, phase=phase)
        if (
            _positive_int(event.get("builder_tag"), label=f"{phase}.builder_tag")
            != replacement_builder
        ):
            raise CanaryArtifactError(f"{phase}: replacement Builder identity changed")
        if event.get("replacement_builder_tag") != replacement_builder:
            raise CanaryArtifactError(f"{phase}: replacement Builder field changed")
        authority = _dispatch_authority(event, phase=phase)
        if event.get("primitive_constructed") is not True:
            raise CanaryArtifactError(f"{phase}: primitive_constructed is not true")
        if authority.get("primitive_constructed") is not True:
            raise CanaryArtifactError(f"{phase}: authority snapshot lacks primitive construction")
    if dispatch.get("primitive_submitted") is not False:
        raise CanaryArtifactError("reset_dispatch_observed prematurely claims submission")
    if submitted.get("primitive_submitted") is not True:
        raise CanaryArtifactError("reset_dispatch_submitted does not prove submission")
    submitted_authority = _dispatch_authority(submitted, phase="reset_dispatch_submitted")
    if submitted_authority.get("primitive_submitted") is not True:
        raise CanaryArtifactError("reset_dispatch_submitted authority snapshot lacks submission")
    submitted_loop = submitted.get("primitive_submitted_game_loop")
    if type(submitted_loop) is not int or submitted_loop < 0:
        raise CanaryArtifactError("reset_dispatch_submitted lacks primitive_submitted_game_loop")
    if submitted_loop < dispatch["game_loop"]:
        raise CanaryArtifactError("primitive submission precedes placement dispatch")
    reset_revision = dispatch.get("observation_revision")
    if not isinstance(reset_revision, str) or not reset_revision:
        raise CanaryArtifactError("reset dispatch lacks an observation revision")
    if reset_revision in {
        open_event.get("observation_revision"),
        rebound.get("observation_revision"),
    }:
        raise CanaryArtifactError("reset placement query reused a stale observation revision")
    if reset_revision != reset_command.get("observation_revision"):
        raise CanaryArtifactError("reset placement query did not use the reset command observation")
    if submitted.get("observation_revision") != reset_revision:
        raise CanaryArtifactError("primitive submission changed the reset observation revision")

    effect = next(event for event in events if event["phase"] == "effect_confirmed")
    _operation(effect, phase="effect_confirmed", expected=operation_id)
    _semantic_action(effect, phase="effect_confirmed", expected=action)
    if effect.get("command_id") != reset_command.get("command_id"):
        raise CanaryArtifactError("effect confirmation command identity changed")
    if effect.get("primitive_submitted") is not True or effect.get("effect_status") != "succeeded":
        raise CanaryArtifactError("effect confirmation does not prove submitted success")
    evidence = effect.get("effect_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("effect_kind") != "build":
        raise CanaryArtifactError("effect confirmation lacks build effect evidence")
    if evidence.get("build_started") is not True:
        raise CanaryArtifactError("effect confirmation lacks real build-start evidence")
    if not evidence.get("observed_structure_tag"):
        raise CanaryArtifactError("effect confirmation lacks observed_structure_tag")
    if evidence.get("confirmation_kind") != "new_structure":
        raise CanaryArtifactError("effect confirmation is not confirmation_kind=new_structure")
    _zero_ownership(effect, phase="effect_confirmed")

    complete = next(event for event in events if event["phase"] == "complete")
    _operation(complete, phase="complete", expected=operation_id)
    _semantic_action(complete, phase="complete", expected=action)
    if complete.get("command_id") != reset_command.get("command_id"):
        raise CanaryArtifactError("complete event command identity changed")
    _zero_ownership(complete, phase="complete")

    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "seed": expected_seed,
        "operation_id": operation_id,
        "semantic_action": action,
        "failure_attempt_ids": failure_attempts,
        "failure_attempt_ordinals": failure_ordinals,
        "reset_attempt_id": reset_attempt,
        "initial_builder_tag": initial_builder,
        "replacement_builder_tag": replacement_builder,
        "failure_count": len(failure_events),
        "open_count": 1,
        "material_reset_count": 1,
        "success_reset_count": 1,
        "reset_count": 2,
        "core_post_open_defer_count": 1,
        "post_open_command_count": 0,
        "post_open_dispatch_count": 0,
        "post_open_rejection_count": 0,
        "post_open_raw_rejection_count": 0,
        "fresh_query_success_count": 1,
        "primitive_submission_count": 1,
        "build_start_count": 1,
        "effect_confirmation_count": 1,
        "phase_count": len(events),
    }


def _runtime_event_id(event: Any) -> int:
    value = getattr(event, "event_id", None)
    if type(value) is not int or value < 0:
        raise CanaryArtifactError("events.jsonl contains an invalid event_id")
    return value


def _runtime_event_type(event: Any) -> str:
    value = getattr(event, "event_type", None)
    if not isinstance(value, str) or not value:
        raise CanaryArtifactError("events.jsonl contains an invalid event_type")
    return value


def _runtime_payload(event: Any) -> Mapping[str, Any]:
    payload = getattr(event, "payload", None)
    if not isinstance(payload, Mapping):
        raise CanaryArtifactError("events.jsonl contains a non-object payload")
    return payload


def _runtime_authoritative(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = payload.get("authoritative_pre_dispatch")
    if isinstance(direct, Mapping):
        return direct
    transition = payload.get("transition")
    if isinstance(transition, Mapping):
        nested = transition.get("authoritative_pre_dispatch")
        if isinstance(nested, Mapping):
            return nested
    return None


def _runtime_identity(
    payload: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    phase: str,
) -> tuple[str, str, int, str, str]:
    operation_id = payload.get("operation_id")
    command_id = payload.get("command_id")
    action_name = payload.get("action_name")
    attempt_id = payload.get("attempt_id")
    attempt_ordinal = payload.get("attempt_ordinal")
    if operation_id != evidence.get("operation_id"):
        raise CanaryArtifactError(f"{phase}: runtime/nested operation identity mismatch")
    if command_id != evidence.get("command_id"):
        raise CanaryArtifactError(f"{phase}: runtime/nested command identity mismatch")
    if action_name != evidence.get("action_name"):
        raise CanaryArtifactError(f"{phase}: runtime/nested action identity mismatch")
    if attempt_id != evidence.get("attempt_id") or attempt_ordinal != evidence.get(
        "attempt_ordinal"
    ):
        raise CanaryArtifactError(f"{phase}: runtime/nested attempt identity mismatch")
    if (
        not isinstance(operation_id, str)
        or _OPERATION_RE.fullmatch(operation_id) is None
        or not isinstance(command_id, str)
        or not command_id
        or not isinstance(action_name, str)
        or action_name != _RUNTIME_ACTION
        or not isinstance(attempt_id, str)
        or not isinstance(attempt_ordinal, int)
        or isinstance(attempt_ordinal, bool)
        or attempt_ordinal < 0
        or attempt_id != _expected_attempt_id(operation_id, command_id, attempt_ordinal)
    ):
        raise CanaryArtifactError(f"{phase}: malformed runtime operation/attempt identity")
    return operation_id, command_id, attempt_ordinal, attempt_id, action_name


def _runtime_action_fields(payload: Mapping[str, Any], *, phase: str) -> None:
    semantic_action = payload.get("semantic_action")
    if semantic_action is not None and semantic_action != _SEMANTIC_ACTION:
        raise CanaryArtifactError(f"{phase}: runtime semantic_action is not {_SEMANTIC_ACTION}")
    runtime_action = payload.get("runtime_action")
    if runtime_action is not None and runtime_action != _RUNTIME_ACTION:
        raise CanaryArtifactError(f"{phase}: runtime_action is not Build_Pylon_Screen")


def _runtime_command_operation(command: Mapping[str, Any]) -> str | None:
    operation_id = command.get("operation_id")
    return operation_id if isinstance(operation_id, str) else None


def _replay_runtime_events(
    runtime_events: Sequence[Any],
    *,
    phase_events: Sequence[Mapping[str, Any]],
    phase_replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the canary's authoritative path from the durable Runtime journal."""

    if not runtime_events:
        raise CanaryArtifactError("events.jsonl is empty")
    ordered = sorted(runtime_events, key=_runtime_event_id)
    run_id = str(phase_replay["run_id"])
    episode_id = str(phase_replay["episode_id"])
    operation_id = str(phase_replay["operation_id"])
    failure_attempts = list(phase_replay["failure_attempt_ids"])
    initial_builder = phase_replay["initial_builder_tag"]
    replacement_builder = phase_replay["replacement_builder_tag"]
    for event in ordered:
        payload = _runtime_payload(event)
        if (
            getattr(event, "run_id", None) != run_id
            or getattr(event, "episode_id", None) != episode_id
        ):
            raise CanaryArtifactError("events.jsonl mixes run or episode identities")
    runtime_operations: set[str] = set()
    for event in ordered:
        payload = _runtime_payload(event)
        nested = payload.get("authoritative_pre_dispatch")
        values: list[Any] = []
        if payload.get("action_name") == _RUNTIME_ACTION:
            values.append(payload.get("operation_id"))
        if isinstance(nested, Mapping):
            if nested.get("action_name") == _RUNTIME_ACTION:
                values.append(nested.get("operation_id"))
        command = payload.get("command")
        if isinstance(command, Mapping):
            if command.get("name", command.get("action_name")) == _RUNTIME_ACTION:
                values.append(command.get("operation_id"))
        runtime_operations.update(
            value for value in values if isinstance(value, str) and _OPERATION_RE.fullmatch(value)
        )
    if runtime_operations != {operation_id}:
        raise CanaryArtifactError("runtime evidence does not contain exactly one operation")

    failures: list[tuple[Any, Mapping[str, Any], Mapping[str, Any]]] = []
    for event in ordered:
        if _runtime_event_type(event) != "execution":
            continue
        payload = _runtime_payload(event)
        if (
            payload.get("operation_id") != operation_id
            or payload.get("action_name") != _RUNTIME_ACTION
        ):
            continue
        _runtime_action_fields(payload, phase="execution")
        if payload.get("failure_code") != "placement_candidate_stale":
            continue
        if payload.get("execution_stage") != "pre_dispatch" or payload.get("status") != "failed":
            raise CanaryArtifactError(
                "stale candidate execution is not a failed pre-dispatch report"
            )
        if payload.get("success") is not False:
            raise CanaryArtifactError("stale candidate execution claims success")
        evidence = _runtime_authoritative(payload)
        if evidence is None:
            raise CanaryArtifactError("stale candidate execution lacks authoritative evidence")
        _runtime_identity(payload, evidence, phase="placement_candidate_stale execution")
        if evidence.get("failure_code") != "placement_candidate_stale":
            raise CanaryArtifactError("authoritative stale candidate evidence has wrong code")
        if evidence.get("threshold") != 3 or evidence.get("duplicate_attempt") is not False:
            raise CanaryArtifactError(
                "authoritative stale candidate threshold/duplicate is invalid"
            )
        material_identity = evidence.get("material_legality_identity")
        world_target = evidence.get("world_target")
        if (
            evidence.get("material_evidence_valid") is not True
            or evidence.get("invalid_evidence_reasons") not in (None, (), [])
            or not isinstance(material_identity, str)
            or _BUILD_LEGALITY_RE.fullmatch(material_identity) is None
            or type(evidence.get("ability_id")) is not int
            or evidence["ability_id"] <= 0
            or not isinstance(world_target, (list, tuple))
            or len(world_target) != 2
            or not all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in world_target
            )
        ):
            raise CanaryArtifactError("authoritative stale candidate material evidence is invalid")
        failures.append((event, payload, evidence))
    if len(failures) != 3:
        raise CanaryArtifactError(
            "events.jsonl must contain exactly three stale candidate executions; "
            f"found {len(failures)}"
        )
    failures.sort(key=lambda item: item[2].get("attempt_ordinal", -1))
    if [item[2].get("attempt_ordinal") for item in failures] != [0, 1, 2]:
        raise CanaryArtifactError(
            "runtime stale candidate attempts are not canonical ordinals 0/1/2"
        )
    if [item[2].get("streak") for item in failures] != [1, 2, 3]:
        raise CanaryArtifactError("runtime authoritative streak is not 1,2,3")
    runtime_attempts = [str(item[2]["attempt_id"]) for item in failures]
    if runtime_attempts != failure_attempts or len(set(runtime_attempts)) != 3:
        raise CanaryArtifactError(
            "runtime stale candidate identities do not match journal stimulus"
        )
    if any(item[2].get("builder_tag") != initial_builder for item in failures):
        raise CanaryArtifactError("runtime failure Builder changed before circuit open")
    transitions = [item[2].get("state_transition") for item in failures]
    if transitions != [None, None, "closed_to_open"]:
        raise CanaryArtifactError(
            "runtime stale candidate transition is not uniquely closed_to_open"
        )
    if sum(value == "closed_to_open" for value in transitions) != 1:
        raise CanaryArtifactError("runtime has more than one closed_to_open transition")
    open_event, open_payload, open_evidence = failures[-1]
    open_event_id = _runtime_event_id(open_event)
    opening_command_id = _required_string(
        open_payload,
        "command_id",
        phase="closed_to_open execution",
    )
    opening_attempt_id = _required_string(
        open_evidence,
        "attempt_id",
        phase="closed_to_open execution",
    )

    defer_events = [
        event
        for event in ordered
        if _runtime_event_type(event) == "authoritative_build_pre_dispatch_circuit_defer"
        and _runtime_payload(event).get("operation_id") == operation_id
    ]
    if len(defer_events) != 1:
        raise CanaryArtifactError("runtime must contain exactly one Core circuit defer")
    defer_event = defer_events[0]
    defer = _runtime_payload(defer_event)
    if _runtime_event_id(defer_event) <= open_event_id:
        raise CanaryArtifactError("Core circuit defer precedes the open boundary")
    if (
        defer.get("action_name") != _RUNTIME_ACTION
        or defer.get("streak") != 3
        or defer.get("threshold") != 3
        or defer.get("failure_count") != 3
        or defer.get("opened_command_id") != failures[-1][1].get("command_id")
        or defer.get("opened_attempt_id") != failures[-1][2].get("attempt_id")
        or defer.get("opened_attempt_ordinal") != 2
    ):
        raise CanaryArtifactError("Core circuit defer does not match the third failure")
    if defer.get("reason") not in {None, "authoritative_build_pre_dispatch_circuit_defer"}:
        raise CanaryArtifactError("Core circuit defer reason is invalid")
    if defer.get("command_count") is not None and defer.get("command_count") != 0:
        raise CanaryArtifactError("runtime Core defer claims a command")
    if (
        defer.get("idle_reason") is not None
        and defer.get("idle_reason") != "plan_commands_deferred"
    ):
        raise CanaryArtifactError("runtime Core defer idle_reason is invalid")
    if defer.get("builder_tag") is not None and defer.get("builder_tag") != initial_builder:
        raise CanaryArtifactError("runtime Core defer Builder changed before rebind")

    reset_events = [
        event
        for event in ordered
        if _runtime_event_type(event) == "authoritative_build_pre_dispatch_circuit_reset"
        and _runtime_payload(event).get("operation_id") == operation_id
    ]
    if len(reset_events) != 1:
        raise CanaryArtifactError("runtime must contain exactly one Core circuit reset")
    reset_event = reset_events[0]
    reset = _runtime_payload(reset_event)
    if _runtime_event_id(reset_event) <= _runtime_event_id(defer_event):
        raise CanaryArtifactError("Core circuit reset precedes the Core defer")
    if (
        reset.get("action_name") != _RUNTIME_ACTION
        or reset.get("reason") != "semantic_legality_material_change"
        or reset.get("reset_from_streak") != 3
        or reset.get("previous_builder_tag") != initial_builder
        or reset.get("bound_builder_tag") != replacement_builder
        or reset.get("operation_epoch_changed") is not False
    ):
        raise CanaryArtifactError("Core circuit reset does not prove the Builder material change")
    for key in (
        "previous_semantic_material_identity",
        "current_semantic_material_identity",
    ):
        if (
            not isinstance(reset.get(key), str)
            or _SEMANTIC_MATERIAL_RE.fullmatch(reset[key]) is None
        ):
            raise CanaryArtifactError(f"Core circuit reset lacks {key}")
    raw_material_identity = reset.get("raw_material_legality_identity")
    if (
        not isinstance(raw_material_identity, str)
        or _BUILD_LEGALITY_RE.fullmatch(raw_material_identity) is None
    ):
        raise CanaryArtifactError("Core circuit reset lacks raw_material_legality_identity")
    if reset["previous_semantic_material_identity"] == reset["current_semantic_material_identity"]:
        raise CanaryArtifactError("Core circuit reset did not change semantic material identity")

    reset_event_id = _runtime_event_id(reset_event)
    post_open_command_ids: set[str] = set()
    post_open_dispatch_ids: set[str] = set()
    post_open_raw_rejections: set[str] = set()
    for event in ordered:
        event_id = _runtime_event_id(event)
        if not open_event_id < event_id < reset_event_id:
            continue
        event_type = _runtime_event_type(event)
        payload = _runtime_payload(event)
        if event_type == "execution" and payload.get("failure_code") in _RAW_CIRCUIT_OPEN_CODES:
            if payload.get("operation_id") == operation_id:
                post_open_raw_rejections.add(str(payload.get("command_id") or f"event:{event_id}"))
        if event_type == "command_lifecycle":
            command = payload.get("command")
            if isinstance(command, Mapping) and _runtime_command_operation(command) == operation_id:
                command_id = str(command.get("command_id") or f"event:{event_id}")
                opening_terminal = (
                    payload.get("status") == "failed"
                    and payload.get("reason") == "placement_candidate_stale"
                    and command_id == opening_command_id
                    and command.get("attempt_id") == opening_attempt_id
                    and command.get("attempt_ordinal") == 2
                    and command.get("name") == _RUNTIME_ACTION
                )
                if opening_terminal:
                    continue
                post_open_command_ids.add(command_id)
                if payload.get("status") == "dispatched":
                    post_open_dispatch_ids.add(command_id)
        elif event_type == "command_lineage" and payload.get("operation_id") == operation_id:
            post_open_command_ids.add(str(payload.get("command_id") or f"event:{event_id}"))
        elif event_type == "decision":
            batch = payload.get("batch")
            commands = (
                batch.get("commands", ())
                if isinstance(batch, Mapping)
                else payload.get("commands", ())
            )
            if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes)):
                for command in commands:
                    if (
                        isinstance(command, Mapping)
                        and _runtime_command_operation(command) == operation_id
                    ):
                        command_id = str(command.get("command_id") or f"event:{event_id}")
                        post_open_command_ids.add(command_id)
                        if command.get("status") == "dispatched":
                            post_open_dispatch_ids.add(command_id)
    if post_open_command_ids or post_open_dispatch_ids or post_open_raw_rejections:
        raise CanaryArtifactError(
            "post-open command/dispatch/raw circuit-open rejection was observed"
        )

    successes: list[tuple[Any, Mapping[str, Any]]] = []
    for event in ordered:
        if _runtime_event_type(event) != "execution":
            continue
        payload = _runtime_payload(event)
        if (
            payload.get("operation_id") != operation_id
            or payload.get("action_name") != _RUNTIME_ACTION
        ):
            continue
        if payload.get("attempt_ordinal") != 3:
            continue
        _runtime_action_fields(payload, phase="reset execution")
        if payload.get("status") != "succeeded" or payload.get("success") is not True:
            raise CanaryArtifactError("ordinal 3 execution is not succeeded")
        attempt_id = payload.get("attempt_id")
        success_command_id = payload.get("command_id")
        if (
            not isinstance(attempt_id, str)
            or not isinstance(success_command_id, str)
            or attempt_id != _expected_attempt_id(operation_id, success_command_id, 3)
        ):
            raise CanaryArtifactError("ordinal 3 execution attempt identity is not canonical")
        effect = payload.get("effect_evidence")
        if not isinstance(effect, Mapping) or effect.get("effect_kind") != "build":
            raise CanaryArtifactError("ordinal 3 execution lacks build effect evidence")
        observed_structure_tag = effect.get("observed_structure_tag") or effect.get(
            "new_structure_tag"
        )
        if (
            effect.get("build_started") is not True
            or not observed_structure_tag
            or effect.get("confirmation_kind") != "new_structure"
        ):
            raise CanaryArtifactError(
                "ordinal 3 execution lacks new_structure build-start evidence"
            )
        successes.append((event, payload))
    if len(successes) != 1:
        raise CanaryArtifactError("runtime must contain exactly one successful ordinal 3 execution")
    success_event, success = successes[0]
    if _runtime_event_id(success_event) <= reset_event_id:
        raise CanaryArtifactError("ordinal 3 execution precedes Core circuit reset")

    raw_reset_events: list[Any] = []
    released_transition_observed = False
    for event in ordered:
        if _runtime_event_type(event) != "placement_ledger_transition":
            continue
        payload = _runtime_payload(event)
        transition = payload.get("transition")
        transition_payload = dict(transition) if isinstance(transition, Mapping) else payload
        evidence = _runtime_authoritative(payload)
        if evidence is None:
            evidence = _runtime_authoritative(transition_payload)
        parent_operation = payload.get("operation_id", transition_payload.get("operation_id"))
        parent_command = payload.get("command_id", transition_payload.get("command_id"))
        parent_action = payload.get("action_name", transition_payload.get("action_name"))
        parent_attempt = payload.get("attempt_id", transition_payload.get("attempt_id"))
        parent_ordinal = payload.get("attempt_ordinal", transition_payload.get("attempt_ordinal"))
        if (
            parent_command == success.get("command_id")
            and parent_action == _RUNTIME_ACTION
            and transition_payload.get("next_state") == "occupied"
            and payload.get("builder_lease_state") == "released"
            and transition_payload.get("release_reason") == "effect_confirmed"
        ):
            if parent_operation not in {None, operation_id}:
                raise CanaryArtifactError("released transition operation identity mismatch")
            if parent_attempt not in {None, success.get("attempt_id")} or parent_ordinal not in {
                None,
                3,
            }:
                raise CanaryArtifactError("released transition attempt identity mismatch")
            released_transition_observed = True
        if evidence is None or evidence.get("status") != "reset":
            continue
        if parent_operation != operation_id and evidence.get("operation_id") != operation_id:
            continue
        if parent_command != success.get("command_id") and evidence.get(
            "command_id"
        ) != success.get("command_id"):
            continue
        if (
            parent_operation != evidence.get("operation_id")
            or parent_command != evidence.get("command_id")
            or parent_action != evidence.get("action_name")
            or parent_attempt != evidence.get("attempt_id")
            or parent_ordinal != evidence.get("attempt_ordinal")
        ):
            raise CanaryArtifactError("raw reset parent/nested identity mismatch")
        if parent_action != _RUNTIME_ACTION:
            raise CanaryArtifactError("raw reset action is not Build_Pylon_Screen")
        if (
            evidence.get("attempt_ordinal") != 3
            or evidence.get("state_transition") != "open_to_reset"
        ):
            raise CanaryArtifactError("raw reset evidence is not the ordinal 3 open_to_reset")
        if evidence.get("attempt_id") != _expected_attempt_id(
            operation_id, str(success["command_id"]), 3
        ):
            raise CanaryArtifactError("raw reset evidence has a non-canonical attempt identity")
        if evidence.get("circuit_open") is True:
            raise CanaryArtifactError("raw reset evidence still claims circuit open")
        raw_reset_events.append(event)
    if not raw_reset_events:
        raise CanaryArtifactError("runtime lacks raw authoritative circuit reset evidence")
    if not released_transition_observed:
        raise CanaryArtifactError("runtime lacks final released ownership transition")

    phase_core = next(event for event in phase_events if event["phase"] == "core_defer_observed")
    phase_reset = next(
        event for event in phase_events if event["phase"] == "reset_dispatch_submitted"
    )
    phase_raw = phase_reset.get("raw_diagnostic")
    if isinstance(phase_raw, Mapping):
        state = phase_raw.get("authoritative_state") or phase_raw.get(
            "authoritative_pre_dispatch_state"
        )
        if isinstance(state, Mapping) and state.get("circuit_open") is not False:
            raise CanaryArtifactError("reset dispatch raw state remains circuit open")
    return {
        "operation_count": 1,
        "runtime_failure_count": len(failures),
        "runtime_open_count": 1,
        "runtime_core_defer_count": len(defer_events),
        "runtime_core_reset_count": len(reset_events),
        "runtime_raw_reset_count": len(raw_reset_events),
        "runtime_released_ownership": released_transition_observed,
        "runtime_success_count": len(successes),
        "post_open_command_count": len(post_open_command_ids),
        "post_open_dispatch_count": len(post_open_dispatch_ids),
        "post_open_raw_rejection_count": len(post_open_raw_rejections),
        "raw_circuit_open_rejection_count": len(post_open_raw_rejections),
        "open_event_id": open_event_id,
        "defer_event_id": _runtime_event_id(defer_event),
        "reset_event_id": reset_event_id,
        "success_event_id": _runtime_event_id(success_event),
        "raw_reset_event_ids": [_runtime_event_id(event) for event in raw_reset_events],
        "observed_structure_tag": str(
            success["effect_evidence"].get("observed_structure_tag")
            or success["effect_evidence"].get("new_structure_tag")
        ),
        "confirmation_kind": success["effect_evidence"]["confirmation_kind"],
        "runtime_action": _RUNTIME_ACTION,
        "semantic_action": _SEMANTIC_ACTION,
        "core_defer_idle_reason": phase_core.get("idle_reason"),
    }


def _summary_artifact_is_canonical(run_dir: Path, *, run_id: str, episode_id: str) -> bool:
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    if not events_path.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = tuple(read_event_log(events_path))
        canonical = json.loads(json.dumps(_build_run_summary(events), sort_keys=True))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if summary != canonical:
        return False
    runs = summary.get("runs") if isinstance(summary, dict) else None
    episodes = runs.get(run_id, {}).get("episodes") if isinstance(runs, dict) else None
    return isinstance(episodes, dict) and episode_id in episodes


def _attestation_file(run_set_dir: Path, run_dir: Path) -> Path:
    for candidate in (
        run_set_dir / "canary-attestation.json",
        run_set_dir / "authoritative-build-circuit-canary-attestation.json",
        run_dir / "canary-attestation.json",
    ):
        if candidate.is_file():
            return candidate
    raise CanaryArtifactError("canary source/hash attestation file is missing")


def _source_attestation_is_exact(source: Mapping[str, Any], expected_git_sha: str) -> bool:
    required = (
        "git_head_before",
        "git_head_after",
        "superproject_dirty_before",
        "superproject_dirty_after",
        "submodule_commit_before",
        "submodule_commit_after",
        "submodule_dirty_before",
        "submodule_dirty_after",
        "submodule_gitlink_before",
        "submodule_gitlink_after",
        "submodule_diff_sha256_before",
        "submodule_diff_sha256_after",
        "reviewed_source_commit_before",
        "reviewed_source_commit_after",
        "reviewed_source_diff_sha256_before",
        "reviewed_source_diff_sha256_after",
        "reviewed_source_tree_sha256_before",
        "reviewed_source_tree_sha256_after",
    )
    if any(not isinstance(source.get(key), str) or not source.get(key) for key in required):
        return False
    if source["git_head_before"] != source["git_head_after"] != expected_git_sha:
        return False
    if source["git_head_before"] != expected_git_sha or any(
        source[key] != "false"
        for key in (
            "superproject_dirty_before",
            "superproject_dirty_after",
            "submodule_dirty_before",
            "submodule_dirty_after",
        )
    ):
        return False
    if not (
        source["submodule_commit_before"]
        == source["submodule_commit_after"]
        == source["submodule_gitlink_before"]
        == source["submodule_gitlink_after"]
        == source["reviewed_source_commit_before"]
        == source["reviewed_source_commit_after"]
    ):
        return False
    if not _SHA40_RE.fullmatch(source["submodule_commit_before"]):
        return False
    hashes_equal = (
        source["submodule_diff_sha256_before"] == source["submodule_diff_sha256_after"]
        and source["reviewed_source_diff_sha256_before"]
        == source["reviewed_source_diff_sha256_after"]
        and source["reviewed_source_tree_sha256_before"]
        == source["reviewed_source_tree_sha256_after"]
    )
    return (
        all(
            _SHA64_RE.fullmatch(source[key]) is not None
            for key in (
                "submodule_diff_sha256_before",
                "submodule_diff_sha256_after",
                "reviewed_source_diff_sha256_before",
                "reviewed_source_diff_sha256_after",
                "reviewed_source_tree_sha256_before",
                "reviewed_source_tree_sha256_after",
            )
        )
        and hashes_equal
    )


def analyze_canary_run(
    run_set_dir: Path,
    *,
    expected_git_sha: str,
    seed: int,
    output: Path | None = None,
) -> dict[str, Any]:
    """Analyze one real-SC2 canary run set and return a derived report."""

    run_set = run_set_dir.expanduser().resolve()
    if _SHA40_RE.fullmatch(expected_git_sha) is None:
        raise CanaryArtifactError("expected_git_sha must be a full 40-character SHA")
    if type(seed) is not int or seed < 0:
        raise CanaryArtifactError("seed must be a non-negative integer")
    canonical_link = run_set / "canonical-run"
    if canonical_link.is_symlink():
        candidate_run_dirs = [canonical_link]
    else:
        candidate_run_dirs = (
            [
                path
                for path in run_set.iterdir()
                if path.is_dir() and (path / _CANARY_JOURNAL_NAME).is_file()
            ]
            if run_set.is_dir()
            else []
        )
        if (run_set / _CANARY_JOURNAL_NAME).is_file():
            candidate_run_dirs.insert(0, run_set)
    if len(candidate_run_dirs) != 1:
        raise CanaryArtifactError("run set must identify exactly one canary run directory")
    run_dir = candidate_run_dirs[0].resolve()
    attestation_path = _attestation_file(run_set, run_dir)
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryArtifactError(f"could not read attestation: {error}") from error
    if not isinstance(attestation, dict):
        raise CanaryArtifactError("canary attestation must be one JSON object")
    if attestation.get("diagnostic_only") is not True:
        raise CanaryArtifactError("canary attestation is not diagnostic_only=true")
    if attestation.get("expected_git_sha") != expected_git_sha or attestation.get("seed") != seed:
        raise CanaryArtifactError("canary attestation does not match expected SHA/seed")
    if Path(str(attestation.get("run_dir", ""))).expanduser().resolve() != run_dir:
        raise CanaryArtifactError("run_dir attestation does not identify the analyzed directory")
    source = attestation.get("source_attestation")
    if not isinstance(source, Mapping) or not _source_attestation_is_exact(
        source, expected_git_sha
    ):
        raise CanaryArtifactError("source/submodule/reviewed-tree attestation is not exact")
    journal = run_dir / _CANARY_JOURNAL_NAME
    expected_journal_path = attestation.get("canary_journal_path")
    expected_events_path = attestation.get("events_path")
    expected_summary_path = attestation.get("summary_path")
    if (
        Path(str(expected_journal_path)).expanduser().resolve() != journal
        or Path(str(expected_events_path)).expanduser().resolve() != run_dir / "events.jsonl"
        or Path(str(expected_summary_path)).expanduser().resolve() != run_dir / "summary.json"
    ):
        raise CanaryArtifactError("run-dir artifact paths do not identify the exact run files")
    expected_journal_sha = attestation.get("canary_journal_sha256")
    expected_events_sha = attestation.get("events_sha256")
    expected_summary_sha = attestation.get("summary_sha256")
    if not (
        isinstance(expected_journal_sha, str)
        and _SHA64_RE.fullmatch(expected_journal_sha)
        and isinstance(expected_events_sha, str)
        and _SHA64_RE.fullmatch(expected_events_sha)
        and isinstance(expected_summary_sha, str)
        and _SHA64_RE.fullmatch(expected_summary_sha)
    ):
        raise CanaryArtifactError("run-dir artifact hashes are missing or malformed")
    try:
        observed_journal_sha = _sha256_file(journal)
        observed_events_sha = _sha256_file(run_dir / "events.jsonl")
        observed_summary_sha = _sha256_file(run_dir / "summary.json")
    except OSError as error:
        raise CanaryArtifactError(f"run-dir artifact is missing: {error}") from error
    if observed_journal_sha != expected_journal_sha:
        raise CanaryArtifactError("canary journal SHA does not match attestation")
    if observed_events_sha != expected_events_sha:
        raise CanaryArtifactError("events.jsonl SHA does not match attestation")
    if observed_summary_sha != expected_summary_sha:
        raise CanaryArtifactError("summary.json SHA does not match attestation")
    if attestation.get("exit_code") != 0:
        raise CanaryArtifactError("real-SC2 canary runner did not exit zero")
    phase_events = _read_phase_journal(journal)
    replay = _replay_phases(phase_events, expected_seed=seed)
    try:
        runtime_events = tuple(read_event_log(run_dir / "events.jsonl"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise CanaryArtifactError(f"could not read events.jsonl: {error}") from error
    runtime_replay = _replay_runtime_events(
        runtime_events,
        phase_events=phase_events,
        phase_replay=replay,
    )
    if not _summary_artifact_is_canonical(
        run_dir, run_id=replay["run_id"], episode_id=replay["episode_id"]
    ):
        raise CanaryArtifactError("events.jsonl/summary.json are not a canonical matching pair")
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "diagnostic_only": True,
        "accepted": True,
        "expected_git_sha": expected_git_sha,
        "seed": seed,
        "run_dir": str(run_dir),
        "canary_journal": str(journal),
        "source_attestation": dict(source),
        "artifact_attestation": {
            "canary_journal_sha256": expected_journal_sha,
            "events_sha256": expected_events_sha,
            "summary_sha256": expected_summary_sha,
        },
        "gates": {
            "diagnostic_only": True,
            "source_attestation_exact": True,
            "run_dir_hash_attestation": True,
            "summary_reconstructed_from_events": True,
            "runtime_events_cross_checked": True,
            "exactly_one_operation": True,
            "three_distinct_failure_attempts_123": True,
            "unique_closed_to_open": True,
            "core_post_open_defer": True,
            "post_open_command_dispatch_rejection_zero": True,
            "open_ownership_empty": True,
            "one_material_builder_reset": True,
            "fresh_sc2_query_success": True,
            "primitive_submitted": True,
            "real_build_start_and_effect_confirmation": True,
            "runtime_raw_circuit_reset": runtime_replay["runtime_raw_reset_count"] > 0,
            "nonzero_failure_open_reset_counts": all(
                replay[key] > 0 for key in ("failure_count", "open_count", "reset_count")
            ),
        },
        "replay": replay,
        "runtime_replay": runtime_replay,
    }
    if not all(report["gates"].values()):
        report["accepted"] = False
    if output is not None:
        output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


# Short aliases make the analyzer convenient for focused unit tests and callers.
analyze_run = analyze_canary_run
build_canary_report = analyze_canary_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_set_dir", type=Path)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = analyze_canary_run(
            arguments.run_set_dir,
            expected_git_sha=arguments.expected_git_sha,
            seed=arguments.seed,
            output=arguments.output,
        )
    except CanaryArtifactError as error:
        failure = {
            "schema_version": "1.0",
            "diagnostic_only": True,
            "accepted": False,
            "expected_git_sha": arguments.expected_git_sha,
            "seed": arguments.seed,
            "error": str(error),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise SystemExit(1) from error
    raise SystemExit(0 if report.get("accepted") is True else 1)


if __name__ == "__main__":
    main()
