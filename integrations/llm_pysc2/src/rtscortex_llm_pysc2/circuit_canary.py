"""Diagnostic-only live stimulus for the authoritative Build circuit.

The controller never fabricates a placement result or gameplay effect.  It only
holds an already-routed Build command for one observation so the normal raw
placement path observes its immutable candidate as stale.  After three such
failures it requires the Core circuit boundary to defer the operation, then
allows a retry only after the Worker binds a different currently ready Builder.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from rtscortex_llm_pysc2.routing import RoutedCommand

_OPERATION_ID = re.compile(r"operation:[0-9a-f]{64}\Z")
_ATTEMPT_ID = re.compile(r"attempt:[0-9a-f]{64}\Z")
_BUILD_LEGALITY_ID = re.compile(r"build-legality:[0-9a-f]{64}\Z")
_MODE = "stale_candidate_then_builder_rebind"
_SEMANTIC_ACTION = "BUILD PYLON"
_RUNTIME_ACTION = "Build_Pylon_Screen"


def _tag(value: Any) -> int:
    try:
        return int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class CanaryRuntimeDecision:
    """Result of inspecting one current-observation Runtime decision."""

    hold_raw_dispatch: bool = False
    core_defer_observed: bool = False


class AuthoritativeBuildCircuitCanary:
    """Drive and journal one bounded real-SC2 authoritative circuit path."""

    def __init__(
        self,
        *,
        run_id: str,
        episode_id: str,
        seed: int,
        journal_path: Path,
        failure_attempts: int = 3,
        hold_observations: int = 1,
    ) -> None:
        if failure_attempts != 3:
            raise ValueError("authoritative Build circuit canary requires threshold 3")
        if hold_observations != 1:
            raise ValueError("authoritative Build circuit canary holds exactly one observation")
        if journal_path.exists():
            raise FileExistsError(f"canary journal already exists: {journal_path}")
        if not journal_path.parent.is_dir():
            raise FileNotFoundError(f"canary journal parent does not exist: {journal_path.parent}")
        self.run_id = str(run_id)
        self.episode_id = str(episode_id)
        self.seed = int(seed)
        self.journal_path = journal_path
        self.failure_attempts = int(failure_attempts)
        self.hold_observations = int(hold_observations)
        self.phase = "awaiting_failure_command"
        self.failure_count = 0
        self.operation_id: str | None = None
        self.semantic_action: str | None = None
        self.initial_builder_tag: int | None = None
        self.replacement_builder_tag: int | None = None
        self.current_command: RoutedCommand | None = None
        self.current_revision: str | None = None
        self.final_command_id: str | None = None
        self._last_attempt_ordinal: int | None = None
        self._hold_pending = False
        self._event_index = 0
        self._write(
            "initialized",
            game_loop=0,
            reason="diagnostic_canary_enabled",
            command_count=0,
            idle_reason=None,
        )

    @property
    def complete(self) -> bool:
        return self.phase == "complete"

    @property
    def requires_builder_rebind(self) -> bool:
        return self.phase in {"rebind_pending", "awaiting_reset_command"}

    @property
    def requires_builder_pin(self) -> bool:
        return bool(
            self.initial_builder_tag is not None
            and self.phase
            in {
                "awaiting_failure_command",
                "awaiting_core_defer",
                "rebind_pending",
                "awaiting_reset_command",
                "awaiting_reset_dispatch",
            }
        )

    @property
    def required_builder_tag(self) -> int | None:
        if self.phase in {"rebind_pending", "awaiting_reset_command", "awaiting_reset_dispatch"}:
            return self.replacement_builder_tag
        return self.initial_builder_tag

    def observe_runtime_decision(
        self,
        commands: Sequence[RoutedCommand],
        *,
        idle_reason: str | None,
        game_loop: int,
        observation_revision: str,
        builder_tag: int | None,
        reservation_count: int,
        leased_builder_tags: Sequence[int],
        effect_inflight_count: int,
        authoritative_state: Any | None,
    ) -> CanaryRuntimeDecision:
        """Inspect a real Runtime result before Raw translation begins."""

        build_commands = tuple(command for command in commands if command.name.startswith("Build_"))
        if len(build_commands) > 1:
            self._fail("multiple_build_commands", game_loop, observation_revision)

        if self.phase == "awaiting_failure_command":
            if not build_commands:
                return CanaryRuntimeDecision()
            if len(commands) != 1:
                self._fail("failure_decision_not_single_build", game_loop, observation_revision)
            if reservation_count or leased_builder_tags or effect_inflight_count:
                self._fail(
                    "failure_hold_execution_ownership_not_empty", game_loop, observation_revision
                )
            command = build_commands[0]
            self._bind_command(command, builder_tag=builder_tag, require_replacement=False)
            self.current_command = command
            self.current_revision = str(observation_revision)
            self._hold_pending = True
            self._write(
                "failure_command_held",
                game_loop=game_loop,
                observation_revision=observation_revision,
                command=command,
                builder_tag=builder_tag,
                reason="hold_until_new_observation_revision",
                command_count=len(commands),
                idle_reason=idle_reason,
                reservation_count=reservation_count,
                leased_builder_tags=sorted(int(tag) for tag in leased_builder_tags),
                effect_inflight_count=effect_inflight_count,
            )
            return CanaryRuntimeDecision(hold_raw_dispatch=True)

        if self.phase == "awaiting_core_defer":
            if commands:
                self._fail(
                    "post_open_command_crossed_core_boundary",
                    game_loop,
                    observation_revision,
                    command=commands[0],
                )
            if idle_reason != "plan_commands_deferred":
                self._fail("core_defer_idle_reason_mismatch", game_loop, observation_revision)
            if builder_tag != self.initial_builder_tag:
                self._fail("core_defer_builder_material_changed", game_loop, observation_revision)
            if authoritative_state is None:
                self._fail("core_defer_raw_circuit_state_missing", game_loop, observation_revision)
            state = self._state_dict(authoritative_state)
            if (
                int(state.get("streak", -1)) != self.failure_attempts
                or int(state.get("threshold", -1)) != self.failure_attempts
                or state.get("circuit_open") is not True
            ):
                self._fail("core_defer_raw_circuit_not_open", game_loop, observation_revision)
            if reservation_count or leased_builder_tags or effect_inflight_count:
                self._fail(
                    "post_open_execution_ownership_not_empty",
                    game_loop,
                    observation_revision,
                )
            self.phase = "rebind_pending"
            self._write(
                "core_defer_observed",
                game_loop=game_loop,
                observation_revision=observation_revision,
                reason="authoritative_build_pre_dispatch_circuit_defer",
                builder_tag=builder_tag,
                command_count=len(commands),
                idle_reason=idle_reason,
                authoritative_state=state,
                reservation_count=reservation_count,
                leased_builder_tags=sorted(int(tag) for tag in leased_builder_tags),
                effect_inflight_count=effect_inflight_count,
            )
            return CanaryRuntimeDecision(core_defer_observed=True)

        if self.phase == "awaiting_reset_command":
            if not build_commands:
                return CanaryRuntimeDecision()
            if len(commands) != 1:
                self._fail("reset_decision_not_single_build", game_loop, observation_revision)
            if reservation_count or leased_builder_tags or effect_inflight_count:
                self._fail(
                    "reset_command_execution_ownership_not_empty", game_loop, observation_revision
                )
            raw_state = self._state_dict(authoritative_state)
            if (
                int(raw_state.get("streak", -1)) != self.failure_attempts
                or int(raw_state.get("threshold", -1)) != self.failure_attempts
                or raw_state.get("circuit_open") is not True
            ):
                self._fail("reset_command_raw_circuit_not_open", game_loop, observation_revision)
            command = build_commands[0]
            self._bind_command(command, builder_tag=builder_tag, require_replacement=True)
            self.current_command = command
            self.final_command_id = command.command_id
            self.phase = "awaiting_reset_dispatch"
            self._write(
                "reset_command_observed",
                game_loop=game_loop,
                observation_revision=observation_revision,
                command=command,
                builder_tag=builder_tag,
                replacement_builder_tag=self.replacement_builder_tag,
                reason="validated_builder_material_change",
                command_count=len(commands),
                idle_reason=idle_reason,
                authoritative_state=raw_state,
                reservation_count=reservation_count,
                leased_builder_tags=sorted(int(tag) for tag in leased_builder_tags),
                effect_inflight_count=effect_inflight_count,
            )
            return CanaryRuntimeDecision()

        if build_commands and self.phase not in {
            "awaiting_reset_dispatch",
            "awaiting_effect",
            "complete",
        }:
            self._fail("unexpected_build_command", game_loop, observation_revision)
        return CanaryRuntimeDecision()

    def should_hold_raw_dispatch(self, *, observation_revision: str) -> bool:
        """Hold only the observation that created the selected stale candidate."""

        should_hold = bool(
            self.phase == "awaiting_failure_command"
            and self.current_command is not None
            and self._hold_pending
            and self.current_revision == str(observation_revision)
        )
        if should_hold:
            self._hold_pending = False
        return should_hold

    def observe_raw_result(
        self,
        *,
        dispatch: Any | None,
        diagnostic: Mapping[str, Any],
        game_loop: int,
        observation_revision: str,
        authoritative_state: Any | None,
        reservation_count: int,
        leased_builder_tags: Sequence[int],
        effect_inflight_count: int,
    ) -> None:
        """Validate the production Raw result after a non-held translation pass."""

        if self.phase == "awaiting_failure_command" and self.current_command is not None:
            if self.current_revision == str(observation_revision):
                self._fail(
                    "held_command_dispatched_in_source_observation", game_loop, observation_revision
                )
            self._observe_authoritative_failure(
                dispatch=dispatch,
                diagnostic=diagnostic,
                game_loop=game_loop,
                observation_revision=observation_revision,
                authoritative_state=authoritative_state,
                reservation_count=reservation_count,
                leased_builder_tags=leased_builder_tags,
                effect_inflight_count=effect_inflight_count,
            )
            return

        if self.phase != "awaiting_reset_dispatch" or dispatch is None:
            return
        command = getattr(dispatch, "command", None)
        if command is None or command.command_id != self.final_command_id:
            self._fail("reset_dispatch_identity_mismatch", game_loop, observation_revision)
        if bool(getattr(dispatch, "approach_only", False)):
            return
        if str(diagnostic.get("placement_query_result", "")) != "Success":
            self._fail("reset_dispatch_query_not_success", game_loop, observation_revision)
        if diagnostic.get("primitive_constructed") is not True:
            self._fail("reset_dispatch_primitive_not_constructed", game_loop, observation_revision)
        if _tag(diagnostic.get("builder_tag")) != self.replacement_builder_tag:
            self._fail("reset_dispatch_builder_mismatch", game_loop, observation_revision)
        raw_state = self._state_dict(authoritative_state)
        if (
            int(raw_state.get("streak", -1)) != self.failure_attempts
            or int(raw_state.get("threshold", -1)) != self.failure_attempts
            or raw_state.get("circuit_open") is not True
        ):
            self._fail("reset_dispatch_raw_circuit_not_open", game_loop, observation_revision)
        if (
            reservation_count != 1
            or tuple(int(tag) for tag in leased_builder_tags) != (self.replacement_builder_tag,)
            or effect_inflight_count != 1
        ):
            self._fail(
                "reset_dispatch_execution_ownership_invalid", game_loop, observation_revision
            )
        self.phase = "awaiting_submission"
        self._write(
            "reset_dispatch_observed",
            game_loop=game_loop,
            observation_revision=observation_revision,
            command=command,
            builder_tag=_tag(diagnostic.get("builder_tag")),
            replacement_builder_tag=self.replacement_builder_tag,
            reason="authoritative_query_success",
            query_result=diagnostic.get("placement_query_result"),
            primitive_constructed=True,
            primitive_submitted=False,
            authoritative_state=raw_state,
            raw_diagnostic=dict(diagnostic),
            reservation_count=reservation_count,
            leased_builder_tags=sorted(int(tag) for tag in leased_builder_tags),
            effect_inflight_count=effect_inflight_count,
        )

    def observe_primitive_submitted(
        self,
        dispatch: Any,
        *,
        game_loop: int,
        diagnostic: Mapping[str, Any],
        reservation_count: int,
        leased_builder_tags: Sequence[int],
        effect_inflight_count: int,
    ) -> None:
        """Record only an actual non-approach env.step submission."""

        if self.phase != "awaiting_submission" or bool(getattr(dispatch, "approach_only", False)):
            return
        command = getattr(dispatch, "command", None)
        if command is None or command.command_id != self.final_command_id:
            self._fail("submitted_primitive_identity_mismatch", game_loop, "")
        if diagnostic.get("primitive_submitted") is not True:
            self._fail("primitive_submission_not_recorded", game_loop, "")
        if (
            reservation_count != 1
            or tuple(int(tag) for tag in leased_builder_tags) != (self.replacement_builder_tag,)
            or effect_inflight_count != 1
        ):
            self._fail("submitted_primitive_execution_ownership_invalid", game_loop, "")
        self.phase = "awaiting_effect"
        self._write(
            "reset_dispatch_submitted",
            game_loop=game_loop,
            observation_revision=str(diagnostic.get("observation_revision", "")),
            command=command,
            builder_tag=_tag(diagnostic.get("builder_tag")),
            replacement_builder_tag=self.replacement_builder_tag,
            reason="pysc2_environment_step_returned",
            query_result=diagnostic.get("placement_query_result"),
            primitive_constructed=True,
            primitive_submitted=True,
            primitive_submitted_game_loop=diagnostic.get("primitive_submitted_game_loop"),
            raw_diagnostic=dict(diagnostic),
            reservation_count=reservation_count,
            leased_builder_tags=sorted(int(tag) for tag in leased_builder_tags),
            effect_inflight_count=effect_inflight_count,
        )

    def observe_effect_reports(
        self,
        reports: Sequence[Mapping[str, Any]],
        *,
        game_loop: int,
        observation_revision: str,
        reservation_count: int,
        leased_builder_tags: Sequence[int],
        effect_inflight_count: int,
        authoritative_state: Any | None,
    ) -> None:
        """Require a real successful build effect and fully released ownership."""

        if self.phase != "awaiting_effect" or self.final_command_id is None:
            return
        report = next(
            (item for item in reports if str(item.get("command_id", "")) == self.final_command_id),
            None,
        )
        if report is None:
            return
        if str(report.get("status", "")) != "succeeded":
            self._fail("reset_build_effect_failed", game_loop, observation_revision)
        evidence = report.get("effect_evidence")
        if not isinstance(evidence, Mapping):
            self._fail("reset_build_effect_evidence_missing", game_loop, observation_revision)
        if (
            evidence.get("build_started") is not True
            or not evidence.get("observed_structure_tag")
            or evidence.get("confirmation_kind") != "new_structure"
        ):
            self._fail("reset_build_effect_not_new_structure", game_loop, observation_revision)
        if reservation_count or leased_builder_tags or effect_inflight_count:
            self._fail("effect_terminal_ownership_not_released", game_loop, observation_revision)
        raw_state = self._state_dict(authoritative_state)
        if int(raw_state.get("streak", -1)) != 0 or raw_state.get("circuit_open") is not False:
            self._fail("effect_terminal_raw_circuit_not_reset", game_loop, observation_revision)
        self._write(
            "effect_confirmed",
            game_loop=game_loop,
            observation_revision=observation_revision,
            reason="real_gameplay_effect_succeeded",
            command_id=self.final_command_id,
            builder_tag=self.replacement_builder_tag,
            replacement_builder_tag=self.replacement_builder_tag,
            primitive_submitted=True,
            effect_status="succeeded",
            effect_evidence=dict(evidence),
            execution_report=dict(report),
            authoritative_state=raw_state,
            reservation_count=reservation_count,
            leased_builder_tags=[],
            effect_inflight_count=effect_inflight_count,
        )
        self.phase = "complete"
        self._write(
            "complete",
            game_loop=game_loop,
            observation_revision=observation_revision,
            reason="authoritative_circuit_real_sc2_path_complete",
            command_id=self.final_command_id,
            reservation_count=0,
            leased_builder_tags=[],
            effect_inflight_count=0,
        )

    def bind_replacement_builder(
        self,
        builder_tag: int,
        *,
        game_loop: int,
        observation_revision: str,
    ) -> None:
        """Bind one distinct ready Builder selected from the current raw observation."""

        if self.phase not in {
            "rebind_pending",
            "awaiting_reset_command",
            "awaiting_reset_dispatch",
        }:
            return
        normalized = int(builder_tag)
        if normalized <= 0 or normalized == self.initial_builder_tag:
            self._fail("replacement_builder_not_distinct", game_loop, observation_revision)
        if self.replacement_builder_tag is not None and normalized != self.replacement_builder_tag:
            self._fail("replacement_builder_changed_before_reset", game_loop, observation_revision)
        if self.replacement_builder_tag is None:
            self.replacement_builder_tag = normalized
            self._write(
                "builder_rebound",
                game_loop=game_loop,
                observation_revision=observation_revision,
                reason="fresh_ready_unleased_builder_observed",
                builder_tag=self.initial_builder_tag,
                replacement_builder_tag=normalized,
            )
        if self.phase == "rebind_pending":
            self.phase = "awaiting_reset_command"

    def observe_terminal(self, *, game_loop: int, observation_revision: str) -> None:
        if not self.complete:
            self._fail("episode_terminal_before_canary_complete", game_loop, observation_revision)

    def _observe_authoritative_failure(
        self,
        *,
        dispatch: Any | None,
        diagnostic: Mapping[str, Any],
        game_loop: int,
        observation_revision: str,
        authoritative_state: Any | None,
        reservation_count: int,
        leased_builder_tags: Sequence[int],
        effect_inflight_count: int,
    ) -> None:
        command = self.current_command
        assert command is not None
        if dispatch is not None:
            self._fail("stale_candidate_produced_dispatch", game_loop, observation_revision)
        if str(diagnostic.get("command_id", "")) != command.command_id:
            self._fail("stale_failure_command_mismatch", game_loop, observation_revision)
        if diagnostic.get("failure_code") != "placement_candidate_stale":
            self._fail("unexpected_authoritative_failure_code", game_loop, observation_revision)
        if diagnostic.get("primitive_constructed") is not False:
            self._fail(
                "authoritative_failure_constructed_primitive", game_loop, observation_revision
            )
        authoritative = diagnostic.get("authoritative_pre_dispatch")
        if not isinstance(authoritative, Mapping):
            self._fail("authoritative_failure_evidence_missing", game_loop, observation_revision)
        expected_streak = self.failure_count + 1
        if int(authoritative.get("streak", -1)) != expected_streak:
            self._fail("authoritative_streak_mismatch", game_loop, observation_revision)
        if int(authoritative.get("threshold", -1)) != self.failure_attempts:
            self._fail("authoritative_threshold_mismatch", game_loop, observation_revision)
        material_identity = authoritative.get("material_legality_identity")
        invalid_reasons = authoritative.get("invalid_evidence_reasons")
        if (
            authoritative.get("material_evidence_valid") is not True
            or invalid_reasons not in (None, (), [])
            or not isinstance(material_identity, str)
            or _BUILD_LEGALITY_ID.fullmatch(material_identity) is None
            or _tag(authoritative.get("ability_id")) <= 0
            or not self._valid_world_target(authoritative.get("world_target"))
        ):
            self._fail("authoritative_material_evidence_invalid", game_loop, observation_revision)
        expected_open = expected_streak == self.failure_attempts
        if bool(authoritative.get("circuit_open")) is not expected_open:
            self._fail("authoritative_open_state_mismatch", game_loop, observation_revision)
        expected_transition = "closed_to_open" if expected_open else None
        if authoritative.get("state_transition") != expected_transition:
            self._fail("authoritative_transition_mismatch", game_loop, observation_revision)
        if authoritative.get("duplicate_attempt") is not False:
            self._fail("authoritative_attempt_was_duplicate", game_loop, observation_revision)
        nested_builder = _tag(authoritative.get("builder_tag"))
        if nested_builder <= 0:
            self._fail("authoritative_builder_missing", game_loop, observation_revision)
        if self.initial_builder_tag is None:
            self.initial_builder_tag = nested_builder
        elif nested_builder != self.initial_builder_tag:
            self._fail("builder_changed_before_circuit_open", game_loop, observation_revision)
        if authoritative_state is None:
            self._fail("authoritative_state_missing", game_loop, observation_revision)
        state_dict = self._state_dict(authoritative_state)
        if int(state_dict.get("streak", -1)) != expected_streak:
            self._fail("authoritative_store_streak_mismatch", game_loop, observation_revision)
        if bool(state_dict.get("circuit_open")) is not expected_open:
            self._fail("authoritative_store_open_mismatch", game_loop, observation_revision)
        if reservation_count or leased_builder_tags or effect_inflight_count:
            self._fail(
                "pre_dispatch_failure_created_execution_ownership", game_loop, observation_revision
            )
        self.failure_count = expected_streak
        self._write(
            "authoritative_failure_observed",
            game_loop=game_loop,
            observation_revision=observation_revision,
            command=command,
            builder_tag=nested_builder,
            reason="placement_candidate_stale",
            authoritative_streak=expected_streak,
            authoritative_threshold=self.failure_attempts,
            circuit_open=expected_open,
            transition=expected_transition,
            duplicate_attempt=False,
            material_legality_identity=material_identity,
            authoritative_state=state_dict,
            raw_diagnostic=dict(diagnostic),
            reservation_count=reservation_count,
            leased_builder_tags=[],
            effect_inflight_count=effect_inflight_count,
        )
        if expected_open:
            self._write(
                "circuit_open_observed",
                game_loop=game_loop,
                observation_revision=observation_revision,
                command=command,
                builder_tag=nested_builder,
                reason="third_unique_authoritative_failure",
                authoritative_streak=expected_streak,
                authoritative_threshold=self.failure_attempts,
                circuit_open=True,
                transition="closed_to_open",
                reservation_count=0,
                leased_builder_tags=[],
                effect_inflight_count=0,
            )
            self.phase = "awaiting_core_defer"
        else:
            self.phase = "awaiting_failure_command"
        self.current_command = None
        self.current_revision = None

    def _bind_command(
        self,
        command: RoutedCommand,
        *,
        builder_tag: int | None,
        require_replacement: bool,
    ) -> None:
        operation_id = command.operation_id
        attempt_id = command.attempt_id
        ordinal = command.attempt_ordinal
        semantic_action = command.semantic_action or command.name
        if command.name != _RUNTIME_ACTION or semantic_action != _SEMANTIC_ACTION:
            raise RuntimeError("canary command is not the dedicated Pylon action")
        if operation_id is None or _OPERATION_ID.fullmatch(operation_id) is None:
            raise RuntimeError("canary Build command has invalid operation identity")
        if attempt_id is None or _ATTEMPT_ID.fullmatch(attempt_id) is None or ordinal is None:
            raise RuntimeError("canary Build command has invalid attempt identity")
        expected_attempt = (
            "attempt:"
            + hashlib.sha256(
                json.dumps(
                    {
                        "operation_id": operation_id,
                        "command_id": command.command_id,
                        "attempt_ordinal": int(ordinal),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        if attempt_id != expected_attempt:
            raise RuntimeError("canary Build command attempt identity is not canonical")
        if self.operation_id is None:
            self.operation_id = operation_id
            self.semantic_action = semantic_action
        elif operation_id != self.operation_id or semantic_action != self.semantic_action:
            raise RuntimeError("canary Build command changed semantic operation")
        expected_ordinal = self.failure_count if not require_replacement else self.failure_attempts
        if int(ordinal) != expected_ordinal:
            raise RuntimeError("canary Build attempt ordinal does not match its phase")
        if (
            self._last_attempt_ordinal is not None
            and int(ordinal) != self._last_attempt_ordinal + 1
        ):
            raise RuntimeError("canary Build attempt ordinals are not consecutive")
        if builder_tag is None or int(builder_tag) <= 0:
            raise RuntimeError("canary Build command has no current Builder")
        if require_replacement:
            if (
                self.replacement_builder_tag is None
                or int(builder_tag) != self.replacement_builder_tag
            ):
                raise RuntimeError("canary reset command is not bound to the replacement Builder")
        elif self.initial_builder_tag is None:
            self.initial_builder_tag = int(builder_tag)
        elif int(builder_tag) != self.initial_builder_tag:
            raise RuntimeError("canary failure command changed Builder before circuit-open")
        self._last_attempt_ordinal = int(ordinal)

    def _fail(
        self,
        reason: str,
        game_loop: int,
        observation_revision: str,
        *,
        command: RoutedCommand | None = None,
    ) -> NoReturn:
        self.phase = "failed"
        self._write(
            "failed",
            game_loop=game_loop,
            observation_revision=observation_revision,
            command=command,
            reason=reason,
        )
        raise RuntimeError(f"authoritative Build circuit canary failed: {reason}")

    @staticmethod
    def _state_dict(value: Any) -> dict[str, Any]:
        if callable(getattr(value, "to_dict", None)):
            result = value.to_dict()
            return dict(result) if isinstance(result, Mapping) else {}
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _valid_world_target(value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        try:
            return all(math.isfinite(float(item)) for item in value)
        except (TypeError, ValueError):
            return False

    def _write(
        self,
        phase: str,
        *,
        game_loop: int | None = None,
        observation_revision: str | None = None,
        command: RoutedCommand | None = None,
        **values: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "event_type": "authoritative_build_circuit_canary_phase",
            "schema_version": "1.0",
            "diagnostic_only": True,
            "mode": _MODE,
            "event_index": self._event_index,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "phase": phase,
            "game_loop": game_loop,
            "observation_revision": observation_revision,
            "operation_id": self.operation_id,
            "semantic_action": self.semantic_action,
        }
        if command is not None:
            payload.update(
                {
                    "command_id": command.command_id,
                    "operation_id": command.operation_id,
                    "attempt_id": command.attempt_id,
                    "attempt_ordinal": command.attempt_ordinal,
                    "semantic_action": command.semantic_action or command.name,
                    "runtime_action": command.name,
                }
            )
        payload.update(values)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
        self._event_index += 1


__all__ = ["AuthoritativeBuildCircuitCanary", "CanaryRuntimeDecision"]
