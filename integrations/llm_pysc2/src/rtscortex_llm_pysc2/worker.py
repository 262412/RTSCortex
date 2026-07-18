"""Importable LLM-PySC2 worker classes for the pinned upstream checkout."""

from __future__ import annotations

import copy
import importlib
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral
from typing import Any, Optional

from rtscortex_llm_pysc2.broker import PrimitiveDispatch, SharedDecisionBroker
from rtscortex_llm_pysc2.clock import FixedRateGameClock, InitialPlanningBarrier
from rtscortex_llm_pysc2.coordinator import BridgeCoordinator
from rtscortex_llm_pysc2.effect_verifier import (
    DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS,
    ActionEffectVerifier,
)
from rtscortex_llm_pysc2.extractor import (
    BUILD_SPECS,
    MINIMAP_POINT_ACTIONS,
    SCREEN_POINT_ACTIONS,
    SELECT_BLINK_ACTION,
    TimeStepExtractor,
    build_screen_candidates,
    builder_move_requires_power,
    is_production_action,
    nexus_placement_footprint_is_visible,
    production_source_tag,
    resolve_screen_build_world_target,
    resolve_screen_point_world_target,
    screen_build_position_is_legal,
    semantic_argument_candidates,
)
from rtscortex_llm_pysc2.frame_stream import RGBFramePublisher, RuntimeFrameUploader
from rtscortex_llm_pysc2.hook import RuntimeQueryMixin
from rtscortex_llm_pysc2.production import ProductionSpec, production_spec
from rtscortex_llm_pysc2.protocol import RuntimeClient

PRODUCTION_CAMERA_SETTLE_MAX_OBSERVATIONS = 4
BUILD_SELECTION_RETRY_MAX_OBSERVATIONS = 2
BUILD_TRANSLATOR_RETRY_FAILURE_CODES = frozenset(
    {"blocked", "need_power", "not_pathable", "no_legal_placement"}
)
DEFAULT_OBSERVATION_GAP_WATCHDOG_GAME_LOOPS = 448
DEFAULT_OBSERVATION_GAP_HARD_LIMIT_GAME_LOOPS = 1792

try:
    _upstream_agents = importlib.import_module("llm_pysc2.agents")
except ModuleNotFoundError as error:
    _UPSTREAM_IMPORT_ERROR: Optional[ModuleNotFoundError] = error
    _MainAgentBase: Any = object
    _LLMAgentBase: Any = object
else:
    _UPSTREAM_IMPORT_ERROR = None
    _MainAgentBase = _upstream_agents.MainAgent
    _LLMAgentBase = _upstream_agents.LLMAgent


@dataclass(frozen=True)
class WorkerSettings:
    run_id: str
    episode_id: str
    socket_path: Optional[str]
    runtime_url: str
    seed: int
    scenario: str = "pvz_task1_level1"
    pending_plan_step_delay_seconds: float = 0.0
    simulation_speed_multiplier: Optional[float] = None
    pause_until_first_plan: bool = False
    runtime_request_timeout_seconds: float = 60.0
    action_effect_timeout_game_loops: int = DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS
    observation_gap_watchdog_game_loops: int = DEFAULT_OBSERVATION_GAP_WATCHDOG_GAME_LOOPS
    observation_gap_hard_limit_game_loops: int = DEFAULT_OBSERVATION_GAP_HARD_LIMIT_GAME_LOOPS
    console_enabled: bool = False
    console_frame_fps: float = 2.0
    console_jpeg_quality: int = 75

    @classmethod
    def from_environment(cls) -> WorkerSettings:
        run_id = os.environ.get("RTSCORTEX_RUN_ID")
        episode_id = os.environ.get("RTSCORTEX_EPISODE_ID")
        if not run_id or not episode_id:
            raise RuntimeError("RTSCORTEX_RUN_ID and RTSCORTEX_EPISODE_ID must be set")
        pending_plan_step_delay_seconds = float(
            os.environ.get("RTSCORTEX_PENDING_PLAN_STEP_DELAY_SECONDS", "0")
        )
        if pending_plan_step_delay_seconds < 0:
            raise RuntimeError("RTSCORTEX_PENDING_PLAN_STEP_DELAY_SECONDS cannot be negative")
        simulation_speed = os.environ.get("RTSCORTEX_SIMULATION_SPEED_MULTIPLIER")
        runtime_request_timeout_seconds = float(
            os.environ.get("RTSCORTEX_RUNTIME_REQUEST_TIMEOUT_SECONDS", "60")
        )
        if runtime_request_timeout_seconds <= 0:
            raise RuntimeError("RTSCORTEX_RUNTIME_REQUEST_TIMEOUT_SECONDS must be positive")
        action_effect_timeout_game_loops = int(
            os.environ.get(
                "RTSCORTEX_ACTION_EFFECT_TIMEOUT_GAME_LOOPS",
                str(DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS),
            )
        )
        if action_effect_timeout_game_loops <= 0:
            raise RuntimeError("RTSCORTEX_ACTION_EFFECT_TIMEOUT_GAME_LOOPS must be positive")
        observation_gap_watchdog_game_loops = int(
            os.environ.get(
                "RTSCORTEX_OBSERVATION_GAP_WATCHDOG_GAME_LOOPS",
                str(DEFAULT_OBSERVATION_GAP_WATCHDOG_GAME_LOOPS),
            )
        )
        observation_gap_hard_limit_game_loops = int(
            os.environ.get(
                "RTSCORTEX_OBSERVATION_GAP_HARD_LIMIT_GAME_LOOPS",
                str(DEFAULT_OBSERVATION_GAP_HARD_LIMIT_GAME_LOOPS),
            )
        )
        if observation_gap_watchdog_game_loops <= 0:
            raise RuntimeError("RTSCORTEX_OBSERVATION_GAP_WATCHDOG_GAME_LOOPS must be positive")
        if observation_gap_hard_limit_game_loops <= observation_gap_watchdog_game_loops:
            raise RuntimeError(
                "RTSCORTEX_OBSERVATION_GAP_HARD_LIMIT_GAME_LOOPS must exceed the watchdog threshold"
            )
        pause_until_first_plan = _environment_bool("RTSCORTEX_PAUSE_UNTIL_FIRST_PLAN", False)
        console_enabled = _environment_bool("RTSCORTEX_CONSOLE_ENABLED", False)
        console_frame_fps = float(os.environ.get("RTSCORTEX_CONSOLE_FRAME_FPS", "2"))
        if console_frame_fps <= 0:
            raise RuntimeError("RTSCORTEX_CONSOLE_FRAME_FPS must be positive")
        console_jpeg_quality = int(os.environ.get("RTSCORTEX_CONSOLE_JPEG_QUALITY", "75"))
        if not 1 <= console_jpeg_quality <= 95:
            raise RuntimeError("RTSCORTEX_CONSOLE_JPEG_QUALITY must be between 1 and 95")
        return cls(
            run_id=run_id,
            episode_id=episode_id,
            socket_path=os.environ.get("RTSCORTEX_RUNTIME_SOCKET")
            or os.environ.get("RTSCORTEX_SOCKET"),
            runtime_url=os.environ.get("RTSCORTEX_RUNTIME_URL", "http://127.0.0.1:8765"),
            seed=int(os.environ.get("RTSCORTEX_SEED", "0")),
            scenario=os.environ.get("RTSCORTEX_SCENARIO", "pvz_task1_level1"),
            pending_plan_step_delay_seconds=pending_plan_step_delay_seconds,
            simulation_speed_multiplier=(
                float(simulation_speed) if simulation_speed is not None else None
            ),
            pause_until_first_plan=pause_until_first_plan,
            runtime_request_timeout_seconds=runtime_request_timeout_seconds,
            action_effect_timeout_game_loops=action_effect_timeout_game_loops,
            observation_gap_watchdog_game_loops=observation_gap_watchdog_game_loops,
            observation_gap_hard_limit_game_loops=observation_gap_hard_limit_game_loops,
            console_enabled=console_enabled,
            console_frame_fps=console_frame_fps,
            console_jpeg_quality=console_jpeg_quality,
        )


class RTSCortexLLMAgent(RuntimeQueryMixin, _LLMAgentBase):  # type: ignore[misc]
    """Real upstream LLMAgent subclass with its model call replaced by a broker."""

    broker: SharedDecisionBroker
    func_list: list[Any]
    action_list: list[dict[str, Any]]

    def __init__(
        self,
        *args: Any,
        broker: SharedDecisionBroker,
        unit_names: Mapping[int, str],
        **kwargs: Any,
    ) -> None:
        _require_upstream()
        super().__init__(*args, **kwargs)
        if not hasattr(self, "last_translation_result"):
            raise RuntimeError(
                "LLM-PySC2 translation-result hook is missing; apply reviewed patch 0004"
            )
        self.broker = broker
        self.unit_names = dict(unit_names)
        self._rtscortex_translation_attempt: Optional[dict[str, Any]] = None
        self._rtscortex_semantic_action: Optional[dict[str, Any]] = None
        self._rtscortex_production_source_tag: Optional[int] = None
        self._rtscortex_production_camera_waits = 0
        self._rtscortex_production_camera_wait_loop: Optional[int] = None
        self._rtscortex_build_selection_retries = 0
        self._rtscortex_camera_settlement_noop = False
        self._rtscortex_rejected_build_positions: dict[str, set[tuple[int, int]]] = {}
        self._rtscortex_rejected_build_targets: dict[str, set[tuple[float, float]]] = {}
        self._rtscortex_active_build_route: Optional[
            tuple[str, tuple[int, int], Optional[tuple[float, float]]]
        ] = None
        broker.register(self)

    def get_func(self, obs: Any) -> Any:
        self._rtscortex_camera_settlement_noop = False
        semantic_action = getattr(self, "_rtscortex_semantic_action", None)
        if (
            not self.func_list
            and self.action_list
            and semantic_action is None
        ):
            self._rtscortex_semantic_action = _isolate_next_action(self.action_list)
            semantic_action = self._rtscortex_semantic_action
            self._rtscortex_production_camera_waits = 0
            self._rtscortex_production_camera_wait_loop = None
            self._rtscortex_build_selection_retries = 0
            if production_spec(str(self.action_list[0].get("name", ""))) is None:
                self._rtscortex_production_source_tag = None
        action: dict[str, Any] = (
            semantic_action
            if semantic_action is not None
            else {"name": getattr(self, "curr_action_name", "")}
        )
        semantic_action_name = str(action.get("name", ""))
        if self._wait_for_production_camera(semantic_action_name, obs):
            self._rtscortex_camera_settlement_noop = True
            return 0, _no_op()
        if not self.func_list and self.action_list:
            action_name = semantic_action_name
            if self._reject_unavailable_production_action(action, obs):
                return 0, _no_op()
            spec = BUILD_SPECS.get(action_name)
            is_screen_build = spec is not None and spec.placement_kind == "screen"
            is_screen_point = action_name in SCREEN_POINT_ACTIONS
            command_id: Optional[str] = None
            resolved: Optional[list[int]] = None
            if is_screen_build or is_screen_point:
                command_id = self.broker.command_id_for(
                    self.name,
                    _execution_team_name(self),
                    action_name,
                )
                provenance = (
                    None if command_id is None else self.broker.screen_route_provenance(command_id)
                )
                if provenance is not None:
                    if is_screen_build:
                        rejected_positions = self._rtscortex_rejected_build_positions.get(
                            action_name, set()
                        )
                        target_key = _build_world_target_key(provenance.world_target)
                        resolved = _resolve_translator_compatible_build_position(
                            action,
                            obs.observation,
                            world_target=provenance.world_target,
                            preferred_anchor_tag=provenance.anchor_tag,
                            excluded_positions=rejected_positions,
                            force_resample=(
                                target_key
                                in self._rtscortex_rejected_build_targets.get(action_name, set())
                            ),
                            screen_size=int(self.size_screen),
                        )
                    else:
                        resolved = _resolve_screen_point_action_position(
                            action,
                            obs.observation,
                            world_target=provenance.world_target,
                            preferred_anchor_tag=provenance.anchor_tag,
                            require_power=(
                                self.name == "Builder"
                                and action_name == "Move_Screen"
                                and builder_move_requires_power(
                                    obs.observation,
                                    self.unit_names,
                                )
                            ),
                        )
            if (is_screen_build or is_screen_point) and resolved is None:
                if is_screen_build:
                    requested = _screen_argument(action)
                    if requested is not None:
                        self._rtscortex_rejected_build_positions.setdefault(action_name, set()).add(
                            _screen_position_key(requested)
                        )
                    if provenance is not None:
                        self._rtscortex_rejected_build_targets.setdefault(action_name, set()).add(
                            _build_world_target_key(provenance.world_target)
                        )
                predispatch_failure_code = (
                    "no_legal_placement" if is_screen_build else "candidate_invalidated"
                )
                predispatch_failure_reason = (
                    "no legal placement remained at dispatch time"
                    if is_screen_build
                    else "screen target could not be reprojected into the current candidate set"
                )
                dispatch = self.broker.claim_primitive(
                    self.name,
                    _execution_team_name(self),
                    action_name,
                    "pre_dispatch",
                    final_primitive=True,
                    origin="translator",
                    ordinal=0,
                    total=1,
                    failure_code=predispatch_failure_code,
                )
                if dispatch is None:
                    self.broker.raise_unattributed_integrity(f"no command owns {action_name!r}")
                self.broker.settle_primitive(
                    dispatch,
                    success=False,
                    failure_reason=predispatch_failure_reason,
                    game_loop=_observation_game_loop(obs.observation),
                )
                self.action_list.pop(0)
                self.func_list.clear()
                self._rtscortex_semantic_action = None
                return 0, _no_op()
            semantic_failure = _semantic_target_failure(
                action,
                obs.observation,
                self.unit_names,
            )
            if semantic_failure is not None:
                semantic_failure_code, semantic_failure_reason = semantic_failure
                dispatch = self.broker.claim_primitive(
                    self.name,
                    _execution_team_name(self),
                    action_name,
                    "pre_dispatch",
                    final_primitive=True,
                    origin="translator",
                    ordinal=0,
                    total=1,
                    failure_code=semantic_failure_code,
                )
                if dispatch is None:
                    self.broker.raise_unattributed_integrity(f"no command owns {action_name!r}")
                self.broker.settle_primitive(
                    dispatch,
                    success=False,
                    failure_reason=semantic_failure_reason,
                    game_loop=_observation_game_loop(obs.observation),
                )
                self.action_list.pop(0)
                self.func_list.clear()
                self._rtscortex_semantic_action = None
                return 0, _no_op()
            if resolved is not None:
                if command_id is not None:
                    self.broker.resolve_arguments(command_id, [resolved])
            build_reselection = self._reselect_builder_for_unavailable_build(
                action,
                obs,
            )
            if build_reselection is not None:
                return build_reselection
        if _next_primitive_is_screen_build(semantic_action_name, self.func_list):
            command_id = self.broker.command_id_for(
                self.name,
                _execution_team_name(self),
                semantic_action_name,
            )
            provenance = (
                None if command_id is None else self.broker.screen_route_provenance(command_id)
            )
            if provenance is not None:
                action["func"] = list(self.func_list)
                rejected_positions = self._rtscortex_rejected_build_positions.get(
                    semantic_action_name,
                    set(),
                )
                target_key = _build_world_target_key(provenance.world_target)
                resolved = _resolve_translator_compatible_build_position(
                    action,
                    obs.observation,
                    world_target=provenance.world_target,
                    preferred_anchor_tag=provenance.anchor_tag,
                    excluded_positions=rejected_positions,
                    force_resample=(
                        target_key
                        in self._rtscortex_rejected_build_targets.get(
                            semantic_action_name,
                            set(),
                        )
                    ),
                    screen_size=int(self.size_screen),
                )
                if resolved is None:
                    requested = _screen_argument(action)
                    if requested is not None:
                        self._rtscortex_rejected_build_positions.setdefault(
                            semantic_action_name,
                            set(),
                        ).add(_screen_position_key(requested))
                    self._rtscortex_rejected_build_targets.setdefault(
                        semantic_action_name,
                        set(),
                    ).add(target_key)
                    dispatch = self.broker.reject_command(
                        self.name,
                        _execution_team_name(self),
                        semantic_action_name,
                        failure_code="no_legal_placement",
                    )
                    if dispatch is None:
                        self.broker.raise_unattributed_integrity(
                            f"no command owns {semantic_action_name!r}"
                        )
                    self.broker.settle_primitive(
                        dispatch,
                        success=False,
                        failure_reason=(
                            "no legal placement remained after camera or builder selection"
                        ),
                        game_loop=_observation_game_loop(obs.observation),
                    )
                    self.func_list.clear()
                    self._rtscortex_semantic_action = None
                    self._rtscortex_build_selection_retries = 0
                    return 0, _no_op()
                self.func_list = list(action["func"])
                assert command_id is not None
                self.broker.resolve_arguments(command_id, [resolved])
        # Upstream consumes and mutates the semantic action while translating it.
        # Keep the already validated request for the final candidate-domain audit.
        candidate_action = copy.deepcopy(action)
        self._rtscortex_translation_attempt = None
        result = super().get_func(obs)
        metadata = getattr(self, "last_translation_result", None)
        if not isinstance(metadata, Mapping):
            self.broker.fail_command_integrity(
                self.name,
                _execution_team_name(self),
                str(getattr(self, "curr_action_name", "")),
                "LLM-PySC2 patch 0004 produced no translation result",
                function_name="translation_result",
                game_loop=_observation_game_loop(obs.observation),
            )
        if _upstream_replaced_production_with_noop(semantic_action_name, metadata):
            self._settle_production_source_failure(
                semantic_action_name,
                obs,
                "upstream translator could not resolve a completed idle production source",
            )
            return result
        attempt = dict(metadata)
        action_name = str(attempt.get("action_name", self.curr_action_name))
        requested_id = int(attempt.get("requested_function_id", 0))
        requested_name = str(attempt.get("requested_function_name") or _function_name(requested_id))
        accepted = bool(attempt.get("accepted", False))
        raw_reason = None if attempt.get("reason") is None else str(attempt["reason"])
        failure_code = _translation_failure_code(raw_reason, action_name)
        ordinal = int(attempt.get("ordinal", 0))
        total = int(attempt.get("total", 1))
        final_primitive = not accepted or ordinal + 1 >= total
        train_spec = production_spec(action_name)
        producer_tag, provenance_failure = self._validated_production_source_tag(
            action_name,
            attempt,
        )
        if provenance_failure is not None:
            failure_code, reason = provenance_failure
            self._settle_production_command_failure(
                action_name,
                obs,
                failure_code=failure_code,
                reason=reason,
            )
            return 0, _no_op()
        if train_spec is not None and final_primitive:
            invalid_reason = _production_source_invalid_reason(
                obs.observation,
                producer_tag,
                train_spec,
                self.unit_names,
            )
            if invalid_reason is not None:
                self._settle_production_command_failure(
                    action_name,
                    obs,
                    failure_code="production_source_invalidated",
                    reason=invalid_reason,
                )
                return 0, _no_op()
        translated_position = _translated_build_position(
            action_name,
            attempt.get("resolved_arguments"),
        )
        translated_build_spec = BUILD_SPECS.get(action_name)
        if (
            not accepted
            and translated_build_spec is not None
            and translated_build_spec.placement_kind == "screen"
            and failure_code in BUILD_TRANSLATOR_RETRY_FAILURE_CODES
        ):
            rejected_position = _screen_argument(candidate_action)
            if rejected_position is not None:
                self._rtscortex_rejected_build_positions.setdefault(
                    action_name,
                    set(),
                ).add(_screen_position_key(rejected_position))
            if provenance is not None:
                self._rtscortex_rejected_build_targets.setdefault(
                    action_name,
                    set(),
                ).add(_build_world_target_key(provenance.world_target))
            # Nothing reached PySC2. Keep the command and primitive ordinal alive
            # so the next observation can use another semantic placement candidate.
            self.func_list = list(candidate_action.get("func", ()))
            self._rtscortex_semantic_action = candidate_action
            self._rtscortex_translation_ordinal = ordinal
            return result
        if (
            action_name == "Build_Nexus_Near"
            and accepted
            and final_primitive
            and (
                translated_position is None
                or not nexus_placement_footprint_is_visible(
                    obs.observation,
                    translated_position,
                )
            )
        ):
            dispatch = self.broker.claim_primitive(
                self.name,
                _execution_team_name(self),
                action_name,
                requested_name,
                final_primitive=True,
                origin="translator",
                ordinal=ordinal,
                total=total,
                failure_code="target_not_visible",
                requested_function_id=requested_id,
                emitted_function_id=0,
            )
            if dispatch is None:
                self.broker.raise_unattributed_integrity(
                    "unscouted Nexus primitive has no active command"
                )
            if translated_position is not None:
                self.broker.resolve_arguments(dispatch.command_id, [translated_position])
            self.broker.settle_primitive(
                dispatch,
                success=False,
                failure_reason="translated Nexus footprint is not fully visible",
                game_loop=_observation_game_loop(obs.observation),
            )
            self.func_list.clear()
            self._rtscortex_semantic_action = None
            return 0, _no_op()
        dispatch = self.broker.claim_primitive(
            self.name,
            _execution_team_name(self),
            action_name,
            requested_name,
            final_primitive=final_primitive,
            origin="translator",
            ordinal=ordinal,
            total=total,
            failure_code=failure_code,
            requested_function_id=requested_id,
            emitted_function_id=int(attempt.get("emitted_function_id", requested_id)),
        )
        if dispatch is None:
            if action_name != "No_Operation":
                self.broker.raise_unattributed_integrity(
                    f"translator primitive for {action_name!r} has no unique active command"
                )
            return result
        if accepted:
            candidate_failure = _candidate_dispatch_failure(
                candidate_action,
                obs.observation,
                self.unit_names,
                final_primitive=dispatch.final_primitive,
                translated_position=translated_position,
            )
            if candidate_failure is not None:
                self.broker.reject_candidate_outside_dispatch(
                    dispatch,
                    candidate_failure,
                    game_loop=_observation_game_loop(obs.observation),
                )
        if not accepted:
            self.broker.settle_primitive(
                dispatch,
                success=False,
                failure_reason=raw_reason or "translator rejected the action",
                game_loop=_observation_game_loop(obs.observation),
            )
            self.func_list.clear()
            self._rtscortex_semantic_action = None
            return result
        if dispatch.final_primitive and translated_position is not None:
            self.broker.resolve_arguments(dispatch.command_id, [translated_position])
            route = self.broker.screen_route_provenance(dispatch.command_id)
            self._rtscortex_active_build_route = (
                action_name,
                _screen_position_key(translated_position),
                None if route is None else _build_world_target_key(route.world_target),
            )
        self._rtscortex_translation_attempt = {
            "dispatch": dispatch,
            "emitted_function_id": int(attempt.get("emitted_function_id", requested_id)),
            "expected_arguments": attempt.get("resolved_arguments", []),
            "candidate_constrained": _is_candidate_constrained_action(action_name),
            "producer_tag": producer_tag,
        }
        if dispatch.final_primitive:
            self._rtscortex_semantic_action = None
            self._rtscortex_production_source_tag = None
            self._rtscortex_production_camera_waits = 0
            self._rtscortex_production_camera_wait_loop = None
            self._rtscortex_build_selection_retries = 0
        return result

    def _reselect_builder_for_unavailable_build(
        self,
        action: Mapping[str, Any],
        obs: Any,
    ) -> Optional[tuple[int, Any]]:
        """Repair a stale feature selection before translating a screen build."""

        action_name = str(action.get("name", ""))
        spec = BUILD_SPECS.get(action_name)
        functions = action.get("func", ())
        if spec is None or spec.placement_kind != "screen" or not functions:
            return None
        requested_function_id = int(functions[0][0])
        available_actions = set(_observation_value(obs.observation, "available_actions", ()))
        if requested_function_id in available_actions:
            self._rtscortex_build_selection_retries = 0
            return None
        if self._rtscortex_build_selection_retries >= BUILD_SELECTION_RETRY_MAX_OBSERVATIONS:
            return None
        builder_tag = _execution_unit_tag(self)
        position = _visible_feature_position(obs.observation, builder_tag)
        if position is None:
            return None
        self._rtscortex_build_selection_retries += 1
        actions = importlib.import_module("pysc2.lib.actions")
        return 2, actions.FUNCTIONS.select_point("select", position)

    def _wait_for_production_camera(self, action_name: str, obs: Any) -> bool:
        """Wait for a post-camera feature observation before selecting a producer."""

        if production_spec(action_name) is None or not self.func_list:
            return False
        next_function_id = int(self.func_list[0][0])
        if next_function_id != 2 or int(self._rtscortex_translation_ordinal) != 1:
            return False
        producer_tag = self._rtscortex_production_source_tag
        if producer_tag is None:
            return False
        if _producer_is_visible(
            obs.observation,
            producer_tag,
            size_screen=int(self.size_screen),
        ):
            self._rtscortex_production_camera_waits = 0
            self._rtscortex_production_camera_wait_loop = None
            return False
        observation_loop = _observation_game_loop(obs.observation)
        if self._rtscortex_production_camera_wait_loop == observation_loop:
            return True
        self._rtscortex_production_camera_wait_loop = observation_loop
        self._rtscortex_production_camera_waits += 1
        if self._rtscortex_production_camera_waits <= (PRODUCTION_CAMERA_SETTLE_MAX_OBSERVATIONS):
            return True
        self._settle_production_command_failure(
            action_name,
            obs,
            failure_code="producer_not_observable",
            reason=(
                f"producer tag {hex(producer_tag)} did not become visible after camera "
                f"settlement ({PRODUCTION_CAMERA_SETTLE_MAX_OBSERVATIONS} observations)"
            ),
        )
        self._rtscortex_production_camera_waits = 0
        self._rtscortex_production_camera_wait_loop = None
        return True

    def _validated_production_source_tag(
        self,
        action_name: str,
        attempt: Mapping[str, Any],
    ) -> tuple[Optional[int], Optional[tuple[str, str]]]:
        """Bind production to the exact raw producer selected by the translator."""

        if production_spec(action_name) is None:
            return None, None
        source_tag = self._rtscortex_production_source_tag
        ordinal = int(attempt.get("ordinal", 0))
        requested_id = int(attempt.get("requested_function_id", 0))
        if ordinal == 0:
            requested_arguments = attempt.get("requested_arguments")
            translator_source_tag = (
                int(requested_arguments[0])
                if isinstance(requested_arguments, (list, tuple))
                and len(requested_arguments) == 1
                and isinstance(requested_arguments[0], int)
                and not isinstance(requested_arguments[0], bool)
                else None
            )
            if requested_id != 573 or translator_source_tag is None:
                return None, (
                    "production_provenance_missing",
                    "production primitive 573 did not expose one exact translator "
                    f"source tag: function {requested_id}, arguments "
                    f"{requested_arguments!r}",
                )
            source_tag = translator_source_tag
            self._rtscortex_production_source_tag = source_tag
        if source_tag is None:
            return None, (
                "production_provenance_missing",
                "production provenance missing before translator dispatch",
            )
        return source_tag, None

    def _reject_unavailable_production_action(
        self,
        action: Mapping[str, Any],
        obs: Any,
    ) -> bool:
        action_name = str(action.get("name", ""))
        if not is_production_action(action_name):
            return False
        source_tag = production_source_tag(
            obs.observation,
            action,
            unit_names=self.unit_names,
            action_source_types=self.broker.extractor.action_source_types,
        )
        if source_tag is not None:
            if production_spec(action_name) is not None:
                self._rtscortex_production_source_tag = source_tag
            return False
        self._settle_production_source_failure(
            action_name,
            obs,
            f"{action_name} has no currently legal production source",
        )
        if self.action_list:
            self.action_list.pop(0)
        return True

    def _settle_production_source_failure(
        self,
        action_name: str,
        obs: Any,
        reason: str,
    ) -> None:
        self._settle_production_command_failure(
            action_name,
            obs,
            failure_code="production_source_unavailable",
            reason=reason,
        )

    def _settle_production_command_failure(
        self,
        action_name: str,
        obs: Any,
        *,
        failure_code: str,
        reason: str,
    ) -> None:
        dispatch = self.broker.reject_command(
            self.name,
            _execution_team_name(self),
            action_name,
            failure_code=failure_code,
        )
        if dispatch is None:
            self.broker.raise_unattributed_integrity(
                f"no command owns unavailable production action {action_name!r}"
            )
        self.broker.settle_primitive(
            dispatch,
            success=False,
            failure_reason=reason,
            game_loop=_observation_game_loop(obs.observation),
        )
        self.func_list.clear()
        self._rtscortex_semantic_action = None
        self._rtscortex_production_source_tag = None
        self._rtscortex_production_camera_waits = 0
        self._rtscortex_production_camera_wait_loop = None


class RTSCortexMainAgent(_MainAgentBase):  # type: ignore[misc]
    """Keep the upstream environment loop and attach RTSCortex at its query seam."""

    def __init__(self) -> None:
        _require_upstream()
        self.worker_settings = WorkerSettings.from_environment()
        self._frame_publisher: Optional[RGBFramePublisher] = None
        self.runtime_client = RuntimeClient(
            base_url=self.worker_settings.runtime_url,
            unix_socket=self.worker_settings.socket_path,
            timeout_seconds=self.worker_settings.runtime_request_timeout_seconds,
        )
        self.runtime_client.health()
        unit_names, building_types = _unit_metadata()
        upgrade_names = _upgrade_metadata()
        coordinator = BridgeCoordinator(
            self.runtime_client,
            effect_verifier=ActionEffectVerifier(
                timeout_game_loops=self.worker_settings.action_effect_timeout_game_loops,
                unit_names=unit_names,
            ),
        )
        extractor = TimeStepExtractor(
            self.worker_settings.run_id,
            self.worker_settings.episode_id,
            unit_names=unit_names,
            upgrade_names=upgrade_names,
            building_types=building_types,
            action_source_types=_production_action_source_types(),
        )
        self.decision_broker = SharedDecisionBroker(
            coordinator,
            extractor,
            metrics_path=os.environ.get("RTSCORTEX_WORKER_METRICS_PATH"),
        )
        self._pending_primitive: Optional[PrimitiveDispatch] = None
        self._pending_primitive_agent: Optional[Any] = None
        self.transport_noop_primitives = 0
        self._observation_watchdog_active = False
        self._observation_watchdog_preempted = False
        self._observation_watchdog_baseline_loop: Optional[int] = None
        self._rtscortex_force_runtime_decision = False
        self._episode_reported = False
        self.initial_planning_barrier = InitialPlanningBarrier()
        self.game_clock = (
            FixedRateGameClock(self.worker_settings.simulation_speed_multiplier)
            if self.worker_settings.simulation_speed_multiplier is not None
            else None
        )

        config = _scenario_config(self.worker_settings.scenario)
        subagent = partial(
            RTSCortexLLMAgent,
            broker=self.decision_broker,
            unit_names=unit_names,
        )
        super().__init__(config, subagent)
        self._rtscortex_accept_visible_team_unit = True
        _apply_scenario_bootstrap(self, self.worker_settings.scenario)
        if self.worker_settings.console_enabled:
            self._frame_publisher = RGBFramePublisher(
                uploader=RuntimeFrameUploader(
                    run_id=self.worker_settings.run_id,
                    episode_id=self.worker_settings.episode_id,
                    base_url=self.worker_settings.runtime_url,
                    unix_socket=self.worker_settings.socket_path,
                ),
                frame_fps=self.worker_settings.console_frame_fps,
                jpeg_quality=self.worker_settings.console_jpeg_quality,
            )

    def step(self, obs: Any) -> Any:
        try:
            return self._step(obs)
        except Exception as error:
            try:
                self._report_error_episode(error)
            finally:
                self._close_frame_publisher()
            raise

    def _step(self, obs: Any) -> Any:
        self._submit_console_frame(obs)
        self._settle_previous_primitive(obs)
        self.decision_broker.observe_effects(obs.observation)
        if _is_terminal(obs):
            return _finish_terminal(self, obs, _base_agent_step, _no_op)
        observation_loop = _observation_game_loop(obs.observation)
        # RTSCortex consumes a global raw snapshot. Do not make every Runtime tick
        # wait for optional upstream camera/selection-based text gathering.
        _release_runtime_observation_barrier(self)
        watchdog_active = self._update_observation_gap_watchdog(observation_loop)
        worker_management_blocked = (
            self.decision_broker.coordinator.effect_verifier.blocks_auto_worker_management
            or watchdog_active
        )
        _prime_deterministic_gas_rebalance(
            self,
            obs.observation,
            blocked=worker_management_blocked,
        )
        upstream_step = super().step
        action = _run_with_auto_worker_management_guard(
            self.config,
            blocked=worker_management_blocked,
            upstream_step=lambda: upstream_step(obs),
        )
        self._consume_execution_aborts(obs)
        if getattr(action, "function", None) == 0:
            self.transport_noop_primitives += 1
        if (
            self.worker_settings.pause_until_first_plan
            and self.initial_planning_barrier.blocks_steps
            and self.decision_broker.initial_decision_started
        ):
            self.decision_broker.wait_for_initial_decision(
                self.worker_settings.runtime_request_timeout_seconds + 5.0
            )
            self.initial_planning_barrier.release()
            if self.game_clock is not None:
                self.game_clock.reset()
        self._capture_primitive(action, obs)
        delay = _pending_plan_idle_delay(
            action,
            planner_pending=self.decision_broker.planner_pending,
            configured_delay_seconds=self.worker_settings.pending_plan_step_delay_seconds,
        )
        if self.game_clock is not None:
            self.game_clock.wait_for_step()
        elif delay:
            time.sleep(delay)
        return action

    def _submit_console_frame(self, obs: Any) -> None:
        publisher = self._frame_publisher
        if publisher is None:
            return
        try:
            publisher.submit(
                obs.observation,
                step_id=int(self.steps),
                game_loop=_observation_game_loop(obs.observation),
            )
        except Exception:
            # Console telemetry is best-effort and cannot fail an environment step.
            return

    def _close_frame_publisher(self) -> None:
        publisher = self._frame_publisher
        if publisher is None:
            return
        self._frame_publisher = None
        publisher.close()

    def _consume_execution_aborts(self, obs: Any) -> None:
        for agent in self.agents.values():
            abort = getattr(agent, "last_execution_abort", None)
            if not isinstance(abort, Mapping):
                continue
            agent.last_execution_abort = None
            team_name = str(abort.get("team_name") or "")
            action_name = str(abort.get("action_name") or "")
            failure_code = str(abort.get("failure_code") or "actor_not_available")
            if not team_name or not action_name:
                self.decision_broker.raise_unattributed_integrity(
                    "upstream abort lacks command identity"
                )
            dispatch = self.decision_broker.reject_command(
                agent.name,
                team_name,
                action_name,
                failure_code=failure_code,
            )
            if dispatch is None:
                self.decision_broker.raise_unattributed_integrity(
                    "upstream aborted an action with no unique command: "
                    f"{agent.name}/{team_name}/{action_name}"
                )
            actor_tag = abort.get("actor_tag")
            tag_detail = "" if actor_tag is None else f" (tag {hex(int(actor_tag))})"
            reason = str(abort.get("failure_reason") or "actor is unavailable")
            self.decision_broker.settle_primitive(
                dispatch,
                success=False,
                failure_reason=(f"{reason}: {agent.name}/{team_name}{tag_detail}"),
                game_loop=_observation_game_loop(obs.observation),
            )

    def _settle_previous_primitive(self, obs: Any) -> None:
        dispatch = self._pending_primitive
        if dispatch is None:
            return
        action_results = list(getattr(obs.observation, "action_result", ()))
        failure_reason = None
        if action_results:
            failure_reason = ", ".join(
                f"PySC2 action result {int(value)}" for value in action_results
            )
            if dispatch.origin == "translator":
                dispatch = replace(
                    dispatch,
                    final_primitive=True,
                    failure_code="pysc2_rejected",
                )
                if self._pending_primitive_agent is not None:
                    self._pending_primitive_agent.func_list.clear()
                    route = getattr(
                        self._pending_primitive_agent,
                        "_rtscortex_active_build_route",
                        None,
                    )
                    if route is not None:
                        action_name, position, world_target = route
                        self._pending_primitive_agent._rtscortex_rejected_build_positions.setdefault(
                            action_name, set()
                        ).add(position)
                        if world_target is not None:
                            self._pending_primitive_agent._rtscortex_rejected_build_targets.setdefault(
                                action_name, set()
                            ).add(world_target)
        self.decision_broker.settle_primitive(
            dispatch,
            success=not action_results,
            failure_reason=failure_reason,
            game_loop=_observation_game_loop(obs.observation),
        )
        self._pending_primitive = None
        if self._pending_primitive_agent is not None:
            self._pending_primitive_agent._rtscortex_active_build_route = None
        self._pending_primitive_agent = None

    def _capture_primitive(self, action: Any, obs: Any) -> None:
        if self._pending_primitive is not None or not self.AGENT_NAMES:
            return
        agent = self.agents[self.AGENT_NAMES[self.agent_id]]
        if bool(getattr(agent, "_rtscortex_camera_settlement_noop", False)):
            agent._rtscortex_camera_settlement_noop = False
            return
        attempt = getattr(agent, "_rtscortex_translation_attempt", None)
        function_id = getattr(action, "function", None)
        if isinstance(attempt, Mapping) and function_id is not None:
            agent._rtscortex_translation_attempt = None
            dispatch = attempt["dispatch"]
            if int(function_id) != int(attempt["emitted_function_id"]):
                self.decision_broker.fail_dispatch_integrity(
                    dispatch,
                    "translator primitive did not match the action returned by MainAgent",
                    game_loop=_observation_game_loop(obs.observation),
                )
            if "expected_arguments" in attempt:
                expected_arguments = _canonical_pysc2_arguments(
                    int(function_id),
                    attempt["expected_arguments"],
                )
                actual_arguments = _canonical_pysc2_arguments(
                    int(function_id),
                    getattr(action, "arguments", ()),
                )
                if actual_arguments != expected_arguments:
                    reason = (
                        "MainAgent changed translator arguments before PySC2 dispatch: "
                        f"expected {expected_arguments!r}, received {actual_arguments!r}"
                    )
                    if bool(attempt.get("candidate_constrained", False)):
                        self.decision_broker.reject_candidate_outside_dispatch(
                            dispatch,
                            reason,
                            game_loop=_observation_game_loop(obs.observation),
                        )
                    self.decision_broker.fail_dispatch_integrity(
                        dispatch,
                        reason,
                        game_loop=_observation_game_loop(obs.observation),
                    )
            if dispatch.final_primitive:
                self.decision_broker.prepare_effect(
                    dispatch,
                    obs.observation,
                    builder_tag=_execution_unit_tag(agent),
                    producer_tag=attempt.get("producer_tag"),
                )
            self._pending_primitive = dispatch
            self._pending_primitive_agent = agent
            return
        team_name = _execution_team_name(agent)
        action_name = getattr(agent, "curr_action_name", "")
        if not action_name or function_id is None:
            return
        action_specification = getattr(agent.translator_a, "ACTION_SPACE_DICT", {}).get(action_name)
        if action_specification is None:
            return
        function_name = _function_name(int(function_id))
        dispatch = self.decision_broker.claim_primitive(
            agent.name,
            team_name,
            str(action_name),
            function_name,
            final_primitive=False,
            origin="orchestration",
            requested_function_id=int(function_id),
            emitted_function_id=int(function_id),
        )
        if dispatch is None:
            return
        self._pending_primitive = dispatch
        self._pending_primitive_agent = agent

    def _update_observation_gap_watchdog(self, game_loop: int) -> bool:
        """Preempt optional upstream automation when Runtime observations stall."""

        last_decision_loop = self.decision_broker.last_decision_game_loop
        if last_decision_loop is None:
            return False
        if (
            self._observation_watchdog_active
            and self._observation_watchdog_baseline_loop is not None
            and last_decision_loop > self._observation_watchdog_baseline_loop
        ):
            self._observation_watchdog_active = False
            self._observation_watchdog_baseline_loop = None
        gap = game_loop - last_decision_loop
        if (
            not self._observation_watchdog_preempted
            and gap > self.worker_settings.observation_gap_watchdog_game_loops
        ):
            self._observation_watchdog_preempted = True
            self._observation_watchdog_active = True
            self._observation_watchdog_baseline_loop = last_decision_loop
            self._rtscortex_force_runtime_decision = True
            self.decision_broker.record_observation_gap_watchdog_trigger()
        if (
            self._observation_watchdog_preempted
            and gap > self.worker_settings.observation_gap_hard_limit_game_loops
        ):
            raise RuntimeError(
                "observation_gap_watchdog_timeout: no Runtime decision for "
                f"{gap} game loops (last decision loop {last_decision_loop})"
            )
        return self._observation_watchdog_active

    def _report_episode(self, obs: Any) -> None:
        reward = float(getattr(obs, "reward", 0.0) or 0.0)
        outcome = "victory" if reward > 0 else "defeat" if reward < 0 else "draw"
        self.decision_broker.end_episode(
            {
                "protocol_version": "1.1",
                "run_id": self.worker_settings.run_id,
                "episode_id": self.worker_settings.episode_id,
                "scenario": self.worker_settings.scenario,
                "seed": self.worker_settings.seed,
                "outcome": outcome,
                "score": reward,
                "steps": int(self.steps),
                "metrics": {
                    "transport_noop_primitives": self.transport_noop_primitives,
                    **self.decision_broker.metrics(),
                },
                "failure_reason": None,
            }
        )
        self._episode_reported = True

    def _report_error_episode(self, error: Exception) -> None:
        if self._episode_reported:
            return
        result = {
            "protocol_version": "1.1",
            "run_id": self.worker_settings.run_id,
            "episode_id": self.worker_settings.episode_id,
            "scenario": self.worker_settings.scenario,
            "seed": self.worker_settings.seed,
            "outcome": "error",
            "score": 0.0,
            "steps": int(self.steps),
            "metrics": {
                "transport_noop_primitives": self.transport_noop_primitives,
                **self.decision_broker.metrics(),
            },
            "failure_reason": f"{type(error).__name__}: {error}",
        }
        try:
            self.decision_broker.end_episode(result)
        except Exception:
            return
        self._episode_reported = True

    def on_episode_truncated(self, total_frames: int) -> None:
        """Report an explicit terminal result when PySC2 reaches its frame limit."""

        if self._episode_reported:
            return
        result = {
            "protocol_version": "1.1",
            "run_id": self.worker_settings.run_id,
            "episode_id": self.worker_settings.episode_id,
            "scenario": self.worker_settings.scenario,
            "seed": self.worker_settings.seed,
            "outcome": "truncated",
            "score": 0.0,
            "steps": int(total_frames),
            "metrics": {
                "transport_noop_primitives": self.transport_noop_primitives,
                **self.decision_broker.metrics(),
            },
            "failure_reason": "max_agent_steps_reached",
        }
        try:
            self.decision_broker.end_episode(result)
            self._episode_reported = True
        finally:
            self._close_frame_publisher()
            self.runtime_client.close()


def _require_upstream() -> None:
    if _UPSTREAM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LLM-PySC2 is unavailable; install the pinned submodule in the worker environment"
        ) from _UPSTREAM_IMPORT_ERROR


def _scenario_config(scenario: str) -> Any:
    definitions = {
        "pvz_task1_level1": (
            "llm_pysc2.agents.configs.llm_pysc2",
            "ConfigPysc2_Harass",
        ),
        "2s3z": ("llm_pysc2.agents.configs.llm_smac", "ConfigSmac_2s3z"),
        "Simple64": ("rtscortex_llm_pysc2.melee", "RTSCortexMeleeConfig"),
    }
    try:
        module_name, class_name = definitions[scenario]
    except KeyError as error:
        supported = ", ".join(sorted(definitions))
        raise ValueError(
            f"unsupported worker scenario {scenario!r}; supported scenarios: {supported}"
        ) from error
    module = importlib.import_module(module_name)
    config = getattr(module, class_name)()
    config.reset_llm(
        model_name="gpt-3.5-turbo",
        api_base="http://127.0.0.1",
        api_key="rtscortex-unused",
    )
    if scenario == "pvz_task1_level1":
        for team in config.AGENTS["CombatGroup7"]["team"]:
            team["task"] = [
                {
                    "time": None,
                    "pos": [52, 32],
                    "info": "Reach minimap [52, 32] while avoiding detection and attacks.",
                },
                {
                    "time": None,
                    "pos": None,
                    "info": "Destroy as many enemy workers as possible.",
                },
            ]
    actions = importlib.import_module("pysc2.lib.actions")
    _ensure_no_operation(config, actions.FUNCTIONS.no_op)
    return config


def _ensure_no_operation(config: Any, no_op_function: Any) -> None:
    """Make the bridge's declared fallback a real upstream action."""

    for agent in config.AGENTS.values():
        action_space = agent["action"]
        for unit_type, unit_actions in list(action_space.items()):
            if any(action["name"] == "No_Operation" for action in unit_actions):
                continue
            action_space[unit_type] = [
                {
                    "name": "No_Operation",
                    "arg": [],
                    "func": [(0, no_op_function, ())],
                },
                *list(unit_actions),
            ]


def _apply_scenario_bootstrap(agent: Any, scenario: str) -> None:
    """Skip strict centering that the small SMAC arena cannot satisfy."""

    if scenario != "2s3z":
        return
    agent.world_xy_calibration = True


def _unit_metadata() -> tuple[dict[int, str], tuple[int, ...]]:
    units = importlib.import_module("pysc2.lib.units")
    names = {}
    for race in (units.Neutral, units.Protoss, units.Terran, units.Zerg):
        names.update({int(value): value.name for value in race})
    utils = importlib.import_module("llm_pysc2.lib.utils")
    return names, tuple(int(value) for value in utils.BUILDING_TYPE)


def _upgrade_metadata() -> dict[int, str]:
    upgrades = importlib.import_module("pysc2.lib.upgrades")
    return {int(value): value.name for value in upgrades.Upgrades}


def _production_action_source_types() -> dict[int, int]:
    actions = importlib.import_module("pysc2.lib.actions")
    llm_action = importlib.import_module("llm_pysc2.lib.llm_action")
    result: dict[int, int] = {}
    for function_id in range(len(actions.FUNCTIONS)):
        source_type = llm_action.find_unit_type_the_func_belongs_to(function_id, "protoss")
        if source_type is not None:
            result[function_id] = int(source_type)
    return result


def _function_name(function_id: int) -> str:
    actions = importlib.import_module("pysc2.lib.actions")
    return str(actions.FUNCTIONS[function_id].name)


def _pending_plan_idle_delay(
    action: Any,
    *,
    planner_pending: bool,
    configured_delay_seconds: float,
) -> float:
    function_id = getattr(action, "function", None)
    if planner_pending and function_id == 0:
        return configured_delay_seconds
    return 0.0


def _run_with_auto_worker_management_guard(
    config: Any,
    *,
    blocked: bool,
    upstream_step: Callable[[], Any],
) -> Any:
    """Prevent upstream automation from reassigning an accepted build's worker."""

    if not blocked:
        return upstream_step()
    worker_management_enabled = config.ENABLE_AUTO_WORKER_MANAGE
    worker_training_enabled = config.ENABLE_AUTO_WORKER_TRAINING
    config.ENABLE_AUTO_WORKER_MANAGE = False
    config.ENABLE_AUTO_WORKER_TRAINING = False
    try:
        return upstream_step()
    finally:
        config.ENABLE_AUTO_WORKER_MANAGE = worker_management_enabled
        config.ENABLE_AUTO_WORKER_TRAINING = worker_training_enabled


def _prime_deterministic_gas_rebalance(
    main_agent: Any,
    observation: Any,
    *,
    blocked: bool,
) -> bool:
    """Choose one exact mineral worker when a completed gas slot is undersaturated."""

    reserved_builder_tags = _reserved_builder_worker_tags(main_agent)
    main_agent._rtscortex_reserved_worker_tags = reserved_builder_tags
    if (
        blocked
        or not bool(getattr(main_agent.config, "ENABLE_AUTO_WORKER_MANAGE", False))
        or bool(getattr(main_agent, "main_loop_lock", False))
        or getattr(main_agent, "stop_worker", None) is not None
        or getattr(main_agent, "stop_worker_nexus_tag", None) is not None
    ):
        return False
    raw_by_tag = {
        int(getattr(unit, "tag", 0)): unit
        for unit in getattr(observation, "raw_units", ())
        if int(getattr(unit, "tag", 0)) > 0
    }
    harvest_order_ids = {102, 103, 154, 356, 357, 358, 359, 360, 361, 362}
    nexus_info_dict = getattr(main_agent, "nexus_info_dict", {})
    for _nexus_key, info in sorted(nexus_info_dict.items(), key=lambda item: int(item[0])):
        nexus = info.get("nexus")
        if nexus is None:
            continue
        for gas_slot, worker_tags in (
            ("g1", info.get("worker_g1_tag_list", ())),
            ("g2", info.get("worker_g2_tag_list", ())),
        ):
            reserved_on_gas = sorted(
                reserved_builder_tags.intersection(int(tag) for tag in worker_tags)
            )
            for worker_tag in reserved_on_gas:
                worker = raw_by_tag.get(worker_tag)
                if worker is None:
                    continue
                main_agent.stop_worker_nexus_tag = int(nexus.tag)
                main_agent.stop_worker_at = gas_slot
                main_agent.stop_worker = worker
                return True

    choices: list[tuple[float, int, int, Any]] = []
    for _nexus_key, info in sorted(nexus_info_dict.items(), key=lambda item: int(item[0])):
        nexus = info.get("nexus")
        if nexus is None:
            continue
        undersaturated = [
            gas
            for gas, workers in (
                (info.get("gas_building_1"), info.get("worker_g1_tag_list", ())),
                (info.get("gas_building_2"), info.get("worker_g2_tag_list", ())),
            )
            if gas is not None
            and float(getattr(gas, "build_progress", 0.0)) in {1.0, 100.0}
            and len(workers) < 3
        ]
        if not undersaturated:
            continue
        target = min(undersaturated, key=lambda gas: int(gas.tag))
        for worker_tag in sorted(set(info.get("worker_m_tag_list", ()))):
            if int(worker_tag) in reserved_builder_tags:
                continue
            worker = raw_by_tag.get(int(worker_tag))
            if worker is None:
                continue
            if int(getattr(worker, "order_id_0", 356)) not in harvest_order_ids:
                continue
            distance = (float(worker.x) - float(target.x)) ** 2 + (
                float(worker.y) - float(target.y)
            ) ** 2
            choices.append((distance, int(worker.tag), int(nexus.tag), worker))
    if not choices:
        return False
    _, _, nexus_tag, worker = min(choices)
    main_agent.stop_worker_nexus_tag = nexus_tag
    main_agent.stop_worker_at = "m"
    main_agent.stop_worker = worker
    return True


def _reserved_builder_worker_tags(main_agent: Any) -> set[int]:
    agents = getattr(main_agent, "agents", {})
    if not isinstance(agents, Mapping):
        return set()
    builder = agents.get("Builder")
    if builder is None:
        return set()

    tags: set[int] = set()
    current = getattr(builder, "team_unit_tag_curr", None)
    if current is not None and int(current) > 0:
        tags.add(int(current))
    for attribute in ("unit_tag_list", "team_unit_tag_list"):
        tags.update(int(tag) for tag in getattr(builder, attribute, ()) if int(tag) > 0)
    for team in getattr(builder, "teams", ()):
        if not isinstance(team, Mapping):
            continue
        tags.update(int(tag) for tag in team.get("unit_tags", ()) if int(tag) > 0)
    return tags


def _release_runtime_observation_barrier(main_agent: Any) -> None:
    """Let current raw state reach Runtime when optional team selection stalls."""

    disappeared = {int(tag) for tag in getattr(main_agent, "unit_uid_disappear", ())}
    agents = getattr(main_agent, "agents", {})
    if not isinstance(agents, Mapping):
        return
    for agent in agents.values():
        if not getattr(agent, "enable", False) or not agent._is_waiting_query():
            continue
        observed_tags = getattr(agent, "team_unit_tag_list", None)
        if not isinstance(observed_tags, list):
            continue
        observed_teams = getattr(agent, "team_unit_team_list", None)
        for team in getattr(agent, "teams", ()):
            if not isinstance(team, Mapping):
                continue
            live_tags = [
                int(tag) for tag in team.get("unit_tags", ()) if int(tag) not in disappeared
            ]
            required_tags = live_tags if team.get("select_type") == "select" else live_tags[:1]
            for tag in required_tags:
                if tag in observed_tags:
                    continue
                observed_tags.append(tag)
                if isinstance(observed_teams, list):
                    observed_teams.append(str(team.get("name", "")))


def _translated_build_position(
    action_name: str,
    arguments: Any,
) -> Optional[list[int]]:
    """Return the actual final screen target emitted by an upstream build translator."""

    if not action_name.startswith("Build_") or not isinstance(arguments, (list, tuple)):
        return None
    for value in reversed(arguments):
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(coordinate, Integral) and not isinstance(coordinate, bool)
                for coordinate in value
            )
        ):
            return [int(value[0]), int(value[1])]
    return None


def _execution_team_name(agent: Any) -> Optional[str]:
    """Recover the implicit Empty actor omitted by the upstream execution loop."""

    team_name = getattr(agent, "team_unit_team_curr", None)
    if team_name is not None:
        return str(team_name)
    unit_tag = getattr(agent, "team_unit_tag_curr", None)
    unit_tags = getattr(agent, "team_unit_tag_list", ())
    if getattr(agent, "flag_enable_empty_unit_group", False) and unit_tag is None and not unit_tags:
        return "Empty"
    return None


def _execution_unit_tag(agent: Any) -> Optional[int]:
    value = getattr(agent, "team_unit_tag_curr", None)
    return None if value is None else int(value)


def _production_source_invalid_reason(
    observation: Any,
    source_tag: Optional[int],
    spec: ProductionSpec,
    unit_names: Mapping[int, str],
) -> Optional[str]:
    if source_tag is None:
        return f"{spec.action_name} producer provenance is unavailable at final dispatch"
    raw_units = (
        observation.get("raw_units", ())
        if isinstance(observation, Mapping)
        else getattr(observation, "raw_units", ())
    )
    source = next(
        (
            unit
            for unit in raw_units
            if int(unit.get("tag", -1) if isinstance(unit, Mapping) else getattr(unit, "tag", -1))
            == source_tag
        ),
        None,
    )
    if source is None:
        return f"{spec.action_name} producer {hex(source_tag)} disappeared before final dispatch"

    def value(name: str, default: Any) -> Any:
        return (
            source.get(name, default)
            if isinstance(source, Mapping)
            else getattr(source, name, default)
        )

    unit_type = value("unit_type", "")
    source_name = (
        str(unit_type)
        if isinstance(unit_type, str)
        else unit_names.get(int(unit_type), f"unit:{int(unit_type)}")
    )
    if int(value("alliance", 0)) != 1 or source_name != spec.producer_type:
        return (
            f"{spec.action_name} producer {hex(source_tag)} changed identity to "
            f"{source_name!r} alliance {int(value('alliance', 0))}"
        )
    progress = float(value("build_progress", 0.0))
    normalized_progress = progress / 100.0 if progress > 1.0 else progress
    if normalized_progress < 1.0:
        return f"{spec.action_name} producer {hex(source_tag)} is no longer complete"
    if int(value("active", 0)) != 0 or int(value("order_length", 0)) != 0:
        return f"{spec.action_name} producer {hex(source_tag)} became busy before final dispatch"
    return None


def _producer_is_visible(
    observation: Any,
    producer_tag: int,
    *,
    size_screen: int,
) -> bool:
    feature_units = (
        observation.get("feature_units", ())
        if isinstance(observation, Mapping)
        else getattr(observation, "feature_units", ())
    )
    for unit in feature_units:
        tag = unit.get("tag", 0) if isinstance(unit, Mapping) else getattr(unit, "tag", 0)
        alliance = (
            unit.get("alliance", 0) if isinstance(unit, Mapping) else getattr(unit, "alliance", 0)
        )
        is_on_screen = (
            unit.get("is_on_screen", True)
            if isinstance(unit, Mapping)
            else getattr(unit, "is_on_screen", True)
        )
        x = unit.get("x", 0) if isinstance(unit, Mapping) else getattr(unit, "x", 0)
        y = unit.get("y", 0) if isinstance(unit, Mapping) else getattr(unit, "y", 0)
        if (
            int(tag) == producer_tag
            and int(alliance) == 1
            and bool(is_on_screen)
            and 0 < float(x) < size_screen
            and 0 < float(y) < size_screen
        ):
            return True
    return False


def _visible_feature_position(
    observation: Any,
    unit_tag: Optional[int],
) -> Optional[tuple[int, int]]:
    """Return the current screen point for one visible friendly feature unit."""

    if unit_tag is None:
        return None
    feature_units = _observation_value(observation, "feature_units", ())
    for unit in feature_units:
        if int(_observation_value(unit, "tag", 0)) != unit_tag:
            continue
        if int(_observation_value(unit, "alliance", 0)) != 1:
            return None
        if not bool(_observation_value(unit, "is_on_screen", True)):
            return None
        return (
            int(_observation_value(unit, "x", 0)),
            int(_observation_value(unit, "y", 0)),
        )
    return None


def _single_position(arguments: Any) -> Optional[tuple[float, float]]:
    if not isinstance(arguments, (list, tuple)):
        return None
    for value in arguments:
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
                for coordinate in value
            )
        ):
            return float(value[0]), float(value[1])
    return None


def _observation_game_loop(observation: Any) -> int:
    value: Any = (
        observation.get("game_loop", 0)
        if isinstance(observation, Mapping)
        else getattr(observation, "game_loop", 0)
    )
    if isinstance(value, (str, bytes)):
        return int(value)
    try:
        if len(value) == 1:
            return int(value[0])
    except (TypeError, IndexError):
        pass
    return int(value)


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be 'true' or 'false'")


def _refresh_build_action_position(action: dict[str, Any], observation: Any) -> bool:
    return _resolve_build_action_position(action, observation) is not None


def _resolve_build_action_position(
    action: dict[str, Any],
    observation: Any,
    *,
    world_target: Optional[tuple[float, float]] = None,
    preferred_anchor_tag: Optional[int] = None,
    excluded_positions: set[tuple[int, int]] | None = None,
    force_resample: bool = False,
) -> Optional[list[int]]:
    action_name = str(action.get("name", ""))
    requested = _screen_argument(action)
    if world_target is not None:
        position = resolve_screen_build_world_target(
            observation,
            action_name,
            world_target,
            preferred_anchor_tag=preferred_anchor_tag,
            excluded_positions=excluded_positions or set(),
            force_resample=force_resample,
        )
        if position is None:
            return None
    else:
        candidates = [
            candidate
            for candidate in build_screen_candidates(observation, action_name)
            if tuple(candidate) not in (excluded_positions or set())
        ]
        if not candidates:
            return None
        position = _nearest_current_build_candidate(observation, candidates, requested)
        if position is None:
            return None
    return _replace_screen_action_position(action, position)


def _resolve_translator_compatible_build_position(
    action: dict[str, Any],
    observation: Any,
    *,
    world_target: tuple[float, float],
    preferred_anchor_tag: Optional[int],
    excluded_positions: set[tuple[int, int]],
    force_resample: bool,
    screen_size: int,
) -> Optional[list[int]]:
    """Resample until the pinned translator accepts the same screen position."""

    action_name = str(action.get("name", ""))
    excluded = set(excluded_positions)
    attempts = len(build_screen_candidates(observation, action_name)) + 1
    for _ in range(attempts):
        position = _resolve_build_action_position(
            action,
            observation,
            world_target=world_target,
            preferred_anchor_tag=preferred_anchor_tag,
            excluded_positions=excluded,
            force_resample=force_resample,
        )
        if position is None:
            excluded_positions.update(excluded)
            return None
        if _translator_screen_build_is_legal(
            observation,
            position,
            screen_size,
            action_name,
        ):
            excluded_positions.update(excluded)
            return position
        excluded.add(_screen_position_key(position))
        force_resample = True
    excluded_positions.update(excluded)
    return None


def _translator_screen_build_is_legal(
    observation: Any,
    position: list[int],
    screen_size: int,
    action_name: str,
) -> bool:
    """Mirror the pinned upstream sampled-grid build argument contract."""

    spec = BUILD_SPECS.get(action_name)
    feature_screen = _observation_value(observation, "feature_screen", None)
    buildable = _observation_value(feature_screen, "buildable", None)
    pathable = _observation_value(feature_screen, "pathable", None)
    player_relative = _observation_value(feature_screen, "player_relative", None)
    power = _observation_value(feature_screen, "power", None)
    if (
        spec is None
        or spec.placement_kind != "screen"
        or screen_size <= 0
        or buildable is None
        or pathable is None
        or player_relative is None
    ):
        return False
    ratio = int(screen_size / 24)
    x0 = int(min(max(0, position[0]), screen_size))
    y0 = int(min(max(0, position[1]), screen_size))
    if spec.requires_power and (power is None or power[y0][x0] == 0):
        return False
    first_x = int(x0 - ratio * (spec.footprint - 1) / 2)
    first_y = int(y0 - ratio * (spec.footprint - 1) / 2)
    for x_index in range(spec.footprint):
        for y_index in range(spec.footprint):
            x = int(first_x + x_index * ratio)
            y = int(first_y + y_index * ratio)
            if not (0 < x < screen_size and 0 < y < screen_size):
                return False
            if buildable[y][x] != 1 or pathable[y][x] != 1:
                return False
            if player_relative[y][x] not in (0, 1):
                return False
    return True


def _build_world_target_key(world_target: tuple[float, float]) -> tuple[float, float]:
    return (round(world_target[0], 3), round(world_target[1], 3))


def _screen_position_key(position: list[int]) -> tuple[int, int]:
    return (int(position[0]), int(position[1]))


def _resolve_screen_point_action_position(
    action: dict[str, Any],
    observation: Any,
    *,
    world_target: tuple[float, float],
    preferred_anchor_tag: Optional[int],
    require_power: bool,
) -> Optional[list[int]]:
    position = resolve_screen_point_world_target(
        observation,
        str(action.get("name", "")),
        world_target,
        preferred_anchor_tag=preferred_anchor_tag,
        require_power=require_power,
    )
    if position is None:
        return None
    return _replace_screen_action_position(action, position)


def _replace_screen_action_position(
    action: dict[str, Any],
    position: list[int],
) -> list[int]:
    action_arguments = action.get("arg", ())
    if isinstance(action_arguments, list) and action_arguments:
        action_arguments[0] = position
    refreshed_functions = []
    for function_id, function, arguments in action.get("func", ()):
        refreshed_arguments = tuple(
            position
            if isinstance(argument, list)
            and len(argument) == 2
            and all(isinstance(value, (int, float)) for value in argument)
            else argument
            for argument in arguments
        )
        refreshed_functions.append((function_id, function, refreshed_arguments))
    action["func"] = refreshed_functions
    return position


def _next_primitive_is_screen_build(
    action_name: str,
    functions: list[Any],
) -> bool:
    spec = BUILD_SPECS.get(action_name)
    if spec is None or spec.placement_kind != "screen" or not functions:
        return False
    function = functions[0]
    if not isinstance(function, (list, tuple)) or len(function) != 3:
        return False
    return str(getattr(function[1], "name", "")).startswith("Build_")


def _isolate_next_action(action_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Detach one routed action from the upstream reusable action templates."""

    action = copy.deepcopy(action_list[0])
    action_list[0] = action
    return action


def _nearest_current_build_candidate(
    observation: Any,
    candidates: list[list[int]],
    requested: Optional[list[int]],
) -> Optional[list[int]]:
    if requested is None:
        return candidates[0]
    if requested in candidates:
        return requested
    feature_screen = _observation_value(observation, "feature_screen", None)
    buildable = _observation_value(feature_screen, "buildable", None)
    shape = getattr(buildable, "shape", ())
    stride = max(4, int(int(shape[0]) / 24)) if shape else 4
    ranked = sorted(
        (
            (candidate[0] - requested[0]) ** 2 + (candidate[1] - requested[1]) ** 2,
            candidate[0],
            candidate[1],
            candidate,
        )
        for candidate in candidates
    )
    if not ranked or ranked[0][0] > (2 * stride) ** 2:
        return None
    return ranked[0][3]


def _screen_argument(action: Mapping[str, Any]) -> Optional[list[int]]:
    arguments = action.get("arg", ())
    values = list(arguments) if isinstance(arguments, (list, tuple)) else []
    functions = action.get("func", ())
    if isinstance(functions, (list, tuple)):
        for triple in functions:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            function_arguments = triple[2]
            if isinstance(function_arguments, (list, tuple)):
                values.extend(function_arguments)
    for value in values:
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(coordinate, (int, float)) for coordinate in value)
        ):
            return [int(value[0]), int(value[1])]
    return None


def _semantic_target_failure(
    action: Mapping[str, Any],
    observation: Any,
    unit_names: Mapping[int, str],
) -> Optional[tuple[str, str]]:
    action_name = str(action.get("name", ""))
    if action_name not in {
        "Attack_Unit",
        *BUILD_SPECS,
        *MINIMAP_POINT_ACTIONS,
        *SCREEN_POINT_ACTIONS,
        SELECT_BLINK_ACTION,
    }:
        return None
    candidates = semantic_argument_candidates(
        observation,
        action_name,
        unit_names=unit_names,
    )
    if action_name in BUILD_SPECS and action_name.endswith("_Screen"):
        requested = _screen_argument(action)
        if requested is not None and screen_build_position_is_legal(
            observation,
            action_name,
            requested,
            unit_names=unit_names,
        ):
            return None
        return "no_legal_placement", f"{action_name} has no legal placement candidate"
    if action_name in SCREEN_POINT_ACTIONS | MINIMAP_POINT_ACTIONS:
        requested = _screen_argument(action)
        legal_positions = {
            tuple(int(coordinate) for coordinate in candidate[0])
            for candidate in candidates or []
            if len(candidate) == 1
            and isinstance(candidate[0], (list, tuple))
            and len(candidate[0]) == 2
        }
        if requested is not None and tuple(requested) in legal_positions:
            return None
        return (
            "candidate_invalidated",
            f"{action_name} arguments are outside the current candidate set",
        )
    if action_name == SELECT_BLINK_ACTION:
        target_tag = _tag_argument(action)
        requested = _screen_argument(action)
        if (
            target_tag is not None
            and requested is not None
            and any(
                candidate
                and len(candidate) == 2
                and int(candidate[0]) == target_tag
                and list(candidate[1]) == requested
                for candidate in candidates or []
            )
        ):
            return None
        return (
            "candidate_invalidated",
            f"{action_name} arguments are outside the current candidate set",
        )
    target_tag = _tag_argument(action)
    if target_tag is None:
        code = "target_not_visible" if action_name == "Attack_Unit" else "translator_rejected"
        return code, f"{action_name} has no valid tag argument"

    candidate_tags = {
        int(candidate[0], 0) if isinstance(candidate[0], str) else int(candidate[0])
        for candidate in candidates or []
        if candidate
    }
    if target_tag in candidate_tags:
        return None

    if action_name == "Attack_Unit":
        units = _observation_value(observation, "feature_units", ())
        target = next(
            (
                unit
                for unit in units
                if int(_observation_value(unit, "tag", -1)) == target_tag
                and bool(_observation_value(unit, "is_on_screen", True))
            ),
            None,
        )
        if target is None:
            return "target_not_visible", f"enemy target {hex(target_tag)} is not visible"
        if int(_observation_value(target, "alliance", 0)) != 4:
            return "friendly_target", f"target {hex(target_tag)} is not an enemy"
        return "target_not_visible", f"enemy target {hex(target_tag)} is not targetable"

    failure_code = (
        "invalid_geyser_tag"
        if action_name == "Build_Assimilator_Near"
        else "invalid_expansion_anchor"
    )
    return failure_code, f"semantic target {hex(target_tag)} is no longer legal"


def _candidate_dispatch_failure(
    action: Mapping[str, Any],
    observation: Any,
    unit_names: Mapping[int, str],
    *,
    final_primitive: bool,
    translated_position: Optional[list[int]],
) -> Optional[str]:
    """Return why an accepted primitive is outside its current semantic domain."""

    action_name = str(action.get("name", ""))
    if not _is_candidate_constrained_action(action_name):
        return None
    failure = _semantic_target_failure(action, observation, unit_names)
    if failure is not None:
        return f"{failure[1]}; accepted primitive would leave the current candidate set"
    spec = BUILD_SPECS.get(action_name)
    if spec is None or spec.placement_kind != "screen" or not final_primitive:
        return None
    if translated_position is not None and screen_build_position_is_legal(
        observation,
        action_name,
        translated_position,
        unit_names=unit_names,
    ):
        return None
    return (
        f"{action_name} translated to {translated_position!r}, outside the current "
        "legal placement domain"
    )


def _is_candidate_constrained_action(action_name: str) -> bool:
    return action_name in {
        "Attack_Unit",
        *BUILD_SPECS,
        *MINIMAP_POINT_ACTIONS,
        *SCREEN_POINT_ACTIONS,
        SELECT_BLINK_ACTION,
    }


def _normalize_pysc2_arguments(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        normalized = tuple(_normalize_pysc2_arguments(item) for item in value)
        return normalized[0] if len(normalized) == 1 else normalized
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return {
            "now": 0,
            "queued": 1,
            "select": 0,
            "toggle": 1,
            "select_all_type": 2,
            "add_all_type": 3,
        }.get(value, value)
    return value


def _canonical_pysc2_arguments(_function_id: int, arguments: Any) -> Any:
    """Normalize translator strings and PySC2 enum encodings to one representation."""

    return _normalize_pysc2_arguments(arguments)


def _tag_argument(action: Mapping[str, Any]) -> Optional[int]:
    values: list[Any] = []
    arguments = action.get("arg", ())
    if isinstance(arguments, (list, tuple)):
        values.extend(arguments)
    functions = action.get("func", ())
    if isinstance(functions, (list, tuple)):
        for triple in functions:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            function_arguments = triple[2]
            if isinstance(function_arguments, (list, tuple)):
                values.extend(function_arguments)
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.startswith("0x"):
            try:
                return int(value, 16)
            except ValueError:
                continue
    return None


def _observation_value(value: Any, name: str, default: Any) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _translation_failure_code(reason: Optional[str], action_name: str) -> Optional[str]:
    if reason is None:
        return None
    normalized = reason.casefold()
    if any(marker in normalized for marker in ("need power", "needs power", "requires power")):
        return "need_power"
    if "not pathable" in normalized:
        return "not_pathable"
    if "resource clearance" in normalized and "Nexus" in action_name:
        return "invalid_expansion_anchor"
    if "not buildable" in normalized or "not blocked" in normalized:
        return "blocked"
    if "is alliance" in normalized:
        return "friendly_target"
    if "cannot find unit" in normalized or "not found" in normalized:
        if action_name == "Attack_Unit":
            return "target_not_visible"
        if "Assimilator" in action_name:
            return "invalid_geyser_tag"
        if "Nexus" in action_name:
            return "invalid_expansion_anchor"
    if "not available" in normalized or "function" in normalized:
        return "translator_rejected"
    return "translator_rejected"


def _upstream_replaced_production_with_noop(
    semantic_action_name: str,
    translation_result: Mapping[str, Any],
) -> bool:
    return (
        is_production_action(semantic_action_name)
        and str(translation_result.get("action_name", "")) == "No_Operation"
        and int(translation_result.get("requested_function_id", -1)) == 0
        and bool(translation_result.get("accepted", False))
    )


def _finish_terminal(
    agent: Any,
    obs: Any,
    base_step: Callable[[Any, Any], None],
    no_op: Callable[[], Any],
) -> Any:
    """Finalize an episode without re-entering the upstream decision loop."""

    base_step(agent, obs)
    if not agent._episode_reported:
        try:
            agent._report_episode(obs)
        finally:
            try:
                close_frames = getattr(agent, "_close_frame_publisher", None)
                if callable(close_frames):
                    close_frames()
            finally:
                agent.runtime_client.close()
    return no_op()


def _base_agent_step(agent: Any, obs: Any) -> None:
    module = importlib.import_module("pysc2.agents.base_agent")
    module.BaseAgent.step(agent, obs)


def _no_op() -> Any:
    actions = importlib.import_module("pysc2.lib.actions")
    return actions.FUNCTIONS.no_op()


def _is_terminal(obs: Any) -> bool:
    last = getattr(obs, "last", None)
    return bool(last()) if callable(last) else False
