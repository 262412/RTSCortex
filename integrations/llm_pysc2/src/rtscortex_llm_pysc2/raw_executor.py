"""Direct PySC2 Raw Action execution for RTSCortex-owned commands."""

from __future__ import annotations

import importlib
import inspect
import math
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.broker import PrimitiveDispatch, SharedDecisionBroker
from rtscortex_llm_pysc2.coordinator import BridgeDecision
from rtscortex_llm_pysc2.extractor import (
    BUILD_RAW_FUNCTION_IDS,
    BUILD_SPECS,
    production_dispatch_failure,
    raw_build_eligibility,
)
from rtscortex_llm_pysc2.observation import split_actor
from rtscortex_llm_pysc2.production import production_spec
from rtscortex_llm_pysc2.raw_placement import (
    RawPlacementFailure,
    RawPlacementService,
    _placement_revision,
)
from rtscortex_llm_pysc2.research import research_spec
from rtscortex_llm_pysc2.routing import RoutedCommand

_BUILD_RAW_FUNCTIONS = {
    "Build_Pylon_Screen": "Build_Pylon_pt",
    "Build_Gateway_Screen": "Build_Gateway_pt",
    "Build_Forge_Screen": "Build_Forge_pt",
    "Build_CyberneticsCore_Screen": "Build_CyberneticsCore_pt",
    "Build_ShieldBattery_Screen": "Build_ShieldBattery_pt",
    "Build_Stargate_Screen": "Build_Stargate_pt",
    "Build_Nexus_Near": "Build_Nexus_pt",
    "Build_Assimilator_Near": "Build_Assimilator_unit",
}
_CONTROL_RAW_FUNCTIONS = {
    "Stop": "Stop_quick",
    "Hold_Position": "HoldPosition_quick",
}
_WARP_RAW_FUNCTIONS = {
    "Warp_Zealot_Near": "TrainWarp_Zealot_pt",
    "Warp_Stalker_Near": "TrainWarp_Stalker_pt",
}
_BUILDER_THREAT_CLEARANCE = 3.0
_NONSPATIAL_BUILD_FAILURE_CODES = frozenset(
    {
        "build_insufficient_minerals",
        "build_insufficient_vespene",
        "build_missing_prerequisite",
        "operation_no_start_circuit_open",
    }
)
_BUILDER_DEFERRAL_CODES = frozenset(
    {
        "actor_not_available",
        "builder_not_ready",
        "builder_unavailable",
        "placement_query_rejected",
        "stale_observation",
    }
)


@dataclass(frozen=True)
class RawDispatch:
    """One exact command-to-raw-primitive transition."""

    command: RoutedCommand
    primitive: PrimitiveDispatch
    action: Any
    actor_tags: tuple[int, ...]
    builder_tag: Optional[int] = None
    producer_tag: Optional[int] = None
    resolved_arguments: tuple[Any, ...] = ()
    reservation_id: Optional[str] = None
    placement_revision: Optional[str] = None


class RawActionExecutor:
    """Translate validated RTSCortex commands without UI selection or camera state."""

    def __init__(
        self,
        broker: SharedDecisionBroker,
        *,
        unit_names: Mapping[int, str],
        placement_service: Optional[RawPlacementService] = None,
    ) -> None:
        self.broker = broker
        self.unit_names = {int(key): str(value) for key, value in unit_names.items()}
        self.placement_service = placement_service or RawPlacementService(
            unit_names=self.unit_names
        )
        self._commands: deque[RoutedCommand] = deque()
        self._inflight: dict[str, RawDispatch] = {}
        self._builder_freshness_barrier: dict[int, int] = {}
        self._deferred_counts: dict[str, int] = {}
        self._diagnostic_snapshot: dict[str, Any] = {}

    @property
    def diagnostic_snapshot(self) -> Mapping[str, Any]:
        """Return the latest JSON-safe pre-dispatch diagnostic snapshot."""

        return deepcopy(self._diagnostic_snapshot)

    @property
    def last_diagnostic_snapshot(self) -> Mapping[str, Any]:
        """Compatibility alias for consumers that treat diagnostics as a last value."""

        return self.diagnostic_snapshot

    @property
    def pending_commands(self) -> int:
        """Deprecated queue-only count; use the explicit lifecycle properties."""

        return self.queued_count

    @property
    def queued_count(self) -> int:
        """Commands accepted from Runtime but not yet translated."""

        return len(self._commands)

    @property
    def effect_inflight_count(self) -> int:
        """Commands dispatched and awaiting terminal effect feedback."""

        return len(self._inflight)

    @property
    def outstanding_work(self) -> bool:
        """Whether queueing or effect verification still owns command work."""

        return bool(self._commands or self._inflight)

    def enqueue(self, decision: BridgeDecision) -> None:
        """Preserve Runtime ActionBatch order across per-agent routes."""

        routed = {
            command.command_id: command
            for route in decision.routes.values()
            for command in route.commands
        }
        for value in decision.action_batch.get("commands", ()):
            command_id = str(value["command_id"])
            command = routed.get(command_id)
            if command is None:
                raise RuntimeError(
                    f"raw_action_integrity_error: command {command_id!r} has no route"
                )
            self._commands.append(command)

    def record_rejection(
        self,
        dispatch: RawDispatch,
        agents: Mapping[str, Any],
        *,
        game_loop: int,
    ) -> None:
        """Quarantine an exact build candidate after SC2 rejects it."""

        self._inflight.pop(dispatch.command.command_id, None)
        self._record_builder_release(dispatch, game_loop=game_loop)
        self._quarantine_failed_build(
            dispatch,
            agents,
            game_loop=game_loop,
            failure_code="pysc2_rejected",
        )

    def observe_reports(
        self,
        reports: Sequence[Mapping[str, Any]],
        agents: Mapping[str, Any],
        *,
        game_loop: int,
    ) -> None:
        """Apply effect-terminal feedback to raw placement memory."""

        for report in reports:
            command_id = str(report.get("command_id", ""))
            dispatch = self._inflight.pop(command_id, None)
            if dispatch is None:
                continue
            self._record_builder_release(dispatch, game_loop=game_loop)
            if str(report.get("status", "")) == "failed":
                failure_code = str(report.get("failure_code") or "effect_failed")
                if self.placement_service.command_target(command_id) is not None:
                    self._quarantine_failed_build(
                        dispatch,
                        agents,
                        game_loop=game_loop,
                        failure_code=failure_code,
                        evidence=(
                            report.get("effect_evidence")
                            if isinstance(report.get("effect_evidence"), Mapping)
                            else report
                        ),
                    )
            else:
                self.placement_service.release_command(command_id)

    def _record_builder_release(self, dispatch: RawDispatch, *, game_loop: int) -> None:
        if dispatch.command.name not in _BUILD_RAW_FUNCTIONS or dispatch.builder_tag is None:
            return
        tag = int(dispatch.builder_tag)
        self._builder_freshness_barrier[tag] = max(
            int(game_loop),
            self._builder_freshness_barrier.get(tag, -1),
        )

    def _quarantine_failed_build(
        self,
        dispatch: RawDispatch,
        agents: Mapping[str, Any],
        *,
        game_loop: int,
        failure_code: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        del agents
        if dispatch.command.name in _BUILD_RAW_FUNCTIONS:
            self._quarantine_command(
                dispatch.command,
                failure_code=failure_code,
                game_loop=game_loop,
                dispatch=dispatch,
                evidence=evidence,
            )

    def _quarantine_command(
        self,
        command: RoutedCommand,
        *,
        failure_code: str,
        game_loop: int,
        dispatch: RawDispatch | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        reservation = self.placement_service.command_target(command.command_id)
        values: dict[str, Any] = {
            "command_id": command.command_id,
            "action_name": command.name,
            "requested_arguments": command.requested_arguments,
            "world_target": command.screen_world_target,
            "failure_code": failure_code,
            "game_loop": game_loop,
            "operation_id": command.operation_id,
            "attempt_ordinal": command.attempt_ordinal,
            "builder_tag": None if dispatch is None else dispatch.builder_tag,
            "placement_revision": command.placement_revision,
            "target_state_revision": None,
            "failure_classification": None,
            "classification_basis": (),
            "target_side_evidence": False,
        }
        if reservation is not None:
            values["operation_id"] = values["operation_id"] or reservation.operation_id
            values["attempt_ordinal"] = (
                reservation.attempt_ordinal
                if reservation.attempt_ordinal is not None
                else values["attempt_ordinal"]
            )
            values["builder_tag"] = values["builder_tag"] or reservation.builder_tag
            values["world_target"] = reservation.world_target
            values["placement_revision"] = reservation.placement_revision
            values["target_state_revision"] = reservation.target_state_revision
        if evidence is not None:
            values["failure_classification"] = evidence.get("failure_classification")
            values["classification_basis"] = tuple(
                str(item) for item in evidence.get("classification_basis", ())
            )
            values["target_side_evidence"] = bool(evidence.get("target_side_evidence", False))
        _call_optional_service_method(
            self.placement_service.quarantine_command,
            values,
        )

    def next_dispatch(
        self,
        observation: Any,
        agents: Mapping[str, Any],
    ) -> Optional[RawDispatch]:
        """Return the next executable raw action, terminalizing invalid commands."""

        self.placement_service.observe(
            observation,
            require_feature_visibility=False,
        )
        while self._commands:
            command = self._commands.popleft()
            try:
                dispatch = self._translate(command, observation, agents)
                self._inflight[command.command_id] = dispatch
                self._deferred_counts.pop(command.command_id, None)
                self._diagnostic_snapshot = _dispatch_snapshot(dispatch, observation)
                return dispatch
            except _RawDispatchDeferral as error:
                self._commands.appendleft(command)
                count = self._deferred_counts.get(command.command_id, 0) + 1
                self._deferred_counts[command.command_id] = count
                self._diagnostic_snapshot = _deferral_snapshot(
                    command,
                    observation,
                    error,
                    count=count,
                )
                # Preserve Runtime ActionBatch order. A stale Builder must not
                # allow a later command to leapfrog it or become accepted.
                return None
            except _RawDispatchFailure as error:
                self._diagnostic_snapshot = _failure_snapshot(command, observation, error)
                if (
                    command.name in _BUILD_RAW_FUNCTIONS
                    and error.code not in _NONSPATIAL_BUILD_FAILURE_CODES
                ):
                    self._quarantine_command(
                        command,
                        failure_code=error.code,
                        game_loop=_game_loop(observation),
                    )
                failure_dispatch = PrimitiveDispatch(
                    command.command_id,
                    "raw_pre_dispatch",
                    True,
                    origin="translator",
                    ordinal=0,
                    total=1,
                    failure_code=error.code,
                    requested_function_id=0,
                    emitted_function_id=0,
                )
                self.broker.settle_primitive(
                    failure_dispatch,
                    success=False,
                    failure_reason=str(error),
                    game_loop=_game_loop(observation),
                )
        return None

    def _translate(
        self,
        command: RoutedCommand,
        observation: Any,
        agents: Mapping[str, Any],
    ) -> RawDispatch:
        actions = importlib.import_module("pysc2.lib.actions")
        actor_tags = _actor_tags(command.actor, observation, agents)
        name = command.name
        arguments = command.resolved_arguments or command.requested_arguments
        builder_tag: Optional[int] = None
        producer_tag: Optional[int] = None
        placement: Any = None
        builder_state: list[dict[str, Any]] = []
        resolved_arguments = tuple(arguments)
        leased_actor_tags = set(actor_tags) & self.placement_service.leased_builder_tags
        if name not in _BUILD_RAW_FUNCTIONS and leased_actor_tags:
            raise _RawDispatchFailure(
                "actor_order_lease_conflict",
                f"{name} would overwrite leased builder tags "
                + ", ".join(hex(tag) for tag in sorted(leased_actor_tags)),
            )

        if name == "No_Operation":
            function = actions.RAW_FUNCTIONS.no_op
            action = function()
        elif name in {"Move_Minimap", "Move_Screen"}:
            _require_actor_tags(name, actor_tags)
            target = (
                command.screen_world_target
                if name == "Move_Screen"
                else _position(arguments, action_name=name)
            )
            if target is None:
                raise _RawDispatchFailure(
                    "candidate_invalidated",
                    f"{name} has no validated raw target",
                )
            function = actions.RAW_FUNCTIONS.Move_Move_pt
            action = function("now", list(actor_tags), _raw_point(target))
            resolved_arguments = (_raw_point(target),)
        elif name == "Attack_Unit":
            _require_actor_tags(name, actor_tags)
            target_tag = _tag_argument(arguments, action_name=name)
            target = _unit_by_tag(observation, target_tag)
            if target is None or int(_value(target, "alliance", 0)) != 4:
                raise _RawDispatchFailure(
                    "target_not_visible",
                    f"{name} target {hex(target_tag)} is not a living visible enemy",
                )
            function = actions.RAW_FUNCTIONS.Attack_Attack_unit
            action = function("now", list(actor_tags), target_tag)
        elif name in _CONTROL_RAW_FUNCTIONS:
            _require_actor_tags(name, actor_tags)
            function = getattr(actions.RAW_FUNCTIONS, _CONTROL_RAW_FUNCTIONS[name])
            action = function("now", list(actor_tags))
        elif name in _BUILD_RAW_FUNCTIONS:
            eligibility = raw_build_eligibility(observation, name, self.unit_names)
            if not eligibility.eligible:
                assert eligibility.failure_code is not None
                raise _RawDispatchFailure(
                    eligibility.failure_code,
                    eligibility.reason or eligibility.failure_code,
                )
            actor_tags, builder_state = _resolve_builder_tags(
                command,
                observation,
                agents,
                unit_names=self.unit_names,
                leased_builder_tags=self.placement_service.leased_builder_tags,
                freshness_blocked_builder_tags=self._freshness_blocked_builders(
                    _game_loop(observation)
                ),
            )
            if name.endswith("_Screen") and (
                command.placement_candidate_id is None or command.placement_revision is None
            ):
                raise _RawDispatchFailure(
                    "placement_provenance_missing",
                    f"{name} has no candidate-stage placement identity",
                )
            available_builders = tuple(
                tag for tag in actor_tags if tag not in self.placement_service.leased_builder_tags
            )
            if not available_builders:
                raise _RawDispatchDeferral(
                    "builder_not_ready",
                    f"{name} has no fresh idle Builder after lease filtering",
                    details={"builder_state": builder_state},
                )
            builder_tag = available_builders[0]
            function = getattr(actions.RAW_FUNCTIONS, _BUILD_RAW_FUNCTIONS[name])
            try:
                placement = self.placement_service.resolve(
                    command_id=command.command_id,
                    action_name=name,
                    requested_arguments=arguments,
                    observation=observation,
                    world_target=command.screen_world_target,
                    preferred_anchor_tag=command.screen_anchor_tag,
                    builder_tags=available_builders,
                    builder_tag=builder_tag,
                    operation_id=command.operation_id,
                    attempt_ordinal=command.attempt_ordinal,
                    ability_name=_BUILD_RAW_FUNCTIONS[name],
                    episode_id=command.episode_id or "unknown",
                    placement_candidate_id=command.placement_candidate_id,
                    candidate_placement_revision=command.placement_revision,
                )
            except RawPlacementFailure as error:
                if command.operation_id is not None and not _operation_retry_allowed(
                    self.placement_service,
                    operation_id=command.operation_id,
                    world_target=command.screen_world_target,
                    target_state_revision=None,
                    builder_tag=builder_tag,
                    observation_revision=_placement_revision(observation),
                ):
                    raise _RawDispatchFailure(
                        "operation_no_start_circuit_open",
                        f"{name} operation {command.operation_id} remains circuit-open",
                    ) from error
                if error.code in _BUILDER_DEFERRAL_CODES:
                    raise _RawDispatchDeferral(
                        error.code,
                        str(error),
                        details={"builder_state": builder_state},
                    ) from error
                raise _RawDispatchFailure(error.code, str(error)) from error
            if command.operation_id is not None and not _operation_retry_allowed(
                self.placement_service,
                operation_id=command.operation_id,
                world_target=placement.world_target,
                target_state_revision=placement.target_state_revision,
                builder_tag=builder_tag,
                observation_revision=placement.placement_revision,
            ):
                self.placement_service.release_command(
                    command.command_id,
                    game_loop=_game_loop(observation),
                    reason="operation_no_start_circuit_open",
                )
                raise _RawDispatchFailure(
                    "operation_no_start_circuit_open",
                    f"{name} operation {command.operation_id} remains circuit-open",
                )
            blockers = _dynamic_target_obstructions(
                observation,
                placement,
                builder_tag=builder_tag,
                unit_names=self.unit_names,
            )
            if blockers:
                self.placement_service.release_command(
                    command.command_id,
                    game_loop=_game_loop(observation),
                    reason="dynamic_target_obstruction",
                )
                raise _RawDispatchDeferral(
                    "dynamic_target_obstruction",
                    f"{name} final emitted target is blocked by dynamic units",
                    details={
                        "builder_state": builder_state,
                        "dynamic_blockers": blockers,
                        "emitted_target": [
                            float(placement.world_target[0]),
                            float(placement.world_target[1]),
                        ],
                        "placement_revision": placement.placement_revision,
                    },
                )
            if name == "Build_Assimilator_Near":
                assert placement.anchor_tag is not None
                action = function("now", [builder_tag], placement.anchor_tag)
            else:
                raw_target = _raw_point(placement.world_target)
                action = function("now", [builder_tag], raw_target)
                resolved_arguments = (raw_target,)
        elif (production := production_spec(name)) is not None:
            function = getattr(
                actions.RAW_FUNCTIONS,
                f"Train_{production.unit_type}_quick",
            )
            contract_failure = production_dispatch_failure(
                observation,
                name,
                unit_names=self.unit_names,
                required_function_id=int(function.id),
            )
            if contract_failure is not None:
                raise _RawDispatchFailure(
                    "production_contract_invalidated",
                    contract_failure,
                )
            producer_tag = _source_tag(
                observation,
                self.unit_names,
                producer_types=(
                    production.producer_type,
                    *production.alternate_producer_types,
                ),
            )
            if producer_tag is None:
                raise _RawDispatchFailure(
                    "production_source_unavailable",
                    f"{name} has no completed idle {production.producer_type}",
                )
            action = function("now", [producer_tag])
        elif (research := research_spec(name)) is not None:
            producer_tag = _source_tag(
                observation,
                self.unit_names,
                producer_types=(research.producer_type,),
            )
            if producer_tag is None:
                raise _RawDispatchFailure(
                    "production_source_unavailable",
                    f"{name} has no completed idle {research.producer_type}",
                )
            function = getattr(
                actions.RAW_FUNCTIONS,
                f"Research_{research.upgrade_name.removesuffix('Research')}_quick",
            )
            action = function("now", [producer_tag])
        elif name in _WARP_RAW_FUNCTIONS:
            producer_tag = _source_tag(
                observation,
                self.unit_names,
                producer_types=("WarpGate",),
            )
            if producer_tag is None:
                raise _RawDispatchFailure(
                    "production_source_unavailable",
                    f"{name} has no completed idle WarpGate",
                )
            target_tag = _tag_argument(arguments, action_name=name)
            target_unit = _unit_by_tag(observation, target_tag)
            if target_unit is None:
                raise _RawDispatchFailure(
                    "target_not_visible",
                    f"{name} anchor {hex(target_tag)} is not observable",
                )
            target = (
                float(_value(target_unit, "x", 0.0)),
                float(_value(target_unit, "y", 0.0)),
            )
            function = getattr(actions.RAW_FUNCTIONS, _WARP_RAW_FUNCTIONS[name])
            action = function("now", [producer_tag], _raw_point(target))
            resolved_arguments = (_raw_point(target),)
        elif name == "Ability_Blink_Screen":
            _require_actor_tags(name, actor_tags)
            if command.screen_world_target is None:
                raise _RawDispatchFailure(
                    "target_not_visible",
                    "Ability_Blink_Screen has no raw target",
                )
            function = actions.RAW_FUNCTIONS.Effect_Blink_Stalker_pt
            action = function("now", list(actor_tags), _raw_point(command.screen_world_target))
            resolved_arguments = (_raw_point(command.screen_world_target),)
        else:
            raise _RawDispatchFailure(
                "raw_action_unsupported",
                f"{name} is not implemented by the Protoss raw executor",
            )

        function_id = int(function.id)
        primitive = PrimitiveDispatch(
            command.command_id,
            str(function.name),
            True,
            origin="translator",
            ordinal=0,
            total=1,
            requested_function_id=function_id,
            emitted_function_id=function_id,
        )
        return RawDispatch(
            command=command,
            primitive=primitive,
            action=action,
            actor_tags=actor_tags,
            builder_tag=builder_tag,
            producer_tag=producer_tag,
            resolved_arguments=resolved_arguments,
            reservation_id=(None if placement is None else placement.reservation_id),
            placement_revision=(None if placement is None else placement.placement_revision),
        )

    def _freshness_blocked_builders(self, game_loop: int) -> frozenset[int]:
        expired = [
            tag
            for tag, release_loop in self._builder_freshness_barrier.items()
            if release_loop < int(game_loop)
        ]
        for tag in expired:
            del self._builder_freshness_barrier[tag]
        return frozenset(self._builder_freshness_barrier)


class _RawDispatchFailure(RuntimeError):
    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code


class _RawDispatchDeferral(_RawDispatchFailure):
    """A pre-dispatch condition that must wait for a fresh observation."""

    def __init__(self, code: str, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(code, reason)
        self.details = dict(details or {})


def _call_optional_service_method(method: Any, values: Mapping[str, Any]) -> Any:
    """Call an optional placement hook while tolerating older service shims."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**dict(values))
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return method(**dict(values))
    accepted = {name: value for name, value in values.items() if name in signature.parameters}
    return method(**accepted)


def _operation_retry_allowed(
    placement_service: Any,
    *,
    operation_id: str,
    world_target: tuple[float, float] | None,
    target_state_revision: str | None,
    builder_tag: int | None,
    observation_revision: str | None,
) -> bool:
    """Honor operation-level no-start circuit state when the hook is available."""

    method = getattr(placement_service, "operation_retry_allowed", None)
    if method is None:
        return True
    result = _call_optional_service_method(
        method,
        {
            "operation_id": operation_id,
            "world_target": world_target,
            "target_state_revision": target_state_revision,
            "builder_tag": builder_tag,
            "builder_ready": builder_tag is not None,
            "observation_revision": observation_revision,
        },
    )
    return True if result is None else bool(result)


def _resolve_builder_tags(
    command: RoutedCommand,
    observation: Any,
    agents: Mapping[str, Any],
    *,
    unit_names: Mapping[int, str],
    leased_builder_tags: Collection[int],
    freshness_blocked_builder_tags: Collection[int],
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Bind a build command to fresh, complete, idle workers in this snapshot."""

    configured = set(_actor_tags(command.actor, observation, agents))
    leases = {int(tag) for tag in leased_builder_tags}
    freshness_blocked = {int(tag) for tag in freshness_blocked_builder_tags}
    feature_visible = {
        int(_value(unit, "tag", 0))
        for unit in _value(observation, "feature_units", ())
        if int(_value(unit, "alliance", 0)) == 1 and bool(_value(unit, "is_on_screen", True))
    }
    actor_hint = command.actor.casefold()
    if "probe" in actor_hint:
        worker_names = {"probe"}
    elif "scv" in actor_hint:
        worker_names = {"scv"}
    elif "drone" in actor_hint:
        worker_names = {"drone"}
    else:
        worker_names = {"probe", "scv", "drone"}

    state: list[dict[str, Any]] = []
    ready: list[int] = []
    for unit in _value(observation, "raw_units", ()):
        tag = int(_value(unit, "tag", 0))
        if tag <= 0 or int(_value(unit, "alliance", 0)) != 1:
            continue
        name = _unit_name(unit, unit_names)
        if name.casefold() not in worker_names:
            continue
        progress = _build_progress(unit)
        order_length = int(_value(unit, "order_length", 0))
        active = int(_value(unit, "active", 0))
        orders = tuple(_orders(unit))
        blocking_build_orders = sorted(set(orders).intersection(BUILD_RAW_FUNCTION_IDS.values()))
        nearby_enemies = _nearby_enemy_threats(
            unit,
            observation,
            unit_names=unit_names,
        )
        reason = None
        if progress < 1.0:
            reason = "incomplete"
        elif float(_value(unit, "health", 1.0)) <= 0:
            reason = "dead"
        elif int(_value(unit, "display_type", 1)) != 1:
            reason = "not_visible"
        elif tag in leases:
            reason = "leased"
        elif tag in freshness_blocked:
            reason = "awaiting_fresh_observation"
        elif blocking_build_orders:
            reason = "prior_build_order"
        elif nearby_enemies:
            reason = "enemy_threat_near_builder"
        else:
            ready.append(tag)
        state.append(
            {
                "tag": hex(tag),
                "unit_type": name,
                "configured": tag in configured,
                "ready": reason is None,
                "reason": reason,
                "order_length": order_length,
                "active": active,
                "orders": [int(value) for value in orders],
                "blocking_build_orders": blocking_build_orders,
                "display_type": int(_value(unit, "display_type", 1)),
                "feature_visible": tag in feature_visible,
                "position": [
                    float(_value(unit, "x", 0.0)),
                    float(_value(unit, "y", 0.0)),
                ],
                "health": float(_value(unit, "health", 1.0)),
                "nearby_enemy_units": nearby_enemies,
            }
        )

    # Preserve the routed worker when it is still valid. If it disappeared or
    # became busy, choose a fresh same-race worker instead of failing the build.
    ready.sort(
        key=lambda tag: (
            tag not in feature_visible,
            tag not in configured,
            tag,
        )
    )
    if ready:
        return (ready[0],), state
    raise _RawDispatchDeferral(
        "builder_not_ready",
        f"{command.name} has no fresh idle Builder in the current observation",
        details={"builder_state": state},
    )


def _nearby_enemy_threats(
    builder: Any,
    observation: Any,
    *,
    unit_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Return visible ground enemies inside the conservative build-order envelope."""

    builder_position = (
        float(_value(builder, "x", 0.0)),
        float(_value(builder, "y", 0.0)),
    )
    builder_radius = max(0.0, float(_value(builder, "radius", 0.375)))
    structure_names = {spec.target_structure.casefold() for spec in BUILD_SPECS.values()}
    threats: list[dict[str, Any]] = []
    for unit in _value(observation, "raw_units", ()):
        if (
            int(_value(unit, "alliance", 0)) != 4
            or int(_value(unit, "display_type", 1)) != 1
            or float(_value(unit, "health", 1.0)) <= 0
            or bool(_value(unit, "is_flying", False))
        ):
            continue
        unit_name = _unit_name(unit, unit_names)
        if bool(_value(unit, "is_structure", False)) or unit_name.casefold() in structure_names:
            continue
        position = (
            float(_value(unit, "x", 0.0)),
            float(_value(unit, "y", 0.0)),
        )
        radius = max(0.0, float(_value(unit, "radius", 0.5)))
        distance = math.dist(position, builder_position)
        if distance > builder_radius + radius + _BUILDER_THREAT_CLEARANCE:
            continue
        threats.append(
            {
                "tag": hex(int(_value(unit, "tag", 0))),
                "unit_type": unit_name,
                "position": [position[0], position[1]],
                "radius": radius,
                "distance": round(distance, 3),
            }
        )
    threats.sort(key=lambda item: (item["distance"], item["tag"]))
    return threats


def _orders(unit: Any) -> tuple[int, ...]:
    result: list[int] = []
    raw_count = max(0, int(_value(unit, "order_length", 0)))
    for index in range(max(4, raw_count)):
        raw_order = int(_value(unit, f"order_id_{index}", 0))
        if raw_order > 0 and raw_order not in result:
            result.append(raw_order)
    value = _value(unit, "orders", ())
    if isinstance(value, Mapping):
        value = (value,)
    for order in value or ():
        if isinstance(order, Mapping):
            order_id = order.get("ability_id", order.get("abilityId", order.get("order_id", 0)))
        else:
            order_id = getattr(
                order,
                "ability_id",
                getattr(order, "abilityId", getattr(order, "order_id", 0)),
            )
        try:
            normalized = int(order_id)
        except (TypeError, ValueError):
            continue
        if normalized > 0 and normalized not in result:
            result.append(normalized)
    return tuple(result)


def _dynamic_target_obstructions(
    observation: Any,
    placement: Any,
    *,
    builder_tag: int,
    unit_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Conservatively reject a final footprint containing another live unit."""

    cells = set(getattr(placement, "occupied_grid_cells", ()))
    if not cells:
        return []
    structure_names = {spec.target_structure.casefold() for spec in BUILD_SPECS.values()}
    blockers: list[dict[str, Any]] = []
    for unit in _value(observation, "raw_units", ()):
        tag = int(_value(unit, "tag", 0))
        if tag <= 0 or tag == int(builder_tag):
            continue
        if getattr(placement, "anchor_tag", None) is not None and tag == int(placement.anchor_tag):
            continue
        if int(_value(unit, "alliance", 0)) not in {1, 2, 3, 4}:
            continue
        if int(_value(unit, "display_type", 1)) != 1:
            continue
        if bool(_value(unit, "is_flying", False)):
            continue
        unit_name = _unit_name(unit, unit_names)
        normalized_name = unit_name.casefold()
        if (
            bool(_value(unit, "is_structure", False))
            or normalized_name in structure_names
            or "mineral" in normalized_name
            or "geyser" in normalized_name
        ):
            continue
        position = (
            float(_value(unit, "x", 0.0)),
            float(_value(unit, "y", 0.0)),
        )
        radius = max(0.0, float(_value(unit, "radius", 0.5)))
        if not _circle_intersects_cells(position, radius, cells):
            continue
        blockers.append(
            {
                "tag": hex(tag),
                "unit_type": unit_name,
                "alliance": int(_value(unit, "alliance", 0)),
                "position": [position[0], position[1]],
                "radius": radius,
            }
        )
    blockers.sort(key=lambda item: item["tag"])
    return blockers


def _circle_intersects_cells(
    position: tuple[float, float],
    radius: float,
    cells: Sequence[tuple[int, int]] | set[tuple[int, int]],
) -> bool:
    for cell_x, cell_y in cells:
        nearest_x = min(max(position[0], cell_x - 0.5), cell_x + 0.5)
        nearest_y = min(max(position[1], cell_y - 0.5), cell_y + 0.5)
        if (position[0] - nearest_x) ** 2 + (position[1] - nearest_y) ** 2 <= radius**2:
            return True
    return False


def _dispatch_snapshot(dispatch: RawDispatch, observation: Any) -> dict[str, Any]:
    reservation_id = dispatch.reservation_id
    return {
        "status": "dispatched",
        "command_id": dispatch.command.command_id,
        "operation_id": dispatch.command.operation_id,
        "action_name": dispatch.command.name,
        "builder_tag": None if dispatch.builder_tag is None else hex(dispatch.builder_tag),
        "builder_tags": [hex(tag) for tag in dispatch.actor_tags],
        "placement_revision": dispatch.placement_revision,
        "observation_revision": _placement_revision(observation),
        "reservation_id": reservation_id,
        "game_loop": _game_loop(observation),
        "placement_query_result": "unavailable_no_controller_access",
        "conservative_final_checks": [
            "builder_readiness",
            "canonical_footprint",
            "available_feature_plane_buildability_pathability_power",
            "reachable_builder_approach",
            "raw_unit_collision_ranges",
        ],
        "emitted_arguments": deepcopy(dispatch.resolved_arguments),
    }


def _deferral_snapshot(
    command: RoutedCommand,
    observation: Any,
    error: _RawDispatchDeferral,
    *,
    count: int,
) -> dict[str, Any]:
    snapshot = {
        "status": "deferred",
        "deferred": True,
        "failure_code": error.code,
        "reason": str(error),
        "command_id": command.command_id,
        "operation_id": command.operation_id,
        "action_name": command.name,
        "builder_tag": None,
        "builder_tags": [],
        "placement_revision": command.placement_revision,
        "observation_revision": _placement_revision(observation),
        "game_loop": _game_loop(observation),
        "deferral_count": int(count),
    }
    snapshot.update(deepcopy(error.details))
    return snapshot


def _failure_snapshot(
    command: RoutedCommand,
    observation: Any,
    error: _RawDispatchFailure,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "deferred": False,
        "failure_code": error.code,
        "reason": str(error),
        "command_id": command.command_id,
        "operation_id": command.operation_id,
        "action_name": command.name,
        "builder_tag": None,
        "builder_tags": [],
        "placement_revision": command.placement_revision,
        "observation_revision": _placement_revision(observation),
        "game_loop": _game_loop(observation),
    }


def _actor_tags(
    actor: str,
    observation: Any,
    agents: Mapping[str, Any],
) -> tuple[int, ...]:
    agent_name, team_name = split_actor(actor)
    agent = agents.get(agent_name)
    if agent is None:
        return ()
    team = next(
        (
            value
            for value in getattr(agent, "teams", ())
            if isinstance(value, Mapping) and str(value.get("name", "")) == team_name
        ),
        None,
    )
    if team is None:
        return ()
    living = {
        int(_value(unit, "tag", 0))
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "alliance", 0)) == 1
        and int(_value(unit, "tag", 0)) > 0
        and _build_progress(unit) >= 1.0
    }
    return tuple(
        sorted(
            {int(tag) for tag in team.get("unit_tags", ()) if int(tag) > 0 and int(tag) in living}
        )
    )


def _source_tag(
    observation: Any,
    unit_names: Mapping[int, str],
    *,
    producer_types: Sequence[str],
) -> Optional[int]:
    wanted = set(producer_types)
    raw_units = list(_value(observation, "raw_units", ()))
    percent_scale = any(_raw_build_progress(unit) > 1.0 for unit in raw_units)
    for unit in raw_units:
        if int(_value(unit, "alliance", 0)) != 1:
            continue
        name = _unit_name(unit, unit_names)
        progress = _raw_build_progress(unit)
        normalized_progress = progress / 100.0 if percent_scale else progress
        if name not in wanted or normalized_progress < 1.0:
            continue
        if int(_value(unit, "order_length", 0)) != 0:
            continue
        tag = int(_value(unit, "tag", 0))
        if tag > 0:
            return tag
    return None


def _require_actor_tags(action_name: str, actor_tags: Sequence[int]) -> None:
    if not actor_tags:
        raise _RawDispatchFailure(
            "actor_not_available",
            f"{action_name} has no living raw actor tags",
        )


def _position(arguments: Sequence[Any], *, action_name: str) -> tuple[float, float]:
    if not arguments:
        raise _RawDispatchFailure("candidate_invalidated", f"{action_name} has no target")
    value = arguments[0]
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        raise _RawDispatchFailure(
            "candidate_invalidated",
            f"{action_name} target is not a two-coordinate point",
        )
    return float(value[0]), float(value[1])


def _tag_argument(arguments: Sequence[Any], *, action_name: str) -> int:
    if not arguments:
        raise _RawDispatchFailure("candidate_invalidated", f"{action_name} has no target tag")
    try:
        value = int(arguments[0], 0) if isinstance(arguments[0], str) else int(arguments[0])
    except (TypeError, ValueError) as error:
        raise _RawDispatchFailure(
            "candidate_invalidated",
            f"{action_name} target is not a tag",
        ) from error
    if value <= 0:
        raise _RawDispatchFailure("candidate_invalidated", f"{action_name} tag must be positive")
    return value


def _raw_point(position: Sequence[float]) -> list[int]:
    return [max(0, int(round(float(position[0])))), max(0, int(round(float(position[1]))))]


def _unit_by_tag(observation: Any, tag: int) -> Optional[Any]:
    return next(
        (
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "tag", -1)) == int(tag)
        ),
        None,
    )


def _unit_name(unit: Any, unit_names: Mapping[int, str]) -> str:
    value = _value(unit, "unit_type", "")
    if isinstance(value, str):
        return value
    return unit_names.get(int(value), f"unit:{int(value)}")


def _build_progress(unit: Any) -> float:
    value = float(_value(unit, "build_progress", 0.0))
    return value / 100.0 if value > 1.0 else value


def _raw_build_progress(unit: Any) -> float:
    return float(_value(unit, "build_progress", 0.0))


def _game_loop(observation: Any) -> int:
    value = _value(observation, "game_loop", 0)
    try:
        return int(value[0]) if len(value) == 1 else int(value)
    except (TypeError, IndexError):
        return int(value)


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = ["RawActionExecutor", "RawDispatch"]
