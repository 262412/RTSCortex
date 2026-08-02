"""Direct PySC2 Raw Action execution for RTSCortex-owned commands."""

from __future__ import annotations

import importlib
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.broker import PrimitiveDispatch, SharedDecisionBroker
from rtscortex_llm_pysc2.coordinator import BridgeDecision
from rtscortex_llm_pysc2.extractor import raw_build_eligibility
from rtscortex_llm_pysc2.observation import split_actor
from rtscortex_llm_pysc2.production import production_spec
from rtscortex_llm_pysc2.raw_placement import RawPlacementFailure, RawPlacementService
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
_NONSPATIAL_BUILD_FAILURE_CODES = frozenset(
    {
        "build_insufficient_minerals",
        "build_insufficient_vespene",
        "build_missing_prerequisite",
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
            if str(report.get("status", "")) == "failed":
                if self.placement_service.command_target(command_id) is not None:
                    self._quarantine_failed_build(
                        dispatch,
                        agents,
                        game_loop=game_loop,
                        failure_code=str(report.get("failure_code") or "effect_failed"),
                    )
            else:
                self.placement_service.release_command(command_id)

    def _quarantine_failed_build(
        self,
        dispatch: RawDispatch,
        agents: Mapping[str, Any],
        *,
        game_loop: int,
        failure_code: str,
    ) -> None:
        del agents
        if dispatch.command.name in _BUILD_RAW_FUNCTIONS:
            self._quarantine_command(
                dispatch.command,
                failure_code=failure_code,
                game_loop=game_loop,
            )

    def _quarantine_command(
        self,
        command: RoutedCommand,
        *,
        failure_code: str,
        game_loop: int,
    ) -> None:
        self.placement_service.quarantine_command(
            command_id=command.command_id,
            action_name=command.name,
            requested_arguments=command.requested_arguments,
            world_target=command.screen_world_target,
            failure_code=failure_code,
            game_loop=game_loop,
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
                return dispatch
            except _RawDispatchFailure as error:
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
            _require_actor_tags(name, available_builders)
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
                    ability_name=_BUILD_RAW_FUNCTIONS[name],
                    episode_id=command.episode_id or "unknown",
                    placement_candidate_id=command.placement_candidate_id,
                    candidate_placement_revision=command.placement_revision,
                )
            except RawPlacementFailure as error:
                raise _RawDispatchFailure(error.code, str(error)) from error
            if name == "Build_Assimilator_Near":
                assert placement.anchor_tag is not None
                action = function("now", [builder_tag], placement.anchor_tag)
            else:
                raw_target = _raw_point(placement.world_target)
                action = function("now", [builder_tag], raw_target)
                resolved_arguments = (raw_target,)
        elif (production := production_spec(name)) is not None:
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
            function = getattr(
                actions.RAW_FUNCTIONS,
                f"Train_{production.unit_type}_quick",
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
        )


class _RawDispatchFailure(RuntimeError):
    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code


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
    for unit in _value(observation, "raw_units", ()):
        if int(_value(unit, "alliance", 0)) != 1:
            continue
        name = _unit_name(unit, unit_names)
        if name not in wanted or _build_progress(unit) < 1.0:
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
