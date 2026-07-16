"""Track expanded PySC2 primitives back to RTSCortex commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rtscortex_llm_pysc2.routing import RoutedActionBatch, RoutedCommand


@dataclass(frozen=True)
class PrimitiveResult:
    function_name: str
    success: bool
    latency_ms: float = 0.0
    failure_reason: Optional[str] = None
    origin: str = "translator"
    ordinal: Optional[int] = None
    total: Optional[int] = None
    game_loop: Optional[int] = None
    failure_code: Optional[str] = None
    requested_function_id: Optional[int] = None
    emitted_function_id: Optional[int] = None


@dataclass
class _TrackedCommand:
    route: RoutedActionBatch
    command: RoutedCommand
    primitives: list[PrimitiveResult] = field(default_factory=list)
    resolved_arguments: tuple[Any, ...] = ()


class ExecutionTracker:
    """Aggregate one or more upstream primitives into one execution report."""

    def __init__(self) -> None:
        self._pending: dict[str, _TrackedCommand] = {}
        self._seen: set[str] = set()

    def has_seen(self, command_id: str) -> bool:
        return command_id in self._seen

    def is_pending(self, command_id: str) -> bool:
        """Return whether a dispatched command still needs one terminal report."""

        return command_id in self._pending

    def register(self, route: RoutedActionBatch) -> None:
        for command in route.commands:
            if command.command_id in self._pending:
                raise ValueError(f"command {command.command_id!r} is already registered")
            if command.command_id in self._seen:
                raise ValueError(f"command {command.command_id!r} was already dispatched")
            self._pending[command.command_id] = _TrackedCommand(
                route=route,
                command=command,
                resolved_arguments=command.resolved_arguments,
            )
            self._seen.add(command.command_id)

    def resolve_arguments(self, command_id: str, arguments: list[Any]) -> None:
        self._get(command_id).resolved_arguments = tuple(arguments)

    def record_primitive(
        self,
        command_id: str,
        function_name: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        failure_reason: Optional[str] = None,
        origin: str = "translator",
        ordinal: Optional[int] = None,
        total: Optional[int] = None,
        game_loop: Optional[int] = None,
        failure_code: Optional[str] = None,
        requested_function_id: Optional[int] = None,
        emitted_function_id: Optional[int] = None,
    ) -> None:
        tracked = self._get(command_id)
        tracked.primitives.append(
            PrimitiveResult(
                function_name=function_name,
                success=success,
                latency_ms=latency_ms,
                failure_reason=failure_reason,
                origin=origin,
                ordinal=ordinal,
                total=total,
                game_loop=game_loop,
                failure_code=failure_code,
                requested_function_id=requested_function_id,
                emitted_function_id=emitted_function_id,
            )
        )

    def complete(
        self,
        command_id: str,
        *,
        game_result: Optional[str] = None,
        failure_reason: Optional[str] = None,
        status: Optional[str] = None,
        execution_stage: Optional[str] = None,
        failure_code: Optional[str] = None,
        effect_evidence: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        tracked = self._pending.pop(command_id, None)
        if tracked is None:
            raise KeyError(f"unknown command {command_id!r}")
        return self._report(
            tracked,
            game_result=game_result,
            terminal_failure_reason=failure_reason,
            terminal_status=status,
            execution_stage=execution_stage,
            terminal_failure_code=failure_code,
            effect_evidence=effect_evidence,
        )

    def primitives_succeeded(self, command_id: str) -> bool:
        """Return whether at least one primitive exists and all were accepted."""

        primitives = [
            item for item in self._get(command_id).primitives if item.origin == "translator"
        ]
        return bool(primitives) and all(item.success for item in primitives)

    def drain_pending(
        self,
        *,
        failure_reason: str,
        game_result: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Fail and remove commands that cannot complete after episode end."""

        tracked_commands = list(self._pending.values())
        self._pending.clear()
        return [
            self._report(
                tracked,
                game_result=game_result,
                terminal_failure_reason=failure_reason,
                terminal_status="cancelled",
                execution_stage="episode_end",
                terminal_failure_code="episode_ended",
            )
            for tracked in tracked_commands
        ]

    @staticmethod
    def _report(
        tracked: _TrackedCommand,
        *,
        game_result: Optional[str],
        terminal_failure_reason: Optional[str] = None,
        terminal_status: Optional[str] = None,
        execution_stage: Optional[str] = None,
        terminal_failure_code: Optional[str] = None,
        effect_evidence: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        primitives = tracked.primitives
        terminal_primitives = [item for item in primitives if item.origin == "translator"]
        success = (
            terminal_failure_reason is None
            and bool(terminal_primitives)
            and all(item.success for item in terminal_primitives)
        )
        failure_reasons = [
            item.failure_reason for item in terminal_primitives if item.failure_reason is not None
        ]
        if terminal_failure_reason is not None:
            failure_reasons.append(terminal_failure_reason)
        elif not primitives:
            failure_reasons.append("no PySC2 primitive recorded")

        status = terminal_status or ("succeeded" if success else "failed")
        primitive_failure_code = next(
            (item.failure_code for item in terminal_primitives if item.failure_code is not None),
            None,
        )
        stage = execution_stage
        if stage is None:
            pre_dispatch_codes = {
                "actor_not_available",
                "actor_not_visible",
                "candidate_invalidated",
                "friendly_target",
                "invalid_expansion_anchor",
                "invalid_geyser_tag",
                "no_legal_placement",
                "production_source_invalidated",
                "production_source_unavailable",
                "target_not_visible",
            }
            if primitive_failure_code in pre_dispatch_codes:
                stage = "pre_dispatch"
            elif primitive_failure_code and primitive_failure_code != "pysc2_rejected":
                stage = "translation"
            else:
                stage = "pysc2_acceptance"

        return {
            "protocol_version": "1.1",
            "run_id": tracked.route.run_id,
            "episode_id": tracked.route.episode_id,
            "step_id": tracked.route.step_id,
            "command_id": tracked.command.command_id,
            "action_name": tracked.command.name,
            "actor": tracked.command.actor,
            "source": tracked.command.source,
            "requested_arguments": list(tracked.command.requested_arguments),
            "resolved_arguments": list(tracked.resolved_arguments),
            "status": status,
            "success": status == "succeeded",
            "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
            "execution_stage": stage,
            "failure_code": terminal_failure_code or primitive_failure_code,
            "pysc2_function": " -> ".join(item.function_name for item in primitives) or None,
            "primitive_trace": [
                {
                    "function_name": item.function_name,
                    "requested_function_id": item.requested_function_id,
                    "emitted_function_id": item.emitted_function_id,
                    "origin": item.origin,
                    "ordinal": item.ordinal,
                    "total": item.total,
                    "game_loop": item.game_loop,
                    "accepted": item.success,
                    "failure_code": item.failure_code,
                    "detail": item.failure_reason,
                }
                for item in primitives
            ],
            "effect_evidence": effect_evidence,
            "latency_ms": sum(item.latency_ms for item in primitives),
            "game_result": game_result,
        }

    def _get(self, command_id: str) -> _TrackedCommand:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown command {command_id!r}") from error
