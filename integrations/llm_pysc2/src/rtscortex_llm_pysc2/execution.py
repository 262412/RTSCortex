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


@dataclass
class _TrackedCommand:
    route: RoutedActionBatch
    command: RoutedCommand
    primitives: list[PrimitiveResult] = field(default_factory=list)


class ExecutionTracker:
    """Aggregate one or more upstream primitives into one execution report."""

    def __init__(self) -> None:
        self._pending: dict[str, _TrackedCommand] = {}

    def register(self, route: RoutedActionBatch) -> None:
        for command in route.commands:
            if command.command_id in self._pending:
                raise ValueError(f"command {command.command_id!r} is already registered")
            self._pending[command.command_id] = _TrackedCommand(route=route, command=command)

    def record_primitive(
        self,
        command_id: str,
        function_name: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        failure_reason: Optional[str] = None,
    ) -> None:
        tracked = self._get(command_id)
        tracked.primitives.append(
            PrimitiveResult(
                function_name=function_name,
                success=success,
                latency_ms=latency_ms,
                failure_reason=failure_reason,
            )
        )

    def complete(self, command_id: str, *, game_result: Optional[str] = None) -> dict[str, Any]:
        tracked = self._pending.pop(command_id, None)
        if tracked is None:
            raise KeyError(f"unknown command {command_id!r}")

        primitives = tracked.primitives
        success = bool(primitives) and all(item.success for item in primitives)
        failure_reasons = [
            item.failure_reason for item in primitives if item.failure_reason is not None
        ]
        if not primitives:
            failure_reasons.append("no PySC2 primitive recorded")

        return {
            "protocol_version": "1.0",
            "run_id": tracked.route.run_id,
            "episode_id": tracked.route.episode_id,
            "step_id": tracked.route.step_id,
            "command_id": command_id,
            "success": success,
            "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
            "pysc2_function": " -> ".join(item.function_name for item in primitives) or None,
            "latency_ms": sum(item.latency_ms for item in primitives),
            "game_result": game_result,
        }

    def _get(self, command_id: str) -> _TrackedCommand:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown command {command_id!r}") from error
