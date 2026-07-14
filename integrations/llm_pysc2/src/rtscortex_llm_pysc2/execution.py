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
        self._seen: set[str] = set()

    def has_seen(self, command_id: str) -> bool:
        return command_id in self._seen

    def register(self, route: RoutedActionBatch) -> None:
        for command in route.commands:
            if command.command_id in self._pending:
                raise ValueError(f"command {command.command_id!r} is already registered")
            self._pending[command.command_id] = _TrackedCommand(route=route, command=command)
            self._seen.add(command.command_id)

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

    def complete(
        self,
        command_id: str,
        *,
        game_result: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> dict[str, Any]:
        tracked = self._pending.pop(command_id, None)
        if tracked is None:
            raise KeyError(f"unknown command {command_id!r}")
        return self._report(
            tracked,
            game_result=game_result,
            terminal_failure_reason=failure_reason,
        )

    def primitives_succeeded(self, command_id: str) -> bool:
        """Return whether at least one primitive exists and all were accepted."""

        primitives = self._get(command_id).primitives
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
            )
            for tracked in tracked_commands
        ]

    @staticmethod
    def _report(
        tracked: _TrackedCommand,
        *,
        game_result: Optional[str],
        terminal_failure_reason: Optional[str] = None,
    ) -> dict[str, Any]:
        primitives = tracked.primitives
        success = (
            terminal_failure_reason is None
            and bool(primitives)
            and all(item.success for item in primitives)
        )
        failure_reasons = [
            item.failure_reason for item in primitives if item.failure_reason is not None
        ]
        if terminal_failure_reason is not None:
            failure_reasons.append(terminal_failure_reason)
        elif not primitives:
            failure_reasons.append("no PySC2 primitive recorded")

        return {
            "protocol_version": "1.0",
            "run_id": tracked.route.run_id,
            "episode_id": tracked.route.episode_id,
            "step_id": tracked.route.step_id,
            "command_id": tracked.command.command_id,
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
