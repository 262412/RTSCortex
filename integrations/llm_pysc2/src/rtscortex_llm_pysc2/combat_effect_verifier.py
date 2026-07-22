"""Verify that accepted combat primitives damage or remove their exact target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.effect_types import EffectVerdict
from rtscortex_llm_pysc2.routing import RoutedCommand


@dataclass
class _PendingCombat:
    command: RoutedCommand
    target_tag: int
    target_type: Optional[str] = None
    dispatched_game_loop: Optional[int] = None
    accepted_game_loop: Optional[int] = None
    latest_game_loop: Optional[int] = None
    baseline_health: Optional[float] = None
    observed_health: Optional[float] = None


class CombatEffectVerifier:
    """Require observable damage instead of treating API acceptance as combat success."""

    def __init__(
        self,
        *,
        timeout_game_loops: int,
        unit_names: Optional[dict[int, str]] = None,
    ) -> None:
        self.timeout_game_loops = int(timeout_game_loops)
        self.unit_names = dict(unit_names or {})
        self._pending: dict[str, _PendingCombat] = {}

    def track(self, command: RoutedCommand) -> bool:
        if command.name != "Attack_Unit":
            return False
        arguments = command.resolved_arguments or command.requested_arguments
        if not arguments:
            raise ValueError("Attack_Unit requires an exact enemy tag")
        target_tag = _parse_tag(arguments[0])
        if target_tag is None:
            raise ValueError("Attack_Unit target must be an integer or hexadecimal tag")
        self._pending[command.command_id] = _PendingCombat(command, target_tag)
        return True

    def is_tracked(self, command_id: str) -> bool:
        return command_id in self._pending

    def resolve_arguments(self, command_id: str, arguments: list[Any]) -> None:
        pending = self._pending[command_id]
        target_tag = _parse_tag(arguments[0]) if arguments else None
        if target_tag is None:
            raise ValueError("Attack_Unit target must remain an exact tag")
        pending.target_tag = target_tag

    def prepare(self, command_id: str, observation: Any) -> None:
        pending = self._pending[command_id]
        target = _unit_by_tag(observation, pending.target_tag)
        pending.dispatched_game_loop = _game_loop(observation)
        pending.latest_game_loop = pending.dispatched_game_loop
        if target is None or int(_value(target, "alliance", 0)) != 4:
            return
        pending.target_type = _unit_name(target, self.unit_names)
        pending.baseline_health = _health_pool(target)
        pending.observed_health = pending.baseline_health

    def accept_primitive(self, command_id: str, *, game_loop: int) -> None:
        pending = self._pending[command_id]
        if pending.dispatched_game_loop is None:
            raise RuntimeError(f"combat baseline was not prepared for command {command_id!r}")
        pending.accepted_game_loop = int(game_loop)

    def observe(self, observation: Any) -> list[EffectVerdict]:
        game_loop = _game_loop(observation)
        verdicts: list[EffectVerdict] = []
        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None:
                continue
            pending.latest_game_loop = game_loop
            target = _unit_by_tag(observation, pending.target_tag)
            if target is not None:
                pending.target_type = pending.target_type or _unit_name(target, self.unit_names)
                pending.observed_health = _health_pool(target)
            if (
                pending.baseline_health is not None
                and pending.observed_health is not None
                and pending.observed_health < pending.baseline_health
            ):
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        True,
                        status="succeeded",
                        evidence=self._evidence(pending, "target_damaged"),
                    )
                )
                del self._pending[command_id]
                continue
            elapsed = game_loop - pending.accepted_game_loop
            if elapsed < self.timeout_game_loops:
                continue
            if pending.baseline_health is None:
                failure_code = "combat_target_baseline_missing"
                failure_reason = (
                    "Attack_Unit target baseline was unavailable before effect "
                    f"verification timed out after {elapsed} game loops"
                )
            elif target is None:
                failure_code = "combat_target_lost"
                failure_reason = (
                    "Attack_Unit target left observation before damage could be "
                    f"confirmed after {elapsed} game loops"
                )
            else:
                failure_code = "combat_effect_not_observed"
                failure_reason = (
                    "Attack_Unit produced no observable target damage after "
                    f"{elapsed} game loops"
                )
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    failure_reason,
                    status="failed",
                    failure_code=failure_code,
                    evidence=self._evidence(pending, None),
                )
            )
            del self._pending[command_id]
        return verdicts

    def cancel(self, command_id: str) -> None:
        self._pending.pop(command_id, None)

    def fail_pending(self, reason: str) -> list[EffectVerdict]:
        verdicts: list[EffectVerdict] = []
        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None:
                continue
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    f"{reason}: combat effect was not observed",
                    status="unconfirmed",
                    failure_code="episode_ended_unconfirmed",
                    evidence=self._evidence(pending, None),
                )
            )
            del self._pending[command_id]
        return verdicts

    def _evidence(
        self,
        pending: _PendingCombat,
        confirmation_kind: Optional[str],
    ) -> dict[str, Any]:
        current_loop = pending.latest_game_loop or pending.dispatched_game_loop
        elapsed = (
            0
            if pending.accepted_game_loop is None or current_loop is None
            else max(0, current_loop - pending.accepted_game_loop)
        )
        delta = (
            None
            if pending.baseline_health is None or pending.observed_health is None
            else max(0.0, pending.baseline_health - pending.observed_health)
        )
        return {
            "effect_kind": "combat",
            "target_type": pending.target_type,
            "target_tag": hex(pending.target_tag),
            "dispatched_loop": pending.dispatched_game_loop,
            "accepted_loop": pending.accepted_game_loop,
            "confirmed_loop": current_loop if confirmation_kind is not None else None,
            "confirmation_kind": confirmation_kind,
            "baseline_target_health": pending.baseline_health,
            "observed_target_health": pending.observed_health,
            "target_health_delta": delta,
            "elapsed_game_loops": elapsed,
            "base_timeout_game_loops": self.timeout_game_loops,
            "effective_timeout_game_loops": self.timeout_game_loops,
        }


def _parse_tag(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value, 0)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _unit_by_tag(observation: Any, tag: int) -> Optional[Any]:
    return next(
        (
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "tag", -1)) == tag
        ),
        None,
    )


def _health_pool(unit: Any) -> float:
    return max(0.0, float(_value(unit, "health", 0.0))) + max(
        0.0,
        float(_value(unit, "shield", 0.0)),
    )


def _unit_name(unit: Any, unit_names: dict[int, str]) -> str:
    value = _value(unit, "unit_type", "")
    if isinstance(value, str):
        return value
    return unit_names.get(int(value), f"unit:{int(value)}")


def _game_loop(observation: Any) -> int:
    value = _value(observation, "game_loop", 0)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 0
    return int(value)


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
