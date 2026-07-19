"""Verify deterministic Queen larva injection on one exact townhall."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.effect_types import EffectVerdict
from rtscortex_llm_pysc2.routing import RoutedCommand

INJECT_ACTION = "Effect_InjectLarva"
INJECT_FEATURE_FUNCTION_ID = 204
INJECT_RAW_FUNCTION_ID = 315
INJECT_ABILITY_ID = 251
INJECT_TARGET_BUFF_ID = 11


@dataclass(frozen=True)
class _InjectEvidence:
    game_loop: int
    queen_orders: Optional[tuple[int, ...]]
    queen_energy: Optional[float]
    target_type: Optional[str]
    target_buffs: Optional[tuple[int, ...]]


@dataclass
class _PendingInject:
    command: RoutedCommand
    target_tag: int
    queen_tag: Optional[int] = None
    baseline: Optional[_InjectEvidence] = None
    latest: Optional[_InjectEvidence] = None
    accepted_game_loop: Optional[int] = None


class InjectEffectVerifier:
    """Verify injection by exact Queen order or the exact townhall timer buff."""

    def __init__(
        self,
        *,
        timeout_game_loops: int,
        unit_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        self.timeout_game_loops = timeout_game_loops
        self.unit_names = {int(key): str(value) for key, value in (unit_names or {}).items()}
        self._pending: dict[str, _PendingInject] = {}

    def track(self, command: RoutedCommand) -> bool:
        if command.name != INJECT_ACTION:
            return False
        if command.command_id in self._pending:
            raise ValueError(f"command {command.command_id!r} is already tracked")
        arguments = command.resolved_arguments or command.requested_arguments
        if len(arguments) != 1:
            raise ValueError(f"{INJECT_ACTION} requires one exact townhall tag")
        target_tag = _parse_tag(arguments[0])
        if target_tag is None:
            raise ValueError(f"{INJECT_ACTION} target must be a positive SC2 tag")
        self._pending[command.command_id] = _PendingInject(command, target_tag)
        return True

    def is_tracked(self, command_id: str) -> bool:
        return command_id in self._pending

    def prepare(self, command_id: str, observation: Any, queen_tag: Optional[int]) -> None:
        pending = self._get(command_id)
        if queen_tag is None:
            raise RuntimeError(f"inject command {command_id!r} has no Queen provenance")
        pending.queen_tag = int(queen_tag)
        pending.baseline = self._evidence(pending, observation)
        pending.latest = pending.baseline
        if pending.baseline.queen_orders is None:
            raise RuntimeError(
                f"inject command {command_id!r} Queen {hex(queen_tag)} is not observable"
            )
        if pending.baseline.target_buffs is None:
            raise RuntimeError(
                f"inject command {command_id!r} target {hex(pending.target_tag)} is not observable"
            )

    def accept_primitive(self, command_id: str, *, game_loop: int) -> None:
        pending = self._get(command_id)
        if pending.baseline is None:
            raise RuntimeError(f"effect baseline was not prepared for command {command_id!r}")
        pending.accepted_game_loop = int(game_loop)

    def cancel(self, command_id: str) -> None:
        self._pending.pop(command_id, None)

    def observe(self, observation: Any) -> list[EffectVerdict]:
        verdicts: list[EffectVerdict] = []
        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None or pending.baseline is None:
                continue
            previous = pending.latest or pending.baseline
            current = self._evidence(pending, observation)
            pending.latest = current
            current_orders = current.queen_orders or ()
            previous_orders = previous.queen_orders or ()
            order_seen = (
                INJECT_RAW_FUNCTION_ID in current_orders
                and INJECT_RAW_FUNCTION_ID not in previous_orders
            )
            current_buffs = current.target_buffs or ()
            previous_buffs = previous.target_buffs or ()
            buff_seen = (
                INJECT_TARGET_BUFF_ID in current_buffs
                and INJECT_TARGET_BUFF_ID not in previous_buffs
            )
            if order_seen or buff_seen:
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        True,
                        status="succeeded",
                        evidence=self._effect_evidence(
                            pending,
                            current,
                            confirmation_kind=("producer_order" if order_seen else "target_buff"),
                        ),
                    )
                )
                del self._pending[command_id]
                continue
            elapsed = current.game_loop - pending.accepted_game_loop
            if elapsed < self.timeout_game_loops:
                continue
            if current.queen_orders is None:
                failure_code = "inject_source_not_observable"
            elif current.target_buffs is None:
                failure_code = "inject_target_not_observable"
            else:
                failure_code = "no_inject_effect_observed"
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    (
                        f"{INJECT_ACTION} was accepted but Queen "
                        f"{hex(int(pending.queen_tag or 0))} did not show order "
                        f"{INJECT_RAW_FUNCTION_ID} and townhall {hex(pending.target_tag)} "
                        f"did not gain buff {INJECT_TARGET_BUFF_ID} after {elapsed} game loops"
                    ),
                    status="failed",
                    failure_code=failure_code,
                    evidence=self._effect_evidence(
                        pending,
                        current,
                        confirmation_kind=None,
                    ),
                )
            )
            del self._pending[command_id]
        return verdicts

    def fail_pending(self, reason: str) -> list[EffectVerdict]:
        verdicts: list[EffectVerdict] = []
        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None:
                del self._pending[command_id]
                continue
            current = pending.latest or pending.baseline
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    f"{reason}: larva injection effect was not observed",
                    status="unconfirmed",
                    failure_code="episode_ended_unconfirmed",
                    evidence=(
                        None
                        if current is None
                        else self._effect_evidence(
                            pending,
                            current,
                            confirmation_kind=None,
                        )
                    ),
                )
            )
            del self._pending[command_id]
        return verdicts

    def _evidence(self, pending: _PendingInject, observation: Any) -> _InjectEvidence:
        raw_units = list(_value(observation, "raw_units", ()))
        queen = next(
            (
                unit
                for unit in raw_units
                if pending.queen_tag is not None
                and int(_value(unit, "tag", -1)) == pending.queen_tag
                and int(_value(unit, "alliance", 0)) == 1
                and self._unit_name(unit) == "Queen"
            ),
            None,
        )
        target = next(
            (
                unit
                for unit in raw_units
                if int(_value(unit, "tag", -1)) == pending.target_tag
                and int(_value(unit, "alliance", 0)) == 1
                and self._unit_name(unit) in {"Hatchery", "Lair", "Hive"}
            ),
            None,
        )
        return _InjectEvidence(
            game_loop=_game_loop(observation),
            queen_orders=None if queen is None else _unit_orders(queen),
            queen_energy=None if queen is None else float(_value(queen, "energy", 0.0)),
            target_type=None if target is None else self._unit_name(target),
            target_buffs=None if target is None else _unit_buffs(target),
        )

    def _effect_evidence(
        self,
        pending: _PendingInject,
        current: _InjectEvidence,
        *,
        confirmation_kind: Optional[str],
    ) -> dict[str, Any]:
        baseline = pending.baseline
        assert baseline is not None
        accepted_loop = pending.accepted_game_loop
        return {
            "effect_kind": "inject",
            "target_type": current.target_type,
            "target_tag": hex(pending.target_tag),
            "builder_tag": None if pending.queen_tag is None else hex(pending.queen_tag),
            "producer_tag": None if pending.queen_tag is None else hex(pending.queen_tag),
            "producer_type": "Queen",
            "producer_observed_type": None if current.queen_orders is None else "Queen",
            "expected_order_id": INJECT_RAW_FUNCTION_ID,
            "baseline_producer_orders": list(baseline.queen_orders or ()),
            "producer_orders": list(current.queen_orders or ()),
            "baseline_target_buff_ids": list(baseline.target_buffs or ()),
            "target_buff_ids": list(current.target_buffs or ()),
            "production_order_seen": confirmation_kind == "producer_order",
            "confirmation_kind": confirmation_kind,
            "dispatched_loop": baseline.game_loop,
            "accepted_loop": accepted_loop,
            "confirmed_loop": current.game_loop if confirmation_kind is not None else None,
            "resource_delta": {
                "queen_energy": int((current.queen_energy or 0) - (baseline.queen_energy or 0))
            },
            "elapsed_game_loops": (
                0 if accepted_loop is None else max(0, current.game_loop - accepted_loop)
            ),
            "base_timeout_game_loops": self.timeout_game_loops,
            "effective_timeout_game_loops": self.timeout_game_loops,
        }

    def _unit_name(self, unit: Any) -> str:
        value = _value(unit, "unit_type", "")
        if isinstance(value, str):
            return value
        return self.unit_names.get(int(value), f"unit:{int(value)}")

    def _get(self, command_id: str) -> _PendingInject:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown inject effect command {command_id!r}") from error


def _unit_orders(unit: Any) -> tuple[int, ...]:
    explicit = _value(unit, "orders", None)
    if explicit is not None:
        values = (
            int(_value(order, "ability_id", _value(order, "order_id", order))) for order in explicit
        )
        return tuple(_normalized_order_id(value) for value in values)
    count = min(max(int(_value(unit, "order_length", 0)), 0), 4)
    return tuple(
        _normalized_order_id(int(_value(unit, f"order_id_{index}", 0))) for index in range(count)
    )


def _normalized_order_id(order_id: int) -> int:
    return INJECT_RAW_FUNCTION_ID if order_id == INJECT_ABILITY_ID else order_id


def _unit_buffs(unit: Any) -> tuple[int, ...]:
    explicit = _value(unit, "buff_ids", None)
    if explicit is not None:
        return tuple(sorted(int(value) for value in explicit if int(value) > 0))
    count = min(max(int(_value(unit, "buff_duration_remain", 0) > 0), 0), 2)
    return tuple(
        int(_value(unit, f"buff_id_{index}", 0))
        for index in range(count)
        if int(_value(unit, f"buff_id_{index}", 0)) > 0
    )


def _parse_tag(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value, 16 if value.casefold().startswith("0x") else 10)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _game_loop(observation: Any) -> int:
    value = _value(observation, "game_loop", 0)
    if isinstance(value, (str, bytes)):
        return int(value)
    try:
        if len(value) == 1:
            return int(value[0])
    except (TypeError, IndexError):
        pass
    return int(value)


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "INJECT_ABILITY_ID",
    "INJECT_ACTION",
    "INJECT_FEATURE_FUNCTION_ID",
    "INJECT_RAW_FUNCTION_ID",
    "INJECT_TARGET_BUFF_ID",
    "InjectEffectVerifier",
]
