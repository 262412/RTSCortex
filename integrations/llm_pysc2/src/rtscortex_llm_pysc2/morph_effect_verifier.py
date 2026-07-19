"""Confirm that an accepted Zerg structure morph started on its exact source."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.effect_types import EffectVerdict
from rtscortex_llm_pysc2.morph import MorphSpec, morph_spec, morph_spec_for_order
from rtscortex_llm_pysc2.routing import RoutedCommand


@dataclass(frozen=True)
class _MorphEvidence:
    game_loop: int
    source_type: Optional[str]
    source_orders: Optional[tuple[int, ...]]
    source_progress: Optional[float]
    minerals: int
    vespene: int


@dataclass
class _PendingMorph:
    command: RoutedCommand
    spec: MorphSpec
    source_tag: Optional[int] = None
    baseline: Optional[_MorphEvidence] = None
    latest: Optional[_MorphEvidence] = None
    accepted_game_loop: Optional[int] = None


class MorphEffectVerifier:
    """Verify an exact-source structure morph by order or same-tag type change."""

    def __init__(
        self,
        *,
        timeout_game_loops: int,
        unit_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        self.timeout_game_loops = timeout_game_loops
        self.unit_names = {int(key): str(value) for key, value in (unit_names or {}).items()}
        self._pending: dict[str, _PendingMorph] = {}
        self._claimed_order_keys: set[tuple[int, int]] = set()

    def track(self, command: RoutedCommand) -> bool:
        spec = morph_spec(command.name)
        if spec is None:
            return False
        if command.command_id in self._pending:
            raise ValueError(f"command {command.command_id!r} is already tracked")
        self._pending[command.command_id] = _PendingMorph(command, spec)
        return True

    def is_tracked(self, command_id: str) -> bool:
        return command_id in self._pending

    def prepare(self, command_id: str, observation: Any, source_tag: Optional[int]) -> None:
        pending = self._get(command_id)
        if source_tag is None:
            raise RuntimeError(f"morph command {command_id!r} has no source provenance")
        pending.source_tag = int(source_tag)
        pending.baseline = self._evidence(pending, observation)
        pending.latest = pending.baseline
        if (
            pending.baseline.source_orders is None
            or pending.baseline.source_type != pending.spec.producer_type
        ):
            raise RuntimeError(
                f"morph command {command_id!r} source {hex(source_tag)} is not an observable "
                f"{pending.spec.producer_type}"
            )

    def accept_primitive(self, command_id: str, *, game_loop: int) -> None:
        pending = self._get(command_id)
        if pending.baseline is None:
            raise RuntimeError(f"effect baseline was not prepared for command {command_id!r}")
        pending.accepted_game_loop = int(game_loop)

    def cancel(self, command_id: str) -> None:
        self._pending.pop(command_id, None)

    def observe(self, observation: Any) -> list[EffectVerdict]:
        self._claimed_order_keys.intersection_update(_active_morph_order_keys(observation))
        verdicts: list[EffectVerdict] = []
        for pending in sorted(
            self._pending.values(),
            key=lambda item: (int(item.accepted_game_loop or 0), item.command.command_id),
        ):
            if pending.accepted_game_loop is None or pending.baseline is None:
                continue
            previous = pending.latest or pending.baseline
            current = self._evidence(pending, observation)
            pending.latest = current
            previous_orders = previous.source_orders or ()
            order_key = (int(pending.source_tag or 0), pending.spec.raw_order_id)
            order_seen = (
                current.source_orders is not None
                and pending.spec.raw_order_id in current.source_orders
                and pending.spec.raw_order_id not in previous_orders
                and order_key not in self._claimed_order_keys
            )
            source_morphed = current.source_type == pending.spec.result_type
            if order_seen or source_morphed:
                kind = "producer_order" if order_seen else "source_morph"
                if order_seen:
                    self._claimed_order_keys.add(order_key)
                verdicts.append(
                    EffectVerdict(
                        pending.command.command_id,
                        True,
                        status="succeeded",
                        evidence=self._effect_evidence(
                            pending,
                            current,
                            confirmation_kind=kind,
                        ),
                    )
                )
                del self._pending[pending.command.command_id]
                continue
            elapsed = current.game_loop - pending.accepted_game_loop
            if elapsed < self.timeout_game_loops:
                continue
            if current.source_orders is None:
                failure_code = "morph_source_not_observable"
            elif current.source_orders:
                failure_code = "morph_order_replaced"
            else:
                failure_code = "no_morph_order_observed"
            verdicts.append(
                EffectVerdict(
                    pending.command.command_id,
                    False,
                    (
                        f"{pending.command.name} was accepted but source "
                        f"{hex(int(pending.source_tag or 0))} did not show order "
                        f"{pending.spec.raw_order_id} or become {pending.spec.result_type} "
                        f"after {elapsed} game loops"
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
            del self._pending[pending.command.command_id]
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
                    f"{reason}: {pending.command.name} morph effect was not observed",
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
        self._claimed_order_keys.clear()
        return verdicts

    def _evidence(self, pending: _PendingMorph, observation: Any) -> _MorphEvidence:
        source = next(
            (
                unit
                for unit in _value(observation, "raw_units", ())
                if pending.source_tag is not None
                and int(_value(unit, "tag", -1)) == pending.source_tag
                and int(_value(unit, "alliance", 0)) == 1
            ),
            None,
        )
        player = _value(observation, "player_common", _value(observation, "player", None))
        if player is None:
            raise ValueError("raw SC2 observation has no player data")
        return _MorphEvidence(
            game_loop=_game_loop(observation),
            source_type=None if source is None else self._unit_name(source),
            source_orders=None if source is None else _unit_orders(source),
            source_progress=None if source is None else _build_progress(source),
            minerals=int(_value(player, "minerals", 0)),
            vespene=int(_value(player, "vespene", 0)),
        )

    def _effect_evidence(
        self,
        pending: _PendingMorph,
        current: _MorphEvidence,
        *,
        confirmation_kind: Optional[str],
    ) -> dict[str, Any]:
        baseline = pending.baseline
        assert baseline is not None
        accepted_loop = pending.accepted_game_loop
        return {
            "effect_kind": "morph",
            "target_type": pending.spec.result_type,
            "target_tag": None if pending.source_tag is None else hex(pending.source_tag),
            "producer_tag": None if pending.source_tag is None else hex(pending.source_tag),
            "producer_type": pending.spec.producer_type,
            "producer_observed_type": current.source_type,
            "expected_order_id": pending.spec.raw_order_id,
            "baseline_producer_orders": list(baseline.source_orders or ()),
            "producer_orders": list(current.source_orders or ()),
            "production_order_seen": confirmation_kind == "producer_order",
            "confirmation_kind": confirmation_kind,
            "dispatched_loop": baseline.game_loop,
            "accepted_loop": accepted_loop,
            "confirmed_loop": current.game_loop if confirmation_kind is not None else None,
            "resource_delta": {
                "minerals": current.minerals - baseline.minerals,
                "vespene": current.vespene - baseline.vespene,
            },
            "source_build_progress": current.source_progress,
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

    def _get(self, command_id: str) -> _PendingMorph:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown morph effect command {command_id!r}") from error


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
    spec = morph_spec_for_order(order_id)
    return spec.raw_order_id if spec is not None else order_id


def _active_morph_order_keys(observation: Any) -> set[tuple[int, int]]:
    return {
        (int(_value(unit, "tag", 0)), order_id)
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "alliance", 0)) == 1
        for order_id in _unit_orders(unit)
        if morph_spec_for_order(order_id) is not None
    }


def _build_progress(unit: Any) -> float:
    value = float(_value(unit, "build_progress", 0.0))
    return value / 100.0 if value > 1.0 else value


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


__all__ = ["MorphEffectVerifier"]
