"""Confirm that accepted direct-training primitives establish production."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.effect_types import EffectVerdict
from rtscortex_llm_pysc2.production import (
    ProductionSpec,
    production_spec,
    production_spec_for_order,
)
from rtscortex_llm_pysc2.routing import RoutedCommand


@dataclass(frozen=True)
class _UnitEvidence:
    tag: int
    position: tuple[float, float]


@dataclass(frozen=True)
class _ProducerEvidence:
    tag: int
    unit_type: str
    position: tuple[float, float]


@dataclass(frozen=True)
class _ProductionEvidence:
    game_loop: int
    units: tuple[_UnitEvidence, ...]
    producer_sources: tuple[_ProducerEvidence, ...]
    producer_orders: Optional[tuple[int, ...]]
    producer_position: Optional[tuple[float, float]]
    producer_observed_type: Optional[str]
    minerals: int
    vespene: int
    supply_used: int


@dataclass
class _PendingProduction:
    command: RoutedCommand
    spec: ProductionSpec
    requested_producer_tag: Optional[int] = None
    producer_tag: Optional[int] = None
    baseline: Optional[_ProductionEvidence] = None
    latest: Optional[_ProductionEvidence] = None
    accepted_game_loop: Optional[int] = None
    production_order_seen: bool = False


class ProductionEffectVerifier:
    """Verify direct unit training by exact producer order or a new unit tag."""

    def __init__(
        self,
        *,
        timeout_game_loops: int,
        unit_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        if timeout_game_loops <= 0:
            raise ValueError("timeout_game_loops must be positive")
        self.timeout_game_loops = timeout_game_loops
        self.unit_names = {int(key): str(value) for key, value in (unit_names or {}).items()}
        self._pending: dict[str, _PendingProduction] = {}
        self._claimed_unit_tags: set[int] = set()
        self._claimed_order_keys: set[tuple[int, int]] = set()

    def track(self, command: RoutedCommand) -> bool:
        spec = production_spec(command.name)
        if spec is None:
            return False
        if command.command_id in self._pending:
            raise ValueError(f"command {command.command_id!r} is already tracked")
        self._pending[command.command_id] = _PendingProduction(command, spec)
        return True

    def is_tracked(self, command_id: str) -> bool:
        return command_id in self._pending

    def prepare(self, command_id: str, observation: Any, producer_tag: Optional[int]) -> None:
        pending = self._get(command_id)
        if producer_tag is None:
            raise RuntimeError(
                f"production command {command_id!r} has no translator producer provenance"
            )
        pending.requested_producer_tag = int(producer_tag)
        pending.producer_tag = int(producer_tag)
        pending.baseline = self._evidence(pending, observation)
        pending.latest = pending.baseline
        if pending.baseline.producer_orders is None:
            raise RuntimeError(
                f"production command {command_id!r} producer {hex(producer_tag)} is not observable"
            )

    def accept_primitive(self, command_id: str, *, game_loop: int) -> None:
        pending = self._get(command_id)
        if pending.baseline is None:
            raise RuntimeError(f"effect baseline was not prepared for command {command_id!r}")
        pending.accepted_game_loop = int(game_loop)

    def cancel(self, command_id: str) -> None:
        self._pending.pop(command_id, None)

    def observe(self, observation: Any) -> list[EffectVerdict]:
        self._claimed_order_keys.intersection_update(_active_production_order_keys(observation))
        accepted = [
            pending
            for pending in self._pending.values()
            if pending.accepted_game_loop is not None and pending.baseline is not None
        ]
        current_by_command = {
            pending.command.command_id: self._evidence(pending, observation) for pending in accepted
        }
        assignments: dict[str, tuple[str, Optional[int], Optional[int]]] = {}

        claimed_orders: set[tuple[int, int]] = set()
        for pending in sorted(
            accepted,
            key=lambda item: (
                int(item.accepted_game_loop or 0),
                item.command.command_id,
            ),
        ):
            command_id = pending.command.command_id
            current = current_by_command[command_id]
            producer_orders = current.producer_orders or ()
            previous_orders = (
                pending.latest.producer_orders
                if pending.latest is not None and pending.latest.producer_orders is not None
                else ()
            )
            order_key = (int(pending.producer_tag or 0), pending.spec.raw_order_id)
            if (
                pending.spec.raw_order_id in producer_orders
                and pending.spec.raw_order_id not in previous_orders
                and order_key not in self._claimed_order_keys
                and order_key not in claimed_orders
            ):
                assignments[command_id] = ("producer_order", None, None)
                claimed_orders.add(order_key)

        for pending in accepted:
            command_id = pending.command.command_id
            if command_id in assignments or not pending.spec.producer_consumed:
                continue
            current = current_by_command[command_id]
            if current.producer_observed_type in pending.spec.intermediate_types:
                assignments[command_id] = ("producer_morph", None, None)

        rebound_pairs: list[tuple[int, str, int]] = []
        for pending in accepted:
            command_id = pending.command.command_id
            if command_id in assignments or not pending.spec.producer_consumed:
                continue
            baseline = pending.baseline
            assert baseline is not None
            baseline_tags = {
                source.tag
                for source in baseline.producer_sources
                if source.unit_type == pending.spec.producer_type
            }
            current = current_by_command[command_id]
            for source in current.producer_sources:
                if (
                    source.tag in baseline_tags
                    and source.unit_type in pending.spec.intermediate_types
                    and source.tag not in self._claimed_unit_tags
                ):
                    rebound_pairs.append(
                        (
                            int(pending.accepted_game_loop or 0),
                            command_id,
                            source.tag,
                        )
                    )
        rebound_pairs.sort()
        rebound_commands = set(assignments)
        rebound_tags: set[int] = set()
        for _, command_id, source_tag in rebound_pairs:
            if command_id in rebound_commands or source_tag in rebound_tags:
                continue
            assignments[command_id] = ("producer_morph", None, source_tag)
            rebound_commands.add(command_id)
            rebound_tags.add(source_tag)

        unit_pairs: list[tuple[float, int, str, int]] = []
        for pending in accepted:
            command_id = pending.command.command_id
            if command_id in assignments:
                continue
            baseline = pending.baseline
            assert baseline is not None
            current = current_by_command[command_id]
            if current.producer_position is None:
                continue
            baseline_tags = {unit.tag for unit in baseline.units}
            for unit in current.units:
                if unit.tag in baseline_tags or unit.tag in self._claimed_unit_tags:
                    continue
                distance = _position_distance(current.producer_position, unit.position)
                if distance <= 8.0:
                    unit_pairs.append(
                        (
                            distance,
                            int(pending.accepted_game_loop or 0),
                            command_id,
                            unit.tag,
                        )
                    )
        unit_pairs.sort()
        claimed_units: set[int] = set()
        assigned_commands = set(assignments)
        for _, _, command_id, unit_tag in unit_pairs:
            if command_id in assigned_commands or unit_tag in claimed_units:
                continue
            assignments[command_id] = ("new_unit", unit_tag, None)
            assigned_commands.add(command_id)
            claimed_units.add(unit_tag)

        self._claimed_order_keys.update(claimed_orders)
        self._claimed_unit_tags.update(claimed_units)
        verdicts: list[EffectVerdict] = []
        self._claimed_unit_tags.update(rebound_tags)
        for command_id, (
            confirmation_kind,
            confirmed_unit_tag,
            rebound_producer_tag,
        ) in assignments.items():
            pending = self._pending.pop(command_id)
            if rebound_producer_tag is not None:
                pending.producer_tag = rebound_producer_tag
                current = self._evidence(pending, observation)
            else:
                current = current_by_command[command_id]
            pending.latest = current
            pending.production_order_seen = confirmation_kind == "producer_order"
            verdicts.append(
                EffectVerdict(
                    command_id,
                    True,
                    status="succeeded",
                    evidence=self._effect_evidence(
                        pending,
                        current,
                        confirmation_kind=confirmation_kind,
                        new_unit_tag=confirmed_unit_tag,
                    ),
                )
            )

        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None or pending.baseline is None:
                continue
            current = current_by_command[command_id]
            pending.latest = current
            elapsed = current.game_loop - pending.accepted_game_loop
            if elapsed < self.timeout_game_loops:
                continue
            producer_missing = current.producer_orders is None
            if producer_missing:
                failure_code = "producer_not_observable"
            elif current.producer_orders:
                failure_code = "production_order_replaced"
            else:
                failure_code = "no_production_order_observed"
            reason = (
                f"{pending.command.name} primitive accepted by PySC2 but producer "
                f"{hex(int(pending.producer_tag or 0))} did not show order "
                f"{pending.spec.raw_order_id} and no new {pending.spec.unit_type} appeared "
                f"after {elapsed} game loops"
            )
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    reason,
                    status="failed",
                    failure_code=failure_code,
                    evidence=self._effect_evidence(
                        pending,
                        current,
                        confirmation_kind=None,
                        new_unit_tag=None,
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
                    f"{reason}: {pending.command.name} production effect was not observed",
                    status="unconfirmed",
                    failure_code="episode_ended_unconfirmed",
                    evidence=(
                        None
                        if current is None
                        else self._effect_evidence(
                            pending,
                            current,
                            confirmation_kind=None,
                            new_unit_tag=None,
                        )
                    ),
                )
            )
            del self._pending[command_id]
        self._claimed_unit_tags.clear()
        self._claimed_order_keys.clear()
        return verdicts

    def _evidence(self, pending: _PendingProduction, observation: Any) -> _ProductionEvidence:
        raw_units = list(_value(observation, "raw_units", ()))
        producer_source_types = {
            pending.spec.producer_type,
            *pending.spec.intermediate_types,
        }
        producer_sources = tuple(
            sorted(
                (
                    _ProducerEvidence(
                        int(_value(unit, "tag", 0)),
                        self._unit_name(unit),
                        (
                            float(_value(unit, "x", 0.0)),
                            float(_value(unit, "y", 0.0)),
                        ),
                    )
                    for unit in raw_units
                    if int(_value(unit, "alliance", 0)) == 1
                    and self._unit_name(unit) in producer_source_types
                    and int(_value(unit, "tag", 0)) > 0
                ),
                key=lambda item: item.tag,
            )
        )
        units = tuple(
            sorted(
                (
                    _UnitEvidence(
                        int(_value(unit, "tag", 0)),
                        (
                            float(_value(unit, "x", 0.0)),
                            float(_value(unit, "y", 0.0)),
                        ),
                    )
                    for unit in raw_units
                    if int(_value(unit, "alliance", 0)) == 1
                    and self._unit_name(unit) == pending.spec.unit_type
                    and int(_value(unit, "tag", 0)) > 0
                ),
                key=lambda item: item.tag,
            )
        )
        producer_unit = next(
            (
                unit
                for unit in raw_units
                if pending.producer_tag is not None
                and int(_value(unit, "tag", -1)) == pending.producer_tag
                and int(_value(unit, "alliance", 0)) == 1
            ),
            None,
        )
        producer_type = None if producer_unit is None else self._unit_name(producer_unit)
        producer_is_valid = producer_type == pending.spec.producer_type or (
            pending.spec.producer_consumed
            and producer_type in pending.spec.intermediate_types
        )
        producer = producer_unit if producer_is_valid else None
        player = _value(observation, "player_common", _value(observation, "player", None))
        if player is None:
            raise ValueError("raw SC2 observation has no player data")
        return _ProductionEvidence(
            game_loop=_game_loop(observation),
            units=units,
            producer_sources=producer_sources,
            producer_orders=None if producer is None else _unit_orders(producer),
            producer_position=(
                None
                if producer is None
                else (
                    float(_value(producer, "x", 0.0)),
                    float(_value(producer, "y", 0.0)),
                )
            ),
            producer_observed_type=producer_type,
            minerals=int(_value(player, "minerals", 0)),
            vespene=int(_value(player, "vespene", 0)),
            supply_used=int(_value(player, "food_used", 0)),
        )

    def _effect_evidence(
        self,
        pending: _PendingProduction,
        current: _ProductionEvidence,
        *,
        confirmation_kind: Optional[str],
        new_unit_tag: Optional[int],
    ) -> dict[str, Any]:
        baseline = pending.baseline
        assert baseline is not None
        accepted_loop = pending.accepted_game_loop
        producer_orders = current.producer_orders or ()
        return {
            "effect_kind": "production",
            "target_type": pending.spec.unit_type,
            "target_position": None,
            "target_tag": None,
            "builder_tag": None,
            "requested_producer_tag": (
                None
                if pending.requested_producer_tag is None
                else hex(pending.requested_producer_tag)
            ),
            "producer_tag": None if pending.producer_tag is None else hex(pending.producer_tag),
            "producer_type": pending.spec.producer_type,
            "producer_observed_type": current.producer_observed_type,
            "producer_consumed": pending.spec.producer_consumed,
            "expected_unit_type": pending.spec.unit_type,
            "expected_order_id": pending.spec.raw_order_id,
            "baseline_structure_tags": [],
            "baseline_unit_tags": [hex(unit.tag) for unit in baseline.units],
            "observed_structure_tag": None,
            "new_unit_tag": None if new_unit_tag is None else hex(new_unit_tag),
            "dispatched_loop": baseline.game_loop,
            "accepted_loop": accepted_loop,
            "confirmed_loop": current.game_loop if confirmation_kind is not None else None,
            "worker_orders": [str(order) for order in producer_orders],
            "baseline_producer_orders": list(baseline.producer_orders or ()),
            "producer_orders": list(producer_orders),
            "production_order_seen": pending.production_order_seen,
            "confirmation_kind": confirmation_kind,
            "resource_delta": {
                "minerals": current.minerals - baseline.minerals,
                "vespene": current.vespene - baseline.vespene,
                "supply_used": current.supply_used - baseline.supply_used,
            },
            "order_seen": pending.production_order_seen,
            "order_last_seen_game_loop": (
                current.game_loop if pending.production_order_seen else None
            ),
            "post_order_grace_game_loops": None,
            "mineral_delta": baseline.minerals - current.minerals,
            "elapsed_game_loops": (
                0 if accepted_loop is None else max(0, current.game_loop - accepted_loop)
            ),
            "base_timeout_game_loops": self.timeout_game_loops,
            "effective_timeout_game_loops": self.timeout_game_loops,
            "active_order_extension": False,
        }

    def _unit_name(self, unit: Any) -> str:
        value = _value(unit, "unit_type", "")
        if isinstance(value, str):
            return value
        return self.unit_names.get(int(value), f"unit:{int(value)}")

    def _get(self, command_id: str) -> _PendingProduction:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown production effect command {command_id!r}") from error


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
    spec = production_spec_for_order(order_id)
    return spec.raw_order_id if spec is not None else order_id


def _active_production_order_keys(observation: Any) -> set[tuple[int, int]]:
    return {
        (int(_value(unit, "tag", 0)), order_id)
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "alliance", 0)) == 1
        for order_id in _unit_orders(unit)
        if production_spec_for_order(order_id) is not None
    }


def _position_distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


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


__all__ = ["ProductionEffectVerifier"]
