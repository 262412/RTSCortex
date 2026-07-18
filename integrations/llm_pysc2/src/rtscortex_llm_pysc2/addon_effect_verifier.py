"""Confirm that accepted Terran add-on primitives establish construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.addon import AddonSpec, addon_spec, addon_spec_for_order
from rtscortex_llm_pysc2.effect_types import EffectVerdict
from rtscortex_llm_pysc2.routing import RoutedCommand


@dataclass(frozen=True)
class _AddonUnit:
    tag: int
    position: tuple[float, float]


@dataclass(frozen=True)
class _AddonEvidence:
    game_loop: int
    addons: tuple[_AddonUnit, ...]
    producer_orders: Optional[tuple[int, ...]]
    producer_position: Optional[tuple[float, float]]
    producer_addon_tag: Optional[int]
    minerals: int
    vespene: int


@dataclass
class _PendingAddon:
    command: RoutedCommand
    spec: AddonSpec
    producer_tag: Optional[int] = None
    baseline: Optional[_AddonEvidence] = None
    latest: Optional[_AddonEvidence] = None
    accepted_game_loop: Optional[int] = None


class AddonEffectVerifier:
    """Verify an add-on by its exact producer order or attached structure tag."""

    def __init__(
        self,
        *,
        timeout_game_loops: int,
        unit_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        self.timeout_game_loops = timeout_game_loops
        self.unit_names = {int(key): str(value) for key, value in (unit_names or {}).items()}
        self._pending: dict[str, _PendingAddon] = {}
        self._claimed_structure_tags: set[int] = set()
        self._claimed_order_keys: set[tuple[int, int]] = set()

    def track(self, command: RoutedCommand) -> bool:
        spec = addon_spec(command.name)
        if spec is None:
            return False
        if command.command_id in self._pending:
            raise ValueError(f"command {command.command_id!r} is already tracked")
        self._pending[command.command_id] = _PendingAddon(command, spec)
        return True

    def is_tracked(self, command_id: str) -> bool:
        return command_id in self._pending

    def prepare(self, command_id: str, observation: Any, producer_tag: Optional[int]) -> None:
        pending = self._get(command_id)
        if producer_tag is None:
            raise RuntimeError(f"add-on command {command_id!r} has no producer provenance")
        pending.producer_tag = int(producer_tag)
        pending.baseline = self._evidence(pending, observation)
        pending.latest = pending.baseline
        if pending.baseline.producer_orders is None:
            raise RuntimeError(
                f"add-on command {command_id!r} producer {hex(producer_tag)} is not observable"
            )

    def accept_primitive(self, command_id: str, *, game_loop: int) -> None:
        pending = self._get(command_id)
        if pending.baseline is None:
            raise RuntimeError(f"effect baseline was not prepared for command {command_id!r}")
        pending.accepted_game_loop = int(game_loop)

    def cancel(self, command_id: str) -> None:
        self._pending.pop(command_id, None)

    def observe(self, observation: Any) -> list[EffectVerdict]:
        self._claimed_order_keys.intersection_update(_active_addon_order_keys(observation))
        accepted = [
            pending
            for pending in self._pending.values()
            if pending.accepted_game_loop is not None and pending.baseline is not None
        ]
        current = {
            pending.command.command_id: self._evidence(pending, observation) for pending in accepted
        }
        assignments: dict[str, tuple[str, Optional[int]]] = {}
        claimed_orders: set[tuple[int, int]] = set()
        claimed_structures: set[int] = set()

        for pending in sorted(
            accepted,
            key=lambda item: (int(item.accepted_game_loop or 0), item.command.command_id),
        ):
            command_id = pending.command.command_id
            evidence = current[command_id]
            baseline = pending.baseline
            assert baseline is not None
            previous_orders = (
                pending.latest.producer_orders
                if pending.latest is not None and pending.latest.producer_orders is not None
                else ()
            )
            orders = evidence.producer_orders or ()
            order_key = (int(pending.producer_tag or 0), pending.spec.raw_order_id)
            if (
                pending.spec.raw_order_id in orders
                and pending.spec.raw_order_id not in previous_orders
                and order_key not in self._claimed_order_keys
                and order_key not in claimed_orders
            ):
                assignments[command_id] = ("producer_order", None)
                claimed_orders.add(order_key)
                continue

            baseline_tags = {item.tag for item in baseline.addons}
            candidates = [
                item
                for item in evidence.addons
                if item.tag not in baseline_tags
                and item.tag not in self._claimed_structure_tags
                and item.tag not in claimed_structures
            ]
            attached = evidence.producer_addon_tag
            selected = next((item for item in candidates if item.tag == attached), None)
            producer_position = evidence.producer_position
            if selected is None and producer_position is not None:
                selected = min(
                    (
                        item
                        for item in candidates
                        if _distance(producer_position, item.position) <= 6.0
                    ),
                    key=lambda item: (
                        _distance(producer_position, item.position),
                        item.tag,
                    ),
                    default=None,
                )
            if selected is not None:
                assignments[command_id] = ("new_structure", selected.tag)
                claimed_structures.add(selected.tag)

        self._claimed_order_keys.update(claimed_orders)
        self._claimed_structure_tags.update(claimed_structures)
        verdicts: list[EffectVerdict] = []
        for command_id, (kind, structure_tag) in assignments.items():
            pending = self._pending.pop(command_id)
            pending.latest = current[command_id]
            verdicts.append(
                EffectVerdict(
                    command_id,
                    True,
                    status="succeeded",
                    evidence=self._effect_evidence(
                        pending,
                        current[command_id],
                        confirmation_kind=kind,
                        structure_tag=structure_tag,
                    ),
                )
            )

        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None or pending.baseline is None:
                continue
            evidence = current[command_id]
            pending.latest = evidence
            elapsed = evidence.game_loop - pending.accepted_game_loop
            if elapsed < self.timeout_game_loops:
                continue
            if evidence.producer_orders is None:
                failure_code = "producer_not_observable"
            elif evidence.producer_orders:
                failure_code = "addon_order_replaced"
            else:
                failure_code = "no_addon_order_observed"
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    (
                        f"{pending.command.name} was accepted but producer "
                        f"{hex(int(pending.producer_tag or 0))} did not show order "
                        f"{pending.spec.raw_order_id} or a new {pending.spec.addon_type} "
                        f"after {elapsed} game loops"
                    ),
                    status="failed",
                    failure_code=failure_code,
                    evidence=self._effect_evidence(
                        pending,
                        evidence,
                        confirmation_kind=None,
                        structure_tag=None,
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
            evidence = pending.latest or pending.baseline
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    f"{reason}: {pending.command.name} add-on effect was not observed",
                    status="unconfirmed",
                    failure_code="episode_ended_unconfirmed",
                    evidence=(
                        None
                        if evidence is None
                        else self._effect_evidence(
                            pending,
                            evidence,
                            confirmation_kind=None,
                            structure_tag=None,
                        )
                    ),
                )
            )
            del self._pending[command_id]
        self._claimed_order_keys.clear()
        self._claimed_structure_tags.clear()
        return verdicts

    def _evidence(self, pending: _PendingAddon, observation: Any) -> _AddonEvidence:
        raw_units = list(_value(observation, "raw_units", ()))
        addons = tuple(
            sorted(
                (
                    _AddonUnit(
                        int(_value(unit, "tag", 0)),
                        (float(_value(unit, "x", 0.0)), float(_value(unit, "y", 0.0))),
                    )
                    for unit in raw_units
                    if int(_value(unit, "alliance", 0)) == 1
                    and self._unit_name(unit) == pending.spec.addon_type
                    and int(_value(unit, "tag", 0)) > 0
                ),
                key=lambda item: item.tag,
            )
        )
        producer = next(
            (
                unit
                for unit in raw_units
                if int(_value(unit, "tag", -1)) == pending.producer_tag
                and int(_value(unit, "alliance", 0)) == 1
                and self._unit_name(unit) == pending.spec.producer_type
            ),
            None,
        )
        player = _value(observation, "player_common", _value(observation, "player", None))
        if player is None:
            raise ValueError("raw SC2 observation has no player data")
        return _AddonEvidence(
            game_loop=_game_loop(observation),
            addons=addons,
            producer_orders=None if producer is None else _unit_orders(producer),
            producer_position=(
                None
                if producer is None
                else (float(_value(producer, "x", 0.0)), float(_value(producer, "y", 0.0)))
            ),
            producer_addon_tag=(
                None if producer is None else int(_value(producer, "add_on_tag", 0))
            ),
            minerals=int(_value(player, "minerals", 0)),
            vespene=int(_value(player, "vespene", 0)),
        )

    def _effect_evidence(
        self,
        pending: _PendingAddon,
        current: _AddonEvidence,
        *,
        confirmation_kind: Optional[str],
        structure_tag: Optional[int],
    ) -> dict[str, Any]:
        baseline = pending.baseline
        assert baseline is not None
        return {
            "effect_kind": "addon",
            "target_type": pending.spec.addon_type,
            "target_position": None,
            "target_tag": None,
            "builder_tag": None,
            "producer_tag": None if pending.producer_tag is None else hex(pending.producer_tag),
            "producer_type": pending.spec.producer_type,
            "expected_unit_type": None,
            "expected_order_id": pending.spec.raw_order_id,
            "baseline_structure_tags": [hex(item.tag) for item in baseline.addons],
            "baseline_unit_tags": [],
            "observed_structure_tag": None if structure_tag is None else hex(structure_tag),
            "new_unit_tag": None,
            "dispatched_loop": baseline.game_loop,
            "accepted_loop": pending.accepted_game_loop,
            "confirmed_loop": current.game_loop if confirmation_kind is not None else None,
            "worker_orders": [str(order) for order in current.producer_orders or ()],
            "baseline_producer_orders": list(baseline.producer_orders or ()),
            "producer_orders": list(current.producer_orders or ()),
            "production_order_seen": confirmation_kind == "producer_order",
            "confirmation_kind": confirmation_kind,
            "resource_delta": {
                "minerals": current.minerals - baseline.minerals,
                "vespene": current.vespene - baseline.vespene,
            },
            "order_seen": confirmation_kind == "producer_order",
            "order_last_seen_game_loop": (
                current.game_loop if confirmation_kind == "producer_order" else None
            ),
            "post_order_grace_game_loops": None,
            "mineral_delta": baseline.minerals - current.minerals,
            "elapsed_game_loops": max(
                0, current.game_loop - int(pending.accepted_game_loop or current.game_loop)
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

    def _get(self, command_id: str) -> _PendingAddon:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown add-on effect command {command_id!r}") from error


def _unit_orders(unit: Any) -> tuple[int, ...]:
    explicit = _value(unit, "orders", None)
    if explicit is not None:
        values = (
            int(_value(order, "ability_id", _value(order, "order_id", order))) for order in explicit
        )
    else:
        count = min(max(int(_value(unit, "order_length", 0)), 0), 4)
        values = (int(_value(unit, f"order_id_{index}", 0)) for index in range(count))
    return tuple(_normalize_order(value) for value in values)


def _normalize_order(order_id: int) -> int:
    spec = addon_spec_for_order(order_id)
    return spec.raw_order_id if spec is not None else order_id


def _active_addon_order_keys(observation: Any) -> set[tuple[int, int]]:
    return {
        (int(_value(unit, "tag", 0)), order_id)
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "alliance", 0)) == 1
        for order_id in _unit_orders(unit)
        if addon_spec_for_order(order_id) is not None
    }


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _game_loop(observation: Any) -> int:
    value = _value(observation, "game_loop", 0)
    if isinstance(value, (str, bytes)):
        return int(value)
    try:
        if len(value) == 1:
            return int(value[0])
    except (TypeError, KeyError):
        pass
    return int(value)


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = ["AddonEffectVerifier"]
