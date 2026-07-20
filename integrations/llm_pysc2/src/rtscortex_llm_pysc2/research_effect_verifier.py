"""Confirm exact-source research by producer order or completed upgrade."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.effect_types import EffectVerdict
from rtscortex_llm_pysc2.research import ResearchSpec, research_spec, research_spec_for_order
from rtscortex_llm_pysc2.routing import RoutedCommand


@dataclass(frozen=True)
class _ResearchEvidence:
    game_loop: int
    producer_type: Optional[str]
    producer_orders: Optional[tuple[int, ...]]
    upgrades: frozenset[int]
    minerals: int
    vespene: int


@dataclass
class _PendingResearch:
    command: RoutedCommand
    spec: ResearchSpec
    producer_tag: Optional[int] = None
    baseline: Optional[_ResearchEvidence] = None
    latest: Optional[_ResearchEvidence] = None
    accepted_game_loop: Optional[int] = None


class ResearchEffectVerifier:
    """Research succeeds only when its exact producer starts it or the upgrade appears."""

    def __init__(
        self,
        *,
        timeout_game_loops: int,
        unit_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        self.timeout_game_loops = timeout_game_loops
        self.unit_names = {int(key): str(value) for key, value in (unit_names or {}).items()}
        self._pending: dict[str, _PendingResearch] = {}
        self._claimed_order_keys: set[tuple[int, int]] = set()

    def track(self, command: RoutedCommand) -> bool:
        spec = research_spec(command.name)
        if spec is None:
            return False
        if command.command_id in self._pending:
            raise ValueError(f"command {command.command_id!r} is already tracked")
        self._pending[command.command_id] = _PendingResearch(command, spec)
        return True

    def is_tracked(self, command_id: str) -> bool:
        return command_id in self._pending

    def prepare(self, command_id: str, observation: Any, producer_tag: Optional[int]) -> None:
        pending = self._get(command_id)
        if producer_tag is None:
            raise RuntimeError(f"research command {command_id!r} has no producer provenance")
        pending.producer_tag = int(producer_tag)
        pending.baseline = self._evidence(pending, observation)
        pending.latest = pending.baseline
        if (
            pending.baseline.producer_orders is None
            or pending.baseline.producer_type != pending.spec.producer_type
        ):
            raise RuntimeError(
                f"research command {command_id!r} producer {hex(producer_tag)} is not an "
                f"observable {pending.spec.producer_type}"
            )

    def accept_primitive(self, command_id: str, *, game_loop: int) -> None:
        pending = self._get(command_id)
        if pending.baseline is None:
            raise RuntimeError(f"effect baseline was not prepared for command {command_id!r}")
        pending.accepted_game_loop = int(game_loop)

    def cancel(self, command_id: str) -> None:
        self._pending.pop(command_id, None)

    def observe(self, observation: Any) -> list[EffectVerdict]:
        self._claimed_order_keys.intersection_update(_active_research_order_keys(observation))
        verdicts: list[EffectVerdict] = []
        for command_id, pending in sorted(self._pending.items()):
            if pending.accepted_game_loop is None or pending.baseline is None:
                continue
            previous = pending.latest or pending.baseline
            current = self._evidence(pending, observation)
            pending.latest = current
            previous_orders = previous.producer_orders or ()
            order_key = (int(pending.producer_tag or 0), pending.spec.raw_order_id)
            order_seen = (
                current.producer_orders is not None
                and pending.spec.raw_order_id in current.producer_orders
                and pending.spec.raw_order_id not in previous_orders
                and order_key not in self._claimed_order_keys
            )
            upgrade_seen = (
                pending.spec.upgrade_id in current.upgrades
                and pending.spec.upgrade_id not in pending.baseline.upgrades
            )
            if order_seen or upgrade_seen:
                if order_seen:
                    self._claimed_order_keys.add(order_key)
                kind = "producer_order" if order_seen else "upgrade_observed"
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        True,
                        status="succeeded",
                        evidence=self._effect_evidence(pending, current, kind),
                    )
                )
                del self._pending[command_id]
                continue
            elapsed = current.game_loop - pending.accepted_game_loop
            if elapsed < self.timeout_game_loops:
                continue
            if current.producer_orders is None:
                failure_code = "research_producer_not_observable"
            elif current.producer_orders:
                failure_code = "research_order_replaced"
            else:
                failure_code = "no_research_order_observed"
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    (
                        f"{pending.command.name} was accepted but producer "
                        f"{hex(int(pending.producer_tag or 0))} did not show order "
                        f"{pending.spec.raw_order_id} and upgrade {pending.spec.upgrade_name} "
                        f"was not observed after {elapsed} game loops"
                    ),
                    status="failed",
                    failure_code=failure_code,
                    evidence=self._effect_evidence(pending, current, None),
                )
            )
            del self._pending[command_id]
        return verdicts

    def fail_pending(self, reason: str) -> list[EffectVerdict]:
        verdicts: list[EffectVerdict] = []
        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is not None:
                current = pending.latest or pending.baseline
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        False,
                        f"{reason}: {pending.command.name} research effect was not observed",
                        status="unconfirmed",
                        failure_code="episode_ended_unconfirmed",
                        evidence=(
                            None
                            if current is None
                            else self._effect_evidence(pending, current, None)
                        ),
                    )
                )
            del self._pending[command_id]
        self._claimed_order_keys.clear()
        return verdicts

    def _evidence(self, pending: _PendingResearch, observation: Any) -> _ResearchEvidence:
        source = next(
            (
                unit
                for unit in _value(observation, "raw_units", ())
                if pending.producer_tag is not None
                and int(_value(unit, "tag", -1)) == pending.producer_tag
                and int(_value(unit, "alliance", 0)) == 1
            ),
            None,
        )
        player = _value(observation, "player_common", _value(observation, "player", None))
        if player is None:
            raise ValueError("raw SC2 observation has no player data")
        return _ResearchEvidence(
            game_loop=_game_loop(observation),
            producer_type=None if source is None else self._unit_name(source),
            producer_orders=None if source is None else _unit_orders(source),
            upgrades=frozenset(int(value) for value in _value(observation, "upgrades", ())),
            minerals=int(_value(player, "minerals", 0)),
            vespene=int(_value(player, "vespene", 0)),
        )

    def _effect_evidence(
        self,
        pending: _PendingResearch,
        current: _ResearchEvidence,
        confirmation_kind: Optional[str],
    ) -> dict[str, Any]:
        baseline = pending.baseline
        assert baseline is not None
        accepted_loop = pending.accepted_game_loop
        return {
            "effect_kind": "research",
            "target_type": pending.spec.upgrade_name,
            "producer_tag": None if pending.producer_tag is None else hex(pending.producer_tag),
            "producer_type": pending.spec.producer_type,
            "producer_observed_type": current.producer_type,
            "expected_order_id": pending.spec.raw_order_id,
            "expected_upgrade": pending.spec.upgrade_name,
            "expected_upgrade_id": pending.spec.upgrade_id,
            "baseline_producer_orders": list(baseline.producer_orders or ()),
            "producer_orders": list(current.producer_orders or ()),
            "baseline_upgrade_ids": sorted(baseline.upgrades),
            "upgrade_ids": sorted(current.upgrades),
            "production_order_seen": confirmation_kind == "producer_order",
            "confirmation_kind": confirmation_kind,
            "dispatched_loop": baseline.game_loop,
            "accepted_loop": accepted_loop,
            "confirmed_loop": current.game_loop if confirmation_kind is not None else None,
            "resource_delta": {
                "minerals": current.minerals - baseline.minerals,
                "vespene": current.vespene - baseline.vespene,
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

    def _get(self, command_id: str) -> _PendingResearch:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown research effect command {command_id!r}") from error


def _unit_orders(unit: Any) -> tuple[int, ...]:
    explicit = _value(unit, "orders", None)
    if explicit is not None:
        values = (
            int(_value(order, "ability_id", _value(order, "order_id", order)))
            for order in explicit
        )
        return tuple(_normalized_order_id(value) for value in values)
    count = min(max(int(_value(unit, "order_length", 0)), 0), 4)
    return tuple(
        _normalized_order_id(int(_value(unit, f"order_id_{index}", 0)))
        for index in range(count)
    )


def _normalized_order_id(order_id: int) -> int:
    spec = research_spec_for_order(order_id)
    return spec.raw_order_id if spec is not None else order_id


def _active_research_order_keys(observation: Any) -> set[tuple[int, int]]:
    return {
        (int(_value(unit, "tag", 0)), order_id)
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "alliance", 0)) == 1
        for order_id in _unit_orders(unit)
        if research_spec_for_order(order_id) is not None
    }


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


__all__ = ["ResearchEffectVerifier"]
