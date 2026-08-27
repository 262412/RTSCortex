"""Verify a Terran MULE calldown by exact Orbital provenance and spawn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.ability import ability_spec
from rtscortex_llm_pysc2.effect_types import EffectVerdict
from rtscortex_llm_pysc2.routing import RoutedCommand

MULE_ACTION = "Effect_CalldownMULE_Screen"


@dataclass(frozen=True)
class _MuleEvidence:
    game_loop: int
    source_type: Optional[str]
    source_energy: Optional[float]
    source_orders: Optional[tuple[int, ...]]
    mule_tags: frozenset[int]


@dataclass
class _PendingMule:
    command: RoutedCommand
    target_position: tuple[float, float]
    source_tag: Optional[int] = None
    baseline: Optional[_MuleEvidence] = None
    latest: Optional[_MuleEvidence] = None
    accepted_game_loop: Optional[int] = None


class MuleEffectVerifier:
    """MULE succeeds only after a new MULE tag appears for one accepted command."""

    def __init__(
        self,
        *,
        timeout_game_loops: int,
        unit_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        self.timeout_game_loops = timeout_game_loops
        self.unit_names = {int(key): str(value) for key, value in (unit_names or {}).items()}
        self._pending: dict[str, _PendingMule] = {}
        self._claimed_mule_tags: set[int] = set()

    def track(self, command: RoutedCommand) -> bool:
        if command.name != MULE_ACTION:
            return False
        arguments = command.resolved_arguments or command.requested_arguments
        target = _position(arguments)
        if target is None:
            raise ValueError(f"{MULE_ACTION} requires one screen position")
        self._pending[command.command_id] = _PendingMule(command, target)
        return True

    def is_tracked(self, command_id: str) -> bool:
        return command_id in self._pending

    def resolve_arguments(self, command_id: str, arguments: list[Any]) -> None:
        target = _position(arguments)
        if target is None:
            raise ValueError(f"{MULE_ACTION} requires one screen position")
        self._get(command_id).target_position = target

    def prepare(self, command_id: str, observation: Any, source_tag: Optional[int]) -> None:
        pending = self._get(command_id)
        if source_tag is None:
            raise RuntimeError(f"MULE command {command_id!r} has no Orbital provenance")
        pending.source_tag = int(source_tag)
        pending.baseline = self._evidence(pending, observation)
        pending.latest = pending.baseline
        if pending.baseline.source_type != "OrbitalCommand":
            raise RuntimeError(
                f"MULE command {command_id!r} source {hex(source_tag)} is not an observable "
                "OrbitalCommand"
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
        claimed_now: set[int] = set()
        for command_id, pending in sorted(
            self._pending.items(),
            key=lambda item: (int(item[1].accepted_game_loop or 0), item[0]),
        ):
            if pending.accepted_game_loop is None or pending.baseline is None:
                continue
            current = self._evidence(pending, observation)
            pending.latest = current
            candidates = sorted(
                current.mule_tags
                - pending.baseline.mule_tags
                - self._claimed_mule_tags
                - claimed_now
            )
            if candidates:
                new_tag = candidates[0]
                claimed_now.add(new_tag)
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        True,
                        status="succeeded",
                        evidence=self._effect_evidence(pending, current, new_tag),
                    )
                )
                del self._pending[command_id]
                continue
            elapsed = current.game_loop - pending.accepted_game_loop
            if elapsed < self.timeout_game_loops:
                continue
            failure_code = (
                "mule_source_not_observable"
                if current.source_type is None
                else "no_mule_spawn_observed"
            )
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    (
                        f"{MULE_ACTION} was accepted but no new MULE appeared after "
                        f"{elapsed} game loops"
                    ),
                    status="failed",
                    failure_code=failure_code,
                    evidence=self._effect_evidence(pending, current, None),
                )
            )
            del self._pending[command_id]
        self._claimed_mule_tags.update(claimed_now)
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
                        f"{reason}: MULE spawn was not observed",
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
        self._claimed_mule_tags.clear()
        return verdicts

    def _evidence(self, pending: _PendingMule, observation: Any) -> _MuleEvidence:
        units = list(_value(observation, "raw_units", ()))
        source = next(
            (
                unit
                for unit in units
                if pending.source_tag is not None
                and int(_value(unit, "tag", -1)) == pending.source_tag
                and int(_value(unit, "alliance", 0)) == 1
            ),
            None,
        )
        return _MuleEvidence(
            game_loop=_game_loop(observation),
            source_type=None if source is None else self._unit_name(source),
            source_energy=None if source is None else float(_value(source, "energy", 0.0)),
            source_orders=None if source is None else _unit_orders(source),
            mule_tags=frozenset(
                int(_value(unit, "tag", 0))
                for unit in units
                if int(_value(unit, "alliance", 0)) == 1
                and self._unit_name(unit) == "MULE"
                and int(_value(unit, "tag", 0)) > 0
            ),
        )

    def _effect_evidence(
        self,
        pending: _PendingMule,
        current: _MuleEvidence,
        new_tag: Optional[int],
    ) -> dict[str, Any]:
        baseline = pending.baseline
        assert baseline is not None
        spec = ability_spec(MULE_ACTION)
        assert spec is not None
        accepted_loop = pending.accepted_game_loop
        return {
            "effect_kind": "ability",
            "target_type": "MULE",
            "target_position": pending.target_position,
            "producer_tag": None if pending.source_tag is None else hex(pending.source_tag),
            "producer_type": "OrbitalCommand",
            "producer_observed_type": current.source_type,
            "expected_order_id": spec.raw_order_id,
            "baseline_unit_tags": [hex(tag) for tag in sorted(baseline.mule_tags)],
            "new_unit_tag": None if new_tag is None else hex(new_tag),
            "baseline_producer_orders": list(baseline.source_orders or ()),
            "producer_orders": list(current.source_orders or ()),
            "confirmation_kind": None if new_tag is None else "new_unit",
            "dispatched_loop": baseline.game_loop,
            "accepted_loop": accepted_loop,
            "confirmed_loop": current.game_loop if new_tag is not None else None,
            "resource_delta": {
                "producer_energy": int((current.source_energy or 0) - (baseline.source_energy or 0))
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

    def _get(self, command_id: str) -> _PendingMule:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown MULE effect command {command_id!r}") from error


def _position(arguments: Any) -> Optional[tuple[float, float]]:
    if not isinstance(arguments, (list, tuple)) or len(arguments) != 1:
        return None
    value = arguments[0]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    return (float(value[0]), float(value[1]))


def _unit_orders(unit: Any) -> tuple[int, ...]:
    count = min(max(int(_value(unit, "order_length", 0)), 0), 4)
    return tuple(int(_value(unit, f"order_id_{index}", 0)) for index in range(count))


def _game_loop(observation: Any) -> int:
    value = _value(observation, "game_loop", 0)
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


__all__ = ["MULE_ACTION", "MuleEffectVerifier"]
