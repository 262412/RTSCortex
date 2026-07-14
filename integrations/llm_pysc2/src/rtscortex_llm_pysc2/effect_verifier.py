"""Confirm that accepted PySC2 build primitives changed the game state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.routing import RoutedCommand

DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS = 112

_MINERAL_COSTS = {
    "Assimilator": 75,
    "CyberneticsCore": 150,
    "Gateway": 150,
    "Nexus": 400,
    "Pylon": 100,
}
_BUILD_ABILITY_IDS = {
    "Assimilator": 882,
    "CyberneticsCore": 894,
    "Gateway": 883,
    "Nexus": 880,
    "Pylon": 881,
}


@dataclass(frozen=True)
class EffectVerdict:
    """One final gameplay-effect verdict for a deferred command."""

    command_id: str
    success: bool
    failure_reason: Optional[str] = None


@dataclass(frozen=True)
class _BuilderEvidence:
    tag: int
    status: str
    orders: tuple[int, ...]
    selected: bool


@dataclass(frozen=True)
class _Evidence:
    game_loop: int
    target_count: int
    target_progress: tuple[float, ...]
    minerals: int
    builder: Optional[_BuilderEvidence]


@dataclass
class _PendingBuild:
    command: RoutedCommand
    target_structure: str
    builder_tag: Optional[int] = None
    baseline: Optional[_Evidence] = None
    latest: Optional[_Evidence] = None
    accepted_game_loop: Optional[int] = None


class ActionEffectVerifier:
    """Defer ``Build_*`` reports until raw SC2 state confirms an effect."""

    def __init__(
        self,
        *,
        timeout_game_loops: int = DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS,
        unit_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        if timeout_game_loops <= 0:
            raise ValueError("timeout_game_loops must be positive")
        self.timeout_game_loops = timeout_game_loops
        self.unit_names = {int(key): str(value) for key, value in (unit_names or {}).items()}
        self._pending: dict[str, _PendingBuild] = {}

    def track(self, command: RoutedCommand) -> bool:
        """Register a build command, returning false for immediate actions."""

        target = _target_structure(command.name)
        if target is None:
            return False
        if command.command_id in self._pending:
            raise ValueError(f"command {command.command_id!r} is already tracked")
        self._pending[command.command_id] = _PendingBuild(command, target)
        return True

    def is_tracked(self, command_id: str) -> bool:
        return command_id in self._pending

    def prepare(self, command_id: str, observation: Any, builder_tag: Optional[int]) -> None:
        """Capture the raw state immediately before the final build primitive."""

        pending = self._get(command_id)
        pending.builder_tag = None if builder_tag is None else int(builder_tag)
        pending.baseline = self._evidence(pending, observation)
        pending.latest = pending.baseline

    def accept_primitive(self, command_id: str, *, game_loop: int) -> None:
        """Mark the PySC2 primitive accepted while keeping the report deferred."""

        pending = self._get(command_id)
        if pending.baseline is None:
            raise RuntimeError(f"effect baseline was not prepared for command {command_id!r}")
        pending.accepted_game_loop = int(game_loop)

    def cancel(self, command_id: str) -> None:
        self._pending.pop(command_id, None)

    def observe(self, observation: Any) -> list[EffectVerdict]:
        """Evaluate all accepted build commands against one raw observation."""

        verdicts: list[EffectVerdict] = []
        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None or pending.baseline is None:
                continue
            current = self._evidence(pending, observation)
            pending.latest = current
            if self._is_confirmed(pending.baseline, current, pending.target_structure):
                verdicts.append(EffectVerdict(command_id, True))
                del self._pending[command_id]
                continue
            elapsed = current.game_loop - pending.accepted_game_loop
            if elapsed >= self.timeout_game_loops:
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        False,
                        self._timeout_reason(pending, current),
                    )
                )
                del self._pending[command_id]
        return verdicts

    def fail_pending(self, reason: str) -> list[EffectVerdict]:
        """Fail accepted commands that cannot receive another observation."""

        verdicts = []
        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None:
                continue
            current = pending.latest or pending.baseline
            detail = self._diagnostic(pending, current) if current is not None else ""
            separator = ": " if detail else ""
            verdicts.append(
                EffectVerdict(command_id, False, f"{reason}{separator}{detail}" or reason)
            )
            del self._pending[command_id]
        return verdicts

    def _evidence(self, pending: _PendingBuild, observation: Any) -> _Evidence:
        raw_units = list(_value(observation, "raw_units", ()))
        target_units = [
            unit
            for unit in raw_units
            if int(_value(unit, "alliance", 0)) == 1
            and self._unit_name(unit) == pending.target_structure
        ]
        builder = next(
            (
                unit
                for unit in raw_units
                if pending.builder_tag is not None
                and int(_value(unit, "tag", -1)) == pending.builder_tag
            ),
            None,
        )
        player = _value(
            observation,
            "player_common",
            _value(observation, "player", None),
        )
        if player is None:
            raise ValueError("raw SC2 observation has no player data")
        return _Evidence(
            game_loop=_game_loop(observation),
            target_count=len(target_units),
            target_progress=tuple(_build_progress(unit) for unit in target_units),
            minerals=int(_value(player, "minerals", 0)),
            builder=None if builder is None else _builder_evidence(builder),
        )

    def _unit_name(self, unit: Any) -> str:
        value = _value(unit, "unit_type", "")
        if isinstance(value, str):
            return value
        return self.unit_names.get(int(value), f"unit:{int(value)}")

    @staticmethod
    def _is_confirmed(
        baseline: _Evidence,
        current: _Evidence,
        target_structure: str,
    ) -> bool:
        if current.target_count > baseline.target_count:
            return True
        expected_cost = _MINERAL_COSTS.get(target_structure)
        minimum_spend = 25 if expected_cost is None else max(25, expected_cost // 2)
        resource_spent = baseline.minerals - current.minerals >= minimum_spend
        expected_ability = _BUILD_ABILITY_IDS.get(target_structure)
        expected_build_order = (
            expected_ability is not None
            and current.builder is not None
            and expected_ability in current.builder.orders
        )
        return resource_spent and expected_build_order

    def _timeout_reason(self, pending: _PendingBuild, current: _Evidence) -> str:
        return (
            f"{pending.command.name} primitive accepted by PySC2 but no gameplay effect "
            f"confirmed within {self.timeout_game_loops} game loops: "
            f"{self._diagnostic(pending, current)}"
        )

    @staticmethod
    def _diagnostic(pending: _PendingBuild, current: _Evidence) -> str:
        baseline = pending.baseline
        if baseline is None:
            return "effect baseline unavailable"
        builder = _builder_change(baseline.builder, current.builder)
        diagnosis = _diagnosis(baseline, current, pending.target_structure)
        progress = f", progress {list(baseline.target_progress)}->{list(current.target_progress)}"
        return (
            f"{pending.target_structure} count {baseline.target_count}->{current.target_count}"
            f"{progress}; minerals {baseline.minerals}->{current.minerals}; {builder}; "
            f"diagnosis: {diagnosis}"
        )

    def _get(self, command_id: str) -> _PendingBuild:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown effect command {command_id!r}") from error


def _target_structure(action_name: str) -> Optional[str]:
    if not action_name.startswith("Build_"):
        return None
    stem = action_name.removeprefix("Build_")
    for suffix in ("_Screen", "_Near"):
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return None


def _builder_evidence(unit: Any) -> _BuilderEvidence:
    orders = _unit_orders(unit)
    return _BuilderEvidence(
        tag=int(_value(unit, "tag", 0)),
        status="active" if int(_value(unit, "order_length", len(orders))) > 0 else "idle",
        orders=orders,
        selected=bool(_value(unit, "is_selected", False)),
    )


def _unit_orders(unit: Any) -> tuple[int, ...]:
    explicit = _value(unit, "orders", None)
    if explicit is not None:
        return tuple(
            int(_value(order, "ability_id", _value(order, "order_id", order))) for order in explicit
        )
    count = min(max(int(_value(unit, "order_length", 0)), 0), 4)
    return tuple(int(_value(unit, f"order_id_{index}", 0)) for index in range(count))


def _build_progress(unit: Any) -> float:
    progress = float(_value(unit, "build_progress", 0.0))
    return progress / 100.0 if progress > 1.0 else progress


def _builder_change(
    baseline: Optional[_BuilderEvidence],
    current: Optional[_BuilderEvidence],
) -> str:
    if baseline is None:
        return "builder tag unavailable at dispatch"
    if current is None:
        return f"builder {hex(baseline.tag)} missing from current observation"
    return (
        f"builder {hex(baseline.tag)} status {baseline.status}->{current.status}, "
        f"orders {list(baseline.orders)}->{list(current.orders)}, "
        f"selected {baseline.selected}->{current.selected}"
    )


def _diagnosis(
    baseline: _Evidence,
    current: _Evidence,
    target_structure: str,
) -> str:
    if baseline.builder is None:
        return "builder tag was unavailable at dispatch; worker selection could not be verified"
    if not baseline.builder.selected:
        return "builder was not selected when the build primitive was dispatched"
    if current.builder is None:
        return "selected builder disappeared before construction became visible"

    expected_ability = _BUILD_ABILITY_IDS.get(target_structure)
    if expected_ability is not None and expected_ability in current.builder.orders:
        return "expected build order appeared but its resource or structure effect was incomplete"
    if current.builder.orders != baseline.builder.orders:
        return (
            "builder was selected at dispatch but now has a different non-build order; "
            "automatic worker management or a later action likely replaced it"
        )
    if current.minerals >= baseline.minerals:
        return (
            "builder was selected, but no build order or resource spend appeared; "
            "feature-action placement likely failed"
        )
    return "resources changed without the expected build order; unrelated spending is likely"


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
