"""Confirm that accepted PySC2 build primitives changed the game state."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from rtscortex_llm_pysc2.addon_effect_verifier import AddonEffectVerifier
from rtscortex_llm_pysc2.combat_effect_verifier import CombatEffectVerifier
from rtscortex_llm_pysc2.effect_lifecycle import (
    DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS as DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS,
)
from rtscortex_llm_pysc2.effect_lifecycle import (
    POST_ORDER_EFFECT_GRACE_GAME_LOOPS,
    build_effect_max_lifetime,
)
from rtscortex_llm_pysc2.effect_types import EffectVerdict
from rtscortex_llm_pysc2.extractor import BUILD_RAW_FUNCTION_IDS, BUILD_SPECS
from rtscortex_llm_pysc2.inject_effect_verifier import InjectEffectVerifier
from rtscortex_llm_pysc2.morph_effect_verifier import MorphEffectVerifier
from rtscortex_llm_pysc2.mule_effect_verifier import MuleEffectVerifier
from rtscortex_llm_pysc2.production_effect_verifier import ProductionEffectVerifier
from rtscortex_llm_pysc2.raw_placement import (
    RawPlacementReservation,
    RawPlacementService,
    _circle_intersects_cells,
    _occupied_cells_for_spec,
    _placement_revision,
)
from rtscortex_llm_pysc2.research_effect_verifier import ResearchEffectVerifier
from rtscortex_llm_pysc2.routing import RoutedCommand

# PySC2 projects SC2's concrete Move ability (16) to Move_Move_pt. The generic
# smart-move function 13 represents a different ability and never appears in
# the raw unit order stream for normal movement.
MOVE_RAW_FUNCTION_ID = 547
MOVE_RAW_FUNCTION_IDS = frozenset({13, MOVE_RAW_FUNCTION_ID})
MOVE_MINIMAP_ARRIVAL_RADIUS = 4.0
MOVE_GAME_LOOPS_PER_MINIMAP_UNIT = 10.0
MOVE_SETTLEMENT_GRACE_GAME_LOOPS = 32
MOVE_ORDER_ACQUISITION_TIMEOUT_GAME_LOOPS = 16


@dataclass(frozen=True)
class _BuilderEvidence:
    tag: int
    alliance: int
    status: str
    orders: tuple[int, ...]
    selected: bool
    position: tuple[float, float]
    health: float
    health_max: float


@dataclass(frozen=True)
class _DynamicUnitEvidence:
    tag: int
    alliance: int
    position: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class _Evidence:
    game_loop: int
    observation_revision: str
    structures: tuple[_StructureEvidence, ...]
    occupants: tuple[tuple[int, tuple[float, float]], ...]
    minerals: int
    builder: Optional[_BuilderEvidence]
    dynamic_units: tuple[_DynamicUnitEvidence, ...]


@dataclass(frozen=True)
class _StructureEvidence:
    tag: int
    position: tuple[float, float]
    screen_position: Optional[tuple[float, float]]
    progress: float


@dataclass
class _PendingBuild:
    command: RoutedCommand
    target_structure: str
    resolved_arguments: tuple[Any, ...]
    builder_tag: Optional[int] = None
    baseline: Optional[_Evidence] = None
    latest: Optional[_Evidence] = None
    accepted_game_loop: Optional[int] = None
    target_tag: Optional[int] = None
    target_position: Optional[tuple[float, float]] = None
    coordinate_space: Optional[str] = None
    order_seen: bool = False
    order_last_seen_game_loop: Optional[int] = None
    build_started: bool = False
    build_start_confirmation_kind: Optional[str] = None
    build_start_confirmed_game_loop: Optional[int] = None
    resource_debit_seen: bool = False
    builder_approach_seen: bool = False
    target_occupancy_seen: bool = False
    active_order_extension: bool = False
    reservation_snapshot: RawPlacementReservation | None = None


@dataclass
class _PendingMove:
    command: RoutedCommand
    resolved_arguments: tuple[Any, ...]
    target_position: tuple[float, float]
    actor_tag: Optional[int] = None
    actor_tags: tuple[int, ...] = ()
    minimap_transform: Optional[tuple[float, float, float, float, float]] = None
    dispatched_game_loop: Optional[int] = None
    accepted_game_loop: Optional[int] = None
    latest_game_loop: Optional[int] = None
    baseline_actor_position: Optional[tuple[float, float]] = None
    latest_actor_position: Optional[tuple[float, float]] = None
    latest_actor_orders: tuple[int, ...] = ()
    move_order_seen: bool = False


class ActionEffectVerifier:
    """Defer commands whose gameplay effect must be observed after API acceptance."""

    def __init__(
        self,
        *,
        timeout_game_loops: int = DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS,
        unit_names: Optional[Mapping[int, str]] = None,
        placement_service: Optional[RawPlacementService] = None,
    ) -> None:
        if timeout_game_loops <= 0:
            raise ValueError("timeout_game_loops must be positive")
        self.timeout_game_loops = timeout_game_loops
        self.unit_names = {int(key): str(value) for key, value in (unit_names or {}).items()}
        self.placement_service = placement_service
        self._pending: dict[str, _PendingBuild] = {}
        self._pending_moves: dict[str, _PendingMove] = {}
        self._claimed_structure_tags: set[int] = set()
        self.production = ProductionEffectVerifier(
            timeout_game_loops=timeout_game_loops,
            unit_names=self.unit_names,
        )
        self.addons = AddonEffectVerifier(
            timeout_game_loops=timeout_game_loops,
            unit_names=self.unit_names,
        )
        self.morphs = MorphEffectVerifier(
            timeout_game_loops=timeout_game_loops,
            unit_names=self.unit_names,
        )
        self.injects = InjectEffectVerifier(
            timeout_game_loops=timeout_game_loops,
            unit_names=self.unit_names,
        )
        self.research = ResearchEffectVerifier(
            timeout_game_loops=timeout_game_loops,
            unit_names=self.unit_names,
        )
        self.mules = MuleEffectVerifier(
            timeout_game_loops=timeout_game_loops,
            unit_names=self.unit_names,
        )
        self.combat = CombatEffectVerifier(
            timeout_game_loops=timeout_game_loops,
            unit_names=self.unit_names,
        )

    def track(self, command: RoutedCommand) -> bool:
        """Register an effectful command, returning false for immediate actions."""

        if self.is_tracked(command.command_id):
            raise ValueError(f"command {command.command_id!r} is already tracked")
        if self.production.track(command):
            return True
        if self.addons.track(command):
            return True
        if self.morphs.track(command):
            return True
        if self.injects.track(command):
            return True
        if self.research.track(command):
            return True
        if self.mules.track(command):
            return True
        if self.combat.track(command):
            return True
        target = _target_structure(command.name)
        if target is None and command.name != "Move_Minimap":
            return False
        if command.name == "Move_Minimap":
            arguments = command.resolved_arguments or command.requested_arguments
            target_position = _position_argument(arguments)
            if target_position is None:
                raise ValueError("Move_Minimap requires one two-coordinate target")
            self._pending_moves[command.command_id] = _PendingMove(
                command,
                arguments,
                target_position,
            )
            return True
        assert target is not None
        self._pending[command.command_id] = _PendingBuild(
            command,
            target,
            command.resolved_arguments or command.requested_arguments,
        )
        return True

    def is_tracked(self, command_id: str) -> bool:
        return (
            command_id in self._pending
            or command_id in self._pending_moves
            or self.production.is_tracked(command_id)
            or self.addons.is_tracked(command_id)
            or self.morphs.is_tracked(command_id)
            or self.injects.is_tracked(command_id)
            or self.research.is_tracked(command_id)
            or self.mules.is_tracked(command_id)
            or self.combat.is_tracked(command_id)
        )

    @property
    def blocks_auto_worker_management(self) -> bool:
        """Keep upstream worker automation off throughout an in-flight build command."""

        return bool(self._pending)

    def resolve_arguments(self, command_id: str, arguments: list[Any]) -> None:
        if self.production.is_tracked(command_id):
            return
        if self.addons.is_tracked(command_id):
            return
        if self.morphs.is_tracked(command_id):
            return
        if self.injects.is_tracked(command_id):
            return
        if self.research.is_tracked(command_id):
            return
        if self.mules.is_tracked(command_id):
            self.mules.resolve_arguments(command_id, arguments)
            return
        if self.combat.is_tracked(command_id):
            self.combat.resolve_arguments(command_id, arguments)
            return
        pending_move = self._pending_moves.get(command_id)
        if pending_move is not None:
            target_position = _position_argument(arguments)
            if target_position is None:
                raise ValueError("Move_Minimap requires one two-coordinate target")
            pending_move.resolved_arguments = tuple(arguments)
            pending_move.target_position = target_position
            return
        self._get(command_id).resolved_arguments = tuple(arguments)

    def prepare(
        self,
        command_id: str,
        observation: Any,
        builder_tag: Optional[int],
        *,
        producer_tag: Optional[int] = None,
        actor_tags: Sequence[int] = (),
        minimap_transform: Optional[tuple[float, float, float, float, float]] = None,
    ) -> None:
        """Capture state immediately before the final effectful primitive."""

        if self.production.is_tracked(command_id):
            self.production.prepare(command_id, observation, producer_tag)
            return
        if self.addons.is_tracked(command_id):
            self.addons.prepare(command_id, observation, producer_tag)
            return
        if self.morphs.is_tracked(command_id):
            self.morphs.prepare(command_id, observation, producer_tag)
            return
        if self.injects.is_tracked(command_id):
            self.injects.prepare(command_id, observation, builder_tag)
            return
        if self.research.is_tracked(command_id):
            self.research.prepare(command_id, observation, producer_tag)
            return
        if self.mules.is_tracked(command_id):
            self.mules.prepare(command_id, observation, producer_tag)
            return
        if self.combat.is_tracked(command_id):
            self.combat.prepare(
                command_id,
                observation,
                tuple(int(tag) for tag in actor_tags),
            )
            return
        pending_move = self._pending_moves.get(command_id)
        if pending_move is not None:
            normalized_tags = tuple(
                dict.fromkeys(
                    int(tag)
                    for tag in (actor_tags or (() if builder_tag is None else (int(builder_tag),)))
                    if int(tag) > 0
                )
            )
            pending_move.actor_tags = normalized_tags
            pending_move.actor_tag = normalized_tags[0] if normalized_tags else None
            pending_move.minimap_transform = minimap_transform
            pending_move.dispatched_game_loop = _game_loop(observation)
            pending_move.latest_game_loop = pending_move.dispatched_game_loop
            actors = _units_by_tags(observation, pending_move.actor_tags)
            pending_move.baseline_actor_position = _movement_centroid(actors, minimap_transform)
            pending_move.latest_actor_position = pending_move.baseline_actor_position
            pending_move.latest_actor_orders = _combined_unit_orders(actors)
            return

        pending = self._get(command_id)
        pending.builder_tag = None if builder_tag is None else int(builder_tag)
        if self.placement_service is not None:
            pending.reservation_snapshot = self.placement_service.command_target(command_id)
        self._resolve_target(pending, observation)
        pending.baseline = self._evidence(pending, observation)
        pending.latest = pending.baseline

    def accept_primitive(self, command_id: str, *, game_loop: int) -> None:
        """Mark the PySC2 primitive accepted while keeping the report deferred."""

        if self.production.is_tracked(command_id):
            self.production.accept_primitive(command_id, game_loop=game_loop)
            return
        if self.addons.is_tracked(command_id):
            self.addons.accept_primitive(command_id, game_loop=game_loop)
            return
        if self.morphs.is_tracked(command_id):
            self.morphs.accept_primitive(command_id, game_loop=game_loop)
            return
        if self.injects.is_tracked(command_id):
            self.injects.accept_primitive(command_id, game_loop=game_loop)
            return
        if self.research.is_tracked(command_id):
            self.research.accept_primitive(command_id, game_loop=game_loop)
            return
        if self.mules.is_tracked(command_id):
            self.mules.accept_primitive(command_id, game_loop=game_loop)
            return
        if self.combat.is_tracked(command_id):
            self.combat.accept_primitive(command_id, game_loop=game_loop)
            return
        pending_move = self._pending_moves.get(command_id)
        if pending_move is not None:
            if pending_move.dispatched_game_loop is None:
                raise RuntimeError(f"effect baseline was not prepared for command {command_id!r}")
            pending_move.accepted_game_loop = int(game_loop)
            return

        pending = self._get(command_id)
        if pending.baseline is None:
            raise RuntimeError(f"effect baseline was not prepared for command {command_id!r}")
        pending.accepted_game_loop = int(game_loop)

    def cancel(self, command_id: str) -> None:
        self._pending.pop(command_id, None)
        self._pending_moves.pop(command_id, None)
        if self.placement_service is not None:
            self.placement_service.release_command(command_id)
        self.production.cancel(command_id)
        self.addons.cancel(command_id)
        self.morphs.cancel(command_id)
        self.injects.cancel(command_id)
        self.research.cancel(command_id)
        self.mules.cancel(command_id)
        self.combat.cancel(command_id)

    def observe(self, observation: Any) -> list[EffectVerdict]:
        """Evaluate all accepted effectful commands against one observation."""

        verdicts = self.production.observe(observation)
        verdicts.extend(self.addons.observe(observation))
        verdicts.extend(self.morphs.observe(observation))
        verdicts.extend(self.injects.observe(observation))
        verdicts.extend(self.research.observe(observation))
        verdicts.extend(self.mules.observe(observation))
        verdicts.extend(self.combat.observe(observation))
        verdicts.extend(self._observe_moves(observation))
        accepted = [
            pending
            for pending in self._pending.values()
            if pending.accepted_game_loop is not None and pending.baseline is not None
        ]
        current_by_command = {
            pending.command.command_id: self._evidence(pending, observation) for pending in accepted
        }
        for pending in accepted:
            current = current_by_command[pending.command.command_id]
            pending.latest = current
            expected_order = BUILD_RAW_FUNCTION_IDS.get(pending.target_structure)
            if (
                expected_order is not None
                and current.builder is not None
                and expected_order in current.builder.orders
            ):
                pending.order_seen = True
                pending.order_last_seen_game_loop = current.game_loop
                self._confirm_build_started(
                    pending,
                    current,
                    confirmation_kind="builder_order",
                )
            self._update_supporting_build_start_evidence(pending, current)
        assignments = self._match_new_structures(accepted, current_by_command)
        self._claimed_structure_tags.update(structure.tag for structure in assignments.values())
        for command_id, structure in assignments.items():
            pending = self._pending.pop(command_id)
            current = current_by_command[command_id]
            pending.latest = current
            self._confirm_build_started(
                pending,
                current,
                confirmation_kind="new_structure",
            )
            evidence = self._effect_evidence(pending, current, structure)
            if self.placement_service is not None:
                self.placement_service.confirm_command(
                    command_id,
                    game_loop=current.game_loop,
                )
            verdicts.append(
                EffectVerdict(
                    command_id,
                    True,
                    status="succeeded",
                    evidence=evidence,
                )
            )

        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None or pending.baseline is None:
                continue
            current = current_by_command[command_id]
            elapsed = current.game_loop - pending.accepted_game_loop
            if elapsed < self.timeout_game_loops:
                continue
            expected_order = BUILD_RAW_FUNCTION_IDS.get(pending.target_structure)
            order_is_active = (
                expected_order is not None
                and current.builder is not None
                and expected_order in current.builder.orders
            )
            hard_timeout = self._active_order_timeout(pending)
            within_order_grace = (
                pending.order_last_seen_game_loop is not None
                and current.game_loop - pending.order_last_seen_game_loop
                < POST_ORDER_EFFECT_GRACE_GAME_LOOPS
            )
            if elapsed < hard_timeout and (
                pending.build_started or order_is_active or within_order_grace
            ):
                pending.active_order_extension = True
                continue
            if elapsed >= self.timeout_game_loops:
                failure_code = self._timeout_code(pending, current)
                evidence = self._effect_evidence(pending, current, None)
                if self.placement_service is not None:
                    reservation = pending.reservation_snapshot
                    self.placement_service.quarantine_command(
                        command_id=command_id,
                        action_name=pending.command.name,
                        requested_arguments=pending.command.requested_arguments,
                        world_target=pending.target_position,
                        failure_code=failure_code,
                        game_loop=current.game_loop,
                        operation_id=pending.command.operation_id,
                        attempt_ordinal=(
                            None if reservation is None else reservation.attempt_ordinal
                        ),
                        builder_tag=pending.builder_tag,
                        placement_revision=(
                            None if reservation is None else reservation.placement_revision
                        ),
                        target_state_revision=(
                            None if reservation is None else reservation.target_state_revision
                        ),
                        failure_classification=evidence.get("failure_classification"),
                        classification_basis=evidence.get("classification_basis", ()),
                        observation=observation,
                    )
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        False,
                        self._timeout_reason(pending, current),
                        status="failed",
                        failure_code=failure_code,
                        evidence=evidence,
                    )
                )
                del self._pending[command_id]
        return verdicts

    def _observe_moves(self, observation: Any) -> list[EffectVerdict]:
        game_loop = _game_loop(observation)
        verdicts: list[EffectVerdict] = []
        for command_id, pending in list(self._pending_moves.items()):
            if pending.accepted_game_loop is None:
                continue
            pending.latest_game_loop = game_loop
            actors = _units_by_tags(observation, pending.actor_tags)
            pending.latest_actor_position = _movement_centroid(actors, pending.minimap_transform)
            pending.latest_actor_orders = _combined_unit_orders(actors)
            if MOVE_RAW_FUNCTION_IDS.intersection(pending.latest_actor_orders):
                pending.move_order_seen = True
            distance_to_target = _optional_position_distance(
                pending.latest_actor_position,
                pending.target_position,
            )
            if distance_to_target is not None and distance_to_target <= MOVE_MINIMAP_ARRIVAL_RADIUS:
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        True,
                        status="succeeded",
                        evidence=self._move_effect_evidence(pending, confirmed=True),
                    )
                )
                del self._pending_moves[command_id]
                continue
            elapsed = game_loop - pending.accepted_game_loop
            if not actors and elapsed >= self.timeout_game_loops:
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        False,
                        (
                            "Move_Minimap did not arrive after "
                            f"{elapsed} game loops (actor is not observable)"
                        ),
                        status="failed",
                        failure_code="actor_not_observable",
                        evidence=self._move_effect_evidence(pending, confirmed=False),
                    )
                )
                del self._pending_moves[command_id]
                continue
            if not pending.move_order_seen and elapsed >= MOVE_ORDER_ACQUISITION_TIMEOUT_GAME_LOOPS:
                verdicts.append(
                    EffectVerdict(
                        command_id,
                        False,
                        (
                            "Move_Minimap was accepted but no bound actor received "
                            f"raw move order {MOVE_RAW_FUNCTION_ID}"
                        ),
                        status="failed",
                        failure_code="move_order_not_observed",
                        evidence=self._move_effect_evidence(pending, confirmed=False),
                    )
                )
                del self._pending_moves[command_id]
                continue
            effective_timeout = self._move_effective_timeout(pending)
            if elapsed < effective_timeout:
                continue
            actor_detail = (
                "actor is not observable"
                if not actors
                else (
                    f"group centroid {pending.latest_actor_position} remained "
                    f"{distance_to_target!r} minimap units from {pending.target_position}"
                )
            )
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    f"Move_Minimap did not arrive after {elapsed} game loops ({actor_detail})",
                    status="failed",
                    failure_code="actor_not_observable" if not actors else "effect_timeout",
                    evidence=self._move_effect_evidence(pending, confirmed=False),
                )
            )
            del self._pending_moves[command_id]
        return verdicts

    def fail_pending(self, reason: str) -> list[EffectVerdict]:
        """Mark accepted commands unconfirmed when no later observation can arrive."""

        verdicts = self.production.fail_pending(reason)
        verdicts.extend(self.addons.fail_pending(reason))
        verdicts.extend(self.morphs.fail_pending(reason))
        verdicts.extend(self.injects.fail_pending(reason))
        verdicts.extend(self.research.fail_pending(reason))
        verdicts.extend(self.mules.fail_pending(reason))
        verdicts.extend(self.combat.fail_pending(reason))
        for command_id, pending in list(self._pending.items()):
            if pending.accepted_game_loop is None:
                continue
            current = pending.latest or pending.baseline
            detail = self._diagnostic(pending, current) if current is not None else ""
            separator = ": " if detail else ""
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    f"{reason}{separator}{detail}" or reason,
                    status="unconfirmed",
                    failure_code="episode_ended_unconfirmed",
                    evidence=(
                        None if current is None else self._effect_evidence(pending, current, None)
                    ),
                )
            )
            if self.placement_service is not None:
                self.placement_service.release_command(
                    command_id,
                    game_loop=(0 if current is None else current.game_loop),
                    reason="episode_ended_unconfirmed",
                )
            del self._pending[command_id]
        for command_id, pending_move in list(self._pending_moves.items()):
            if pending_move.accepted_game_loop is None:
                continue
            verdicts.append(
                EffectVerdict(
                    command_id,
                    False,
                    f"{reason}: Move_Minimap movement was not observed",
                    status="unconfirmed",
                    failure_code="episode_ended_unconfirmed",
                    evidence=self._move_effect_evidence(pending_move, confirmed=False),
                )
            )
            del self._pending_moves[command_id]
        self._claimed_structure_tags.clear()
        return verdicts

    def _move_effect_evidence(
        self,
        pending: _PendingMove,
        *,
        confirmed: bool,
    ) -> dict[str, Any]:
        current_loop = pending.latest_game_loop or pending.dispatched_game_loop
        elapsed = (
            0
            if pending.accepted_game_loop is None or current_loop is None
            else max(0, current_loop - pending.accepted_game_loop)
        )
        return {
            "effect_kind": "move",
            "target_type": "Move_Minimap",
            "target_position": pending.target_position,
            "target_tag": None,
            "actor_tag": None if pending.actor_tag is None else hex(pending.actor_tag),
            "actor_tags": [hex(tag) for tag in pending.actor_tags],
            "builder_tag": None if pending.actor_tag is None else hex(pending.actor_tag),
            "baseline_structure_tags": [],
            "observed_structure_tag": None,
            "dispatched_loop": pending.dispatched_game_loop,
            "accepted_loop": pending.accepted_game_loop,
            "confirmed_loop": current_loop if confirmed else None,
            "worker_orders": [str(order) for order in pending.latest_actor_orders],
            "resource_delta": {},
            "order_seen": pending.move_order_seen,
            "order_last_seen_game_loop": None,
            "post_order_grace_game_loops": None,
            "mineral_delta": None,
            "elapsed_game_loops": elapsed,
            "base_timeout_game_loops": self.timeout_game_loops,
            "effective_timeout_game_loops": self._move_effective_timeout(pending),
            "active_order_extension": False,
            "baseline_actor_position": pending.baseline_actor_position,
            "observed_actor_position": pending.latest_actor_position,
            "actor_displacement": _optional_position_distance(
                pending.baseline_actor_position,
                pending.latest_actor_position,
            ),
            "baseline_builder_position": pending.baseline_actor_position,
            "observed_builder_position": pending.latest_actor_position,
            "builder_displacement": _optional_position_distance(
                pending.baseline_actor_position,
                pending.latest_actor_position,
            ),
            "move_order_seen": pending.move_order_seen,
        }

    def _move_effective_timeout(self, pending: _PendingMove) -> int:
        distance = _optional_position_distance(
            pending.baseline_actor_position,
            pending.target_position,
        )
        if distance is None:
            return self.timeout_game_loops
        travel_budget = math.ceil(
            distance * MOVE_GAME_LOOPS_PER_MINIMAP_UNIT + MOVE_SETTLEMENT_GRACE_GAME_LOOPS
        )
        return max(self.timeout_game_loops, travel_budget)

    def _evidence(self, pending: _PendingBuild, observation: Any) -> _Evidence:
        raw_units = list(_value(observation, "raw_units", ()))
        screen_by_tag = {
            int(_value(unit, "tag", 0)): (
                float(_value(unit, "x", 0.0)),
                float(_value(unit, "y", 0.0)),
            )
            for unit in _value(observation, "feature_units", ())
            if bool(_value(unit, "is_on_screen", True))
        }
        target_units = [
            unit
            for unit in raw_units
            if int(_value(unit, "alliance", 0)) == 1
            and self._unit_name(unit) in _target_structure_names(pending.target_structure)
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
            observation_revision=_placement_revision(observation),
            structures=tuple(
                _StructureEvidence(
                    tag=int(_value(unit, "tag", 0)),
                    position=(
                        float(_value(unit, "x", 0.0)),
                        float(_value(unit, "y", 0.0)),
                    ),
                    screen_position=screen_by_tag.get(int(_value(unit, "tag", 0))),
                    progress=_build_progress(unit),
                )
                for unit in target_units
            ),
            occupants=tuple(
                (
                    int(_value(unit, "tag", 0)),
                    (
                        float(_value(unit, "x", 0.0)),
                        float(_value(unit, "y", 0.0)),
                    ),
                )
                for unit in raw_units
                if int(_value(unit, "alliance", 0)) == 1
            ),
            minerals=int(_value(player, "minerals", 0)),
            builder=None if builder is None else _builder_evidence(builder),
            dynamic_units=tuple(
                _DynamicUnitEvidence(
                    tag=int(_value(unit, "tag", 0)),
                    alliance=int(_value(unit, "alliance", 0)),
                    position=(
                        float(_value(unit, "x", 0.0)),
                        float(_value(unit, "y", 0.0)),
                    ),
                    radius=max(0.0, float(_value(unit, "radius", 0.5))),
                )
                for unit in raw_units
                if int(_value(unit, "tag", 0)) > 0
                and int(_value(unit, "tag", 0)) != pending.builder_tag
                and int(_value(unit, "tag", 0)) != pending.target_tag
                and int(_value(unit, "display_type", 1)) == 1
                and not bool(_value(unit, "is_structure", False))
                and not bool(_value(unit, "is_flying", False))
                and not _is_resource_name(self._unit_name(unit))
            ),
        )

    def _unit_name(self, unit: Any) -> str:
        value = _value(unit, "unit_type", "")
        if isinstance(value, str):
            return value
        return self.unit_names.get(int(value), f"unit:{int(value)}")

    def _resolve_target(self, pending: _PendingBuild, observation: Any) -> None:
        if self.placement_service is not None:
            placement = self.placement_service.command_target(pending.command.command_id)
            if placement is not None:
                pending.target_position = placement.world_target
                pending.target_tag = placement.anchor_tag
                pending.coordinate_space = "world"
                return
        if pending.command.screen_world_target is not None:
            pending.target_position = (
                float(pending.command.screen_world_target[0]),
                float(pending.command.screen_world_target[1]),
            )
            pending.coordinate_space = "world"
            return
        if not pending.resolved_arguments:
            return
        raw_units = list(_value(observation, "raw_units", ()))
        if pending.command.name.endswith("_Near") and pending.command.requested_arguments:
            pending.target_tag = _parse_tag(pending.command.requested_arguments[0])
            spec = BUILD_SPECS.get(pending.command.name)
            if (
                spec is not None
                and spec.placement_kind == "geyser"
                and pending.target_tag is not None
            ):
                for unit in raw_units:
                    if int(_value(unit, "tag", -1)) == pending.target_tag:
                        pending.target_position = (
                            float(_value(unit, "x", 0.0)),
                            float(_value(unit, "y", 0.0)),
                        )
                        pending.coordinate_space = "world"
                        return
        value = pending.resolved_arguments[0]
        if isinstance(value, (list, tuple)):
            if len(value) == 2 and all(
                isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
                for coordinate in value
            ):
                pending.target_position = _screen_to_world(
                    observation,
                    (float(value[0]), float(value[1])),
                    pending.builder_tag,
                )
                pending.coordinate_space = "world"
            return
        tag = _parse_tag(value)
        if tag is None:
            return
        pending.target_tag = tag
        for unit in raw_units:
            if int(_value(unit, "tag", -1)) == tag:
                pending.target_position = (
                    float(_value(unit, "x", 0.0)),
                    float(_value(unit, "y", 0.0)),
                )
                pending.coordinate_space = "world"
                if spec is not None and spec.placement_kind == "expansion":
                    resources = [
                        candidate
                        for candidate in raw_units
                        if int(_value(candidate, "alliance", 0)) == 3
                        and _is_resource_name(self._unit_name(candidate))
                        and _position_distance(
                            pending.target_position,
                            (
                                float(_value(candidate, "x", 0.0)),
                                float(_value(candidate, "y", 0.0)),
                            ),
                        )
                        <= 12.0
                    ]
                    if resources:
                        pending.target_position = (
                            sum(float(_value(item, "x", 0.0)) for item in resources)
                            / len(resources),
                            sum(float(_value(item, "y", 0.0)) for item in resources)
                            / len(resources),
                        )
                break

    def _match_new_structures(
        self,
        pending_builds: list[_PendingBuild],
        current_by_command: Mapping[str, _Evidence],
    ) -> dict[str, _StructureEvidence]:
        pairs: list[tuple[float, str, int, _StructureEvidence]] = []
        for pending in pending_builds:
            command_id = pending.command.command_id
            baseline = pending.baseline
            if baseline is None:
                continue
            baseline_tags = {structure.tag for structure in baseline.structures}
            for structure in current_by_command[command_id].structures:
                if structure.tag in baseline_tags or structure.tag in self._claimed_structure_tags:
                    continue
                distance = self._target_distance(pending, structure)
                if distance is not None:
                    pairs.append((distance, command_id, structure.tag, structure))
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        assigned_commands: set[str] = set()
        assigned_structures: set[int] = set()
        assignments = {}
        for _, command_id, structure_tag, structure in pairs:
            if command_id in assigned_commands or structure_tag in assigned_structures:
                continue
            assignments[command_id] = structure
            assigned_commands.add(command_id)
            assigned_structures.add(structure_tag)
        return assignments

    @staticmethod
    def _target_distance(
        pending: _PendingBuild,
        structure: _StructureEvidence,
    ) -> Optional[float]:
        target = pending.target_position
        if target is None:
            return None
        distance = _position_distance(target, structure.position)
        if pending.command.name.endswith("_Screen"):
            spec = BUILD_SPECS.get(pending.command.name)
            tolerance = 2.0 if spec is None else max(2.0, spec.footprint / 2.0 + 1.0)
        else:
            spec = BUILD_SPECS.get(pending.command.name)
            tolerance = 1.5 if spec is not None and spec.placement_kind == "geyser" else 4.0
        return distance if distance <= tolerance else None

    def _timeout_reason(self, pending: _PendingBuild, current: _Evidence) -> str:
        accepted_loop = pending.accepted_game_loop
        elapsed = 0 if accepted_loop is None else current.game_loop - accepted_loop
        maximum = (
            self._active_order_timeout(pending)
            if pending.active_order_extension
            else self.timeout_game_loops
        )
        return (
            f"{pending.command.name} primitive accepted by PySC2 but no gameplay effect "
            f"confirmed after {elapsed} game loops "
            f"(base timeout {self.timeout_game_loops}, maximum {maximum}): "
            f"{self._diagnostic(pending, current)}"
        )

    @staticmethod
    def _timeout_code(pending: _PendingBuild, current: _Evidence) -> str:
        del current
        if pending.build_started:
            return "build_started_effect_missing"
        return "no_build_start_evidence"

    @staticmethod
    def _diagnostic(pending: _PendingBuild, current: _Evidence) -> str:
        baseline = pending.baseline
        if baseline is None:
            return "effect baseline unavailable"
        builder = _builder_change(baseline.builder, current.builder)
        diagnosis = _diagnosis(
            baseline,
            current,
            pending.target_structure,
            order_seen=pending.order_seen,
        )
        baseline_tags = [hex(structure.tag) for structure in baseline.structures]
        current_tags = [hex(structure.tag) for structure in current.structures]
        return (
            f"{pending.target_structure} tags {baseline_tags}->{current_tags}; "
            f"minerals {baseline.minerals}->{current.minerals}; {builder}; "
            f"diagnosis: {diagnosis}"
        )

    def _effect_evidence(
        self,
        pending: _PendingBuild,
        current: _Evidence,
        structure: Optional[_StructureEvidence],
    ) -> dict[str, Any]:
        baseline = pending.baseline
        reservation = pending.reservation_snapshot
        failure_classification, classification_basis = self._failure_classification(
            pending,
            current,
            structure,
        )
        nearby_units = self._nearby_dynamic_units(pending, current)
        nearby_enemies = self._nearby_enemy_units(pending, current)
        return {
            "effect_kind": "build",
            "target_type": pending.target_structure,
            "target_position": pending.target_position,
            "requested_target_position": (
                None if reservation is None else reservation.requested_world_target
            ),
            "final_validated_target_position": (
                None if reservation is None else reservation.final_validated_world_target
            ),
            "validated_target_position": (
                None if reservation is None else reservation.final_validated_world_target
            ),
            "emitted_target_position": (None if reservation is None else reservation.world_target),
            "verified_target_position": pending.target_position,
            "target_tag": None if pending.target_tag is None else hex(pending.target_tag),
            "builder_tag": None if pending.builder_tag is None else hex(pending.builder_tag),
            "reservation_id": None if reservation is None else reservation.reservation_id,
            "placement_revision": (None if reservation is None else reservation.placement_revision),
            "placement_state": (None if reservation is None else reservation.placement_state),
            "placement_episode_id": (None if reservation is None else reservation.episode_id),
            "footprint_width": (None if reservation is None else reservation.footprint_width),
            "footprint_height": (None if reservation is None else reservation.footprint_height),
            "occupied_grid_cells": (
                [] if reservation is None else sorted(reservation.occupied_grid_cells)
            ),
            "baseline_builder_orders": (
                list(baseline.builder.orders)
                if baseline is not None and baseline.builder is not None
                else []
            ),
            "failure_classification": failure_classification,
            "classification_basis": classification_basis,
            "nearby_enemy_units": [hex(unit.tag) for unit in nearby_enemies],
            "nearby_dynamic_occupants": [hex(unit.tag) for unit in nearby_units],
            "builder_status": (None if current.builder is None else current.builder.status),
            "baseline_builder_status": (
                None if baseline is None or baseline.builder is None else baseline.builder.status
            ),
            "builder_alliance": (None if current.builder is None else current.builder.alliance),
            "builder_health": (None if current.builder is None else current.builder.health),
            "builder_health_max": (None if current.builder is None else current.builder.health_max),
            "observation_revision": (None if baseline is None else baseline.observation_revision),
            "failure_observation_revision": current.observation_revision,
            "placement_query_result": "unavailable_no_controller_access",
            "baseline_structure_tags": (
                [] if baseline is None else [hex(item.tag) for item in baseline.structures]
            ),
            "observed_structure_tag": None if structure is None else hex(structure.tag),
            "dispatched_loop": None if baseline is None else baseline.game_loop,
            "accepted_loop": pending.accepted_game_loop,
            "confirmed_loop": current.game_loop if structure is not None else None,
            "worker_orders": (
                [] if current.builder is None else [str(order) for order in current.builder.orders]
            ),
            "resource_delta": {
                "minerals": 0 if baseline is None else current.minerals - baseline.minerals,
            },
            "order_seen": pending.order_seen,
            "build_started": pending.build_started,
            "build_start_confirmation_kind": pending.build_start_confirmation_kind,
            "build_start_confirmed_game_loop": pending.build_start_confirmed_game_loop,
            "resource_debit_seen": pending.resource_debit_seen,
            "builder_approach_seen": pending.builder_approach_seen,
            "target_occupancy_seen": pending.target_occupancy_seen,
            "order_last_seen_game_loop": pending.order_last_seen_game_loop,
            "post_order_grace_game_loops": POST_ORDER_EFFECT_GRACE_GAME_LOOPS,
            "mineral_delta": 0 if baseline is None else baseline.minerals - current.minerals,
            "elapsed_game_loops": (
                0
                if pending.accepted_game_loop is None
                else current.game_loop - pending.accepted_game_loop
            ),
            "base_timeout_game_loops": self.timeout_game_loops,
            "effective_timeout_game_loops": (
                self._active_order_timeout(pending)
                if pending.active_order_extension
                else self.timeout_game_loops
            ),
            "active_order_extension": pending.active_order_extension,
            "baseline_builder_position": (
                None if baseline is None or baseline.builder is None else baseline.builder.position
            ),
            "observed_builder_position": (
                None if current.builder is None else current.builder.position
            ),
            "builder_displacement": (
                None
                if baseline is None or baseline.builder is None or current.builder is None
                else _position_distance(
                    baseline.builder.position,
                    current.builder.position,
                )
            ),
            "confirmation_kind": (
                "new_structure" if structure is not None else pending.build_start_confirmation_kind
            ),
        }

    def _failure_classification(
        self,
        pending: _PendingBuild,
        current: _Evidence,
        structure: Optional[_StructureEvidence],
    ) -> tuple[str | None, list[str]]:
        if structure is not None:
            return None, []
        if pending.build_started:
            return "gameplay_effect_missing_after_start", ["build_start_evidence_without_structure"]
        baseline_builder = None if pending.baseline is None else pending.baseline.builder
        if baseline_builder is None:
            return "builder_not_ready", ["builder_missing_at_dispatch"]
        if baseline_builder.alliance != 1:
            return "builder_not_ready", ["builder_not_owned_at_dispatch"]
        if baseline_builder.health <= 0 or current.builder is None:
            return "builder_not_ready", ["builder_unavailable_after_dispatch"]
        build_order_ids = frozenset(BUILD_RAW_FUNCTION_IDS.values())
        if build_order_ids.intersection(baseline_builder.orders):
            return "builder_not_ready", ["builder_had_prior_build_order"]
        if self._nearby_dynamic_units(pending, current):
            return "dynamic_target_obstruction", ["dynamic_unit_inside_footprint"]
        if self._nearby_enemy_units(pending, current):
            return "gameplay_no_start_unknown", [
                "nearby_enemy_threat_without_target_obstruction",
                "no_authoritative_rejection_evidence",
            ]
        return "gameplay_no_start_unknown", ["no_authoritative_rejection_evidence"]

    @staticmethod
    def _nearby_dynamic_units(
        pending: _PendingBuild,
        current: _Evidence,
    ) -> tuple[_DynamicUnitEvidence, ...]:
        target = pending.target_position
        if target is None:
            return ()
        spec = BUILD_SPECS.get(pending.command.name)
        if pending.reservation_snapshot is not None:
            footprint_cells = pending.reservation_snapshot.occupied_grid_cells
        elif spec is not None:
            footprint_cells = _occupied_cells_for_spec(target, spec)
        else:
            return ()
        candidates = (
            (*pending.baseline.dynamic_units, *current.dynamic_units)
            if pending.baseline is not None
            else current.dynamic_units
        )
        by_tag: dict[int, _DynamicUnitEvidence] = {}
        for unit in candidates:
            if _circle_intersects_cells(unit.position, unit.radius, footprint_cells):
                by_tag[unit.tag] = unit
        return tuple(by_tag[tag] for tag in sorted(by_tag))

    @staticmethod
    def _nearby_enemy_units(
        pending: _PendingBuild,
        current: _Evidence,
    ) -> tuple[_DynamicUnitEvidence, ...]:
        anchors = [pending.target_position]
        if pending.baseline is not None and pending.baseline.builder is not None:
            anchors.append(pending.baseline.builder.position)
        if current.builder is not None:
            anchors.append(current.builder.position)
        positions = [anchor for anchor in anchors if anchor is not None]
        if not positions:
            return ()
        candidates = (
            (*pending.baseline.dynamic_units, *current.dynamic_units)
            if pending.baseline is not None
            else current.dynamic_units
        )
        by_tag = {
            unit.tag: unit
            for unit in candidates
            if unit.alliance == 4
            and any(
                _position_distance(unit.position, anchor) <= 8.0 + unit.radius
                for anchor in positions
            )
        }
        return tuple(by_tag[tag] for tag in sorted(by_tag))

    def _confirm_build_started(
        self,
        pending: _PendingBuild,
        current: _Evidence,
        *,
        confirmation_kind: str,
    ) -> None:
        if pending.build_started:
            return
        pending.build_started = True
        pending.build_start_confirmation_kind = confirmation_kind
        pending.build_start_confirmed_game_loop = current.game_loop
        if self.placement_service is not None:
            accepted = pending.accepted_game_loop or current.game_loop
            self.placement_service.mark_build_started(
                pending.command.command_id,
                game_loop=current.game_loop,
                expires_game_loop=(
                    accepted
                    + build_effect_max_lifetime(
                        self.timeout_game_loops,
                        pending.command.name,
                    )
                    + POST_ORDER_EFFECT_GRACE_GAME_LOOPS
                ),
            )

    def _update_supporting_build_start_evidence(
        self,
        pending: _PendingBuild,
        current: _Evidence,
    ) -> None:
        baseline = pending.baseline
        target = pending.target_position
        if baseline is None:
            return
        spec = BUILD_SPECS.get(pending.command.name)
        mineral_cost = 0 if spec is None else spec.mineral_cost
        pending.resource_debit_seen = pending.resource_debit_seen or (
            mineral_cost > 0 and baseline.minerals - current.minerals >= mineral_cost
        )
        if baseline.builder is not None and current.builder is not None and target is not None:
            baseline_distance = _position_distance(baseline.builder.position, target)
            current_distance = _position_distance(current.builder.position, target)
            pending.builder_approach_seen = pending.builder_approach_seen or (
                current_distance + 1.0 <= baseline_distance
                or current_distance <= max(3.0, float(spec.footprint if spec else 2))
            )
        if target is not None:
            baseline_tags = {tag for tag, _ in baseline.occupants}
            footprint_radius = max(1.5, float(spec.footprint if spec else 2) / 2.0)
            pending.target_occupancy_seen = pending.target_occupancy_seen or any(
                tag not in baseline_tags
                and _position_distance(position, target) <= footprint_radius
                for tag, position in current.occupants
            )
        if pending.resource_debit_seen and (
            pending.builder_approach_seen or pending.target_occupancy_seen
        ):
            self._confirm_build_started(
                pending,
                current,
                confirmation_kind="supporting_quorum",
            )

    def _active_order_timeout(self, pending: _PendingBuild) -> int:
        return build_effect_max_lifetime(self.timeout_game_loops, pending.command.name)

    def _get(self, command_id: str) -> _PendingBuild:
        try:
            return self._pending[command_id]
        except KeyError as error:
            raise KeyError(f"unknown effect command {command_id!r}") from error


def _target_structure(action_name: str) -> Optional[str]:
    spec = BUILD_SPECS.get(action_name)
    if spec is not None:
        return spec.target_structure
    if not action_name.startswith("Build_"):
        return None
    stem = action_name.removeprefix("Build_")
    for suffix in ("_Screen", "_Near"):
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return None


def _target_structure_names(target_structure: str) -> frozenset[str]:
    if target_structure.startswith("CreepTumor"):
        return frozenset({"CreepTumor", "CreepTumorBurrowed", "CreepTumorQueen"})
    return frozenset({target_structure})


def _position_argument(values: Sequence[Any]) -> Optional[tuple[float, float]]:
    for value in values:
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
                for coordinate in value
            )
        ):
            return float(value[0]), float(value[1])
    return None


def _unit_by_tag(observation: Any, tag: Optional[int]) -> Optional[Any]:
    if tag is None:
        return None
    return next(
        (
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "tag", -1)) == tag
        ),
        None,
    )


def _units_by_tags(observation: Any, tags: Sequence[int]) -> tuple[Any, ...]:
    wanted = set(tags)
    return tuple(
        unit
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "tag", -1)) in wanted and int(_value(unit, "alliance", 1)) == 1
    )


def _combined_unit_orders(units: Sequence[Any]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(order for unit in units for order in _unit_orders(unit)))


def _minimap_centroid(
    units: Sequence[Any],
    transform: tuple[float, float, float, float, float],
) -> Optional[tuple[float, float]]:
    positions = [_world_to_minimap_position(_unit_position(unit), transform) for unit in units]
    present = [position for position in positions if position is not None]
    if not present:
        return None
    return (
        sum(position[0] for position in present) / len(present),
        sum(position[1] for position in present) / len(present),
    )


def _movement_centroid(
    units: Sequence[Any],
    transform: Optional[tuple[float, float, float, float, float]],
) -> Optional[tuple[float, float]]:
    """Use raw-grid positions directly in raw mode, transform legacy world data."""

    if transform is None:
        positions = [_unit_position(unit) for unit in units]
        present = [position for position in positions if position is not None]
        if not present:
            return None
        return (
            sum(position[0] for position in present) / len(present),
            sum(position[1] for position in present) / len(present),
        )
    return _minimap_centroid(units, transform)


def _world_to_minimap_position(
    position: Optional[tuple[float, float]],
    transform: tuple[float, float, float, float, float],
) -> Optional[tuple[float, float]]:
    if position is None:
        return None
    scale, x_offset, y_offset, world_range, maximum = transform
    return (
        max(0.0, min(maximum, (position[0] + x_offset) * scale)),
        max(0.0, min(maximum, (world_range - position[1] + y_offset) * scale)),
    )


def _unit_position(unit: Optional[Any]) -> Optional[tuple[float, float]]:
    if unit is None:
        return None
    return float(_value(unit, "x", 0.0)), float(_value(unit, "y", 0.0))


def _parse_tag(value: Any) -> Optional[int]:
    try:
        return int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return None


def _builder_evidence(unit: Any) -> _BuilderEvidence:
    orders = _unit_orders(unit)
    return _BuilderEvidence(
        tag=int(_value(unit, "tag", 0)),
        alliance=int(_value(unit, "alliance", 0)),
        status="active" if int(_value(unit, "order_length", len(orders))) > 0 else "idle",
        orders=orders,
        selected=bool(_value(unit, "is_selected", False)),
        position=(
            float(_value(unit, "x", 0.0)),
            float(_value(unit, "y", 0.0)),
        ),
        health=float(_value(unit, "health", 1.0)),
        health_max=float(_value(unit, "health_max", 1.0)),
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


def _position_distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _optional_position_distance(
    left: Optional[tuple[float, float]],
    right: Optional[tuple[float, float]],
) -> Optional[float]:
    if left is None or right is None:
        return None
    return _position_distance(left, right)


def _screen_to_world(
    observation: Any,
    target: tuple[float, float],
    builder_tag: Optional[int],
) -> Optional[tuple[float, float]]:
    raw_by_tag = {
        int(_value(unit, "tag", 0)): unit for unit in _value(observation, "raw_units", ())
    }
    feature_by_tag = {
        int(_value(unit, "tag", 0)): unit
        for unit in _value(observation, "feature_units", ())
        if bool(_value(unit, "is_on_screen", True))
    }
    shared_tags = sorted(raw_by_tag.keys() & feature_by_tag.keys())
    if not shared_tags:
        return None
    reference_tag = builder_tag if builder_tag in shared_tags else shared_tags[0]
    assert reference_tag is not None
    raw_reference = raw_by_tag[reference_tag]
    feature_reference = feature_by_tag[reference_tag]
    feature_screen = _value(observation, "feature_screen", None)
    buildable = _value(feature_screen, "buildable", None)
    shape = getattr(buildable, "shape", ())
    screen_size = float(shape[0]) if shape else 128.0
    scale = screen_size / 24.0
    return (
        float(_value(raw_reference, "x", 0.0))
        + (target[0] - float(_value(feature_reference, "x", 0.0))) / scale,
        float(_value(raw_reference, "y", 0.0))
        + (target[1] - float(_value(feature_reference, "y", 0.0))) / scale,
    )


def _is_resource_name(name: str) -> bool:
    normalized = name.casefold()
    return "mineralfield" in normalized or "geyser" in normalized


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
        f"orders {list(baseline.orders)}->{list(current.orders)}"
    )


def _diagnosis(
    baseline: _Evidence,
    current: _Evidence,
    target_structure: str,
    *,
    order_seen: bool,
) -> str:
    if baseline.builder is None:
        return "builder tag was unavailable at RAW dispatch"
    if current.builder is None:
        return "RAW builder disappeared before construction became visible"

    expected_order = BUILD_RAW_FUNCTION_IDS.get(target_structure)
    if expected_order is not None and expected_order in current.builder.orders:
        return "expected build order remains active but the target structure is not visible"
    if order_seen and current.builder.orders:
        return "expected build order was observed and later changed to a different order"
    if order_seen:
        return "expected build order was observed and later ended without a target structure"
    if current.minerals >= baseline.minerals:
        return (
            "expected build order was never observed and no net resource spend remains; "
            "the primitive did not establish construction"
        )
    return (
        "expected build order was never observed; resources changed, but that change alone "
        "cannot establish construction"
    )


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


__all__ = ["ActionEffectVerifier", "EffectVerdict"]
