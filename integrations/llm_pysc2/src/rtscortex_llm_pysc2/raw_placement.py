"""Single world-space placement authority for PySC2 raw build actions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional

from rtscortex_llm_pysc2.effect_lifecycle import build_reservation_expiry


@dataclass(frozen=True)
class RawPlacement:
    action_name: str
    world_target: tuple[float, float]
    anchor_tag: int | None = None


@dataclass(frozen=True)
class RawPlacementReservation:
    command_id: str
    operation_id: str | None
    action_name: str
    builder_tag: int | None
    ability_name: str
    requested_world_target: tuple[float, float] | None
    final_validated_world_target: tuple[float, float]
    world_target: tuple[float, float]
    anchor_tag: int | None
    placement_revision: str
    target_state_revision: str
    baseline_builder_orders: tuple[int, ...]
    structure_type: str
    footprint_width: int
    footprint_height: int
    occupied_grid_cells: frozenset[tuple[int, int]]
    world_center: tuple[float, float]
    placement_state: str
    episode_id: str
    expires_game_loop: int
    attempt_ordinal: int | None = None

    @property
    def reservation_id(self) -> str:
        payload = {
            "command_id": self.command_id,
            "operation_id": self.operation_id,
            "action_name": self.action_name,
            "builder_tag": self.builder_tag,
            "ability_name": self.ability_name,
            "world_target": self.world_target,
            "anchor_tag": self.anchor_tag,
            "placement_revision": self.placement_revision,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"placement:{digest}"


@dataclass(frozen=True)
class _SpatialExclusion:
    world_target: tuple[float, float]
    occupied_grid_cells: frozenset[tuple[int, int]]
    reason: str


@dataclass(frozen=True)
class _TemporarySuppression:
    world_target: tuple[float, float]
    occupied_grid_cells: frozenset[tuple[int, int]]
    expires_game_loop: int
    reason: str
    target_state_revision: str | None = None
    anchor_tag: int | None = None


@dataclass(frozen=True)
class RawPlacementCandidates:
    argument_candidates: list[list[Any]]
    screen_provenance: list[Any]
    unavailable_reason: str | None = None


class RawPlacementFailure(RuntimeError):
    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code


class RawPlacementNoStartStatus(str, Enum):  # noqa: UP042 - PySC2 bridge supports Python 3.9
    """Typed disposition for an operation whose accepted build never started."""

    RETRY = "retry"
    DEFER_REPLAN = "defer_replan"
    DUPLICATE = "duplicate"
    RESET = "reset"


@dataclass(frozen=True)
class RawPlacementNoStartState:
    """Auditable operation-level no-start streak snapshot."""

    operation_id: str
    streak: int
    threshold: int
    circuit_open: bool
    last_status: str
    last_command_id: str | None = None
    last_attempt_ordinal: int | None = None
    last_builder_tag: int | None = None
    last_world_target: tuple[float, float] | None = None
    last_placement_revision: str | None = None
    last_target_state_revision: str | None = None
    last_observation_revision: str | None = None
    blocked_target_state_revision: str | None = None
    blocked_builder_tag: int | None = None
    blocked_placement_revision: str | None = None
    blocked_observation_revision: str | None = None
    blocked_failure_classification: str | None = None
    blocked_target_side_evidence: bool = False
    seen_attempt_ordinals: tuple[int, ...] = ()
    seen_command_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawPlacementNoStartDecision:
    """One operation-level no-start update returned by ``quarantine_command``."""

    operation_id: str | None
    command_id: str
    failure_code: str
    status: str
    streak: int
    threshold: int
    circuit_open: bool
    duplicate_attempt: bool
    suppressed_target: bool
    target_side_evidence: bool
    next_action: str
    attempt_ordinal: int | None = None
    failure_classification: str | None = None
    classification_basis: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def deferred(self) -> bool:
        return self.status == RawPlacementNoStartStatus.DEFER_REPLAN.value

    @property
    def replan_required(self) -> bool:
        return self.deferred

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "command_id": self.command_id,
            "failure_code": self.failure_code,
            "status": self.status,
            "streak": self.streak,
            "threshold": self.threshold,
            "circuit_open": self.circuit_open,
            "duplicate_attempt": self.duplicate_attempt,
            "suppressed_target": self.suppressed_target,
            "target_side_evidence": self.target_side_evidence,
            "next_action": self.next_action,
            "attempt_ordinal": self.attempt_ordinal,
            "failure_classification": self.failure_classification,
            "classification_basis": list(self.classification_basis),
            "evidence": dict(self.evidence),
        }


@dataclass
class _OperationNoStartLedger:
    operation_id: str
    threshold: int
    streak: int = 0
    circuit_open: bool = False
    last_status: str = RawPlacementNoStartStatus.RESET.value
    last_command_id: str | None = None
    last_attempt_ordinal: int | None = None
    last_builder_tag: int | None = None
    last_world_target: tuple[float, float] | None = None
    last_placement_revision: str | None = None
    last_target_state_revision: str | None = None
    last_observation_revision: str | None = None
    blocked_target_state_revision: str | None = None
    blocked_builder_tag: int | None = None
    blocked_placement_revision: str | None = None
    blocked_observation_revision: str | None = None
    blocked_failure_classification: str | None = None
    blocked_target_side_evidence: bool = False
    seen_attempt_ordinals: set[int] = field(default_factory=set)
    seen_command_ids: set[str] = field(default_factory=set)


def _footprint_cells(
    world_center: tuple[float, float],
    width: int,
    height: int,
) -> frozenset[tuple[int, int]]:
    """Return the exact discrete build-grid cells covered by a world rectangle."""

    if width < 1 or height < 1:
        raise ValueError("placement footprint dimensions must be positive")
    start_x = math.floor(float(world_center[0]) - width / 2 + 0.5)
    start_y = math.floor(float(world_center[1]) - height / 2 + 0.5)
    return frozenset(
        (x, y) for y in range(start_y, start_y + height) for x in range(start_x, start_x + width)
    )


def _occupied_cells_for_spec(
    world_center: tuple[float, float],
    spec: Any,
) -> frozenset[tuple[int, int]]:
    size = int(spec.footprint)
    main_cells = _footprint_cells(world_center, size, size)
    if not bool(spec.reserves_addon_space):
        return main_cells
    max_x = max(cell[0] for cell in main_cells)
    min_y = min(cell[1] for cell in main_cells)
    max_y = max(cell[1] for cell in main_cells)
    addon_clearance = {(x, y) for x in range(max_x + 1, max_x + 3) for y in range(min_y, max_y + 1)}
    return frozenset((*main_cells, *addon_clearance))


def _placement_candidate_id(
    action_name: str,
    world_target: tuple[float, float],
    anchor_tag: int,
    placement_revision: str,
    footprint: int,
    reserves_addon_space: bool,
) -> str:
    payload = {
        "action_name": action_name,
        "world_target": [float(world_target[0]), float(world_target[1])],
        "anchor_tag": int(anchor_tag),
        "placement_revision": placement_revision,
        "footprint": int(footprint),
        "reserves_addon_space": bool(reserves_addon_space),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"placement-candidate:{digest}"


class RawPlacementService:
    """Generate, resolve, execute and verify one canonical placement identity.

    The wire protocol may still carry a feature-screen coordinate for backward
    compatibility, but raw execution and effect verification only consume the
    world target retained here.
    """

    def __init__(
        self,
        *,
        unit_names: Mapping[int, str],
        transition_sink: Callable[[dict[str, Any]], None] | None = None,
        effect_timeout_game_loops: int = 112,
        no_start_streak_threshold: int = 3,
    ) -> None:
        if effect_timeout_game_loops < 1:
            raise ValueError("effect_timeout_game_loops must be positive")
        if no_start_streak_threshold < 1:
            raise ValueError("no_start_streak_threshold must be positive")
        self.unit_names = {int(key): str(value) for key, value in unit_names.items()}
        self.effect_timeout_game_loops = int(effect_timeout_game_loops)
        self.no_start_streak_threshold = int(no_start_streak_threshold)
        self._known_resources: dict[int, dict[str, Any]] = {}
        self._visible_resource_tags: set[int] = set()
        self._resource_presence_generation: dict[int, int] = {}
        self._target_state_memory: dict[
            tuple[str, tuple[float, float]],
            tuple[str, int],
        ] = {}
        self._suppressed_clusters: set[int] = set()
        self._suppressed_resource_tags: set[int] = set()
        self._permanent_exclusions: list[_SpatialExclusion] = []
        self._temporary_suppressions: dict[str, list[_TemporarySuppression]] = {}
        self._operation_no_start: dict[str, _OperationNoStartLedger] = {}
        self._command_targets: dict[str, RawPlacementReservation] = {}
        self._observed_occupancy: dict[int, _SpatialExclusion] = {}
        self._builder_leases: dict[int, str] = {}
        self._placement_diagnostics: dict[str, str] = {}
        self._transition_history: dict[str, list[dict[str, Any]]] = {}
        self._transition_sink = transition_sink
        self._transition_sequence = 0
        self._runtime_run_id: str | None = None
        self._runtime_episode_id: str | None = None
        self._runtime_step_id = 0
        self._runtime_game_loop = 0
        self._last_transition_loop: dict[str, int] = {}
        self._world_to_minimap_transform: tuple[float, float, float, float, float] | None = None

    def set_world_to_minimap_transform(
        self,
        transform: tuple[float, float, float, float, float] | None,
    ) -> None:
        self._world_to_minimap_transform = transform

    @property
    def transitions_are_durable(self) -> bool:
        return self._transition_sink is not None

    def set_runtime_context(
        self,
        *,
        run_id: str,
        episode_id: str,
        step_id: int,
        game_loop: int,
    ) -> None:
        self._runtime_run_id = run_id
        self._runtime_episode_id = episode_id
        self._runtime_step_id = int(step_id)
        self._runtime_game_loop = max(self._runtime_game_loop, int(game_loop))

    @property
    def known_resources(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._known_resources[tag] for tag in sorted(self._known_resources))

    @property
    def suppressed_anchors(self) -> frozenset[int]:
        return frozenset((*self._suppressed_clusters, *self._suppressed_resource_tags))

    @property
    def quarantined_targets(self) -> Mapping[str, Sequence[tuple[float, float]]]:
        targets = [exclusion.world_target for exclusion in self._permanent_exclusions]
        result = {"*": tuple(targets)} if targets else {}
        result.update(
            {
                action: tuple(item.world_target for item in suppressions)
                for action, suppressions in self._temporary_suppressions.items()
                if suppressions
            }
        )
        return result

    @property
    def leased_builder_tags(self) -> frozenset[int]:
        return frozenset(self._builder_leases)

    @property
    def active_reservation_count(self) -> int:
        return len(self._command_targets)

    @property
    def placement_alerts(self) -> tuple[str, ...]:
        return tuple(
            f"placement_unavailable:{action}:{reason}"
            for action, reason in sorted(self._placement_diagnostics.items())
        )

    def reset_diagnostics(self) -> None:
        self._placement_diagnostics.clear()

    def operation_no_start_state(
        self,
        operation_id: str,
    ) -> RawPlacementNoStartState | None:
        """Return the current auditable no-start state for one operation."""

        ledger = self._operation_no_start.get(str(operation_id))
        if ledger is None:
            return None
        return RawPlacementNoStartState(
            operation_id=ledger.operation_id,
            streak=ledger.streak,
            threshold=ledger.threshold,
            circuit_open=ledger.circuit_open,
            last_status=ledger.last_status,
            last_command_id=ledger.last_command_id,
            last_attempt_ordinal=ledger.last_attempt_ordinal,
            last_builder_tag=ledger.last_builder_tag,
            last_world_target=ledger.last_world_target,
            last_placement_revision=ledger.last_placement_revision,
            last_target_state_revision=ledger.last_target_state_revision,
            last_observation_revision=ledger.last_observation_revision,
            blocked_target_state_revision=ledger.blocked_target_state_revision,
            blocked_builder_tag=ledger.blocked_builder_tag,
            blocked_placement_revision=ledger.blocked_placement_revision,
            blocked_observation_revision=ledger.blocked_observation_revision,
            blocked_failure_classification=ledger.blocked_failure_classification,
            blocked_target_side_evidence=ledger.blocked_target_side_evidence,
            seen_attempt_ordinals=tuple(sorted(ledger.seen_attempt_ordinals)),
            seen_command_ids=tuple(sorted(ledger.seen_command_ids)),
        )

    def no_start_streak(self, operation_id: str) -> int:
        state = self.operation_no_start_state(operation_id)
        return 0 if state is None else state.streak

    def operation_retry_allowed(
        self,
        operation_id: str,
        *,
        world_target: tuple[float, float] | None = None,
        target_state_revision: str | None = None,
        failure_classification: str | None = None,
        target_side_evidence: bool = False,
        builder_tag: int | None = None,
        builder_ready: bool = False,
        observation_revision: str | None = None,
    ) -> bool:
        """Whether a circuit-open operation has acquired a genuinely new target state."""

        ledger = self._operation_no_start.get(str(operation_id))
        if ledger is None or not ledger.circuit_open:
            return True
        effective_classification = (
            failure_classification
            if failure_classification is not None
            else ledger.blocked_failure_classification
        )
        effective_target_evidence = target_side_evidence or ledger.blocked_target_side_evidence
        same_target = (
            world_target is not None
            and ledger.last_world_target is not None
            and math.dist(world_target, ledger.last_world_target) <= 0.25
        )
        target_changed = (
            same_target
            and target_state_revision is not None
            and ledger.blocked_target_state_revision is not None
            and target_state_revision != ledger.blocked_target_state_revision
        )
        builder_changed = (
            builder_ready
            and builder_tag is not None
            and (
                ledger.blocked_builder_tag is None or int(builder_tag) != ledger.blocked_builder_tag
            )
            and observation_revision is not None
            and ledger.blocked_observation_revision is not None
            and observation_revision != ledger.blocked_observation_revision
        )
        if not self._allows_target_state_reopen(
            effective_classification,
            target_side_evidence=effective_target_evidence,
            target_changed=target_changed,
            builder_changed=builder_changed,
        ):
            return False
        ledger.circuit_open = False
        ledger.blocked_target_state_revision = None
        ledger.blocked_builder_tag = None
        ledger.blocked_placement_revision = None
        ledger.blocked_observation_revision = None
        ledger.blocked_failure_classification = None
        ledger.blocked_target_side_evidence = False
        ledger.last_status = RawPlacementNoStartStatus.RETRY.value
        return True

    def reset_operation_no_start(
        self,
        operation_id: str,
        *,
        reason: str = "build_started",
    ) -> RawPlacementNoStartState | None:
        """Clear an operation's consecutive no-start counter after success."""

        ledger = self._operation_no_start.get(str(operation_id))
        if ledger is None:
            return None
        ledger.streak = 0
        ledger.circuit_open = False
        ledger.blocked_target_state_revision = None
        ledger.blocked_builder_tag = None
        ledger.blocked_placement_revision = None
        ledger.blocked_observation_revision = None
        ledger.blocked_failure_classification = None
        ledger.blocked_target_side_evidence = False
        ledger.last_status = RawPlacementNoStartStatus.RESET.value
        return self.operation_no_start_state(operation_id)

    def record_no_start_failure(
        self,
        *,
        operation_id: str | None,
        command_id: str,
        failure_code: str = "no_build_start_evidence",
        attempt_ordinal: int | None = None,
        builder_tag: int | None = None,
        world_target: tuple[float, float] | None = None,
        placement_revision: str | None = None,
        target_state_revision: str | None = None,
        failure_classification: str | None = None,
        classification_basis: Sequence[str] = (),
        target_side_evidence: bool = False,
        suppressed_target: bool = False,
        builder_ready: bool = False,
        observation_revision: str | None = None,
    ) -> RawPlacementNoStartDecision:
        """Record one accepted gameplay no-start attempt by operation identity.

        Coordinates, placement revisions, command IDs and builder tags are retained
        as audit fields only; they never partition the streak.  Attempt ordinals
        and command IDs de-duplicate replayed historical reports.
        """

        normalized_operation = None if operation_id is None else str(operation_id)
        basis = tuple(str(item) for item in classification_basis)
        if normalized_operation is None:
            return RawPlacementNoStartDecision(
                operation_id=None,
                command_id=str(command_id),
                failure_code=str(failure_code),
                status=RawPlacementNoStartStatus.RETRY.value,
                streak=0,
                threshold=self.no_start_streak_threshold,
                circuit_open=False,
                duplicate_attempt=False,
                suppressed_target=suppressed_target,
                target_side_evidence=target_side_evidence,
                next_action="retry",
                attempt_ordinal=attempt_ordinal,
                failure_classification=failure_classification,
                classification_basis=basis,
                evidence={
                    "operation_id": None,
                    "command_id": str(command_id),
                    "failure_code": str(failure_code),
                    "streak": 0,
                    "threshold": self.no_start_streak_threshold,
                    "circuit_open": False,
                    "duplicate_attempt": False,
                    "suppressed_target": suppressed_target,
                    "target_side_evidence": target_side_evidence,
                    "next_action": "retry",
                    "attempt_ordinal": attempt_ordinal,
                    "builder_tag": None if builder_tag is None else hex(int(builder_tag)),
                    "builder_ready": builder_ready,
                    "observation_revision": observation_revision,
                    "world_target": world_target,
                    "placement_revision": placement_revision,
                    "target_state_revision": target_state_revision,
                    "failure_classification": failure_classification,
                    "classification_basis": list(basis),
                },
            )

        ledger = self._operation_no_start.setdefault(
            normalized_operation,
            _OperationNoStartLedger(
                operation_id=normalized_operation,
                threshold=self.no_start_streak_threshold,
            ),
        )
        duplicate = (
            str(command_id) in ledger.seen_command_ids
            or attempt_ordinal is not None
            and (
                int(attempt_ordinal) in ledger.seen_attempt_ordinals
                or (
                    bool(ledger.seen_attempt_ordinals)
                    and int(attempt_ordinal) < max(ledger.seen_attempt_ordinals)
                )
            )
        )
        if duplicate:
            status = RawPlacementNoStartStatus.DUPLICATE.value
            next_action = "defer_replan" if ledger.circuit_open else "retry"
        else:
            if ledger.circuit_open and not self.operation_retry_allowed(
                normalized_operation,
                world_target=world_target,
                target_state_revision=target_state_revision,
                failure_classification=failure_classification,
                target_side_evidence=target_side_evidence,
                builder_tag=builder_tag,
                builder_ready=builder_ready,
                observation_revision=observation_revision,
            ):
                if attempt_ordinal is not None:
                    ledger.seen_attempt_ordinals.add(int(attempt_ordinal))
                ledger.seen_command_ids.add(str(command_id))
                status = RawPlacementNoStartStatus.DEFER_REPLAN.value
                next_action = "replan"
            else:
                if attempt_ordinal is not None:
                    ledger.seen_attempt_ordinals.add(int(attempt_ordinal))
                ledger.seen_command_ids.add(str(command_id))
                ledger.streak += 1
                ledger.last_command_id = str(command_id)
                ledger.last_attempt_ordinal = (
                    None if attempt_ordinal is None else int(attempt_ordinal)
                )
                ledger.last_builder_tag = None if builder_tag is None else int(builder_tag)
                ledger.last_world_target = world_target
                ledger.last_placement_revision = placement_revision
                ledger.last_target_state_revision = target_state_revision
                ledger.last_observation_revision = observation_revision
                if ledger.streak >= ledger.threshold:
                    ledger.circuit_open = True
                    ledger.blocked_target_state_revision = target_state_revision
                    ledger.blocked_builder_tag = None if builder_tag is None else int(builder_tag)
                    ledger.blocked_placement_revision = placement_revision
                    ledger.blocked_observation_revision = observation_revision
                    ledger.blocked_failure_classification = failure_classification
                    ledger.blocked_target_side_evidence = target_side_evidence
                    status = RawPlacementNoStartStatus.DEFER_REPLAN.value
                    next_action = "replan"
                else:
                    status = RawPlacementNoStartStatus.RETRY.value
                    next_action = "retry"
                ledger.last_status = status

        evidence = {
            "operation_id": normalized_operation,
            "command_id": str(command_id),
            "failure_code": str(failure_code),
            "streak": ledger.streak,
            "threshold": ledger.threshold,
            "circuit_open": ledger.circuit_open,
            "duplicate_attempt": duplicate,
            "suppressed_target": suppressed_target,
            "target_side_evidence": target_side_evidence,
            "next_action": next_action,
            "attempt_ordinal": None if attempt_ordinal is None else int(attempt_ordinal),
            "builder_tag": None if builder_tag is None else hex(int(builder_tag)),
            "builder_ready": builder_ready,
            "observation_revision": observation_revision,
            "world_target": world_target,
            "placement_revision": placement_revision,
            "target_state_revision": target_state_revision,
            "failure_classification": failure_classification,
            "classification_basis": list(basis),
        }
        return RawPlacementNoStartDecision(
            operation_id=normalized_operation,
            command_id=str(command_id),
            failure_code=str(failure_code),
            status=status,
            streak=ledger.streak,
            threshold=ledger.threshold,
            circuit_open=ledger.circuit_open,
            duplicate_attempt=duplicate,
            suppressed_target=suppressed_target,
            target_side_evidence=target_side_evidence,
            next_action=next_action,
            attempt_ordinal=None if attempt_ordinal is None else int(attempt_ordinal),
            failure_classification=failure_classification,
            classification_basis=basis,
            evidence=evidence,
        )

    def observe(self, observation: Any, *, require_feature_visibility: bool) -> None:
        self._runtime_game_loop = max(self._runtime_game_loop, _game_loop(observation))
        self.observe_units(
            _value(observation, "raw_units", ()),
            _value(observation, "feature_units", ()),
            require_feature_visibility=require_feature_visibility,
        )
        for action_name, suppressions in self._temporary_suppressions.items():
            for suppression in suppressions:
                if suppression.target_state_revision is not None:
                    self._target_state_revision(
                        observation,
                        action_name,
                        suppression.world_target,
                        anchor_tag=suppression.anchor_tag,
                    )

    def observe_units(
        self,
        raw_units: Sequence[Any],
        feature_units: Sequence[Any],
        *,
        require_feature_visibility: bool,
    ) -> None:
        from rtscortex_llm_pysc2.extractor import BUILD_SPECS

        spec_by_structure = {spec.target_structure: spec for spec in BUILD_SPECS.values()}
        observed_occupancy: dict[int, _SpatialExclusion] = {}
        previously_visible_resources = self._visible_resource_tags
        currently_visible_resources: set[int] = set()
        visible_feature_tags = {
            int(_value(unit, "tag", 0))
            for unit in feature_units
            if int(_value(unit, "alliance", 0)) == 3
            and bool(_value(unit, "is_on_screen", True))
            and int(_value(unit, "display_type", 1)) == 1
        }
        for unit in raw_units:
            tag = int(_value(unit, "tag", 0))
            name = _unit_name(unit, self.unit_names)
            if (
                tag > 0
                and int(_value(unit, "alliance", 0)) in {1, 2, 4}
                and int(_value(unit, "display_type", 1)) == 1
                and _build_progress(unit) > 0.0
                and name in spec_by_structure
            ):
                center = (
                    float(_value(unit, "x", 0.0)),
                    float(_value(unit, "y", 0.0)),
                )
                spec = spec_by_structure[name]
                observed_occupancy[tag] = _SpatialExclusion(
                    center,
                    _occupied_cells_for_spec(center, spec),
                    "observed_structure",
                )
            if (
                tag <= 0
                or int(_value(unit, "alliance", 0)) != 3
                or int(_value(unit, "display_type", 1)) != 1
                or not _is_resource(name)
                or require_feature_visibility
                and tag not in visible_feature_tags
            ):
                continue
            currently_visible_resources.add(tag)
            self._known_resources[tag] = {
                "tag": tag,
                "unit_type": name,
                "alliance": 3,
                "x": float(_value(unit, "x", 0.0)),
                "y": float(_value(unit, "y", 0.0)),
                "display_type": 1,
            }
        for tag in previously_visible_resources.symmetric_difference(currently_visible_resources):
            self._resource_presence_generation[tag] = (
                self._resource_presence_generation.get(tag, 0) + 1
            )
        self._visible_resource_tags = currently_visible_resources
        self._observed_occupancy = observed_occupancy
        observed_cells = {
            cell
            for exclusion in observed_occupancy.values()
            for cell in exclusion.occupied_grid_cells
        }
        for command_id, reservation in list(self._command_targets.items()):
            if (
                reservation.placement_state == "occupied"
                and reservation.occupied_grid_cells & observed_cells
            ):
                self.release_command(
                    command_id,
                    game_loop=0,
                    reason="observed_structure_occupancy",
                )

    def candidates(
        self,
        observation: Any,
        action_name: str,
        *,
        builder_tags: Collection[int] = (),
    ) -> RawPlacementCandidates:
        # Lazy import avoids making the extractor depend on itself while all
        # public raw placement entry points remain centralized in this service.
        from rtscortex_llm_pysc2.extractor import (
            BUILD_SPECS,
            _gas_structure_candidates,
            build_screen_candidates,
            raw_build_eligibility,
            screen_candidate_provenance,
        )

        spec = BUILD_SPECS[action_name]
        self.observe(observation, require_feature_visibility=False)
        eligibility = raw_build_eligibility(observation, spec, self.unit_names)
        if not eligibility.eligible:
            assert eligibility.failure_code is not None
            self._remember_diagnostic(action_name, eligibility.failure_code)
            return RawPlacementCandidates(
                argument_candidates=[],
                screen_provenance=[],
                unavailable_reason=eligibility.failure_code,
            )
        game_loop = _game_loop(observation)
        self._expire_temporary_suppressions(game_loop)
        if spec.placement_kind == "screen":
            screen_candidates = build_screen_candidates(
                observation,
                action_name,
                unit_names=self.unit_names,
                builder_tags=builder_tags,
            )
            provenance = screen_candidate_provenance(observation, screen_candidates)
            revision = _placement_revision(observation)
            provenance = [
                replace(
                    item,
                    placement_candidate_id=_placement_candidate_id(
                        action_name,
                        item.world_target,
                        item.anchor_tag,
                        revision,
                        int(spec.footprint),
                        bool(spec.reserves_addon_space),
                    ),
                    placement_revision=revision,
                )
                for item in provenance
            ]
            kept = [
                (candidate, item)
                for candidate, item in zip(screen_candidates, provenance)  # noqa: B905 - Python 3.9
                if not self.is_quarantined(
                    action_name,
                    item.world_target,
                    game_loop=game_loop,
                    observation=observation,
                    anchor_tag=item.anchor_tag,
                )
            ]
            reason = None if kept else self._screen_unavailable_reason(observation, spec)
            self._remember_diagnostic(action_name, reason)
            return RawPlacementCandidates(
                argument_candidates=[[candidate] for candidate, _ in kept],
                screen_provenance=[item for _, item in kept],
                unavailable_reason=reason,
            )
        if spec.placement_kind == "geyser":
            candidates = _gas_structure_candidates(
                observation,
                self.unit_names,
                target_structure=spec.target_structure,
            )
            candidates = [
                tag
                for tag in candidates
                if (target := _unit_by_tag(observation, tag)) is not None
                and not self.is_quarantined(
                    action_name,
                    (
                        float(_value(target, "x", 0.0)),
                        float(_value(target, "y", 0.0)),
                    ),
                    game_loop=game_loop,
                    observation=observation,
                    anchor_tag=tag,
                )
            ]
            reason = None if candidates else "no_unoccupied_geyser"
            self._remember_diagnostic(action_name, reason)
            return RawPlacementCandidates(
                argument_candidates=[[tag] for tag in candidates],
                screen_provenance=[],
                unavailable_reason=reason,
            )
        candidates = self._expansion_candidates(observation, action_name)
        reason = None if candidates else "no_unoccupied_resource_cluster"
        self._remember_diagnostic(action_name, reason)
        return RawPlacementCandidates(
            argument_candidates=[[tag] for tag in candidates],
            screen_provenance=[],
            unavailable_reason=reason,
        )

    def resolve(
        self,
        *,
        command_id: str,
        action_name: str,
        requested_arguments: Sequence[Any],
        observation: Any,
        world_target: tuple[float, float] | None,
        preferred_anchor_tag: int | None = None,
        builder_tags: Collection[int] = (),
        builder_tag: int | None = None,
        operation_id: str | None = None,
        attempt_ordinal: int | None = None,
        ability_name: str | None = None,
        episode_id: str = "unknown",
        expires_game_loop: int | None = None,
        placement_candidate_id: str | None = None,
        candidate_placement_revision: str | None = None,
    ) -> RawPlacementReservation:
        from rtscortex_llm_pysc2.extractor import (
            BUILD_SPECS,
            raw_build_eligibility,
            resolve_screen_build_world_target,
            screen_to_world_target,
            world_build_target_is_legal,
        )

        spec = BUILD_SPECS[action_name]
        self.observe(observation, require_feature_visibility=False)
        eligibility = raw_build_eligibility(observation, spec, self.unit_names)
        if not eligibility.eligible:
            assert eligibility.failure_code is not None
            raise RawPlacementFailure(
                eligibility.failure_code,
                eligibility.reason or eligibility.failure_code,
            )
        game_loop = _game_loop(observation)
        current_revision = _placement_revision(observation)
        self._expire_temporary_suppressions(game_loop)
        normalized_builder = None if builder_tag is None else int(builder_tag)
        baseline_builder_orders: tuple[int, ...] = ()
        if normalized_builder is not None:
            lease_owner = self._builder_leases.get(normalized_builder)
            if lease_owner is not None and lease_owner != command_id:
                raise RawPlacementFailure(
                    "builder_unavailable",
                    f"builder {hex(normalized_builder)} is leased by {lease_owner}",
                )
            builder = _unit_by_tag(observation, normalized_builder)
            if builder is None or int(_value(builder, "alliance", 0)) != 1:
                raise RawPlacementFailure(
                    "builder_unavailable",
                    f"builder {hex(normalized_builder)} is not observable as an own unit",
                )
            if builder_tags and normalized_builder not in {int(tag) for tag in builder_tags}:
                raise RawPlacementFailure(
                    "builder_unavailable",
                    f"builder {hex(normalized_builder)} is not bound to the final target",
                )
            baseline_builder_orders = _orders(builder)
        final_builder_tags = (
            tuple(int(tag) for tag in builder_tags)
            if builder_tags
            else (() if normalized_builder is None else (normalized_builder,))
        )
        if spec.placement_kind == "screen":
            if (placement_candidate_id is None) != (candidate_placement_revision is None):
                raise RawPlacementFailure(
                    "placement_provenance_missing",
                    f"{action_name} has no immutable candidate placement identity",
                )
            if world_target is None:
                raise RawPlacementFailure(
                    "no_legal_placement",
                    f"{action_name} has no validated world-space target",
                )
            target = (float(world_target[0]), float(world_target[1]))
            if placement_candidate_id is not None:
                assert candidate_placement_revision is not None
                if candidate_placement_revision != current_revision:
                    raise RawPlacementFailure(
                        "placement_candidate_stale",
                        f"{action_name} candidate belongs to an older observation revision",
                    )
                expected_candidate_id = _placement_candidate_id(
                    action_name,
                    target,
                    0 if preferred_anchor_tag is None else preferred_anchor_tag,
                    candidate_placement_revision,
                    int(spec.footprint),
                    bool(spec.reserves_addon_space),
                )
                if placement_candidate_id != expected_candidate_id:
                    raise RawPlacementFailure(
                        "placement_provenance_mismatch",
                        f"{action_name} candidate placement identity does not match "
                        "its world target",
                    )
            emitted_target = (float(round(target[0])), float(round(target[1])))
            if self.is_quarantined(
                action_name,
                emitted_target,
                game_loop=game_loop,
                observation=observation,
                anchor_tag=preferred_anchor_tag,
            ):
                raise RawPlacementFailure(
                    "no_legal_placement",
                    f"{action_name} target {target} is permanently quarantined",
                )
            if _value(observation, "feature_screen", None) is not None:
                resolved_screen = resolve_screen_build_world_target(
                    observation,
                    action_name,
                    emitted_target,
                    preferred_anchor_tag=preferred_anchor_tag,
                    unit_names=self.unit_names,
                    builder_tags=final_builder_tags,
                )
                resolved = (
                    None
                    if resolved_screen is None
                    else screen_to_world_target(
                        observation,
                        resolved_screen,
                        preferred_anchor_tag=preferred_anchor_tag,
                    )
                )
                if resolved is None or math.dist(emitted_target, resolved.world_target) > 0.75:
                    self.suppress_world_target_temporarily(
                        action_name,
                        emitted_target,
                        expires_game_loop=game_loop + self.effect_timeout_game_loops,
                        reason="no_legal_placement",
                        target_state_revision=self._target_state_revision(
                            observation,
                            action_name,
                            emitted_target,
                            anchor_tag=preferred_anchor_tag,
                        ),
                        anchor_tag=preferred_anchor_tag,
                    )
                    raise RawPlacementFailure(
                        "no_legal_placement",
                        f"{action_name} world target {target} is no longer legal",
                    )
            requested_target = target
            emitted = emitted_target
            anchor_tag = None
        elif spec.placement_kind == "geyser":
            anchor = _tag_argument(requested_arguments, action_name=action_name)
            target_unit = _unit_by_tag(observation, anchor)
            if (
                target_unit is None
                or int(_value(target_unit, "alliance", 0)) != 3
                or int(_value(target_unit, "display_type", 1)) != 1
                or not _is_gas(_unit_name(target_unit, self.unit_names))
            ):
                raise RawPlacementFailure(
                    "invalid_geyser_tag",
                    f"{action_name} target {hex(anchor)} is not a visible neutral geyser",
                )
            emitted = (
                float(_value(target_unit, "x", 0.0)),
                float(_value(target_unit, "y", 0.0)),
            )
            requested_target = emitted
            anchor_tag = anchor
            if self.is_quarantined(
                action_name,
                emitted,
                game_loop=game_loop,
                observation=observation,
                anchor_tag=anchor_tag,
            ):
                raise RawPlacementFailure(
                    "no_legal_placement",
                    f"{action_name} geyser {hex(anchor)} has unchanged failed or occupied state",
                )
        else:
            anchor = _tag_argument(requested_arguments, action_name=action_name)
            cluster_id = self._cluster_id(anchor)
            if (
                cluster_id is None
                or self._cluster_is_suppressed(self._resource_cluster(cluster_id))
                or not self._cluster_is_current(cluster_id, observation)
            ):
                raise RawPlacementFailure(
                    "invalid_expansion_anchor",
                    f"{action_name} anchor {hex(anchor)} is permanently suppressed",
                )
            expansion_target = self._expansion_target(cluster_id, observation)
            if expansion_target is None:
                raise RawPlacementFailure(
                    "no_legal_placement",
                    f"{action_name} anchor {hex(anchor)} has no legal world placement",
                )
            requested_target = (
                float(expansion_target[0]),
                float(expansion_target[1]),
            )
            emitted = (float(round(expansion_target[0])), float(round(expansion_target[1])))
            anchor_tag = cluster_id
            quarantined = self.is_quarantined(
                action_name,
                emitted,
                game_loop=game_loop,
                observation=observation,
                anchor_tag=anchor_tag,
            )
            exact_target_legal = world_build_target_is_legal(
                observation,
                action_name,
                emitted,
                preferred_anchor_tag=anchor_tag,
                unit_names=self.unit_names,
                builder_tags=final_builder_tags,
                world_to_minimap_transform=self._world_to_minimap_transform,
            )
            if quarantined or not exact_target_legal:
                if not quarantined:
                    self.suppress_world_target_temporarily(
                        action_name,
                        emitted,
                        expires_game_loop=game_loop + self.effect_timeout_game_loops,
                        reason="no_legal_placement",
                        target_state_revision=self._target_state_revision(
                            observation,
                            action_name,
                            emitted,
                            anchor_tag=anchor_tag,
                        ),
                        anchor_tag=anchor_tag,
                    )
                raise RawPlacementFailure(
                    "no_legal_placement",
                    f"{action_name} anchor {hex(anchor)} has no current legal exact footprint",
                )
        target_state_revision = self._target_state_revision(
            observation,
            action_name,
            emitted,
            anchor_tag=anchor_tag,
        )
        reservation = RawPlacementReservation(
            command_id=command_id,
            operation_id=operation_id,
            action_name=action_name,
            builder_tag=normalized_builder,
            ability_name=ability_name or action_name,
            requested_world_target=requested_target,
            final_validated_world_target=emitted,
            world_target=emitted,
            anchor_tag=anchor_tag,
            placement_revision=current_revision,
            target_state_revision=target_state_revision,
            baseline_builder_orders=baseline_builder_orders,
            structure_type=spec.target_structure,
            footprint_width=(
                int(spec.footprint) + 2 if spec.reserves_addon_space else int(spec.footprint)
            ),
            footprint_height=int(spec.footprint),
            occupied_grid_cells=_occupied_cells_for_spec(emitted, spec),
            world_center=emitted,
            placement_state="reserved",
            episode_id=episode_id,
            expires_game_loop=(
                int(expires_game_loop)
                if expires_game_loop is not None
                else build_reservation_expiry(
                    game_loop,
                    action_name,
                    base_timeout_game_loops=self.effect_timeout_game_loops,
                )
            ),
            attempt_ordinal=(None if attempt_ordinal is None else int(attempt_ordinal)),
        )
        self._command_targets[command_id] = reservation
        self._record_transition(
            command_id,
            reservation=reservation,
            previous_state="unreserved",
            next_state="reserved",
            game_loop=game_loop,
        )
        if normalized_builder is not None:
            self._builder_leases[normalized_builder] = command_id
        return reservation

    def command_target(self, command_id: str) -> RawPlacementReservation | None:
        return self._command_targets.get(command_id)

    def drain_transition_history(self, command_id: str) -> list[dict[str, Any]]:
        return self._transition_history.pop(command_id, [])

    def quarantine_command(
        self,
        *,
        command_id: str,
        action_name: str,
        requested_arguments: Sequence[Any],
        world_target: tuple[float, float] | None,
        failure_code: str = "invalid_terrain",
        game_loop: int | None = None,
        operation_id: str | None = None,
        attempt_ordinal: int | None = None,
        builder_tag: int | None = None,
        placement_revision: str | None = None,
        target_state_revision: str | None = None,
        failure_classification: str | None = None,
        classification_basis: Sequence[str] = (),
        target_side_evidence: bool = False,
        builder_ready: bool = False,
        observation_revision: str | None = None,
        observation: Any | None = None,
    ) -> RawPlacementNoStartDecision | None:
        placement = self._command_targets.get(command_id)
        operation_id = operation_id or (None if placement is None else placement.operation_id)
        attempt_ordinal = (
            attempt_ordinal
            if attempt_ordinal is not None
            else (None if placement is None else placement.attempt_ordinal)
        )
        builder_tag = (
            builder_tag
            if builder_tag is not None
            else (None if placement is None else placement.builder_tag)
        )
        placement_revision = placement_revision or (
            None if placement is None else placement.placement_revision
        )
        target_state_revision = target_state_revision or (
            None if placement is None else placement.target_state_revision
        )
        target = None if placement is None else placement.world_target
        anchor = None if placement is None else placement.anchor_tag
        if target is None and world_target is not None:
            target = (float(world_target[0]), float(world_target[1]))
        if anchor is None and action_name.endswith("_Near") and requested_arguments:
            try:
                anchor = _tag_argument(requested_arguments, action_name=action_name)
            except RawPlacementFailure:
                anchor = None
        if observation is not None and target is not None:
            self.observe(observation, require_feature_visibility=False)
            target_state_revision = self._target_state_revision(
                observation,
                action_name,
                target,
                anchor_tag=anchor,
            )
            observation_revision = _placement_revision(observation)
        permanent_spatial_codes = {
            "blocked",
            "invalid_terrain",
            "invalid_expansion_anchor",
            "not_pathable",
            "placement_occupied",
        }
        retryable_spatial_codes = {"no_legal_placement"}
        classification = None if failure_classification is None else str(failure_classification)
        basis = tuple(str(item) for item in classification_basis)
        target_side_evidence = target_side_evidence or self._target_side_evidence(
            failure_code=failure_code,
            failure_classification=classification,
            classification_basis=basis,
        )
        temporary_suppression = target is not None and (
            failure_code in retryable_spatial_codes
            and classification
            not in {
                "builder_not_ready",
                "gameplay_no_start_unknown",
                "gameplay_effect_missing_after_start",
            }
            or target_side_evidence
            and classification in {"dynamic_target_obstruction", "placement_invalid"}
        )
        no_start_decision: RawPlacementNoStartDecision | None = None
        if failure_code == "no_build_start_evidence":
            no_start_decision = self.record_no_start_failure(
                operation_id=operation_id,
                command_id=command_id,
                failure_code=failure_code,
                attempt_ordinal=attempt_ordinal,
                builder_tag=builder_tag,
                world_target=target,
                placement_revision=placement_revision,
                target_state_revision=target_state_revision,
                failure_classification=classification,
                classification_basis=basis,
                target_side_evidence=target_side_evidence,
                suppressed_target=temporary_suppression,
                builder_ready=builder_ready,
                observation_revision=observation_revision,
            )
        transition_state = "released"
        failure_class = "nonspatial"
        if (
            target is not None
            and failure_code in permanent_spatial_codes
            and classification != "dynamic_target_obstruction"
            and (classification is None or target_side_evidence)
        ):
            self.suppress_world_target(action_name, target, reason=failure_code)
            transition_state = "permanent_invalid"
            failure_class = "spatial_permanent"
        elif temporary_suppression:
            assert target is not None
            current_loop = 0 if game_loop is None else int(game_loop)
            self.suppress_world_target_temporarily(
                action_name,
                target,
                expires_game_loop=current_loop + self.effect_timeout_game_loops,
                reason=failure_code,
                target_state_revision=target_state_revision,
                anchor_tag=anchor,
            )
            transition_state = "temporary_suppressed"
            failure_class = "spatial_retryable"
        if (
            anchor is not None
            and failure_code in permanent_spatial_codes
            and any(townhall in action_name for townhall in ("Nexus", "CommandCenter", "Hatchery"))
        ):
            self.suppress_anchor(anchor)
        self._record_transition(
            command_id,
            reservation=placement,
            action_name=action_name,
            world_target=target,
            previous_state=("unreserved" if placement is None else placement.placement_state),
            next_state=transition_state,
            failure_class=failure_class,
            actor_failure=failure_code in {"builder_unavailable", "actor_not_available"}
            or classification == "builder_not_ready",
            game_loop=0 if game_loop is None else int(game_loop),
            release_reason=failure_code,
            no_start_decision=no_start_decision,
        )
        self.release_command(command_id, record_transition=False)
        return no_start_decision

    @staticmethod
    def _target_side_evidence(
        *,
        failure_code: str,
        failure_classification: str | None,
        classification_basis: Sequence[str],
    ) -> bool:
        if failure_classification in {
            "builder_not_ready",
            "gameplay_no_start_unknown",
            "gameplay_effect_missing_after_start",
        }:
            return False
        if failure_classification == "placement_invalid":
            return True
        if failure_classification == "dynamic_target_obstruction":
            return any(
                basis
                in {
                    "dynamic_unit_inside_footprint",
                    "target_occupancy",
                    "target_obstruction",
                    "placement_occupied",
                }
                for basis in classification_basis
            )
        return failure_code in {
            "blocked",
            "invalid_terrain",
            "invalid_expansion_anchor",
            "not_pathable",
            "placement_occupied",
            "no_legal_placement",
        }

    @staticmethod
    def _allows_target_state_reopen(
        failure_classification: str | None,
        *,
        target_side_evidence: bool,
        target_changed: bool,
        builder_changed: bool,
    ) -> bool:
        if failure_classification in {"builder_not_ready", "gameplay_no_start_unknown"}:
            return builder_changed or target_changed and target_side_evidence
        if failure_classification in {
            "dynamic_target_obstruction",
            "placement_invalid",
        }:
            return target_changed and target_side_evidence
        # Legacy callers have no classification.  Their target-state revision is
        # the only available proof that retrying is meaningful.
        return failure_classification is None and (target_changed or builder_changed)

    def suppress_anchor(self, tag: int) -> None:
        cluster = self._resource_cluster(int(tag))
        cluster_id = self._cluster_id(int(tag))
        self._suppressed_clusters.add(int(tag) if cluster_id is None else cluster_id)
        self._suppressed_resource_tags.update(int(unit["tag"]) for unit in cluster)

    def confirm_command(self, command_id: str, *, game_loop: int | None = None) -> None:
        placement = self._command_targets.get(command_id)
        if placement is None:
            return
        if placement.operation_id is not None:
            self.reset_operation_no_start(placement.operation_id, reason="effect_confirmed")
        if placement.anchor_tag is not None and any(
            townhall in placement.action_name for townhall in ("Nexus", "CommandCenter", "Hatchery")
        ):
            self.suppress_anchor(placement.anchor_tag)
            self.suppress_world_target(
                placement.action_name,
                placement.world_target,
                reason="confirmed_townhall_occupancy",
            )
        self._command_targets[command_id] = replace(
            placement,
            placement_state="occupied",
            expires_game_loop=max(placement.expires_game_loop, 2**31 - 1),
        )
        self._record_transition(
            command_id,
            reservation=placement,
            previous_state=placement.placement_state,
            next_state="occupied",
            game_loop=self._resolved_transition_loop(
                placement.reservation_id,
                game_loop,
            ),
            release_reason="effect_confirmed",
        )
        self._release_builder_lease(placement)

    def mark_build_started(
        self,
        command_id: str,
        *,
        game_loop: int,
        expires_game_loop: int,
    ) -> None:
        """Renew active placement ownership when RAW order/effect evidence appears."""

        placement = self._command_targets.get(command_id)
        if placement is None:
            return
        renewed = replace(
            placement,
            placement_state="build_started",
            expires_game_loop=max(placement.expires_game_loop, int(expires_game_loop)),
        )
        self._command_targets[command_id] = renewed
        if placement.operation_id is not None:
            self.reset_operation_no_start(placement.operation_id, reason="build_started")
        if placement.placement_state == "reserved":
            self._record_transition(
                command_id,
                reservation=placement,
                previous_state="reserved",
                next_state="build_started",
                game_loop=self._resolved_transition_loop(
                    placement.reservation_id,
                    game_loop,
                ),
                release_reason="build_start_observed",
            )

    def release_command(
        self,
        command_id: str,
        *,
        game_loop: int | None = None,
        reason: str = "released",
        record_transition: bool = True,
    ) -> None:
        placement = self._command_targets.pop(command_id, None)
        if placement is None:
            return
        if record_transition:
            self._record_transition(
                command_id,
                reservation=placement,
                previous_state=placement.placement_state,
                next_state="released",
                game_loop=self._resolved_transition_loop(
                    placement.reservation_id,
                    game_loop,
                ),
                release_reason=reason,
            )
        self._release_builder_lease(placement)

    def _release_builder_lease(self, placement: RawPlacementReservation) -> None:
        if placement.builder_tag is None:
            return
        if self._builder_leases.get(placement.builder_tag) == placement.command_id:
            del self._builder_leases[placement.builder_tag]

    def suppress_world_target(
        self,
        action_name: str,
        target: Sequence[int | float],
        *,
        reason: str = "invalid_terrain",
        radius: float = 1.5,
    ) -> None:
        if len(target) != 2:
            return
        point = (float(target[0]), float(target[1]))
        cells = self._cells_for_action(action_name, point, fallback_radius=radius)
        if not any(
            cells == previous.occupied_grid_cells for previous in self._permanent_exclusions
        ):
            self._permanent_exclusions.append(_SpatialExclusion(point, cells, reason))

    def suppress_world_target_temporarily(
        self,
        action_name: str,
        target: Sequence[int | float],
        *,
        expires_game_loop: int,
        reason: str,
        target_state_revision: str | None = None,
        anchor_tag: int | None = None,
    ) -> None:
        if len(target) != 2:
            return
        point = (float(target[0]), float(target[1]))
        cells = self._cells_for_action(action_name, point)
        stored = self._temporary_suppressions.setdefault(action_name, [])
        stored.append(
            _TemporarySuppression(
                point,
                cells,
                int(expires_game_loop),
                reason,
                target_state_revision,
                anchor_tag,
            )
        )

    def is_quarantined(
        self,
        action_name: str,
        target: tuple[float, float],
        *,
        radius: float | None = None,
        game_loop: int | None = None,
        observation: Any | None = None,
        anchor_tag: int | None = None,
    ) -> bool:
        if game_loop is not None:
            self._expire_temporary_suppressions(game_loop)
        cells = self._cells_for_action(
            action_name,
            target,
            fallback_radius=1.5 if radius is None else radius,
        )
        if any(cells & exclusion.occupied_grid_cells for exclusion in self._permanent_exclusions):
            return True
        suppressions = self._temporary_suppressions.get(action_name, [])
        if observation is not None and suppressions:
            current_state_revision = self._target_state_revision(
                observation,
                action_name,
                target,
                anchor_tag=anchor_tag,
            )
            suppressions = [
                suppression
                for suppression in suppressions
                if not cells & suppression.occupied_grid_cells
                or suppression.target_state_revision is None
                or suppression.target_state_revision == current_state_revision
            ]
            if suppressions:
                self._temporary_suppressions[action_name] = suppressions
            else:
                self._temporary_suppressions.pop(action_name, None)
        if any(cells & suppression.occupied_grid_cells for suppression in suppressions):
            return True
        if any(
            cells & exclusion.occupied_grid_cells for exclusion in self._observed_occupancy.values()
        ):
            return True
        return any(
            cells & reservation.occupied_grid_cells
            for reservation in self._command_targets.values()
        )

    @staticmethod
    def _cells_for_action(
        action_name: str,
        target: tuple[float, float],
        *,
        fallback_radius: float = 1.5,
    ) -> frozenset[tuple[int, int]]:
        from rtscortex_llm_pysc2.extractor import BUILD_SPECS

        spec = BUILD_SPECS.get(action_name)
        if spec is not None:
            return _occupied_cells_for_spec(target, spec)
        size = max(1, int(round(fallback_radius * 2)))
        return _footprint_cells(target, size, size)

    def _expire_temporary_suppressions(self, game_loop: int) -> None:
        for command_id, reservation in list(self._command_targets.items()):
            if reservation.placement_state != "occupied" and reservation.expires_game_loop <= int(
                game_loop
            ):
                self.release_command(
                    command_id,
                    game_loop=int(game_loop),
                    reason="reservation_expired",
                )
        for action_name, suppressions in list(self._temporary_suppressions.items()):
            kept = [
                suppression
                for suppression in suppressions
                if suppression.target_state_revision is not None
                or suppression.expires_game_loop > int(game_loop)
            ]
            if kept:
                self._temporary_suppressions[action_name] = kept
            else:
                del self._temporary_suppressions[action_name]

    def _record_transition(
        self,
        command_id: str,
        *,
        reservation: RawPlacementReservation | None,
        previous_state: str,
        next_state: str,
        game_loop: int,
        action_name: str | None = None,
        world_target: tuple[float, float] | None = None,
        failure_class: str | None = None,
        actor_failure: bool = False,
        release_reason: str | None = None,
        no_start_decision: RawPlacementNoStartDecision | None = None,
    ) -> None:
        from rtscortex_llm_pysc2.extractor import BUILD_SPECS

        resolved_action = reservation.action_name if reservation is not None else action_name
        if resolved_action is None:
            return
        spec = BUILD_SPECS.get(resolved_action)
        if spec is None:
            return
        target = reservation.world_target if reservation is not None else world_target
        if target is None:
            return
        cells = (
            reservation.occupied_grid_cells
            if reservation is not None
            else _occupied_cells_for_spec(target, spec)
        )
        reservation_id = (
            reservation.reservation_id
            if reservation is not None
            else "placement:"
            + hashlib.sha256(
                json.dumps(
                    {
                        "command_id": command_id,
                        "action_name": resolved_action,
                        "world_target": target,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        resolved_loop = self._resolved_transition_loop(reservation_id, game_loop)
        transition: dict[str, Any] = {
            "reservation_id": reservation_id,
            "structure_type": spec.target_structure,
            "footprint_cells": sorted(cells),
            "previous_state": previous_state,
            "next_state": next_state,
            "failure_class": failure_class,
            "actor_failure": actor_failure,
            "game_loop": resolved_loop,
            "release_reason": release_reason,
        }
        if reservation is not None:
            transition["target_state_revision"] = reservation.target_state_revision
        if no_start_decision is not None:
            transition["placement_no_start"] = no_start_decision.to_dict()
        self._transition_sequence += 1
        transition_id = (
            "placement-transition:"
            + hashlib.sha256(
                json.dumps(
                    {
                        "command_id": command_id,
                        "transition": transition,
                        "sequence": self._transition_sequence,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        builder_tag = None if reservation is None else reservation.builder_tag
        lease_state = (
            "acquired"
            if next_state == "reserved" and builder_tag is not None
            else "released"
            if builder_tag is not None
            and next_state
            in {
                "occupied",
                "released",
                "temporary_suppressed",
                "permanent_invalid",
            }
            else None
        )
        if self._transition_sink is None:
            self._transition_history.setdefault(command_id, []).append(transition)
            return
        if self._runtime_run_id is None or self._runtime_episode_id is None:
            raise RuntimeError("durable placement transition has no Runtime context")
        self._transition_sink(
            {
                "protocol_version": "1.1",
                "run_id": self._runtime_run_id,
                "episode_id": self._runtime_episode_id,
                "step_id": self._runtime_step_id,
                "command_id": command_id,
                "action_name": resolved_action,
                "transition_id": transition_id,
                "builder_tag": None if builder_tag is None else hex(builder_tag),
                "builder_lease_state": lease_state,
                "transition": transition,
            }
        )

    def _resolved_transition_loop(
        self,
        reservation_id: str,
        game_loop: int | None,
    ) -> int:
        resolved = self._runtime_game_loop if game_loop is None else int(game_loop)
        resolved = max(
            0,
            resolved,
            self._runtime_game_loop,
            self._last_transition_loop.get(reservation_id, 0),
        )
        self._runtime_game_loop = max(self._runtime_game_loop, resolved)
        self._last_transition_loop[reservation_id] = resolved
        return resolved

    def _expansion_target(
        self,
        anchor_tag: int,
        observation: Any,
    ) -> Optional[tuple[float, float]]:
        resources = self._resource_cluster(anchor_tag)
        if sum(_is_mineral(str(unit["unit_type"])) for unit in resources) < 5:
            return None
        center = (
            sum(float(unit["x"]) for unit in resources) / len(resources),
            sum(float(unit["y"]) for unit in resources) / len(resources),
        )
        occupied = [
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "alliance", 0)) in {1, 2, 4}
            and int(_value(unit, "display_type", 1)) == 1
            and _build_progress(unit) > 0.0
        ]
        candidates: list[tuple[float, float, int, int]] = []
        for y in range(math.floor(center[1] - 12), math.ceil(center[1] + 12) + 1):
            for x in range(math.floor(center[0] - 12), math.ceil(center[0] + 12) + 1):
                point = (float(x), float(y))
                if any(
                    math.dist(
                        point,
                        (
                            float(_value(unit, "x", 0.0)),
                            float(_value(unit, "y", 0.0)),
                        ),
                    )
                    < 4.0
                    for unit in occupied
                ):
                    continue
                distances = [
                    math.dist(point, (float(unit["x"]), float(unit["y"]))) for unit in resources
                ]
                if any(
                    distance < (6.5 if _is_gas(str(unit["unit_type"])) else 5.5)
                    for unit, distance in zip(resources, distances)  # noqa: B905 - Python 3.9
                ):
                    continue
                nearby_minerals = sum(
                    _is_mineral(str(unit["unit_type"])) and distance <= 12.0
                    for unit, distance in zip(resources, distances)  # noqa: B905 - Python 3.9
                )
                if nearby_minerals < 5:
                    continue
                score = sum(
                    abs(distance - (8.5 if _is_gas(str(unit["unit_type"])) else 7.5))
                    for unit, distance in zip(resources, distances)  # noqa: B905 - Python 3.9
                )
                candidates.append((score, math.dist(point, center), x, y))
        if not candidates:
            return None
        _, _, x, y = min(candidates)
        return float(x), float(y)

    def _target_state_revision(
        self,
        observation: Any,
        action_name: str,
        target: tuple[float, float],
        *,
        anchor_tag: int | None,
    ) -> str:
        cells = self._cells_for_action(action_name, target)
        occupants = sorted(
            (
                int(tag),
                exclusion.world_target,
                exclusion.reason,
            )
            for tag, exclusion in self._observed_occupancy.items()
            if cells & exclusion.occupied_grid_cells
        )
        resource_tags: set[int] = set()
        if anchor_tag is not None:
            resource_tags = {
                int(unit["tag"]) for unit in self._resource_cluster(int(anchor_tag))
            } or {int(anchor_tag)}
        exact_world_legal: bool | None = None
        if (
            _value(observation, "feature_screen", None) is not None
            or self._world_to_minimap_transform is not None
        ):
            from rtscortex_llm_pysc2.extractor import world_build_target_is_legal

            exact_world_legal = bool(
                world_build_target_is_legal(
                    observation,
                    action_name,
                    target,
                    preferred_anchor_tag=anchor_tag,
                    unit_names=self.unit_names,
                    world_to_minimap_transform=self._world_to_minimap_transform,
                )
            )
        payload = {
            "action_name": action_name,
            "world_target": [float(target[0]), float(target[1])],
            "occupants": occupants,
            "dynamic_target_state": _dynamic_target_state(
                observation,
                cells,
                target,
                anchor_tag=anchor_tag,
                unit_names=self.unit_names,
            ),
            "exact_world_legal": exact_world_legal,
            "resource_presence": [
                [
                    tag,
                    tag in self._visible_resource_tags,
                    self._resource_presence_generation.get(tag, 0),
                ]
                for tag in sorted(resource_tags)
            ],
        }
        signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        key = (
            action_name,
            (round(float(target[0]), 3), round(float(target[1]), 3)),
        )
        previous_signature, generation = self._target_state_memory.get(
            key,
            ("", 0),
        )
        if signature != previous_signature:
            generation += 1
            self._target_state_memory[key] = (signature, generation)
        return hashlib.sha256(f"{signature}:{generation}".encode()).hexdigest()

    def _cluster_is_current(self, anchor_tag: int, observation: Any) -> bool:
        current_resources = {
            int(_value(unit, "tag", 0)): unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "alliance", 0)) == 3
            and int(_value(unit, "display_type", 1)) == 1
            and _is_resource(_unit_name(unit, self.unit_names))
        }
        cluster = self._resource_cluster(anchor_tag)
        current_cluster = [unit for unit in cluster if int(unit["tag"]) in current_resources]
        if not current_cluster:
            return (
                _value(observation, "player_common", _value(observation, "player", None)) is None
                and sum(_is_mineral(str(unit["unit_type"])) for unit in cluster) >= 5
            )
        center = (
            sum(float(unit["x"]) for unit in current_cluster) / len(current_cluster),
            sum(float(unit["y"]) for unit in current_cluster) / len(current_cluster),
        )
        occupied_cluster = any(
            int(_value(unit, "alliance", 0)) in {1, 2, 4}
            and int(_value(unit, "display_type", 1)) == 1
            and _unit_name(unit, self.unit_names).casefold()
            in {
                "commandcenter",
                "hatchery",
                "hive",
                "lair",
                "nexus",
                "orbitalcommand",
                "planetaryfortress",
            }
            and math.dist(
                center,
                (
                    float(_value(unit, "x", 0.0)),
                    float(_value(unit, "y", 0.0)),
                ),
            )
            < 12.0
            for unit in _value(observation, "raw_units", ())
        )
        return (
            int(anchor_tag) in current_resources
            and sum(_is_mineral(str(unit["unit_type"])) for unit in current_cluster) >= 5
            and not occupied_cluster
        )

    def _resource_cluster(self, anchor_tag: int) -> list[dict[str, Any]]:
        anchor = self._known_resources.get(anchor_tag)
        if anchor is None:
            return []
        resources = [anchor]
        tags = {anchor_tag}
        while True:
            connected = [
                unit
                for tag, unit in self._known_resources.items()
                if tag not in tags
                and any(
                    math.dist(
                        (float(unit["x"]), float(unit["y"])),
                        (float(existing["x"]), float(existing["y"])),
                    )
                    <= 12.0
                    for existing in resources
                )
            ]
            if not connected:
                return resources
            resources.extend(connected)
            tags.update(int(unit["tag"]) for unit in connected)

    def _cluster_id(self, anchor_tag: int) -> int | None:
        cluster = self._resource_cluster(anchor_tag)
        if not cluster:
            return None
        return min(int(unit["tag"]) for unit in cluster)

    def _resource_clusters(self) -> list[tuple[int, list[dict[str, Any]]]]:
        remaining = set(self._known_resources)
        clusters: list[tuple[int, list[dict[str, Any]]]] = []
        while remaining:
            cluster = self._resource_cluster(min(remaining))
            tags = {int(unit["tag"]) for unit in cluster}
            remaining.difference_update(tags)
            cluster_id = min(tags)
            clusters.append((cluster_id, cluster))
        return sorted(clusters, key=lambda item: item[0])

    def _expansion_candidates(self, observation: Any, action_name: str) -> list[int]:
        from rtscortex_llm_pysc2.extractor import world_build_target_is_legal

        own_townhalls = [
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "alliance", 0)) == 1
            and int(_value(unit, "display_type", 1)) == 1
            and _unit_name(unit, self.unit_names).casefold()
            in {
                "commandcenter",
                "hatchery",
                "hive",
                "lair",
                "nexus",
                "orbitalcommand",
                "planetaryfortress",
            }
        ]
        ranked: list[tuple[float, int]] = []
        for cluster_id, cluster in self._resource_clusters():
            if self._cluster_is_suppressed(cluster):
                continue
            if sum(_is_mineral(str(unit["unit_type"])) for unit in cluster) < 5:
                continue
            if not self._cluster_is_current(cluster_id, observation):
                continue
            center = (
                sum(float(unit["x"]) for unit in cluster) / len(cluster),
                sum(float(unit["y"]) for unit in cluster) / len(cluster),
            )
            if any(
                math.dist(
                    center,
                    (
                        float(_value(townhall, "x", 0.0)),
                        float(_value(townhall, "y", 0.0)),
                    ),
                )
                < 12.0
                for townhall in own_townhalls
            ):
                continue
            expansion_target = self._expansion_target(cluster_id, observation)
            if (
                expansion_target is None
                or self.is_quarantined(
                    action_name,
                    expansion_target,
                    game_loop=_game_loop(observation),
                    observation=observation,
                    anchor_tag=cluster_id,
                )
                or not world_build_target_is_legal(
                    observation,
                    action_name,
                    expansion_target,
                    preferred_anchor_tag=cluster_id,
                    unit_names=self.unit_names,
                    world_to_minimap_transform=self._world_to_minimap_transform,
                )
            ):
                continue
            base_distance = min(
                (
                    math.dist(
                        center,
                        (
                            float(_value(townhall, "x", 0.0)),
                            float(_value(townhall, "y", 0.0)),
                        ),
                    )
                    for townhall in own_townhalls
                ),
                default=math.inf,
            )
            ranked.append((base_distance, cluster_id))
        ranked.sort()
        return [cluster_id for _, cluster_id in ranked[:4]]

    def _cluster_is_suppressed(self, cluster: Sequence[Mapping[str, Any]]) -> bool:
        tags = {int(unit["tag"]) for unit in cluster}
        return bool(
            tags.intersection(self._suppressed_resource_tags)
            or tags.intersection(self._suppressed_clusters)
        )

    def _remember_diagnostic(self, action_name: str, reason: str | None) -> None:
        if reason is None:
            self._placement_diagnostics.pop(action_name, None)
        else:
            self._placement_diagnostics[action_name] = reason

    @staticmethod
    def _screen_unavailable_reason(observation: Any, spec: Any) -> str:
        feature_screen = _value(observation, "feature_screen", None)
        if feature_screen is None:
            return "out_of_view"
        if spec.requires_power and not _plane_has_positive(_value(feature_screen, "power", None)):
            return "need_power"
        if spec.requires_creep and not _plane_has_positive(_value(feature_screen, "creep", None)):
            return "need_creep"
        if not _plane_has_positive(_value(feature_screen, "buildable", None)):
            return "not_buildable"
        if not _plane_has_positive(_value(feature_screen, "pathable", None)):
            return "not_pathable"
        return "occupied_or_unreachable"


def _tag_argument(arguments: Sequence[Any], *, action_name: str) -> int:
    if not arguments:
        raise RawPlacementFailure(
            "candidate_invalidated",
            f"{action_name} has no target tag",
        )
    try:
        value = int(arguments[0], 0) if isinstance(arguments[0], str) else int(arguments[0])
    except (TypeError, ValueError) as error:
        raise RawPlacementFailure(
            "candidate_invalidated",
            f"{action_name} target is not a tag",
        ) from error
    if value <= 0:
        raise RawPlacementFailure(
            "candidate_invalidated",
            f"{action_name} tag must be positive",
        )
    return value


def _unit_by_tag(observation: Any, tag: int) -> Optional[Any]:
    return next(
        (
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "tag", -1)) == tag
        ),
        None,
    )


def _unit_name(unit: Any, unit_names: Mapping[int, str]) -> str:
    value = _value(unit, "unit_type", "")
    if isinstance(value, str):
        return value
    return unit_names.get(int(value), f"unit:{int(value)}")


def _is_resource(name: str) -> bool:
    return _is_mineral(name) or _is_gas(name)


def _is_mineral(name: str) -> bool:
    return "mineral" in name.casefold()


def _is_gas(name: str) -> bool:
    normalized = name.casefold()
    return "vespene" in normalized or "geyser" in normalized


def _build_progress(unit: Any) -> float:
    value = float(_value(unit, "build_progress", 0.0))
    return value / 100.0 if value > 1.0 else value


def _orders(unit: Any) -> tuple[int, ...]:
    count = max(0, int(_value(unit, "order_length", 0)))
    return tuple(
        order
        for index in range(max(4, count))
        if (order := int(_value(unit, f"order_id_{index}", 0))) > 0
    )


def _game_loop(observation: Any) -> int:
    value = _value(observation, "game_loop", 0)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 0
    return int(value)


def _dynamic_target_state(
    observation: Any,
    footprint_cells: Collection[tuple[int, int]],
    target: tuple[float, float],
    *,
    anchor_tag: int | None,
    unit_names: Mapping[int, str],
) -> list[list[Any]]:
    """Return stable ground-blocker and nearby-enemy evidence for one target."""

    footprint = set(footprint_cells)
    state: list[list[Any]] = []
    for unit in _value(observation, "raw_units", ()):
        tag = int(_value(unit, "tag", 0))
        alliance = int(_value(unit, "alliance", 0))
        if (
            tag <= 0
            or tag == anchor_tag
            or alliance not in {1, 2, 3, 4}
            or int(_value(unit, "display_type", 1)) != 1
        ):
            continue
        name = _unit_name(unit, unit_names)
        if (
            _is_resource(name)
            or bool(_value(unit, "is_structure", False))
            or bool(_value(unit, "is_flying", False))
        ):
            continue
        position = (
            float(_value(unit, "x", 0.0)),
            float(_value(unit, "y", 0.0)),
        )
        radius = max(0.0, float(_value(unit, "radius", 0.5)))
        footprint_blocker = _circle_intersects_cells(position, radius, footprint)
        nearby_enemy = alliance == 4 and math.dist(position, target) <= 8.0 + radius
        if not footprint_blocker and not nearby_enemy:
            continue
        state.append(
            [
                tag,
                alliance,
                name,
                round(position[0] * 2.0) / 2.0,
                round(position[1] * 2.0) / 2.0,
                round(radius, 2),
                "footprint" if footprint_blocker else "threat",
            ]
        )
    return sorted(state, key=lambda item: (item[0], item[6]))


def _circle_intersects_cells(
    position: tuple[float, float],
    radius: float,
    cells: Collection[tuple[int, int]],
) -> bool:
    for cell_x, cell_y in cells:
        nearest_x = min(max(position[0], cell_x - 0.5), cell_x + 0.5)
        nearest_y = min(max(position[1], cell_y - 0.5), cell_y + 0.5)
        if math.dist(position, (nearest_x, nearest_y)) <= radius:
            return True
    return False


def _placement_revision(observation: Any) -> str:
    units = [
        (
            int(_value(unit, "tag", 0)),
            int(_value(unit, "alliance", 0)),
            str(_value(unit, "unit_type", "")),
            round(float(_value(unit, "x", 0.0)), 3),
            round(float(_value(unit, "y", 0.0)), 3),
            round(_build_progress(unit), 3),
        )
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "display_type", 1)) == 1
    ]
    payload = {"game_loop": _game_loop(observation), "units": sorted(units)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _plane_has_positive(plane: Any) -> bool:
    shape: Sequence[Any] = getattr(plane, "shape", ())
    if plane is None or len(shape) != 2:
        return False
    return any(int(plane[y][x]) > 0 for y in range(int(shape[0])) for x in range(int(shape[1])))


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "RawPlacement",
    "RawPlacementCandidates",
    "RawPlacementFailure",
    "RawPlacementReservation",
    "RawPlacementService",
]
