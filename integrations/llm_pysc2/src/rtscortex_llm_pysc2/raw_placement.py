"""Single world-space placement authority for PySC2 raw build actions."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from copy import deepcopy
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
    attempt_id: str | None = None
    attempt_ordinal: int | None = None
    ability_id: int | None = None
    target_legality_fingerprint: str | None = None
    material_legality_identity: str | None = None
    available_ability_query: str | None = None
    placement_query_result: str | None = None
    build_authorization_details: Mapping[str, Any] = field(default_factory=dict)
    primitive_constructed_game_loop: int | None = None
    primitive_submitted_game_loop: int | None = None
    action_result: tuple[int, ...] | None = None

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


_AUTHORITATIVE_PRE_DISPATCH_FAILURE_CODES = frozenset(
    {
        "placement_query_rejected",
        "placement_query_rejected_cached",
        "placement_candidate_stale",
        "no_legal_placement",
    }
)
_AUTHORITATIVE_PRE_DISPATCH_STREAK_THRESHOLD = 3
_OPERATION_IDENTITY = re.compile(r"^operation:[0-9a-f]{64}$")
_ATTEMPT_IDENTITY = re.compile(r"^attempt:[0-9a-f]{64}$")
_BUILD_LEGALITY_IDENTITY = re.compile(r"^build-legality:[0-9a-f]{64}$")


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
    last_material_legality_identity: str | None = None
    blocked_material_legality_identity: str | None = None
    failed_material_legality_identities: tuple[str, ...] = ()
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
    material_duplicate: bool = False

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
            "material_duplicate": self.material_duplicate,
        }


class RawAuthoritativePreDispatchStatus(str, Enum):  # noqa: UP042 - PySC2 bridge supports Python 3.9
    """Typed disposition for an authoritative build pre-dispatch failure."""

    RETRY = "retry"
    DEFER_REPLAN = "defer_replan"
    DUPLICATE = "duplicate"
    RESET = "reset"


@dataclass(frozen=True)
class RawAuthoritativePreDispatchState:
    """Auditable operation-level circuit state before a build reservation."""

    operation_id: str
    action_name: str | None
    streak: int
    threshold: int
    circuit_open: bool
    last_status: str
    last_command_id: str | None = None
    last_attempt_ordinal: int | None = None
    last_builder_tag: int | None = None
    last_ability_id: int | None = None
    last_world_target: tuple[float, float] | None = None
    last_failure_code: str | None = None
    last_placement_revision: str | None = None
    last_target_state_revision: str | None = None
    last_observation_revision: str | None = None
    last_observation_game_loop: int | None = None
    last_material_legality_identity: str | None = None
    blocked_material_legality_identity: str | None = None
    failed_authorization_material_legality_identities: tuple[str, ...] = ()
    seen_attempt_ordinals: tuple[int, ...] = ()
    seen_command_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "action_name": self.action_name,
            "streak": self.streak,
            "threshold": self.threshold,
            "circuit_open": self.circuit_open,
            "last_status": self.last_status,
            "last_command_id": self.last_command_id,
            "last_attempt_ordinal": self.last_attempt_ordinal,
            "last_builder_tag": self.last_builder_tag,
            "last_ability_id": self.last_ability_id,
            "last_world_target": (
                None
                if self.last_world_target is None
                else [float(self.last_world_target[0]), float(self.last_world_target[1])]
            ),
            "last_failure_code": self.last_failure_code,
            "last_placement_revision": self.last_placement_revision,
            "last_target_state_revision": self.last_target_state_revision,
            "last_observation_revision": self.last_observation_revision,
            "last_observation_game_loop": self.last_observation_game_loop,
            "last_material_legality_identity": self.last_material_legality_identity,
            "blocked_material_legality_identity": self.blocked_material_legality_identity,
            "failed_authorization_material_legality_identities": list(
                self.failed_authorization_material_legality_identities
            ),
            "seen_attempt_ordinals": list(self.seen_attempt_ordinals),
            "seen_command_ids": list(self.seen_command_ids),
        }


@dataclass(frozen=True)
class RawAuthoritativePreDispatchDecision:
    """One authoritative pre-dispatch circuit update."""

    operation_id: str | None
    action_name: str
    command_id: str
    failure_code: str
    status: str
    streak: int
    threshold: int
    circuit_open: bool
    duplicate_attempt: bool
    attempt_id: str | None = None
    attempt_ordinal: int | None = None
    builder_tag: int | None = None
    ability_id: int | None = None
    world_target: tuple[float, float] | None = None
    placement_revision: str | None = None
    target_state_revision: str | None = None
    observation_revision: str | None = None
    observation_game_loop: int | None = None
    material_legality_identity: str | None = None
    material_evidence_valid: bool = False
    invalid_evidence_reasons: tuple[str, ...] = ()
    state_transition: str | None = None
    reset_reason: str | None = None
    material_change_reason: str | None = None
    next_action: str = "retry"
    material_duplicate: bool = False
    operation_epoch_changed: bool = False

    @property
    def deferred(self) -> bool:
        return self.status == RawAuthoritativePreDispatchStatus.DEFER_REPLAN.value

    @property
    def transition(self) -> str | None:
        """Compatibility alias for callers that inspect the typed decision."""

        return self.state_transition

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "action_name": self.action_name,
            "command_id": self.command_id,
            "failure_code": self.failure_code,
            "status": self.status,
            "streak": self.streak,
            "threshold": self.threshold,
            "circuit_open": self.circuit_open,
            "duplicate_attempt": self.duplicate_attempt,
            "attempt_id": self.attempt_id,
            "attempt_ordinal": self.attempt_ordinal,
            "builder_tag": self.builder_tag,
            "ability_id": self.ability_id,
            "world_target": (
                None
                if self.world_target is None
                else [float(self.world_target[0]), float(self.world_target[1])]
            ),
            "placement_revision": self.placement_revision,
            "target_state_revision": self.target_state_revision,
            "observation_revision": self.observation_revision,
            "observation_game_loop": self.observation_game_loop,
            "material_legality_identity": self.material_legality_identity,
            "material_evidence_valid": self.material_evidence_valid,
            "invalid_evidence_reasons": list(self.invalid_evidence_reasons),
            "material_duplicate": self.material_duplicate,
            "operation_epoch_changed": self.operation_epoch_changed,
            "state_transition": self.state_transition,
            "reset_reason": self.reset_reason,
            "material_change_reason": self.material_change_reason,
            "next_action": self.next_action,
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
    last_material_legality_identity: str | None = None
    blocked_material_legality_identity: str | None = None
    failed_material_legality_identities: set[str] = field(default_factory=set)
    seen_attempt_ordinals: set[int] = field(default_factory=set)
    seen_command_ids: set[str] = field(default_factory=set)


@dataclass
class _OperationAuthoritativePreDispatchLedger:
    operation_id: str
    threshold: int
    action_name: str | None = None
    streak: int = 0
    circuit_open: bool = False
    last_status: str = RawAuthoritativePreDispatchStatus.RESET.value
    last_command_id: str | None = None
    last_attempt_ordinal: int | None = None
    last_builder_tag: int | None = None
    last_ability_id: int | None = None
    last_world_target: tuple[float, float] | None = None
    last_failure_code: str | None = None
    last_placement_revision: str | None = None
    last_target_state_revision: str | None = None
    last_observation_revision: str | None = None
    last_observation_game_loop: int | None = None
    last_material_legality_identity: str | None = None
    blocked_material_legality_identity: str | None = None
    failed_authorization_material_legality_identities: set[str] = field(default_factory=set)
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


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_point(value: Any) -> tuple[float, float] | None:
    if value is None or not isinstance(value, Sequence) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _required_point(value: Any) -> tuple[float, float]:
    point = _optional_point(value)
    if point is None:
        raise ValueError("checkpoint placement target must be a two-coordinate point")
    return point


def _required_grid_cell(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("checkpoint placement cell must contain two coordinates")
    return int(value[0]), int(value[1])


def _is_positive_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _is_finite_point(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


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


def build_material_legality_identity(
    *,
    operation_id: str | None,
    builder_tag: int | None,
    ability_id: int | None,
    world_target: tuple[float, float] | None,
    target_legality_fingerprint: str | None,
    target_state_revision: str | None = None,
) -> str:
    """Build the stable identity used by operation-level no-start tombstones."""

    payload = {
        "operation_id": None if operation_id is None else str(operation_id),
        "builder_tag": None if builder_tag is None else int(builder_tag),
        "ability_id": None if ability_id is None else int(ability_id),
        "world_target": (
            None if world_target is None else [float(world_target[0]), float(world_target[1])]
        ),
        "target_legality_fingerprint": target_legality_fingerprint,
        "target_state_revision": target_state_revision,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"build-legality:{digest}"


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
        self._operation_authoritative_pre_dispatch: dict[
            str,
            _OperationAuthoritativePreDispatchLedger,
        ] = {}
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
        self._failed_build_authorizations: set[str] = set()
        self._failed_build_authorization_materials: dict[str, str] = {}

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

    def builder_lease_owner(self, builder_tag: int) -> str | None:
        """Return the command that owns a builder lease, if any."""

        return self._builder_leases.get(int(builder_tag))

    @staticmethod
    def build_authorization_identity(
        *,
        operation_id: str | None,
        builder_tag: int,
        ability_id: int,
        world_target: tuple[float, float],
        target_state_revision: str | None,
    ) -> str:
        payload = {
            "operation_id": operation_id,
            "builder_tag": int(builder_tag),
            "ability_id": int(ability_id),
            "world_target": [float(world_target[0]), float(world_target[1])],
            "target_state_revision": target_state_revision,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"build-authorization:{digest}"

    def record_failed_build_authorization(
        self,
        *,
        operation_id: str | None,
        builder_tag: int,
        ability_id: int,
        world_target: tuple[float, float],
        target_state_revision: str | None,
        material_legality_identity: str | None = None,
    ) -> str:
        identity = self.build_authorization_identity(
            operation_id=operation_id,
            builder_tag=builder_tag,
            ability_id=ability_id,
            world_target=world_target,
            target_state_revision=target_state_revision,
        )
        self._failed_build_authorizations.add(identity)
        if material_legality_identity is not None:
            self._failed_build_authorization_materials[identity] = str(material_legality_identity)
        return identity

    def failed_build_authorization_material_identity(
        self,
        *,
        operation_id: str | None,
        builder_tag: int,
        ability_id: int,
        world_target: tuple[float, float],
        target_state_revision: str | None,
    ) -> str | None:
        """Return the exact material identity retained for an auth-cache hit."""

        identity = self.build_authorization_identity(
            operation_id=operation_id,
            builder_tag=builder_tag,
            ability_id=ability_id,
            world_target=world_target,
            target_state_revision=target_state_revision,
        )
        return self._failed_build_authorization_materials.get(identity)

    def build_authorization_was_rejected(
        self,
        *,
        operation_id: str | None,
        builder_tag: int,
        ability_id: int,
        world_target: tuple[float, float],
        target_state_revision: str | None,
    ) -> bool:
        return (
            self.build_authorization_identity(
                operation_id=operation_id,
                builder_tag=builder_tag,
                ability_id=ability_id,
                world_target=world_target,
                target_state_revision=target_state_revision,
            )
            in self._failed_build_authorizations
        )

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

    def checkpoint_state(self) -> dict[str, Any]:
        """Return JSON-safe placement/tombstone state for episode checkpoints."""

        return {
            "version": 1,
            "no_start_operations": {
                operation_id: {
                    "operation_id": ledger.operation_id,
                    "threshold": ledger.threshold,
                    "streak": ledger.streak,
                    "circuit_open": ledger.circuit_open,
                    "last_status": ledger.last_status,
                    "last_command_id": ledger.last_command_id,
                    "last_attempt_ordinal": ledger.last_attempt_ordinal,
                    "last_builder_tag": ledger.last_builder_tag,
                    "last_world_target": ledger.last_world_target,
                    "last_placement_revision": ledger.last_placement_revision,
                    "last_target_state_revision": ledger.last_target_state_revision,
                    "last_observation_revision": ledger.last_observation_revision,
                    "last_material_legality_identity": ledger.last_material_legality_identity,
                    "failed_material_legality_identities": sorted(
                        ledger.failed_material_legality_identities
                    ),
                    "blocked_target_state_revision": ledger.blocked_target_state_revision,
                    "blocked_builder_tag": ledger.blocked_builder_tag,
                    "blocked_placement_revision": ledger.blocked_placement_revision,
                    "blocked_observation_revision": ledger.blocked_observation_revision,
                    "blocked_material_legality_identity": ledger.blocked_material_legality_identity,
                    "blocked_failure_classification": ledger.blocked_failure_classification,
                    "blocked_target_side_evidence": ledger.blocked_target_side_evidence,
                    "seen_attempt_ordinals": sorted(ledger.seen_attempt_ordinals),
                    "seen_command_ids": sorted(ledger.seen_command_ids),
                }
                for operation_id, ledger in self._operation_no_start.items()
            },
            "authoritative_pre_dispatch_operations": {
                operation_id: {
                    "operation_id": ledger.operation_id,
                    "threshold": ledger.threshold,
                    "action_name": ledger.action_name,
                    "streak": ledger.streak,
                    "circuit_open": ledger.circuit_open,
                    "last_status": ledger.last_status,
                    "last_command_id": ledger.last_command_id,
                    "last_attempt_ordinal": ledger.last_attempt_ordinal,
                    "last_builder_tag": ledger.last_builder_tag,
                    "last_ability_id": ledger.last_ability_id,
                    "last_world_target": ledger.last_world_target,
                    "last_failure_code": ledger.last_failure_code,
                    "last_placement_revision": ledger.last_placement_revision,
                    "last_target_state_revision": ledger.last_target_state_revision,
                    "last_observation_revision": ledger.last_observation_revision,
                    "last_observation_game_loop": ledger.last_observation_game_loop,
                    "last_material_legality_identity": ledger.last_material_legality_identity,
                    "blocked_material_legality_identity": ledger.blocked_material_legality_identity,
                    "failed_authorization_material_legality_identities": sorted(
                        ledger.failed_authorization_material_legality_identities
                    ),
                    "seen_attempt_ordinals": sorted(ledger.seen_attempt_ordinals),
                    "seen_command_ids": sorted(ledger.seen_command_ids),
                }
                for operation_id, ledger in self._operation_authoritative_pre_dispatch.items()
            },
            "permanent_exclusions": [
                {
                    "world_target": exclusion.world_target,
                    "occupied_grid_cells": sorted(exclusion.occupied_grid_cells),
                    "reason": exclusion.reason,
                }
                for exclusion in self._permanent_exclusions
            ],
            "temporary_suppressions": {
                action_name: [
                    {
                        "world_target": suppression.world_target,
                        "occupied_grid_cells": sorted(suppression.occupied_grid_cells),
                        "expires_game_loop": suppression.expires_game_loop,
                        "reason": suppression.reason,
                        "target_state_revision": suppression.target_state_revision,
                        "anchor_tag": suppression.anchor_tag,
                    }
                    for suppression in suppressions
                ]
                for action_name, suppressions in self._temporary_suppressions.items()
            },
            "failed_build_authorizations": sorted(self._failed_build_authorizations),
            "failed_build_authorization_materials": dict(
                self._failed_build_authorization_materials
            ),
            "target_state_memory": [
                {
                    "action_name": action_name,
                    "world_target": list(world_target),
                    "signature": signature,
                    "generation": generation,
                }
                for (action_name, world_target), (
                    signature,
                    generation,
                ) in self._target_state_memory.items()
            ],
        }

    export_state = checkpoint_state

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore durable operation tombstones and target suppressions."""

        operations = state.get("no_start_operations", {})
        if isinstance(operations, Mapping):
            for operation_id, raw in operations.items():
                if not isinstance(raw, Mapping):
                    continue
                no_start_ledger = _OperationNoStartLedger(
                    operation_id=str(raw.get("operation_id", operation_id)),
                    threshold=int(raw.get("threshold", self.no_start_streak_threshold)),
                    streak=int(raw.get("streak", 0)),
                    circuit_open=bool(raw.get("circuit_open", False)),
                    last_status=str(raw.get("last_status", RawPlacementNoStartStatus.RESET.value)),
                    last_command_id=_optional_str(raw.get("last_command_id")),
                    last_attempt_ordinal=_optional_int(raw.get("last_attempt_ordinal")),
                    last_builder_tag=_optional_int(raw.get("last_builder_tag")),
                    last_world_target=_optional_point(raw.get("last_world_target")),
                    last_placement_revision=_optional_str(raw.get("last_placement_revision")),
                    last_target_state_revision=_optional_str(raw.get("last_target_state_revision")),
                    last_observation_revision=_optional_str(raw.get("last_observation_revision")),
                    last_material_legality_identity=_optional_str(
                        raw.get("last_material_legality_identity")
                    ),
                    blocked_target_state_revision=_optional_str(
                        raw.get("blocked_target_state_revision")
                    ),
                    blocked_builder_tag=_optional_int(raw.get("blocked_builder_tag")),
                    blocked_placement_revision=_optional_str(raw.get("blocked_placement_revision")),
                    blocked_observation_revision=_optional_str(
                        raw.get("blocked_observation_revision")
                    ),
                    blocked_material_legality_identity=_optional_str(
                        raw.get("blocked_material_legality_identity")
                    ),
                    blocked_failure_classification=_optional_str(
                        raw.get("blocked_failure_classification")
                    ),
                    blocked_target_side_evidence=bool(
                        raw.get("blocked_target_side_evidence", False)
                    ),
                    failed_material_legality_identities={
                        str(value) for value in raw.get("failed_material_legality_identities", ())
                    },
                    seen_attempt_ordinals={
                        int(value) for value in raw.get("seen_attempt_ordinals", ())
                    },
                    seen_command_ids={str(value) for value in raw.get("seen_command_ids", ())},
                )
                self._operation_no_start[str(operation_id)] = no_start_ledger
        authoritative_operations = state.get("authoritative_pre_dispatch_operations", {})
        if isinstance(authoritative_operations, Mapping):
            for operation_id, raw in authoritative_operations.items():
                if not isinstance(raw, Mapping):
                    continue
                restored_threshold = int(
                    raw.get("threshold", _AUTHORITATIVE_PRE_DISPATCH_STREAK_THRESHOLD)
                )
                if restored_threshold != _AUTHORITATIVE_PRE_DISPATCH_STREAK_THRESHOLD:
                    raise ValueError(
                        "authoritative pre-dispatch checkpoint threshold must be exactly 3"
                    )
                authoritative_ledger = _OperationAuthoritativePreDispatchLedger(
                    operation_id=str(raw.get("operation_id", operation_id)),
                    threshold=_AUTHORITATIVE_PRE_DISPATCH_STREAK_THRESHOLD,
                    action_name=_optional_str(raw.get("action_name")),
                    streak=int(raw.get("streak", 0)),
                    circuit_open=bool(raw.get("circuit_open", False)),
                    last_status=str(
                        raw.get(
                            "last_status",
                            RawAuthoritativePreDispatchStatus.RESET.value,
                        )
                    ),
                    last_command_id=_optional_str(raw.get("last_command_id")),
                    last_attempt_ordinal=_optional_int(raw.get("last_attempt_ordinal")),
                    last_builder_tag=_optional_int(raw.get("last_builder_tag")),
                    last_ability_id=_optional_int(raw.get("last_ability_id")),
                    last_world_target=_optional_point(raw.get("last_world_target")),
                    last_failure_code=_optional_str(raw.get("last_failure_code")),
                    last_placement_revision=_optional_str(raw.get("last_placement_revision")),
                    last_target_state_revision=_optional_str(raw.get("last_target_state_revision")),
                    last_observation_revision=_optional_str(raw.get("last_observation_revision")),
                    last_observation_game_loop=_optional_int(raw.get("last_observation_game_loop")),
                    last_material_legality_identity=_optional_str(
                        raw.get("last_material_legality_identity")
                    ),
                    blocked_material_legality_identity=_optional_str(
                        raw.get("blocked_material_legality_identity")
                    ),
                    failed_authorization_material_legality_identities={
                        str(value)
                        for value in raw.get(
                            "failed_authorization_material_legality_identities", ()
                        )
                    },
                    seen_attempt_ordinals={
                        int(value) for value in raw.get("seen_attempt_ordinals", ())
                    },
                    seen_command_ids={str(value) for value in raw.get("seen_command_ids", ())},
                )
                self._operation_authoritative_pre_dispatch[str(operation_id)] = authoritative_ledger
        permanent = state.get("permanent_exclusions", ())
        if isinstance(permanent, Sequence) and not isinstance(permanent, (str, bytes)):
            self._permanent_exclusions = [
                _SpatialExclusion(
                    _required_point(item.get("world_target")),
                    frozenset(
                        _required_grid_cell(value) for value in item.get("occupied_grid_cells", ())
                    ),
                    str(item.get("reason", "invalid_terrain")),
                )
                for item in permanent
                if isinstance(item, Mapping)
            ]
        temporary = state.get("temporary_suppressions", {})
        if isinstance(temporary, Mapping):
            self._temporary_suppressions = {
                str(action_name): [
                    _TemporarySuppression(
                        _required_point(item.get("world_target")),
                        frozenset(
                            _required_grid_cell(value)
                            for value in item.get("occupied_grid_cells", ())
                        ),
                        int(item.get("expires_game_loop", 0)),
                        str(item.get("reason", "no_legal_placement")),
                        _optional_str(item.get("target_state_revision")),
                        _optional_int(item.get("anchor_tag")),
                    )
                    for item in values
                    if isinstance(item, Mapping)
                ]
                for action_name, values in temporary.items()
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
            }
        failed_authorizations = state.get("failed_build_authorizations", ())
        self._failed_build_authorizations = (
            {str(value) for value in failed_authorizations}
            if isinstance(failed_authorizations, (list, tuple, set, frozenset))
            else set()
        )
        failed_authorization_materials = state.get("failed_build_authorization_materials", {})
        self._failed_build_authorization_materials = (
            {str(key): str(value) for key, value in failed_authorization_materials.items()}
            if isinstance(failed_authorization_materials, Mapping)
            else {}
        )
        target_state_memory = state.get("target_state_memory", ())
        if isinstance(target_state_memory, Sequence) and not isinstance(
            target_state_memory,
            (str, bytes),
        ):
            self._target_state_memory = {
                (
                    str(item.get("action_name")),
                    _required_point(item.get("world_target")),
                ): (
                    str(item.get("signature", "")),
                    int(item.get("generation", 0)),
                )
                for item in target_state_memory
                if isinstance(item, Mapping)
            }

    restore_state = restore_checkpoint_state

    def authoritative_pre_dispatch_state(
        self,
        operation_id: str,
    ) -> RawAuthoritativePreDispatchState | None:
        """Return the durable authoritative pre-dispatch circuit state."""

        ledger = self._operation_authoritative_pre_dispatch.get(str(operation_id))
        if ledger is None:
            return None
        return RawAuthoritativePreDispatchState(
            operation_id=ledger.operation_id,
            action_name=ledger.action_name,
            streak=ledger.streak,
            threshold=ledger.threshold,
            circuit_open=ledger.circuit_open,
            last_status=ledger.last_status,
            last_command_id=ledger.last_command_id,
            last_attempt_ordinal=ledger.last_attempt_ordinal,
            last_builder_tag=ledger.last_builder_tag,
            last_ability_id=ledger.last_ability_id,
            last_world_target=ledger.last_world_target,
            last_failure_code=ledger.last_failure_code,
            last_placement_revision=ledger.last_placement_revision,
            last_target_state_revision=ledger.last_target_state_revision,
            last_observation_revision=ledger.last_observation_revision,
            last_observation_game_loop=ledger.last_observation_game_loop,
            last_material_legality_identity=ledger.last_material_legality_identity,
            blocked_material_legality_identity=ledger.blocked_material_legality_identity,
            failed_authorization_material_legality_identities=tuple(
                sorted(ledger.failed_authorization_material_legality_identities)
            ),
            seen_attempt_ordinals=tuple(sorted(ledger.seen_attempt_ordinals)),
            seen_command_ids=tuple(sorted(ledger.seen_command_ids)),
        )

    def target_state_revision_for(
        self,
        observation: Any,
        action_name: str,
        world_target: tuple[float, float] | None,
        *,
        anchor_tag: int | None = None,
    ) -> str | None:
        """Compute target state for a pre-dispatch identity without reserving it."""

        if world_target is None:
            return None
        self.observe(observation, require_feature_visibility=False)
        return self._target_state_revision(
            observation,
            action_name,
            (float(world_target[0]), float(world_target[1])),
            anchor_tag=anchor_tag,
        )

    @staticmethod
    def authoritative_pre_dispatch_material_identity(
        *,
        operation_id: str | None,
        builder_tag: int | None,
        ability_id: int | None,
        world_target: tuple[float, float] | None,
        target_state_revision: str | None,
        material_legality_identity: str | None = None,
    ) -> str | None:
        """Return a stable material identity when all raw identity fields exist."""

        if (
            operation_id is None
            or _OPERATION_IDENTITY.fullmatch(str(operation_id)) is None
            or not _is_positive_int(builder_tag)
            or not _is_positive_int(ability_id)
            or not _is_finite_point(world_target)
            or target_state_revision is None
            or not str(target_state_revision)
        ):
            return None
        if material_legality_identity is not None:
            normalized = str(material_legality_identity)
            return normalized if _BUILD_LEGALITY_IDENTITY.fullmatch(normalized) else None
        return build_material_legality_identity(
            operation_id=operation_id,
            builder_tag=builder_tag,
            ability_id=ability_id,
            world_target=world_target,
            target_legality_fingerprint=None,
            target_state_revision=target_state_revision,
        )

    def authoritative_pre_dispatch_retry_allowed(
        self,
        operation_id: str,
        *,
        action_name: str | None = None,
        builder_tag: int | None = None,
        ability_id: int | None = None,
        world_target: tuple[float, float] | None = None,
        target_state_revision: str | None = None,
        material_legality_identity: str | None = None,
        placement_revision: str | None = None,
        observation_revision: str | None = None,
        observation_game_loop: int | None = None,
    ) -> bool:
        """Allow a circuit-open operation only with a changed stable material identity."""

        ledger = self._operation_authoritative_pre_dispatch.get(str(operation_id))
        if ledger is None or not ledger.circuit_open:
            return True
        current_identity = self.authoritative_pre_dispatch_material_identity(
            operation_id=operation_id,
            builder_tag=builder_tag,
            ability_id=ability_id,
            world_target=world_target,
            target_state_revision=target_state_revision,
            material_legality_identity=material_legality_identity,
        )
        if current_identity is None or ledger.blocked_material_legality_identity is None:
            return False
        return (
            self._authoritative_pre_dispatch_material_change_reason(
                ledger,
                builder_tag=builder_tag,
                ability_id=ability_id,
                world_target=world_target,
                target_state_revision=target_state_revision,
            )
            is not None
        )

    @staticmethod
    def _authoritative_pre_dispatch_material_change_reason(
        ledger: _OperationAuthoritativePreDispatchLedger,
        *,
        builder_tag: int | None,
        ability_id: int | None,
        world_target: tuple[float, float] | None,
        target_state_revision: str | None,
    ) -> str | None:
        """Recognize only state changes that can alter the failed authorization."""

        normalized_builder = None if builder_tag is None else int(builder_tag)
        if (
            normalized_builder is not None
            and ledger.last_builder_tag is not None
            and normalized_builder != ledger.last_builder_tag
        ):
            return "builder_changed"
        normalized_ability = None if ability_id is None else int(ability_id)
        if (
            normalized_ability is not None
            and ledger.last_ability_id is not None
            and normalized_ability != ledger.last_ability_id
        ):
            return "ability_changed"
        same_target = bool(
            world_target is not None
            and ledger.last_world_target is not None
            and all(
                abs(float(world_target[index]) - float(ledger.last_world_target[index])) <= 1e-3
                for index in range(2)
            )
        )
        if (
            same_target
            and target_state_revision is not None
            and ledger.last_target_state_revision is not None
            and str(target_state_revision) != ledger.last_target_state_revision
        ):
            return "target_state_changed"
        return None

    def record_authoritative_pre_dispatch_failure(
        self,
        *,
        operation_id: str | None,
        action_name: str,
        command_id: str,
        failure_code: str,
        attempt_id: str | None = None,
        attempt_ordinal: int | None = None,
        builder_tag: int | None = None,
        ability_id: int | None = None,
        world_target: tuple[float, float] | None = None,
        placement_revision: str | None = None,
        target_state_revision: str | None = None,
        observation_revision: str | None = None,
        observation_game_loop: int | None = None,
        material_legality_identity: str | None = None,
    ) -> RawAuthoritativePreDispatchDecision:
        """Record one of the four authoritative failures before reservation/lease."""

        normalized_operation = None if operation_id is None else str(operation_id)
        normalized_attempt = None if attempt_id is None else str(attempt_id)
        if normalized_operation is not None and _OPERATION_IDENTITY.fullmatch(normalized_operation):
            expected_attempt = (
                None
                if attempt_ordinal is None
                else "attempt:"
                + hashlib.sha256(
                    json.dumps(
                        {
                            "operation_id": normalized_operation,
                            "command_id": str(command_id),
                            "attempt_ordinal": int(attempt_ordinal),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )
        else:
            expected_attempt = None
        identity = self.authoritative_pre_dispatch_material_identity(
            operation_id=normalized_operation,
            builder_tag=builder_tag,
            ability_id=ability_id,
            world_target=world_target,
            target_state_revision=target_state_revision,
            material_legality_identity=material_legality_identity,
        )
        invalid_evidence_reasons = tuple(
            reason
            for reason, missing in (
                ("operation_id_missing", normalized_operation is None),
                (
                    "operation_id_invalid",
                    normalized_operation is not None
                    and _OPERATION_IDENTITY.fullmatch(normalized_operation) is None,
                ),
                ("attempt_id_missing", normalized_attempt is None),
                (
                    "attempt_id_invalid",
                    normalized_attempt is not None
                    and (
                        _ATTEMPT_IDENTITY.fullmatch(normalized_attempt) is None
                        or expected_attempt is None
                        or normalized_attempt != expected_attempt
                    ),
                ),
                ("builder_tag_missing", not _is_positive_int(builder_tag)),
                ("ability_id_missing", not _is_positive_int(ability_id)),
                ("world_target_missing", not _is_finite_point(world_target)),
                (
                    "target_state_revision_missing",
                    target_state_revision is None or not str(target_state_revision),
                ),
                ("material_legality_identity_invalid", identity is None),
            )
            if missing
        )
        material_evidence_valid = not invalid_evidence_reasons
        if normalized_operation is None:
            decision = RawAuthoritativePreDispatchDecision(
                operation_id=None,
                action_name=str(action_name),
                command_id=str(command_id),
                failure_code=str(failure_code),
                status=RawAuthoritativePreDispatchStatus.RETRY.value,
                streak=0,
                threshold=_AUTHORITATIVE_PRE_DISPATCH_STREAK_THRESHOLD,
                circuit_open=False,
                duplicate_attempt=False,
                attempt_id=normalized_attempt,
                attempt_ordinal=attempt_ordinal,
                builder_tag=None if builder_tag is None else int(builder_tag),
                ability_id=None if ability_id is None else int(ability_id),
                world_target=world_target,
                placement_revision=placement_revision,
                target_state_revision=target_state_revision,
                observation_revision=observation_revision,
                observation_game_loop=observation_game_loop,
                material_legality_identity=identity,
                material_evidence_valid=material_evidence_valid,
                invalid_evidence_reasons=invalid_evidence_reasons,
            )
            return decision

        ledger = self._operation_authoritative_pre_dispatch.setdefault(
            normalized_operation,
            _OperationAuthoritativePreDispatchLedger(
                operation_id=normalized_operation,
                threshold=_AUTHORITATIVE_PRE_DISPATCH_STREAK_THRESHOLD,
            ),
        )
        duplicate_attempt = (
            str(command_id) in ledger.seen_command_ids
            or attempt_ordinal is not None
            and int(attempt_ordinal) in ledger.seen_attempt_ordinals
        )
        material_duplicate = (
            str(failure_code) == "placement_query_rejected_cached"
            and identity is not None
            and identity in ledger.failed_authorization_material_legality_identities
        )
        transition: str | None = None
        reset_reason: str | None = None
        material_change_reason: str | None = None
        if duplicate_attempt:
            status = RawAuthoritativePreDispatchStatus.DUPLICATE.value
            next_action = "defer_replan" if ledger.circuit_open else "retry"
        elif ledger.circuit_open:
            material_change_reason = (
                self._authoritative_pre_dispatch_material_change_reason(
                    ledger,
                    builder_tag=builder_tag,
                    ability_id=ability_id,
                    world_target=world_target,
                    target_state_revision=target_state_revision,
                )
                if material_evidence_valid and ledger.blocked_material_legality_identity is not None
                else None
            )
            if material_change_reason is not None:
                ledger.streak = 0
                ledger.circuit_open = False
                ledger.blocked_material_legality_identity = None
                transition = "open_to_reset"
                reset_reason = "material_state_changed"
                ledger.last_status = RawAuthoritativePreDispatchStatus.RESET.value
            else:
                ledger.seen_command_ids.add(str(command_id))
                if attempt_ordinal is not None:
                    ledger.seen_attempt_ordinals.add(int(attempt_ordinal))
                status = RawAuthoritativePreDispatchStatus.DEFER_REPLAN.value
                next_action = "replan"
        if not duplicate_attempt and not ledger.circuit_open:
            ledger.seen_command_ids.add(str(command_id))
            if attempt_ordinal is not None:
                ledger.seen_attempt_ordinals.add(int(attempt_ordinal))
            ledger.action_name = str(action_name)
            ledger.streak += 1
            ledger.last_command_id = str(command_id)
            ledger.last_attempt_ordinal = None if attempt_ordinal is None else int(attempt_ordinal)
            ledger.last_builder_tag = None if builder_tag is None else int(builder_tag)
            ledger.last_ability_id = None if ability_id is None else int(ability_id)
            ledger.last_world_target = world_target
            ledger.last_failure_code = str(failure_code)
            ledger.last_placement_revision = placement_revision
            ledger.last_target_state_revision = target_state_revision
            ledger.last_observation_revision = observation_revision
            ledger.last_observation_game_loop = observation_game_loop
            ledger.last_material_legality_identity = identity
            if str(failure_code) == "placement_query_rejected" and identity is not None:
                ledger.failed_authorization_material_legality_identities.add(identity)
            if ledger.streak >= ledger.threshold:
                ledger.circuit_open = True
                ledger.blocked_material_legality_identity = identity
                status = RawAuthoritativePreDispatchStatus.DEFER_REPLAN.value
                next_action = "replan"
                transition = transition or "closed_to_open"
            else:
                status = RawAuthoritativePreDispatchStatus.RETRY.value
                next_action = "retry"
            ledger.last_status = status
        elif duplicate_attempt:
            ledger.seen_command_ids.add(str(command_id))
        decision = RawAuthoritativePreDispatchDecision(
            operation_id=normalized_operation,
            action_name=str(action_name),
            command_id=str(command_id),
            failure_code=str(failure_code),
            status=status,
            streak=ledger.streak,
            threshold=ledger.threshold,
            circuit_open=ledger.circuit_open,
            duplicate_attempt=duplicate_attempt,
            attempt_id=normalized_attempt,
            attempt_ordinal=None if attempt_ordinal is None else int(attempt_ordinal),
            builder_tag=None if builder_tag is None else int(builder_tag),
            ability_id=None if ability_id is None else int(ability_id),
            world_target=world_target,
            placement_revision=placement_revision,
            target_state_revision=target_state_revision,
            observation_revision=observation_revision,
            observation_game_loop=observation_game_loop,
            material_legality_identity=identity,
            material_evidence_valid=material_evidence_valid,
            invalid_evidence_reasons=invalid_evidence_reasons,
            state_transition=transition,
            reset_reason=reset_reason,
            material_change_reason=material_change_reason,
            next_action=next_action,
            material_duplicate=material_duplicate,
        )
        return decision

    def reset_authoritative_pre_dispatch(
        self,
        operation_id: str,
        *,
        reason: str = "build_started",
        command_id: str = "",
        action_name: str = "",
        attempt_id: str | None = None,
        attempt_ordinal: int | None = None,
        builder_tag: int | None = None,
        ability_id: int | None = None,
        world_target: tuple[float, float] | None = None,
        placement_revision: str | None = None,
        target_state_revision: str | None = None,
        observation_revision: str | None = None,
        observation_game_loop: int | None = None,
    ) -> RawAuthoritativePreDispatchDecision | None:
        if reason not in {"build_started", "effect_confirmed"}:
            raise ValueError(
                "authoritative pre-dispatch reset requires build start or confirmation evidence"
            )
        ledger = self._operation_authoritative_pre_dispatch.get(str(operation_id))
        if ledger is None:
            return None
        had_state = ledger.streak > 0 or ledger.circuit_open
        was_open = ledger.circuit_open
        ledger.streak = 0
        ledger.circuit_open = False
        ledger.blocked_material_legality_identity = None
        ledger.failed_authorization_material_legality_identities.clear()
        ledger.seen_attempt_ordinals.clear()
        ledger.seen_command_ids.clear()
        ledger.last_status = RawAuthoritativePreDispatchStatus.RESET.value
        if not had_state:
            return None
        decision = RawAuthoritativePreDispatchDecision(
            operation_id=str(operation_id),
            action_name=action_name or (ledger.action_name or ""),
            command_id=command_id,
            failure_code="build_started",
            status=RawAuthoritativePreDispatchStatus.RESET.value,
            streak=0,
            threshold=ledger.threshold,
            circuit_open=False,
            duplicate_attempt=False,
            attempt_id=attempt_id,
            attempt_ordinal=attempt_ordinal,
            builder_tag=builder_tag,
            ability_id=ability_id,
            world_target=world_target,
            placement_revision=placement_revision,
            target_state_revision=target_state_revision,
            observation_revision=observation_revision,
            observation_game_loop=observation_game_loop,
            state_transition=("open_to_reset" if was_open else None),
            reset_reason=str(reason),
            next_action="retry",
        )
        return decision

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
            last_material_legality_identity=ledger.last_material_legality_identity,
            blocked_material_legality_identity=ledger.blocked_material_legality_identity,
            failed_material_legality_identities=tuple(
                sorted(ledger.failed_material_legality_identities)
            ),
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
        material_legality_identity: str | None = None,
    ) -> bool:
        """Whether a circuit-open operation has acquired a genuinely new target state."""

        ledger = self._operation_no_start.get(str(operation_id))
        if ledger is None or not ledger.circuit_open:
            return material_legality_identity not in (
                set() if ledger is None else ledger.failed_material_legality_identities
            )
        if (
            material_legality_identity is not None
            and material_legality_identity in ledger.failed_material_legality_identities
        ):
            return False
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
        material_changed = (
            material_legality_identity is not None
            and ledger.blocked_material_legality_identity is not None
            and material_legality_identity != ledger.blocked_material_legality_identity
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
        # Once a durable material identity is available, observation churn alone
        # cannot reopen the circuit.  Only a changed exact builder/ability/target
        # legality identity is a valid new attempt.
        if ledger.blocked_material_legality_identity is not None:
            target_changed = material_changed
            builder_changed = False
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
        ledger.blocked_material_legality_identity = None
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
        ledger.blocked_material_legality_identity = None
        ledger.failed_material_legality_identities.clear()
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
        material_legality_identity: str | None = None,
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
                    "material_legality_identity": material_legality_identity,
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
        material_duplicate = (
            material_legality_identity is not None
            and material_legality_identity in ledger.failed_material_legality_identities
        )
        if duplicate or material_duplicate:
            status = RawPlacementNoStartStatus.DUPLICATE.value
            next_action = "defer_replan" if ledger.circuit_open or material_duplicate else "retry"
            if material_duplicate and not duplicate:
                ledger.seen_command_ids.add(str(command_id))
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
                material_legality_identity=material_legality_identity,
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
                ledger.last_material_legality_identity = material_legality_identity
                if material_legality_identity is not None:
                    ledger.failed_material_legality_identities.add(material_legality_identity)
                if ledger.streak >= ledger.threshold:
                    ledger.circuit_open = True
                    ledger.blocked_target_state_revision = target_state_revision
                    ledger.blocked_builder_tag = None if builder_tag is None else int(builder_tag)
                    ledger.blocked_placement_revision = placement_revision
                    ledger.blocked_observation_revision = observation_revision
                    ledger.blocked_failure_classification = failure_classification
                    ledger.blocked_target_side_evidence = target_side_evidence
                    ledger.blocked_material_legality_identity = material_legality_identity
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
            "material_duplicate": material_duplicate,
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
            "material_legality_identity": material_legality_identity,
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
            material_duplicate=material_duplicate,
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
        attempt_id: str | None = None,
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
                anchor_tag=None,
                exclude_command_id=command_id,
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
                exclude_command_id=command_id,
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
                exclude_command_id=command_id,
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
            attempt_id=attempt_id,
            attempt_ordinal=(None if attempt_ordinal is None else int(attempt_ordinal)),
        )
        existing = self._command_targets.get(command_id)
        if existing is not None and existing.reservation_id == reservation.reservation_id:
            refreshed = replace(
                existing,
                requested_world_target=requested_target,
                final_validated_world_target=emitted,
                placement_revision=current_revision,
                target_state_revision=target_state_revision,
                baseline_builder_orders=baseline_builder_orders,
                expires_game_loop=max(existing.expires_game_loop, int(expires_game_loop or 0)),
            )
            self._command_targets[command_id] = refreshed
            return refreshed
        if existing is not None:
            self._record_transition(
                command_id,
                reservation=existing,
                previous_state=existing.placement_state,
                next_state="released",
                game_loop=game_loop,
                release_reason="reservation_replaced",
            )
            self._release_builder_lease(existing)
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

    def record_build_authorization(
        self,
        command_id: str,
        *,
        ability_id: int,
        authorization: Any,
        game_loop: int,
    ) -> None:
        """Attach the exact read-only SC2 authority result to a reservation."""

        placement = self._command_targets.get(command_id)
        if placement is None:
            return
        placement_result = getattr(authorization, "placement_query_status", None)
        if placement_result is None:
            placement_result = getattr(authorization, "placement_result", None)
        available_query = getattr(authorization, "available_ability_query", None)
        fingerprint = getattr(authorization, "target_legality_fingerprint", None)
        details = getattr(authorization, "details", {})
        if not isinstance(details, Mapping):
            details = {}
        material_identity = build_material_legality_identity(
            operation_id=placement.operation_id,
            builder_tag=placement.builder_tag,
            ability_id=int(ability_id),
            world_target=placement.world_target,
            target_legality_fingerprint=(None if fingerprint is None else str(fingerprint)),
            target_state_revision=placement.target_state_revision,
        )
        self._command_targets[command_id] = replace(
            placement,
            ability_id=int(ability_id),
            target_legality_fingerprint=(None if fingerprint is None else str(fingerprint)),
            material_legality_identity=material_identity,
            available_ability_query=(None if available_query is None else str(available_query)),
            placement_query_result=(None if placement_result is None else str(placement_result)),
            build_authorization_details=deepcopy(dict(details)),
        )

    def mark_primitive_constructed(self, command_id: str, *, game_loop: int) -> None:
        placement = self._command_targets.get(command_id)
        if placement is None:
            return
        self._command_targets[command_id] = replace(
            placement,
            primitive_constructed_game_loop=int(game_loop),
        )

    def mark_primitive_submitted(self, command_id: str, *, game_loop: int) -> None:
        placement = self._command_targets.get(command_id)
        if placement is None:
            return
        self._command_targets[command_id] = replace(
            placement,
            primitive_submitted_game_loop=int(game_loop),
        )

    def record_action_result(self, command_id: str, results: Sequence[Any]) -> None:
        placement = self._command_targets.get(command_id)
        if placement is None:
            return
        self._command_targets[command_id] = replace(
            placement,
            action_result=tuple(int(value) for value in results),
        )

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
        ability_id: int | None = None,
        placement_revision: str | None = None,
        target_state_revision: str | None = None,
        failure_classification: str | None = None,
        classification_basis: Sequence[str] = (),
        target_side_evidence: bool = False,
        builder_ready: bool = False,
        observation_revision: str | None = None,
        observation: Any | None = None,
        material_legality_identity: str | None = None,
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
        material_legality_identity = material_legality_identity or (
            None if placement is None else placement.material_legality_identity
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
        authoritative_pre_dispatch_decision: RawAuthoritativePreDispatchDecision | None = None
        if failure_code in _AUTHORITATIVE_PRE_DISPATCH_FAILURE_CODES:
            authoritative_pre_dispatch_decision = self.record_authoritative_pre_dispatch_failure(
                operation_id=operation_id,
                action_name=action_name,
                command_id=command_id,
                failure_code=failure_code,
                attempt_id=(None if placement is None else placement.attempt_id),
                attempt_ordinal=attempt_ordinal,
                builder_tag=builder_tag,
                ability_id=(
                    ability_id
                    if ability_id is not None
                    else (None if placement is None else placement.ability_id)
                ),
                world_target=target,
                placement_revision=placement_revision,
                target_state_revision=target_state_revision,
                observation_revision=observation_revision,
                observation_game_loop=(0 if game_loop is None else int(game_loop)),
                material_legality_identity=material_legality_identity,
            )
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
                material_legality_identity=material_legality_identity,
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
            authoritative_pre_dispatch_decision=authoritative_pre_dispatch_decision,
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
        authoritative_reset = (
            None
            if placement.operation_id is None
            else self.reset_authoritative_pre_dispatch(
                placement.operation_id,
                reason="effect_confirmed",
                command_id=command_id,
                action_name=placement.action_name,
                attempt_id=placement.attempt_id,
                attempt_ordinal=placement.attempt_ordinal,
                builder_tag=placement.builder_tag,
                ability_id=placement.ability_id,
                world_target=placement.world_target,
                placement_revision=placement.placement_revision,
                target_state_revision=placement.target_state_revision,
                observation_game_loop=game_loop,
            )
        )
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
            authoritative_pre_dispatch_decision=authoritative_reset,
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
        authoritative_reset = (
            None
            if placement.operation_id is None
            else self.reset_authoritative_pre_dispatch(
                placement.operation_id,
                reason="build_started",
                command_id=command_id,
                action_name=placement.action_name,
                attempt_id=placement.attempt_id,
                attempt_ordinal=placement.attempt_ordinal,
                builder_tag=placement.builder_tag,
                ability_id=placement.ability_id,
                world_target=placement.world_target,
                placement_revision=placement.placement_revision,
                target_state_revision=placement.target_state_revision,
                observation_game_loop=game_loop,
            )
        )
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
                authoritative_pre_dispatch_decision=authoritative_reset,
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
        exclude_command_id: str | None = None,
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
            for command_id, reservation in self._command_targets.items()
            if command_id != exclude_command_id
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
        authoritative_pre_dispatch_decision: RawAuthoritativePreDispatchDecision | None = None,
    ) -> None:
        from rtscortex_llm_pysc2.extractor import BUILD_SPECS

        resolved_action = reservation.action_name if reservation is not None else action_name
        if resolved_action is None:
            return
        spec = BUILD_SPECS.get(resolved_action)
        if spec is None:
            return
        target = reservation.world_target if reservation is not None else world_target
        if target is None and authoritative_pre_dispatch_decision is not None:
            target = authoritative_pre_dispatch_decision.world_target
        cells = (
            reservation.occupied_grid_cells
            if reservation is not None
            else _occupied_cells_for_spec(target, spec)
            if target is not None
            else frozenset()
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
        resolved_operation_id = (
            reservation.operation_id
            if reservation is not None
            else (
                None
                if authoritative_pre_dispatch_decision is None
                else authoritative_pre_dispatch_decision.operation_id
            )
        )
        resolved_attempt_id = (
            reservation.attempt_id
            if reservation is not None
            else (
                None
                if authoritative_pre_dispatch_decision is None
                else authoritative_pre_dispatch_decision.attempt_id
            )
        )
        resolved_attempt_ordinal = (
            reservation.attempt_ordinal
            if reservation is not None
            else (
                None
                if authoritative_pre_dispatch_decision is None
                else authoritative_pre_dispatch_decision.attempt_ordinal
            )
        )
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
            if reservation.ability_id is not None:
                transition["ability_id"] = reservation.ability_id
            if reservation.available_ability_query is not None:
                transition["available_ability_query"] = reservation.available_ability_query
            if reservation.placement_query_result is not None:
                transition["placement_query_result"] = reservation.placement_query_result
            if reservation.target_legality_fingerprint is not None:
                transition["target_legality_fingerprint"] = reservation.target_legality_fingerprint
            if reservation.material_legality_identity is not None:
                transition["material_legality_identity"] = reservation.material_legality_identity
            if reservation.primitive_constructed_game_loop is not None:
                transition["primitive_constructed_game_loop"] = (
                    reservation.primitive_constructed_game_loop
                )
            if reservation.primitive_submitted_game_loop is not None:
                transition["primitive_submitted_game_loop"] = (
                    reservation.primitive_submitted_game_loop
                )
            if reservation.action_result is not None:
                transition["action_result"] = list(reservation.action_result)
        if no_start_decision is not None:
            transition["placement_no_start"] = no_start_decision.to_dict()
        if authoritative_pre_dispatch_decision is not None:
            transition.update(
                {
                    "operation_id": resolved_operation_id,
                    "command_id": command_id,
                    "action_name": resolved_action,
                    "attempt_id": resolved_attempt_id,
                    "attempt_ordinal": resolved_attempt_ordinal,
                }
            )
            transition["authoritative_pre_dispatch"] = authoritative_pre_dispatch_decision.to_dict()
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
                **(
                    {
                        "operation_id": resolved_operation_id,
                        "attempt_id": resolved_attempt_id,
                        "attempt_ordinal": resolved_attempt_ordinal,
                    }
                    if authoritative_pre_dispatch_decision is not None
                    else {}
                ),
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
    "RawAuthoritativePreDispatchDecision",
    "RawAuthoritativePreDispatchState",
    "RawAuthoritativePreDispatchStatus",
    "RawPlacementNoStartDecision",
    "RawPlacementNoStartState",
    "RawPlacementReservation",
    "RawPlacementService",
    "build_material_legality_identity",
]
