"""Fail-closed engineering acceptance for one natural-terminal live run."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rtscortex.memory import StoredEvent
from rtscortex.placement import (
    CANONICAL_PLACEMENT_SPECS,
    canonical_footprint_cells,
)

ENGINEERING_GATES_FILENAME = "engineering-gates.json"
REQUIRED_ENGINEERING_GATES = (
    "runtime_crash_zero",
    "friendly_target_zero",
    "lineage_complete",
    "minimum_production_exposure",
    "production_confirmation_complete",
    "minimum_build_exposure",
    "build_start_coverage",
    "build_confirmation_rate",
    "build_failure_rate",
    "placement_identity_complete",
    "builder_provenance_complete",
    "builder_lease_complete",
    "placement_ledger_evidence_present",
    "placement_ledger_coverage",
    "placement_ledger_transitions_valid",
    "placement_ledger_canonical_footprint_complete",
    "placement_ledger_terminal_complete",
    "cross_type_footprint_overlap_zero",
    "nonspatial_quarantine_zero",
    "invalid_footprint_redispatch_zero",
    "repeated_retreat_arrival_zero",
    "unchanged_attack_redispatch_zero",
    "health_delta_engagement_attribution_valid",
    "expansion_terminal_immediate_rearm_zero",
    "defense_inventory_evidence_present",
    "defense_inventory_within_cap",
    "subscriber_callback_under_durable_lock_zero",
    "recovery_evidence_present",
    "checkpoint_tail_recovery_bounded",
    "postgame_semantic_event_coverage",
    "effective_loops_per_second",
    "natural_run_disk_reduction_ratio",
    "analysis_evidence_retention_bounded",
)

_NEGATIVE_BUILD_EFFECT_CODES = frozenset(
    {
        "effect_timeout",
        "no_build_order_observed",
        "worker_order_replaced",
        "target_not_created",
        "builder_not_observable",
        "no_build_start_evidence",
        "build_started_effect_missing",
    }
)
_RETAINED_EVENT_TYPES = frozenset(
    {
        "episode_result",
        "execution",
        "command_lifecycle",
        "command_lineage",
        "placement_ledger_transition",
        "tactical_actor_state",
        "expansion_commitment_started",
        "expansion_commitment_terminal",
        "defense_inventory_evaluated",
        "event_store_performance",
        "runtime_recovery_completed",
        "strategic_consequence_attributed",
        "postgame_review_completed",
    }
)
DEFAULT_MAX_RETAINED_ENGINEERING_EVENTS = 100_000


@dataclass
class EngineeringAccumulator:
    """Retain only command-scale acceptance evidence while scanning a journal."""

    events: list[StoredEvent] = field(default_factory=list)
    event_count: int = 0
    max_game_loop: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    retention_limit: int = DEFAULT_MAX_RETAINED_ENGINEERING_EVENTS
    retention_overflow_count: int = 0

    def ingest(self, event: StoredEvent) -> None:
        self.event_count += 1
        try:
            timestamp = datetime.fromisoformat(event.created_at)
        except ValueError:
            timestamp = None
        if timestamp is not None:
            if self.first_timestamp is None:
                self.first_timestamp = timestamp
            self.last_timestamp = timestamp
        game_loop = event.payload.get("game_loop")
        if isinstance(game_loop, int | float) and not isinstance(game_loop, bool):
            self.max_game_loop = max(self.max_game_loop, int(game_loop))
        if event.event_type in _RETAINED_EVENT_TYPES:
            if len(self.events) < self.retention_limit:
                self.events.append(event)
            else:
                self.retention_overflow_count += 1

    @classmethod
    def from_events(cls, events: Iterable[StoredEvent]) -> EngineeringAccumulator:
        accumulator = cls()
        for event in events:
            accumulator.ingest(event)
        return accumulator

    @property
    def elapsed_seconds(self) -> float:
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0.0
        return max(0.0, (self.last_timestamp - self.first_timestamp).total_seconds())


def build_engineering_gate_report(
    events: Iterable[StoredEvent] | EngineeringAccumulator,
    *,
    run_dir: Path,
    natural_run_baseline_bytes_per_loop: float | None,
    recovery_tail_limit: int = 4096,
    recovery_evidence: dict[str, Any] | None = None,
    expected_git_sha: str | None = None,
) -> dict[str, Any]:
    """Derive every SCX-PT-039 engineering gate without permissive defaults."""

    accumulator = (
        events
        if isinstance(events, EngineeringAccumulator)
        else EngineeringAccumulator.from_events(events)
    )
    retained = accumulator.events
    result = _last_payload(retained, "episode_result")
    reports = [
        (event.event_id, event.payload) for event in retained if event.event_type == "execution"
    ]
    dispatches = [
        event.payload
        for event in retained
        if event.event_type == "command_lifecycle" and event.payload.get("status") == "dispatched"
    ]
    lineages = {
        str(event.payload.get("command_id")): event.payload.get(
            "lineage",
            event.payload,
        )
        for event in retained
        if event.event_type == "command_lineage"
    }
    dispatched_ids = {
        str(command.get("command_id"))
        for payload in dispatches
        if isinstance((command := payload.get("command")), dict)
    }
    accepted_builds = [
        (event_id, payload)
        for event_id, payload in reports
        if _is_build(payload) and _pysc2_accepted(payload)
    ]
    build_started = sum(_build_started(payload) for _, payload in accepted_builds)
    build_confirmed = sum(
        payload.get("status") == "succeeded"
        and payload.get("execution_stage") == "effect_verification"
        for _, payload in accepted_builds
    )
    build_failures = sum(
        payload.get("failure_code") in _NEGATIVE_BUILD_EFFECT_CODES
        for _, payload in accepted_builds
    )
    placement_complete = sum(
        _placement_identity_complete(payload) for _, payload in accepted_builds
    )
    builder_complete = sum(_builder_provenance_complete(payload) for _, payload in accepted_builds)
    production_reports = [
        payload for _, payload in reports if _is_production(payload) and _pysc2_accepted(payload)
    ]
    production_confirmed = sum(_production_confirmed(payload) for payload in production_reports)
    ledger = _placement_ledger_audit(retained, accepted_builds=accepted_builds)
    retreat_repeats = _repeated_retreat_arrivals(retained)
    unchanged_attacks = _unchanged_attack_redispatches(retained)
    health_delta_collisions = _health_delta_engagement_collisions(reports)
    expansion_rearms = _expansion_immediate_rearms(retained)
    defense_evaluations = [
        event.payload for event in retained if event.event_type == "defense_inventory_evaluated"
    ]
    defense_audit = _defense_inventory_audit(defense_evaluations)
    performance = _last_payload(retained, "event_store_performance")
    recovery_events = [
        event.payload for event in retained if event.event_type == "runtime_recovery_completed"
    ]
    recovery_artifact_valid = (
        isinstance(recovery_evidence, dict)
        and recovery_evidence.get("passed") is True
        and isinstance(recovery_evidence.get("git_sha"), str)
        and (expected_git_sha is None or recovery_evidence.get("git_sha") == expected_git_sha)
    )
    max_game_loop = accumulator.max_game_loop
    if max_game_loop == 0 and isinstance(result, dict):
        max_game_loop = int(result.get("steps", 0))
    elapsed = accumulator.elapsed_seconds
    loops_per_second = max_game_loop / elapsed if elapsed > 0 else None
    durable_bytes = sum(
        path.stat().st_size
        for path in (run_dir / "events.sqlite3", run_dir / "events.jsonl")
        if path.is_file()
    )
    bytes_per_loop = durable_bytes / max_game_loop if max_game_loop > 0 else None
    disk_ratio = (
        natural_run_baseline_bytes_per_loop / bytes_per_loop
        if natural_run_baseline_bytes_per_loop is not None
        and bytes_per_loop is not None
        and bytes_per_loop > 0
        else None
    )
    attributed_consequences = sum(
        event.event_type == "strategic_consequence_attributed" for event in retained
    )
    review = _last_payload(retained, "postgame_review_completed")
    postgame_coverage = (
        None
        if review is None
        else float(int(review.get("strategic_consequence_count", -1)) == attributed_consequences)
    )
    lineage_complete = (
        bool(dispatched_ids)
        and set(lineages) == dispatched_ids
        and all(
            isinstance(lineages[command_id], dict)
            and bool(lineages[command_id].get("responsibility"))
            for command_id in dispatched_ids
        )
    )
    build_count = len(accepted_builds)
    production_count = len(production_reports)
    metrics: dict[str, bool | int | float | None] = {
        "runtime_crash_zero": (
            isinstance(result, dict)
            and result.get("outcome") in {"victory", "defeat", "draw"}
            and not result.get("failure_reason")
        ),
        "friendly_target_zero": all(
            payload.get("failure_code") != "friendly_target" for _, payload in reports
        ),
        "lineage_complete": lineage_complete,
        "minimum_production_exposure": production_count >= 1,
        "production_confirmation_complete": (
            None if production_count == 0 else production_confirmed / production_count
        ),
        "minimum_build_exposure": build_count >= 1,
        "build_start_coverage": _ratio_or_none(build_started, build_count),
        "build_confirmation_rate": _ratio_or_none(build_confirmed, build_count),
        "build_failure_rate": _ratio_or_none(build_failures, build_count),
        "placement_identity_complete": (
            None if build_count == 0 else placement_complete == build_count
        ),
        "builder_provenance_complete": (
            None if build_count == 0 else builder_complete == build_count
        ),
        "builder_lease_complete": (
            None
            if build_count == 0
            else int(ledger["builder_lease_complete_count"] or 0) == build_count
        ),
        "placement_ledger_evidence_present": int(ledger["transition_count"] or 0) > 0,
        "placement_ledger_coverage": ledger["coverage"],
        "placement_ledger_transitions_valid": int(ledger["invalid_transition_count"] or 0) == 0,
        "placement_ledger_canonical_footprint_complete": (
            None
            if build_count == 0
            else int(ledger["canonical_footprint_complete_count"] or 0) == build_count
        ),
        "placement_ledger_terminal_complete": int(ledger["orphan_reservation_count"] or 0) == 0,
        "cross_type_footprint_overlap_zero": int(ledger["overlap_count"] or 0) == 0,
        "nonspatial_quarantine_zero": int(ledger["nonspatial_quarantine_count"] or 0) == 0,
        "invalid_footprint_redispatch_zero": int(ledger["invalid_redispatch_count"] or 0) == 0,
        "repeated_retreat_arrival_zero": retreat_repeats == 0,
        "unchanged_attack_redispatch_zero": unchanged_attacks == 0,
        "health_delta_engagement_attribution_valid": health_delta_collisions == 0,
        "expansion_terminal_immediate_rearm_zero": expansion_rearms == 0,
        "defense_inventory_evidence_present": bool(defense_evaluations),
        "defense_inventory_within_cap": (
            None
            if not defense_evaluations
            else defense_audit["invalid_count"] == 0 and defense_audit["cap_violation_count"] == 0
        ),
        "subscriber_callback_under_durable_lock_zero": (
            None
            if performance is None
            else int(performance.get("subscriber_callbacks_under_durable_lock", -1)) == 0
        ),
        "recovery_evidence_present": bool(recovery_events) or recovery_artifact_valid,
        "checkpoint_tail_recovery_bounded": (
            all(
                int(item.get("tail_event_count", recovery_tail_limit + 1))
                <= int(item.get("tail_event_limit", recovery_tail_limit))
                for item in recovery_events
            )
            if recovery_events
            else True
            if recovery_artifact_valid
            else None
        ),
        "postgame_semantic_event_coverage": postgame_coverage,
        "effective_loops_per_second": loops_per_second,
        "natural_run_disk_reduction_ratio": disk_ratio,
        "analysis_evidence_retention_bounded": accumulator.retention_overflow_count == 0,
    }
    thresholds = _thresholds()
    gates = {
        name: {
            "value": metrics.get(name),
            "comparison": comparison,
            "threshold": threshold,
            "passed": _passes(metrics.get(name), comparison, threshold),
        }
        for name, (comparison, threshold) in thresholds.items()
    }
    missing = [
        name for name in REQUIRED_ENGINEERING_GATES if name not in metrics or metrics[name] is None
    ]
    return {
        "format_version": "1.1",
        "metrics": metrics,
        "diagnostics": {
            "accepted_build_count": build_count,
            "accepted_production_count": production_count,
            "production_confirmed_count": production_confirmed,
            "build_start_count": build_started,
            "build_confirmed_count": build_confirmed,
            "build_failure_count": build_failures,
            **ledger,
            "repeated_retreat_arrival_count": retreat_repeats,
            "unchanged_attack_redispatch_count": unchanged_attacks,
            "health_delta_engagement_collision_count": health_delta_collisions,
            "expansion_terminal_immediate_rearm_count": expansion_rearms,
            "defense_inventory_evaluation_count": len(defense_evaluations),
            "defense_inventory_invalid_count": defense_audit["invalid_count"],
            "defense_inventory_cap_violation_count": defense_audit["cap_violation_count"],
            "max_game_loop": max_game_loop,
            "durable_bytes": durable_bytes,
            "durable_bytes_per_game_loop": bytes_per_loop,
            "natural_run_baseline_bytes_per_game_loop": natural_run_baseline_bytes_per_loop,
            "retained_event_count": len(retained),
            "scanned_event_count": accumulator.event_count,
            "retention_limit": accumulator.retention_limit,
            "retention_overflow_count": accumulator.retention_overflow_count,
        },
        "gates": gates,
        "missing_required_metrics": missing,
        "accepted": not missing and all(item["passed"] is True for item in gates.values()),
    }


def _thresholds() -> dict[str, tuple[str, bool | int | float]]:
    rate_names = {
        "production_confirmation_complete",
        "build_start_coverage",
        "build_confirmation_rate",
        "build_failure_rate",
        "placement_ledger_coverage",
        "postgame_semantic_event_coverage",
        "effective_loops_per_second",
        "natural_run_disk_reduction_ratio",
    }
    thresholds: dict[str, tuple[str, bool | int | float]] = {
        name: ("==", True) for name in REQUIRED_ENGINEERING_GATES if name not in rate_names
    }
    thresholds.update(
        {
            "production_confirmation_complete": (">=", 1.0),
            "build_start_coverage": (">=", 1.0),
            "build_confirmation_rate": (">=", 0.90),
            "build_failure_rate": ("<=", 0.10),
            "placement_ledger_coverage": (">=", 1.0),
            "postgame_semantic_event_coverage": (">=", 1.0),
            "effective_loops_per_second": (">=", 2.0),
            "natural_run_disk_reduction_ratio": (">=", 4.0),
        }
    )
    return thresholds


def _last_payload(events: Sequence[StoredEvent], event_type: str) -> dict[str, Any] | None:
    return next(
        (event.payload for event in reversed(events) if event.event_type == event_type),
        None,
    )


def _is_build(payload: dict[str, Any]) -> bool:
    return str(payload.get("action_name", "")).startswith("Build_")


def _is_production(payload: dict[str, Any]) -> bool:
    return str(payload.get("action_name", "")).startswith(("Train_", "Warp_"))


def _pysc2_accepted(payload: dict[str, Any]) -> bool:
    trace = payload.get("primitive_trace")
    translator = (
        [item for item in trace if isinstance(item, dict) and item.get("origin") == "translator"]
        if isinstance(trace, list)
        else []
    )
    return (
        bool(translator and translator[-1].get("accepted") is True)
        or payload.get("execution_stage") == "effect_verification"
    )


def _evidence(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("effect_evidence")
    return value if isinstance(value, dict) else {}


def _production_confirmed(payload: dict[str, Any]) -> bool:
    evidence = _evidence(payload)
    return (
        payload.get("status") == "succeeded"
        and payload.get("execution_stage") == "effect_verification"
        and evidence.get("confirmation_kind") in {"producer_order", "producer_morph", "new_unit"}
    )


def _build_started(payload: dict[str, Any]) -> bool:
    evidence = _evidence(payload)
    return evidence.get("build_started") is True or bool(evidence.get("new_structure_tag"))


def _placement_identity_complete(payload: dict[str, Any]) -> bool:
    evidence = _evidence(payload)
    validated = evidence.get("validated_target_position")
    emitted = evidence.get("emitted_target_position")
    verified = evidence.get("target_position")
    return (
        isinstance(validated, (list, tuple))
        and isinstance(emitted, (list, tuple))
        and isinstance(verified, (list, tuple))
        and tuple(validated) == tuple(emitted) == tuple(verified)
    )


def _builder_provenance_complete(payload: dict[str, Any]) -> bool:
    return bool(_evidence(payload).get("builder_tag"))


def _placement_ledger_audit(
    events: Sequence[StoredEvent],
    *,
    accepted_builds: Sequence[tuple[int, dict[str, Any]]],
) -> dict[str, int | float | None]:
    active: dict[str, tuple[str, frozenset[tuple[int, int]]]] = {}
    states: dict[str, str] = {}
    identities: dict[str, tuple[str, frozenset[tuple[int, int]], str]] = {}
    command_by_reservation: dict[str, str] = {}
    transitions_by_reservation: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    last_loop: dict[str, int] = {}
    transition_ids: set[str] = set()
    permanent_cells: set[tuple[int, int]] = set()
    invalid_transitions = 0
    overlaps = 0
    invalid_redispatches = 0
    nonspatial_quarantines = 0
    transition_count = 0
    allowed = {
        ("unreserved", "reserved"),
        ("unreserved", "permanent_invalid"),
        ("unreserved", "temporary_suppressed"),
        ("unreserved", "released"),
        ("reserved", "occupied"),
        ("reserved", "permanent_invalid"),
        ("reserved", "temporary_suppressed"),
        ("reserved", "released"),
        ("occupied", "released"),
    }
    for event in events:
        if event.event_type != "placement_ledger_transition":
            continue
        transition_count += 1
        payload = event.payload
        reservation_id = payload.get("reservation_id")
        command_id = payload.get("command_id")
        transition_id = payload.get("transition_id")
        raw_cells = payload.get("footprint_cells")
        if (
            not isinstance(reservation_id, str)
            or not isinstance(command_id, str)
            or not isinstance(transition_id, str)
            or not transition_id.startswith("placement-transition:")
            or not isinstance(raw_cells, list)
        ):
            invalid_transitions += 1
            continue
        if transition_id in transition_ids:
            invalid_transitions += 1
        transition_ids.add(transition_id)
        try:
            cells = frozenset((int(cell[0]), int(cell[1])) for cell in raw_cells)
        except (IndexError, TypeError, ValueError):
            invalid_transitions += 1
            continue
        previous = str(payload.get("previous_state", ""))
        next_state = str(payload.get("next_state", ""))
        structure_type = str(payload.get("structure_type", ""))
        game_loop = payload.get("game_loop")
        if not isinstance(game_loop, int) or isinstance(game_loop, bool):
            invalid_transitions += 1
            continue
        identity = (structure_type, cells, command_id)
        if reservation_id in identities and identities[reservation_id] != identity:
            invalid_transitions += 1
        identities.setdefault(reservation_id, identity)
        if game_loop < last_loop.get(reservation_id, 0):
            invalid_transitions += 1
        last_loop[reservation_id] = game_loop
        command_by_reservation.setdefault(reservation_id, command_id)
        transitions_by_reservation[reservation_id].append(payload)
        observed_previous = states.get(reservation_id, "unreserved")
        if (
            not cells
            or not structure_type
            or previous != observed_previous
            or (previous, next_state) not in allowed
        ):
            invalid_transitions += 1
        states[reservation_id] = next_state
        if next_state == "reserved":
            if cells & permanent_cells:
                invalid_redispatches += 1
            for other_id, (_, other_cells) in active.items():
                if other_id != reservation_id and cells & other_cells:
                    overlaps += 1
            active[reservation_id] = (structure_type, cells)
        elif next_state == "occupied":
            if reservation_id not in active:
                invalid_transitions += 1
            active[reservation_id] = (structure_type, cells)
        elif next_state in {
            "released",
            "temporary_suppressed",
            "permanent_invalid",
        }:
            active.pop(reservation_id, None)
        if next_state == "permanent_invalid":
            permanent_cells.update(cells)
            if payload.get("actor_failure") is True or payload.get("failure_class") == "nonspatial":
                nonspatial_quarantines += 1
    complete_builds = 0
    complete_builder_leases = 0
    complete_canonical_footprints = 0
    for _, report in accepted_builds:
        command_id = report.get("command_id")
        evidence = _evidence(report)
        reservation_id = evidence.get("reservation_id")
        if not isinstance(command_id, str) or not isinstance(reservation_id, str):
            continue
        chain = transitions_by_reservation.get(reservation_id, [])
        if not chain or command_by_reservation.get(reservation_id) != command_id:
            continue
        starts_reserved = (
            chain[0].get("previous_state") == "unreserved"
            and chain[0].get("next_state") == "reserved"
        )
        terminal_state = chain[-1].get("next_state")
        confirmed = report.get("status") == "succeeded"
        occupied = any(item.get("next_state") == "occupied" for item in chain)
        terminal_complete = terminal_state in {
            "released",
            "temporary_suppressed",
            "permanent_invalid",
        }
        effect_builder = evidence.get("builder_tag")
        lease_complete = (
            isinstance(effect_builder, str)
            and bool(effect_builder)
            and chain[0].get("builder_tag") == effect_builder
            and chain[0].get("builder_lease_state") == "acquired"
            and all(item.get("builder_tag") == effect_builder for item in chain)
            and any(
                item.get("builder_tag") == effect_builder
                and item.get("builder_lease_state") == "released"
                for item in chain[1:]
            )
        )
        canonical_complete = _canonical_placement_complete(report, chain)
        complete_builder_leases += lease_complete
        complete_canonical_footprints += canonical_complete
        if (
            starts_reserved
            and terminal_complete
            and lease_complete
            and canonical_complete
            and (not confirmed or occupied)
        ):
            complete_builds += 1
    accepted_build_count = len(accepted_builds)
    return {
        "transition_count": transition_count,
        "invalid_transition_count": invalid_transitions,
        "overlap_count": overlaps,
        "invalid_redispatch_count": invalid_redispatches,
        "nonspatial_quarantine_count": nonspatial_quarantines,
        "accepted_build_ledger_count": complete_builds,
        "builder_lease_complete_count": complete_builder_leases,
        "canonical_footprint_complete_count": complete_canonical_footprints,
        "canonical_footprint_mismatch_count": (
            accepted_build_count - complete_canonical_footprints
        ),
        "orphan_reservation_count": len(active),
        "coverage": (None if accepted_build_count == 0 else complete_builds / accepted_build_count),
    }


def _canonical_placement_complete(
    report: dict[str, Any],
    chain: Sequence[dict[str, Any]],
) -> bool:
    action_name = report.get("action_name")
    spec = CANONICAL_PLACEMENT_SPECS.get(action_name) if isinstance(action_name, str) else None
    evidence = _evidence(report)
    emitted = _world_position(evidence.get("emitted_target_position"))
    effect_cells = _grid_cells(evidence.get("occupied_grid_cells"))
    if spec is None or emitted is None or effect_cells is None:
        return False
    expected_cells = canonical_footprint_cells(emitted, spec)
    expected_width = spec.footprint + (2 if spec.reserves_addon_space else 0)
    return (
        evidence.get("footprint_width") == expected_width
        and evidence.get("footprint_height") == spec.footprint
        and effect_cells == expected_cells
        and all(
            item.get("structure_type") == spec.structure_type
            and _grid_cells(item.get("footprint_cells")) == expected_cells
            for item in chain
        )
    )


def _world_position(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _grid_cells(value: Any) -> frozenset[tuple[int, int]] | None:
    if not isinstance(value, list):
        return None
    try:
        cells = frozenset((int(cell[0]), int(cell[1])) for cell in value)
    except (IndexError, TypeError, ValueError):
        return None
    return cells if cells else None


def _repeated_retreat_arrivals(events: Sequence[StoredEvent]) -> int:
    counts: Counter[str] = Counter()
    for event in events:
        if (
            event.event_type == "tactical_actor_state"
            and event.payload.get("state") == "retreat_arrived"
        ):
            identity = event.payload.get("commitment_id")
            if isinstance(identity, str):
                counts[identity] += 1
    return sum(max(0, count - 1) for count in counts.values())


def _unchanged_attack_redispatches(events: Sequence[StoredEvent]) -> int:
    active: dict[tuple[str, str, str], str] = {}
    identity_by_command: dict[str, tuple[str, str, str]] = {}
    violations = 0
    for event in events:
        payload = event.payload
        if event.event_type == "execution":
            command_id = payload.get("command_id")
            if not isinstance(command_id, str):
                continue
            identity = identity_by_command.pop(command_id, None)
            if identity is not None and active.get(identity) == command_id:
                del active[identity]
            continue
        if event.event_type != "command_lifecycle" or payload.get("status") != "dispatched":
            continue
        command = payload.get("command")
        if not isinstance(command, dict) or command.get("name") != "Attack_Unit":
            continue
        arguments = command.get("arguments")
        operation_id = command.get("operation_id")
        command_id = command.get("command_id")
        if (
            not isinstance(arguments, list)
            or not arguments
            or not isinstance(operation_id, str)
            or not isinstance(command_id, str)
        ):
            continue
        identity = (
            str(command.get("actor", "")),
            str(arguments[0]).casefold(),
            operation_id,
        )
        if identity in active:
            violations += 1
        active[identity] = command_id
        identity_by_command[command_id] = identity
    return violations


def _health_delta_engagement_collisions(
    reports: Sequence[tuple[int, dict[str, Any]]],
) -> int:
    engagements: defaultdict[tuple[Any, ...], set[str]] = defaultdict(set)
    for _, payload in reports:
        evidence = _evidence(payload)
        if evidence.get("effect_kind") != "combat" or not evidence.get("target_health_delta"):
            continue
        engagement = evidence.get("engagement_id")
        if not isinstance(engagement, str):
            continue
        key = (
            evidence.get("target_tag"),
            evidence.get("confirmed_game_loop"),
            evidence.get("baseline_target_health"),
            evidence.get("observed_target_health"),
        )
        engagements[key].add(engagement)
    return sum(max(0, len(values) - 1) for values in engagements.values())


def _expansion_immediate_rearms(events: Sequence[StoredEvent]) -> int:
    terminal_by_generation: dict[int, int] = {}
    violations = 0
    for event in events:
        if event.event_type == "expansion_commitment_terminal":
            generation = event.payload.get("generation_id")
            if isinstance(generation, int):
                terminal_by_generation[generation] = int(event.payload.get("terminal_game_loop", 0))
        elif event.event_type == "expansion_commitment_started":
            generation = event.payload.get("generation_id")
            started = event.payload.get("started_game_loop")
            if (
                isinstance(generation, int)
                and isinstance(started, int)
                and generation in terminal_by_generation
                and started <= terminal_by_generation[generation] + 1
            ):
                violations += 1
    return violations


def _defense_inventory_audit(
    evaluations: Sequence[dict[str, Any]],
) -> dict[str, int]:
    invalid = 0
    cap_violations = 0
    count_fields = (
        "completed",
        "constructing_or_training",
        "queued",
        "reserved",
        "dispatched_not_terminal",
        "effective_count",
        "hard_cap",
    )
    for item in evaluations:
        if (
            not isinstance(item.get("item_type"), str)
            or any(
                not isinstance(item.get(field), int)
                or isinstance(item.get(field), bool)
                or int(item[field]) < 0
                for field in count_fields
            )
            or int(item.get("hard_cap", 0)) < 1
        ):
            invalid += 1
            continue
        completed = int(item["completed"])
        constructing = int(item["constructing_or_training"])
        queued = int(item["queued"])
        reserved = int(item["reserved"])
        dispatched = int(item["dispatched_not_terminal"])
        unobserved_dispatches = max(0, dispatched - constructing - queued)
        expected = completed + constructing + queued + reserved + unobserved_dispatches
        hard_cap = int(item["hard_cap"])
        expected_decision = (
            "over_cap"
            if expected > hard_cap
            else "at_cap"
            if expected == hard_cap
            else "within_cap"
        )
        if (
            int(item["effective_count"]) != expected
            or item.get("decision") != expected_decision
            or (
                "dispatches_not_already_observed" in item
                and item["dispatches_not_already_observed"] != unobserved_dispatches
            )
        ):
            invalid += 1
            continue
        cap_violations += expected > hard_cap
    return {
        "invalid_count": invalid,
        "cap_violation_count": cap_violations,
    }


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _passes(
    value: bool | int | float | None,
    comparison: str,
    threshold: bool | int | float,
) -> bool:
    if value is None:
        return False
    if comparison == "==":
        return value == threshold
    if comparison == ">=":
        return value >= threshold
    return value <= threshold
