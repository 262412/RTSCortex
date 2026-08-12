"""Fail-closed engineering acceptance for one natural-terminal live run."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rtscortex.contracts import (
    authoritative_build_preflight_authorization_id,
    authoritative_build_preflight_request_id,
)
from rtscortex.cortex.terminal import TerminalCollapseReason
from rtscortex.memory import StoredEvent
from rtscortex.placement import (
    CANONICAL_PLACEMENT_SPECS,
    canonical_footprint_cells,
)

ENGINEERING_GATES_FILENAME = "engineering-gates.json"
SEMANTIC_BUILD_FAILURE_STREAK_LIMIT = 3
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
    "terminal_collapse_non_recovery_macro_dispatch_count",
    "semantic_build_failure_streak_bounded",
    "authoritative_build_pre_dispatch_circuit_bounded",
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
    "unchanged_failed_target_redispatch_zero",
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
        "macro_frontier_obsolete",
        "terminal_collapse_macro_hold_released",
        "terminal_collapse_non_recovery_macro_dispatch",
        "authoritative_build_pre_dispatch_circuit_defer",
        "authoritative_build_pre_dispatch_circuit_reset",
        "authoritative_build_pre_dispatch_preflight",
        "authoritative_build_pre_dispatch_preflight_requested",
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
    evidence: dict[str, Any] | None = None,
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
    semantic_build_audit = _semantic_build_operation_audit(retained)
    authoritative_pre_dispatch_audit = authoritative_build_pre_dispatch_audit(retained)
    terminal_collapse_macro_audit = _terminal_collapse_macro_dispatch_audit(retained)
    performance = _last_payload(retained, "event_store_performance")
    recovery_events = [
        event.payload for event in retained if event.event_type == "runtime_recovery_completed"
    ]
    recovery_artifact_valid = (
        isinstance(recovery_evidence, dict)
        and recovery_evidence.get("passed") is True
        and recovery_evidence.get("recovery_evidence_present") is True
        and recovery_evidence.get("checkpoint_tail_recovery_bounded") is True
        and isinstance(recovery_evidence.get("git_sha"), str)
        and isinstance(expected_git_sha, str)
        and recovery_evidence.get("git_sha") == expected_git_sha
        and recovery_evidence.get("expected_git_sha") == expected_git_sha
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
        "terminal_collapse_non_recovery_macro_dispatch_count": (
            None
            if terminal_collapse_macro_audit["unknown_count"] > 0
            else terminal_collapse_macro_audit["violation_count"]
        ),
        "semantic_build_failure_streak_bounded": (
            None
            if semantic_build_audit["operation_count"] == 0
            else semantic_build_audit["max_failure_streak"] <= SEMANTIC_BUILD_FAILURE_STREAK_LIMIT
        ),
        "authoritative_build_pre_dispatch_circuit_bounded": (
            authoritative_pre_dispatch_audit["max_failure_streak"]
            <= SEMANTIC_BUILD_FAILURE_STREAK_LIMIT
            and authoritative_pre_dispatch_audit["post_open_command_count"] == 0
            and authoritative_pre_dispatch_audit["post_open_dispatch_count"] == 0
            and authoritative_pre_dispatch_audit["post_open_rejection_count"] == 0
            and authoritative_pre_dispatch_audit["post_open_primitive_count"] == 0
            and authoritative_pre_dispatch_audit["post_open_approach_primitive_count"] == 0
            and authoritative_pre_dispatch_audit["missing_identity_count"] == 0
            and authoritative_pre_dispatch_audit["identity_inconsistency_count"] == 0
            and authoritative_pre_dispatch_audit["invalid_transition_count"] == 0
            and authoritative_pre_dispatch_audit["missing_open_count"] == 0
            and authoritative_pre_dispatch_audit["threshold_violation_count"] == 0
            and authoritative_pre_dispatch_audit["producer_inconsistency_count"] == 0
            and authoritative_pre_dispatch_audit["raw_circuit_open_violation_count"] == 0
            and accumulator.retention_overflow_count == 0
        ),
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
        "unchanged_failed_target_redispatch_zero": (
            int(ledger["unchanged_failed_target_redispatch_count"] or 0) == 0
        ),
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
        "format_version": "1.2",
        "evidence": evidence,
        "metrics": metrics,
        "diagnostics": {
            "accepted_build_count": build_count,
            "accepted_production_count": production_count,
            "production_confirmed_count": production_confirmed,
            "build_start_count": build_started,
            "build_confirmed_count": build_confirmed,
            "build_failure_count": build_failures,
            "terminal_collapse_non_recovery_macro_dispatch_count": (
                terminal_collapse_macro_audit["violation_count"]
            ),
            "terminal_collapse_macro_lineage_unknown_count": (
                terminal_collapse_macro_audit["unknown_count"]
            ),
            "terminal_collapse_macro_dispatched_violation_count": (
                terminal_collapse_macro_audit["dispatched_violation_count"]
            ),
            "terminal_collapse_macro_dispatch_guard_violation_count": (
                terminal_collapse_macro_audit["guard_violation_count"]
            ),
            "terminal_collapse_macro_raw_boundary_violation_count": (
                terminal_collapse_macro_audit["raw_boundary_violation_count"]
            ),
            "semantic_build_operation_count": semantic_build_audit["operation_count"],
            "semantic_build_failure_count": semantic_build_audit["failure_count"],
            "semantic_build_operation_max_failure_streak": semantic_build_audit[
                "max_failure_streak"
            ],
            "semantic_build_circuit_breaker_count": semantic_build_audit["circuit_breaker_count"],
            "semantic_build_cross_revision_retry_count": semantic_build_audit[
                "cross_revision_retry_count"
            ],
            "semantic_build_failure_classification": semantic_build_audit["failure_classification"],
            "semantic_build_operations": semantic_build_audit["operations"],
            "authoritative_build_pre_dispatch_operation_count": (
                authoritative_pre_dispatch_audit["operation_count"]
            ),
            "authoritative_build_pre_dispatch_failure_count": (
                authoritative_pre_dispatch_audit["failure_count"]
            ),
            "authoritative_build_pre_dispatch_max_failure_streak": (
                authoritative_pre_dispatch_audit["max_failure_streak"]
            ),
            "authoritative_build_pre_dispatch_circuit_open_count": (
                authoritative_pre_dispatch_audit["circuit_open_count"]
            ),
            "authoritative_build_pre_dispatch_circuit_reset_count": (
                authoritative_pre_dispatch_audit["circuit_reset_count"]
            ),
            "authoritative_build_pre_dispatch_post_open_command_count": (
                authoritative_pre_dispatch_audit["post_open_command_count"]
            ),
            "authoritative_build_pre_dispatch_post_open_dispatch_count": (
                authoritative_pre_dispatch_audit["post_open_dispatch_count"]
            ),
            "authoritative_build_pre_dispatch_post_open_rejection_count": (
                authoritative_pre_dispatch_audit["post_open_rejection_count"]
            ),
            "authoritative_build_pre_dispatch_post_open_primitive_count": (
                authoritative_pre_dispatch_audit["post_open_primitive_count"]
            ),
            "authoritative_build_pre_dispatch_post_open_approach_primitive_count": (
                authoritative_pre_dispatch_audit["post_open_approach_primitive_count"]
            ),
            "authoritative_build_pre_dispatch_raw_circuit_open_violation_count": (
                authoritative_pre_dispatch_audit["raw_circuit_open_violation_count"]
            ),
            "authoritative_build_pre_dispatch_missing_identity_count": (
                authoritative_pre_dispatch_audit["missing_identity_count"]
            ),
            "authoritative_build_pre_dispatch_identity_inconsistency_count": (
                authoritative_pre_dispatch_audit["identity_inconsistency_count"]
            ),
            "authoritative_build_pre_dispatch_invalid_transition_count": (
                authoritative_pre_dispatch_audit["invalid_transition_count"]
            ),
            "authoritative_build_pre_dispatch_missing_open_count": (
                authoritative_pre_dispatch_audit["missing_open_count"]
            ),
            "authoritative_build_pre_dispatch_threshold_violation_count": (
                authoritative_pre_dispatch_audit["threshold_violation_count"]
            ),
            "authoritative_build_pre_dispatch_producer_inconsistency_count": (
                authoritative_pre_dispatch_audit["producer_inconsistency_count"]
            ),
            "authoritative_build_pre_dispatch_duplicate_attempt_count": (
                authoritative_pre_dispatch_audit["duplicate_attempt_count"]
            ),
            "authoritative_build_pre_dispatch_duplicate_envelope_count": (
                authoritative_pre_dispatch_audit["duplicate_envelope_count"]
            ),
            "authoritative_build_pre_dispatch_cross_revision_retry_count": (
                authoritative_pre_dispatch_audit["cross_revision_retry_count"]
            ),
            "authoritative_build_pre_dispatch_operation_epoch_change_count": (
                authoritative_pre_dispatch_audit["operation_epoch_change_count"]
            ),
            "authoritative_build_pre_dispatch_failure_codes": (
                authoritative_pre_dispatch_audit["failure_codes"]
            ),
            "authoritative_build_pre_dispatch_reset_reasons": (
                authoritative_pre_dispatch_audit["reset_reasons"]
            ),
            "authoritative_build_pre_dispatch_operations": (
                authoritative_pre_dispatch_audit["operations"]
            ),
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
            "terminal_collapse_non_recovery_macro_dispatch_count": ("==", 0),
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


def _terminal_collapse_macro_dispatch_audit(
    events: Sequence[StoredEvent],
) -> dict[str, int]:
    dispatched_ids = {
        str(command["command_id"])
        for event in events
        if event.event_type == "command_lifecycle"
        and event.payload.get("status") == "dispatched"
        and isinstance((command := event.payload.get("command")), dict)
        and isinstance(command.get("command_id"), str)
    }
    dispatched_violation_ids: set[str] = set()
    unknown_count = 0
    for event in events:
        if event.event_type != "command_lineage":
            continue
        payload = event.payload
        lineage = payload.get("lineage", payload)
        if not isinstance(lineage, dict) or lineage.get("source_role") != "macro":
            continue
        command_id = payload.get("command_id", lineage.get("command_id"))
        if not isinstance(command_id, str) or command_id not in dispatched_ids:
            continue
        terminal_collapse = payload.get("terminal_collapse")
        townhall_recovery = payload.get("townhall_recovery")
        if not isinstance(terminal_collapse, bool) or not isinstance(
            townhall_recovery,
            bool,
        ):
            unknown_count += 1
            continue
        if terminal_collapse and not townhall_recovery:
            dispatched_violation_ids.add(command_id)
    guard_violation_ids = {
        str(command_id)
        for event in events
        if event.event_type == "terminal_collapse_non_recovery_macro_dispatch"
        and (command_id := event.payload.get("command_id")) is not None
    }
    raw_boundary_violation_ids = {
        str(command_id)
        for event in events
        if event.event_type == "execution"
        and event.payload.get("failure_code")
        == TerminalCollapseReason.NON_RECOVERY_MACRO_DISPATCH.value
        and (command_id := event.payload.get("command_id")) is not None
    }
    violation_ids = dispatched_violation_ids | guard_violation_ids | raw_boundary_violation_ids
    return {
        "violation_count": len(violation_ids),
        "unknown_count": unknown_count,
        "dispatched_violation_count": len(dispatched_violation_ids),
        "guard_violation_count": len(guard_violation_ids),
        "raw_boundary_violation_count": len(raw_boundary_violation_ids),
    }


_AUTHORITATIVE_BUILD_PRE_DISPATCH_CODES = frozenset(
    {
        "placement_query_rejected",
        "placement_query_rejected_cached",
        "placement_candidate_stale",
        "no_legal_placement",
    }
)


def _typed_digest_identity(value: str | None, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    digest = value[len(prefix) :]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _finite_world_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        for item in value
    ):
        return None
    return float(value[0]), float(value[1])


def _positive_unit_tag(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value, 0)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _authoritative_attempt_identity(
    operation_id: str,
    command_id: str,
    attempt_ordinal: int,
) -> str:
    encoded = json.dumps(
        {
            "operation_id": operation_id,
            "command_id": command_id,
            "attempt_ordinal": attempt_ordinal,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"attempt:{hashlib.sha256(encoded).hexdigest()}"


def _authoritative_pre_dispatch_evidence(payload: dict[str, Any]) -> dict[str, Any] | None:
    direct = payload.get("authoritative_pre_dispatch")
    if isinstance(direct, dict):
        return direct
    transition = payload.get("transition")
    if isinstance(transition, dict):
        nested = transition.get("authoritative_pre_dispatch")
        if isinstance(nested, dict):
            return nested
    return None


def authoritative_build_pre_dispatch_audit(
    events: Sequence[StoredEvent],
) -> dict[str, Any]:
    """Replay the raw Build boundary instead of trusting producer summaries.

    A journal contains the same decision in several envelopes (execution,
    placement transition and sometimes the Runtime's producer/defer event).  The
    replay below orders those envelopes by ``event_id`` and deduplicates by the
    command/attempt identity before changing an operation's state.  This keeps a
    repeated producer row or a raw ``circuit_open`` rejection from becoming a
    fourth failure.
    """

    ordered = sorted(events, key=lambda event: event.event_id)
    records_by_event: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    missing_identity: set[str] = set()
    identity_inconsistency: set[str] = set()
    failure_codes: Counter[str] = Counter()
    reset_reasons: Counter[str] = Counter()
    producer_inconsistency: set[str] = set()
    missing_open: set[str] = set()
    threshold_violations: set[str] = set()
    operations: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "authoritative_failure_count": 0,
            "max_failure_streak": 0,
            "circuit_open_count": 0,
            "circuit_reset_count": 0,
            "cross_revision_retry_count": 0,
            "duplicate_attempt_count": 0,
            "success_reset_count": 0,
            "invalid_transition_count": 0,
            "producer_defer_count": 0,
            "producer_reset_count": 0,
            "duplicate_envelope_count": 0,
            "action_name": None,
            "failure_codes": Counter(),
            "reset_reasons": Counter(),
            "operation_epoch_change_count": 0,
        }
    )

    def _parent_fields(event: StoredEvent) -> dict[str, Any]:
        payload = event.payload
        transition = payload.get("transition")
        parent: dict[str, Any] = dict(transition) if isinstance(transition, dict) else {}
        parent.update({key: value for key, value in payload.items() if key != "transition"})
        command = payload.get("command")
        if isinstance(command, dict):
            parent = {**command, **parent}
        return parent

    def _make_record(event: StoredEvent, evidence: dict[str, Any]) -> dict[str, Any]:
        parent = _parent_fields(event)
        operation_id = _string_value(evidence.get("operation_id"))
        command_id = _string_value(evidence.get("command_id"))
        action_name = _string_value(evidence.get("action_name"))
        attempt_ordinal = evidence.get("attempt_ordinal")
        attempt_id = _string_value(evidence.get("attempt_id"))
        parent_mismatch = False
        # Both execution and placement-ledger envelopes are authoritative
        # parents.  Every nested identity field must be present and exact;
        # accepting a child-only identity would allow replay under a different
        # operation or attempt.
        for key, value in (
            ("operation_id", operation_id),
            ("command_id", command_id),
            ("action_name", action_name),
            ("attempt_ordinal", attempt_ordinal),
        ):
            if key in parent and parent.get(key) is not None and parent.get(key) != value:
                parent_mismatch = True
        parent_operation = _string_value(parent.get("operation_id"))
        parent_command = _string_value(parent.get("command_id"))
        parent_action = _string_value(parent.get("action_name"))
        parent_attempt_id = _string_value(parent.get("attempt_id"))
        if event.event_type in {"execution", "placement_ledger_transition"}:
            parent_mismatch = parent_mismatch or any(
                (
                    parent_operation != operation_id,
                    parent_command != command_id,
                    parent_action != action_name,
                    parent.get("attempt_ordinal") != attempt_ordinal,
                )
            )
            if not _typed_digest_identity(parent_attempt_id, "attempt:"):
                parent_mismatch = True
            if attempt_id != parent_attempt_id:
                parent_mismatch = True
            attempt_id = parent_attempt_id
        else:
            if attempt_id is None:
                attempt_id = parent_attempt_id

        status = _string_value(evidence.get("status")) or ""
        failure_code = _string_value(evidence.get("failure_code"))
        if failure_code is None:
            failure_code = _string_value(event.payload.get("failure_code"))
        parent_failure_code = _string_value(parent.get("failure_code"))
        if parent_failure_code is not None and parent_failure_code != failure_code:
            parent_mismatch = True
        if event.event_type == "execution":
            parent_status = _string_value(parent.get("status"))
            if status == "reset" and parent_status != "succeeded":
                parent_mismatch = True
            if status != "reset" and parent_status == "succeeded":
                parent_mismatch = True
            if isinstance(parent.get("success"), bool):
                if status == "reset" and parent.get("success") is not True:
                    parent_mismatch = True
                if status != "reset" and parent.get("success") is not False:
                    parent_mismatch = True
        material_identity = _string_value(evidence.get("material_legality_identity"))
        builder_tag = evidence.get("builder_tag")
        ability_id = evidence.get("ability_id")
        world_target = _finite_world_point(evidence.get("world_target"))
        material_evidence_valid = evidence.get("material_evidence_valid")
        invalid_evidence_reasons = evidence.get("invalid_evidence_reasons")
        parent_material_identity = _string_value(parent.get("material_legality_identity"))
        if parent_material_identity is not None and parent_material_identity != material_identity:
            parent_mismatch = True
        transition = _string_value(evidence.get("state_transition")) or _string_value(
            evidence.get("transition")
        )
        reset_reason = _string_value(evidence.get("reset_reason"))
        if event.event_type == "placement_ledger_transition" and status == "reset":
            next_state = _string_value(parent.get("next_state"))
            if not (
                reset_reason == "build_started"
                and next_state == "build_started"
                or reset_reason == "effect_confirmed"
                and next_state == "occupied"
            ):
                parent_mismatch = True
            if any(
                (
                    _positive_unit_tag(parent.get("builder_tag")) != builder_tag,
                    parent.get("ability_id") != ability_id,
                    _finite_world_point(parent.get("world_target")) != world_target,
                    _string_value(parent.get("target_state_revision"))
                    != _string_value(evidence.get("target_state_revision")),
                    parent_material_identity != material_identity,
                )
            ):
                parent_mismatch = True
        failure = failure_code in _AUTHORITATIVE_BUILD_PRE_DISPATCH_CODES and status != "reset"
        success_reset = status == "reset"
        material_identity_required = failure or success_reset
        attempt_ordinal_valid = isinstance(attempt_ordinal, int) and not isinstance(
            attempt_ordinal, bool
        )
        expected_attempt_identity = (
            _authoritative_attempt_identity(operation_id, command_id, attempt_ordinal)
            if isinstance(operation_id, str)
            and isinstance(command_id, str)
            and isinstance(attempt_ordinal, int)
            and not isinstance(attempt_ordinal, bool)
            else None
        )
        identity_attempt_valid = (
            _typed_digest_identity(attempt_id, "attempt:")
            and _typed_digest_identity(operation_id, "operation:")
            and command_id is not None
            and attempt_ordinal_valid
            and attempt_id == expected_attempt_identity
        )
        identity_complete = (
            _typed_digest_identity(operation_id, "operation:")
            and command_id is not None
            and action_name is not None
            and action_name.startswith("Build_")
            and len(action_name) > len("Build_")
            and identity_attempt_valid
            and attempt_ordinal_valid
            and (
                not material_identity_required
                or _typed_digest_identity(material_identity, "build-legality:")
            )
            and (not success_reset or bool(_string_value(evidence.get("target_state_revision"))))
            and (
                not material_identity_required
                or (
                    isinstance(builder_tag, int)
                    and not isinstance(builder_tag, bool)
                    and builder_tag > 0
                    and isinstance(ability_id, int)
                    and not isinstance(ability_id, bool)
                    and ability_id > 0
                    and world_target is not None
                    and material_evidence_valid is True
                    and invalid_evidence_reasons == []
                )
            )
            and not parent_mismatch
        )
        record = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "operation_id": operation_id,
            "command_id": command_id,
            "action_name": action_name,
            "attempt_id": attempt_id,
            "attempt_ordinal": attempt_ordinal,
            "status": status,
            "failure_code": failure_code,
            "failure": failure,
            "threshold": evidence.get("threshold"),
            "streak": evidence.get("streak"),
            "circuit_open": evidence.get("circuit_open"),
            "duplicate_attempt": evidence.get("duplicate_attempt"),
            "material_duplicate": evidence.get("material_duplicate") is True,
            "material_legality_identity": material_identity,
            "builder_tag": builder_tag,
            "ability_id": ability_id,
            "world_target": world_target,
            "material_evidence_valid": material_evidence_valid,
            "invalid_evidence_reasons": invalid_evidence_reasons,
            "state_transition": transition,
            "reset_reason": reset_reason,
            "material_change_reason": _string_value(evidence.get("material_change_reason")),
            "placement_revision": _string_value(evidence.get("placement_revision")),
            "operation_epoch_changed": evidence.get("operation_epoch_changed") is True,
            "identity_complete": identity_complete,
            "parent_mismatch": parent_mismatch,
        }
        return record

    for event in ordered:
        evidence = _authoritative_pre_dispatch_evidence(event.payload)
        if evidence is None:
            continue
        record = _make_record(event, evidence)
        records_by_event[event.event_id].append(record)

    command_owner: dict[str, str] = {}
    attempt_owner: dict[str, str] = {}
    for records in records_by_event.values():
        for record in records:
            record["replay_valid"] = record["identity_complete"]
            command_id = record["command_id"]
            operation_id = record["operation_id"]
            if command_id is not None and _typed_digest_identity(operation_id, "operation:"):
                previous_owner = command_owner.setdefault(command_id, operation_id)
                if previous_owner != operation_id:
                    identity_inconsistency.add(command_id)
                    missing_identity.add(command_id)
            attempt_id = record["attempt_id"]
            if attempt_id is not None and _typed_digest_identity(operation_id, "operation:"):
                previous_attempt_owner = attempt_owner.setdefault(attempt_id, operation_id)
                if previous_attempt_owner != operation_id:
                    identity_inconsistency.add(attempt_id)
                    missing_identity.add(attempt_id)
            if record["status"] not in {"retry", "defer_replan", "duplicate", "reset"}:
                missing_identity.add(record["command_id"] or f"event:{record['event_id']}")
                identity_inconsistency.add(record["command_id"] or f"event:{record['event_id']}")
            if (
                record["status"] != "reset"
                and record["failure_code"] not in _AUTHORITATIVE_BUILD_PRE_DISPATCH_CODES
            ):
                missing_identity.add(record["command_id"] or f"event:{record['event_id']}")
                identity_inconsistency.add(record["command_id"] or f"event:{record['event_id']}")
            if record["failure"] or record["status"] == "reset":
                command_id = record["command_id"] or f"event:{record['event_id']}"
                if not record["replay_valid"]:
                    missing_identity.add(command_id)
                if record["parent_mismatch"]:
                    identity_inconsistency.add(command_id)

    records_by_event_id = records_by_event
    states: dict[str, dict[str, Any]] = {}
    post_open_commands: set[str] = set()
    post_open_dispatches: set[str] = set()
    post_open_rejections: set[str] = set()
    raw_circuit_open_violations: set[str] = set()
    post_open_unknown: set[str] = set()
    post_open_primitives: set[tuple[int, int]] = set()
    post_open_approach_primitives: set[tuple[int, int]] = set()
    invalid_transition_count = 0
    operation_epoch_change_count = 0
    previous_revision_by_operation: dict[str, str] = {}
    seen_producer_defers: set[tuple[str, str | None, str | None, Any, Any]] = set()
    seen_producer_resets: set[tuple[str, str | None, str | None]] = set()
    preflight_requests: dict[str, dict[str, Any]] = {}
    released_preflight_authorizations: dict[str, dict[str, Any]] = {}

    def _state(operation_id: str) -> dict[str, Any]:
        return states.setdefault(
            operation_id,
            {
                "streak": 0,
                "open": False,
                "action_name": None,
                "opener_command": None,
                "opener_attempt": None,
                "opener_ordinal": None,
                "opener_event_id": None,
                "opener_material": None,
                "opener_builder_tag": None,
                "opener_ability_id": None,
                "opener_world_target": None,
                "blocked_semantic_material_identity": None,
                "producer_reset_pending": False,
                "producer_reset_previous_material": None,
                "pending_preflight_authorization": None,
                "seen_commands": set(),
                "seen_attempts": set(),
                "seen_command_records": {},
                "seen_reset_keys": set(),
            },
        )

    def _invalid(operation_id: str | None) -> None:
        nonlocal invalid_transition_count
        invalid_transition_count += 1
        if operation_id is not None:
            operations[operation_id]["invalid_transition_count"] += 1

    def _clear_state(operation_id: str, state: dict[str, Any], reason: str) -> None:
        state["streak"] = 0
        state["open"] = False
        state["action_name"] = None
        state["opener_command"] = None
        state["opener_attempt"] = None
        state["opener_ordinal"] = None
        state["opener_event_id"] = None
        state["opener_material"] = None
        state["opener_builder_tag"] = None
        state["opener_ability_id"] = None
        state["opener_world_target"] = None
        state["blocked_semantic_material_identity"] = None
        state["producer_reset_pending"] = False
        state["producer_reset_previous_material"] = None
        state["pending_preflight_authorization"] = None
        state["seen_commands"].clear()
        state["seen_attempts"].clear()
        state["seen_command_records"].clear()
        previous_revision_by_operation.pop(operation_id, None)
        operations[operation_id]["circuit_reset_count"] += 1
        operations[operation_id]["reset_reasons"][reason] += 1
        reset_reasons[reason] += 1

    def _record_failure(record: dict[str, Any]) -> None:
        nonlocal operation_epoch_change_count
        operation_id = record["operation_id"]
        if not _typed_digest_identity(operation_id, "operation:"):
            return
        if not record.get("replay_valid", False):
            # Preserve an explicitly claimed open boundary for downstream
            # post-open checks, but do not count malformed evidence as a real
            # failure or let it advance the independently replayed streak.
            if record["streak"] == 3 and record["state_transition"] == "closed_to_open":
                state = _state(operation_id)
                state["open"] = True
                state["opener_command"] = record["command_id"]
                state["opener_attempt"] = record["attempt_id"] or (
                    "ordinal",
                    record["attempt_ordinal"],
                )
                state["opener_ordinal"] = record["attempt_ordinal"]
                state["opener_event_id"] = record["event_id"]
                state["opener_material"] = record["material_legality_identity"]
                state["opener_builder_tag"] = record["builder_tag"]
                state["opener_ability_id"] = record["ability_id"]
                state["opener_world_target"] = record["world_target"]
            return
        operation = operations[operation_id]
        state = _state(operation_id)
        if not isinstance(record["threshold"], int) or isinstance(record["threshold"], bool):
            _invalid(operation_id)
            return
        if not isinstance(record["streak"], int) or isinstance(record["streak"], bool):
            _invalid(operation_id)
            return
        if record["duplicate_attempt"] is not None and not isinstance(
            record["duplicate_attempt"], bool
        ):
            _invalid(operation_id)
            return
        if record["operation_epoch_changed"]:
            operation["operation_epoch_change_count"] += 1
            operation_epoch_change_count += 1
            if state["streak"] and record["state_transition"] != "open_to_reset":
                _invalid(operation_id)
        command_id = record["command_id"]
        if state["action_name"] is not None and record["action_name"] != state["action_name"]:
            _invalid(operation_id)
        elif state["action_name"] is None:
            state["action_name"] = record["action_name"]
            operation["action_name"] = record["action_name"]
        attempt_key = record["attempt_id"] or (
            "ordinal",
            record["attempt_ordinal"],
        )
        duplicate = command_id in state["seen_commands"] or attempt_key in state["seen_attempts"]
        previous_record = state["seen_command_records"].get(command_id)
        envelope_duplicate = bool(
            duplicate
            and previous_record is not None
            and previous_record.get("event_type") != record.get("event_type")
            and previous_record.get("failure_code") == record.get("failure_code")
            and previous_record.get("streak") == record.get("streak")
            and previous_record.get("state_transition") == record.get("state_transition")
        )
        if (
            record["duplicate_attempt"] is not None
            and bool(record["duplicate_attempt"]) != duplicate
            and not envelope_duplicate
        ):
            _invalid(operation_id)
            if bool(record["duplicate_attempt"]) and not duplicate:
                # A producer cannot turn a first attempt into a counted
                # failure merely by labelling it duplicate.
                return
        if duplicate:
            if envelope_duplicate:
                operation["duplicate_envelope_count"] += 1
            else:
                operation["duplicate_attempt_count"] += 1
            if not envelope_duplicate and (
                record["status"] != "duplicate" or record["state_transition"] is not None
            ):
                _invalid(operation_id)
            if record["streak"] != state["streak"] or record["circuit_open"] is not state["open"]:
                _invalid(operation_id)
            return
        if record["status"] == "duplicate":
            _invalid(operation_id)
            return
        if command_id is not None:
            state["seen_commands"].add(command_id)
            state["seen_command_records"][command_id] = record
        state["seen_attempts"].add(attempt_key)

        material_reset_consumed = False
        if state["producer_reset_pending"] and not state["open"]:
            producer_reset_followup = (
                record["state_transition"] == "open_to_reset"
                and record["reset_reason"] == "material_state_changed"
                and record["material_change_reason"]
                in {
                    "builder_changed",
                    "ability_changed",
                    "target_state_changed",
                    "material_state_changed",
                    "semantic_legality_material_change",
                }
                and _typed_digest_identity(record["material_legality_identity"], "build-legality:")
                and record["material_legality_identity"]
                != state["producer_reset_previous_material"]
            )
            if producer_reset_followup:
                state["producer_reset_pending"] = False
                state["producer_reset_previous_material"] = None
                material_reset_consumed = True
            else:
                _invalid(operation_id)
        if state["open"]:
            if command_id is not None and (
                state["opener_event_id"] is None or record["event_id"] > state["opener_event_id"]
            ):
                post_open_rejections.add(command_id)
            if record["state_transition"] is not None:
                _invalid(operation_id)
            if record["status"] != "defer_replan" or record["streak"] != 3:
                _invalid(operation_id)
            if record["circuit_open"] is not True:
                _invalid(operation_id)
            return

        expected = state["streak"] + 1
        if expected > 3:
            _invalid(operation_id)
            return
        if record["threshold"] != 3:
            threshold_violations.add(operation_id)
            _invalid(operation_id)
        expected_status = "defer_replan" if expected == 3 else "retry"
        expected_open = expected == 3
        if record["streak"] != expected or record["status"] != expected_status:
            _invalid(operation_id)
        if record["circuit_open"] is not expected_open:
            _invalid(operation_id)
        transition = record["state_transition"]
        if expected == 3:
            if transition != "closed_to_open":
                missing_open.add(operation_id)
                _invalid(operation_id)
            else:
                operation["circuit_open_count"] += 1
            state["open"] = True
            state["opener_command"] = command_id
            state["opener_attempt"] = attempt_key
            state["opener_ordinal"] = record["attempt_ordinal"]
            state["opener_event_id"] = record["event_id"]
            state["opener_material"] = record["material_legality_identity"]
            state["opener_builder_tag"] = record["builder_tag"]
            state["opener_ability_id"] = record["ability_id"]
            state["opener_world_target"] = record["world_target"]
        elif record["streak"] == 3 and transition == "closed_to_open":
            # Keep replay fail-closed while still treating an explicitly
            # claimed third-failure boundary as open for downstream checks.
            # This lets us report a post-open dispatch even when the journal
            # omitted attempts 1/2; it never increments the valid open count.
            state["open"] = True
            state["opener_command"] = command_id
            state["opener_attempt"] = attempt_key
            state["opener_ordinal"] = record["attempt_ordinal"]
            state["opener_event_id"] = record["event_id"]
            state["opener_material"] = record["material_legality_identity"]
            state["opener_builder_tag"] = record["builder_tag"]
            state["opener_ability_id"] = record["ability_id"]
            state["opener_world_target"] = record["world_target"]
        elif transition is not None and not material_reset_consumed:
            _invalid(operation_id)
        state["streak"] = expected
        operation["authoritative_failure_count"] += 1
        operation["failure_codes"][record["failure_code"]] += 1
        failure_codes[record["failure_code"]] += 1
        operation["max_failure_streak"] = max(operation["max_failure_streak"], expected)
        revision = record["placement_revision"]
        previous_revision = previous_revision_by_operation.get(operation_id)
        if revision is not None and previous_revision is not None and revision != previous_revision:
            operation["cross_revision_retry_count"] += 1
        if revision is not None:
            previous_revision_by_operation[operation_id] = revision

    def _record_reset(record: dict[str, Any]) -> None:
        nonlocal operation_epoch_change_count
        operation_id = record["operation_id"]
        if not _typed_digest_identity(operation_id, "operation:"):
            return
        if not record.get("replay_valid", False):
            return
        operation = operations[operation_id]
        state = _state(operation_id)
        if state["action_name"] is not None and record["action_name"] != state["action_name"]:
            _invalid(operation_id)
        reset_key = (
            record["command_id"],
            record["reset_reason"],
            record["state_transition"],
        )
        if reset_key in state["seen_reset_keys"]:
            operation["duplicate_envelope_count"] += 1
            return
        state["seen_reset_keys"].add(reset_key)
        if record["operation_epoch_changed"]:
            operation["operation_epoch_change_count"] += 1
            operation_epoch_change_count += 1
        if record["threshold"] != 3:
            threshold_violations.add(operation_id)
        if record["threshold"] != 3 or record["status"] != "reset":
            _invalid(operation_id)
        if record["failure_code"] != "build_started":
            _invalid(operation_id)
        if record["circuit_open"] is not False or record["streak"] != 0:
            _invalid(operation_id)
        reason = record["reset_reason"]
        if not reason:
            _invalid(operation_id)
            return
        transition = record["state_transition"]
        authorization = released_preflight_authorizations.get(operation_id)
        if authorization is not None and not state["open"] and not state["streak"]:
            authorized_success = bool(
                transition is None
                and reason in {"build_started", "effect_confirmed"}
                and record["action_name"] == authorization.get("action_name")
                and record["builder_tag"] == authorization.get("builder_tag")
                and record["ability_id"] == authorization.get("ability_id")
                and record["world_target"] == _finite_world_point(authorization.get("world_target"))
                and record["material_legality_identity"]
                == authorization.get("material_legality_identity")
            )
            if not authorized_success:
                _invalid(operation_id)
            else:
                operation["success_reset_count"] += 1
                released_preflight_authorizations.pop(operation_id, None)
            return
        if state["producer_reset_pending"]:
            if transition != "open_to_reset" or reason not in {
                "build_started",
                "effect_confirmed",
            }:
                _invalid(operation_id)
            else:
                state["producer_reset_pending"] = False
                state["producer_reset_previous_material"] = None
                operation["success_reset_count"] += 1
            return
        if state["open"]:
            if transition != "open_to_reset":
                _invalid(operation_id)
        elif state["streak"]:
            if transition is not None:
                _invalid(operation_id)
        else:
            _invalid(operation_id)
        had_state = state["open"] or state["streak"]
        if had_state:
            _clear_state(operation_id, state, reason)
            operation["success_reset_count"] += 1

    def _canonical_preflight_request(request: dict[str, Any]) -> bool:
        arguments = request.get("requested_arguments")
        if not isinstance(arguments, list):
            return False
        try:
            expected = authoritative_build_preflight_request_id(
                operation_id=str(request["operation_id"]),
                operation_epoch=int(request["operation_epoch"]),
                action_name=str(request["action_name"]),
                actor=str(request["actor"]),
                requested_arguments=arguments,
                opened_command_id=str(request["opened_command_id"]),
                opened_attempt_id=str(request["opened_attempt_id"]),
                opened_attempt_ordinal=int(request["opened_attempt_ordinal"]),
                blocked_material_legality_identity=str(
                    request["blocked_material_legality_identity"]
                ),
                observation_revision=str(request["observation_revision"]),
                observation_game_loop=int(request["observation_game_loop"]),
            )
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            request.get("request_id") == expected
            and _typed_digest_identity(
                _string_value(request.get("operation_id")),
                "operation:",
            )
            and _typed_digest_identity(
                _string_value(request.get("opened_attempt_id")),
                "attempt:",
            )
            and _typed_digest_identity(
                _string_value(request.get("blocked_material_legality_identity")),
                "build-legality:",
            )
            and isinstance(request.get("operation_epoch"), int)
            and not isinstance(request.get("operation_epoch"), bool)
            and request["operation_epoch"] >= 0
            and isinstance(request.get("observation_game_loop"), int)
            and request["observation_game_loop"] >= 0
            and str(request.get("action_name", "")).startswith("Build_")
            and bool(request.get("actor"))
        )

    def _canonical_preflight_authorization(result: dict[str, Any]) -> bool:
        world_target = _finite_world_point(result.get("world_target"))
        if world_target is None:
            return False
        try:
            expected = authoritative_build_preflight_authorization_id(
                request_id=str(result["request_id"]),
                operation_id=str(result["operation_id"]),
                operation_epoch=int(result["operation_epoch"]),
                action_name=str(result["action_name"]),
                builder_tag=int(result["builder_tag"]),
                ability_id=int(result["ability_id"]),
                world_target=world_target,
                target_state_revision=str(result["target_state_revision"]),
                material_legality_identity=str(result["material_legality_identity"]),
                observation_revision=str(result["observation_revision"]),
                observation_game_loop=int(result["observation_game_loop"]),
                expires_game_loop=int(result["expires_game_loop"]),
            )
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            result.get("authorization_id") == expected
            and result.get("authorized") is True
            and result.get("status") == "authorized"
            and result.get("circuit_open") is False
            and result.get("state_transition") == "open_to_reset"
            and _positive_unit_tag(result.get("builder_tag")) is not None
            and isinstance(result.get("ability_id"), int)
            and not isinstance(result.get("ability_id"), bool)
            and result["ability_id"] > 0
            and _typed_digest_identity(
                _string_value(result.get("material_legality_identity")),
                "build-legality:",
            )
            and isinstance(result.get("observation_game_loop"), int)
            and isinstance(result.get("expires_game_loop"), int)
            and result["expires_game_loop"] >= result["observation_game_loop"]
            and result.get("invalid_evidence_reasons") == []
        )

    for event in ordered:
        payload = event.payload
        for record in records_by_event_id.get(event.event_id, ()):
            operation_id = record["operation_id"]
            if operation_id is None or not _typed_digest_identity(operation_id, "operation:"):
                continue
            if record["failure"]:
                _record_failure(record)
            elif record["status"] == "reset":
                _record_reset(record)
            elif record["state_transition"] is not None:
                _invalid(operation_id)

        if event.event_type == "authoritative_build_pre_dispatch_preflight_requested":
            request = payload.get("request")
            operation_id = (
                _string_value(request.get("operation_id")) if isinstance(request, dict) else None
            )
            request_id = (
                _string_value(request.get("request_id")) if isinstance(request, dict) else None
            )
            state = states.get(operation_id or "")
            valid_request = bool(
                isinstance(request, dict)
                and request_id is not None
                and request_id not in preflight_requests
                and _canonical_preflight_request(request)
                and state is not None
                and state["open"]
                and request.get("action_name") == state["action_name"]
                and request.get("opened_command_id") == state["opener_command"]
                and request.get("opened_attempt_id") == state["opener_attempt"]
                and request.get("opened_attempt_ordinal") == state["opener_ordinal"]
                and request.get("blocked_material_legality_identity") == state["opener_material"]
                and payload.get("side_effect_free") is True
            )
            if not valid_request:
                producer_inconsistency.add(operation_id or f"event:{event.event_id}")
                _invalid(operation_id)
            else:
                assert isinstance(request, dict)
                assert request_id is not None
                preflight_requests[request_id] = request

        if event.event_type == "authoritative_build_pre_dispatch_preflight":
            result = payload.get("result")
            if not isinstance(result, dict):
                producer_inconsistency.add(f"event:{event.event_id}")
                _invalid(None)
                continue
            operation_id = _string_value(result.get("operation_id"))
            request_id = _string_value(result.get("request_id"))
            request = preflight_requests.get(request_id or "")
            state = states.get(operation_id or "")
            shared_contract = bool(
                request is not None
                and state is not None
                and state["open"]
                and payload.get("accepted_by_core") is True
                and payload.get("invalid_evidence_reasons") == []
                and result.get("operation_id") == request.get("operation_id")
                and result.get("operation_epoch") == request.get("operation_epoch")
                and result.get("action_name") == request.get("action_name")
                and result.get("actor") == request.get("actor")
                and result.get("requested_arguments") == request.get("requested_arguments")
                and result.get("opened_command_id") == state["opener_command"]
                and result.get("opened_attempt_id") == state["opener_attempt"]
                and result.get("opened_attempt_ordinal") == state["opener_ordinal"]
                and result.get("blocked_material_legality_identity") == state["opener_material"]
            )
            if not shared_contract:
                producer_inconsistency.add(operation_id or f"event:{event.event_id}")
                _invalid(operation_id)
                continue
            assert request is not None
            assert state is not None
            if result.get("authorized") is True:
                world_target = _finite_world_point(result.get("world_target"))
                material_reason = _string_value(result.get("material_change_reason"))
                material_change_valid = bool(
                    material_reason == "builder_changed"
                    and _positive_unit_tag(result.get("builder_tag")) != state["opener_builder_tag"]
                    or material_reason == "ability_changed"
                    and result.get("ability_id") != state["opener_ability_id"]
                    or material_reason == "target_state_changed"
                    and world_target == state["opener_world_target"]
                    and result.get("material_legality_identity") != state["opener_material"]
                    or material_reason == "operation_epoch_changed"
                    and result.get("operation_epoch", 0) > 0
                )
                if not _canonical_preflight_authorization(result) or not material_change_valid:
                    producer_inconsistency.add(operation_id or f"event:{event.event_id}")
                    _invalid(operation_id)
                else:
                    state["pending_preflight_authorization"] = result
            elif not (
                result.get("status") == "deferred"
                and result.get("authorization_id") is None
                and result.get("state_transition") is None
                and result.get("circuit_open") is True
                and isinstance(result.get("invalid_evidence_reasons"), list)
                and bool(result.get("invalid_evidence_reasons"))
            ):
                producer_inconsistency.add(operation_id or f"event:{event.event_id}")
                _invalid(operation_id)

        if event.event_type == "authoritative_build_pre_dispatch_circuit_defer":
            operation_id = _string_value(payload.get("operation_id"))
            defer_key = (
                operation_id or "",
                _string_value(payload.get("material_legality_identity")),
                _string_value(payload.get("opened_command_id")),
                payload.get("streak"),
                payload.get("threshold"),
            )
            if defer_key in seen_producer_defers:
                continue
            seen_producer_defers.add(defer_key)
            if not _typed_digest_identity(operation_id, "operation:"):
                producer_inconsistency.add(f"event:{event.event_id}")
            else:
                assert operation_id is not None
                operation = operations[operation_id]
                state = _state(operation_id)
                operation["producer_defer_count"] += 1
                opened_attempt_id = _string_value(payload.get("opened_attempt_id"))
                blocked_semantic_identity = _string_value(
                    payload.get("blocked_semantic_material_identity")
                )
                producer_material_valid = payload.get("material_evidence_valid")
                producer_invalid_reasons = payload.get("invalid_evidence_reasons")
                producer_material_contract = (
                    producer_material_valid is True
                    and _typed_digest_identity(
                        blocked_semantic_identity,
                        "semantic-build-material:",
                    )
                    and producer_invalid_reasons == []
                    or producer_material_valid is False
                    and blocked_semantic_identity is None
                    and isinstance(producer_invalid_reasons, list)
                    and bool(producer_invalid_reasons)
                    and all(
                        isinstance(reason, str) and reason for reason in producer_invalid_reasons
                    )
                )
                if (
                    not state["open"]
                    or not (
                        (_string_value(payload.get("action_name", "")) or "").startswith("Build_")
                        and len(_string_value(payload.get("action_name", "")) or "") > len("Build_")
                    )
                    or state["action_name"] != _string_value(payload.get("action_name"))
                    or payload.get("threshold") != 3
                    or payload.get("streak") != 3
                    or payload.get("failure_count") != 3
                    or _string_value(payload.get("opened_command_id")) != state["opener_command"]
                    or payload.get("opened_attempt_ordinal") != state["opener_ordinal"]
                    or not isinstance(payload.get("next_action"), str)
                    or not payload.get("next_action")
                    or not _typed_digest_identity(opened_attempt_id, "attempt:")
                    or opened_attempt_id != state["opener_attempt"]
                    or _string_value(payload.get("material_legality_identity"))
                    != state["opener_material"]
                    or not producer_material_contract
                ):
                    producer_inconsistency.add(operation_id)
                    missing_open.add(operation_id)
                elif producer_material_valid is True:
                    state["blocked_semantic_material_identity"] = blocked_semantic_identity

        elif event.event_type == "authoritative_build_pre_dispatch_circuit_reset":
            operation_id = _string_value(payload.get("operation_id"))
            reason = _string_value(payload.get("reason"))
            reset_key = (operation_id or "", reason, _string_value(payload.get("action_name")))
            if reset_key in seen_producer_resets:
                continue
            seen_producer_resets.add(reset_key)
            if not _typed_digest_identity(operation_id, "operation:") or not reason:
                producer_inconsistency.add(f"event:{event.event_id}")
            else:
                assert operation_id is not None
                operation = operations[operation_id]
                state = _state(operation_id)
                operation["producer_reset_count"] += 1
                authorization = state.get("pending_preflight_authorization")
                reset_contract_valid = bool(
                    state["open"]
                    and reason == "raw_preflight_authorized"
                    and isinstance(authorization, dict)
                    and payload.get("state_transition") == "open_to_reset"
                    and payload.get("authorization_id") == authorization.get("authorization_id")
                    and payload.get("request_id") == authorization.get("request_id")
                    and payload.get("operation_epoch") == authorization.get("operation_epoch")
                    and payload.get("action_name")
                    == authorization.get("action_name")
                    == state["action_name"]
                    and payload.get("opened_command_id") == state["opener_command"]
                    and payload.get("opened_attempt_id") == state["opener_attempt"]
                    and payload.get("opened_attempt_ordinal") == state["opener_ordinal"]
                    and payload.get("blocked_material_legality_identity")
                    == state["opener_material"]
                    and payload.get("reset_from_streak") == state["streak"] == 3
                    and payload.get("builder_tag") == authorization.get("builder_tag")
                    and payload.get("ability_id") == authorization.get("ability_id")
                    and _finite_world_point(payload.get("world_target"))
                    == _finite_world_point(authorization.get("world_target"))
                    and payload.get("target_state_revision")
                    == authorization.get("target_state_revision")
                    and payload.get("material_legality_identity")
                    == authorization.get("material_legality_identity")
                    and payload.get("observation_revision")
                    == authorization.get("observation_revision")
                    and payload.get("authorization_game_loop")
                    == authorization.get("observation_game_loop")
                    and payload.get("expires_game_loop") == authorization.get("expires_game_loop")
                    and payload.get("material_change_reason")
                    == authorization.get("material_change_reason")
                )
                if not reset_contract_valid:
                    producer_inconsistency.add(operation_id)
                    _invalid(operation_id)
                else:
                    assert isinstance(authorization, dict)
                    _clear_state(operation_id, state, reason)
                    released_preflight_authorizations[operation_id] = authorization

        # A producer/Runtime command is forbidden once the replay has opened.
        if event.event_type == "command_lineage":
            lineage = payload.get("lineage", payload)
            if isinstance(lineage, dict):
                operation_id = _string_value(lineage.get("operation_id"))
                command_id = _string_value(lineage.get("command_id")) or _string_value(
                    payload.get("command_id")
                )
                parent_operation_id = _string_value(payload.get("operation_id"))
                parent_command_id = _string_value(payload.get("command_id"))
                if (parent_operation_id is not None and parent_operation_id != operation_id) or (
                    parent_command_id is not None and parent_command_id != command_id
                ):
                    identity_inconsistency.add(command_id or f"event:{event.event_id}")
                    missing_identity.add(command_id or f"event:{event.event_id}")
                lineage_state = states.get(operation_id or "")
                if (
                    lineage_state is not None
                    and lineage_state["open"]
                    and (
                        lineage_state["opener_event_id"] is None
                        or event.event_id > lineage_state["opener_event_id"]
                    )
                ):
                    if command_id is None:
                        post_open_unknown.add(f"event:{event.event_id}")
                    else:
                        post_open_commands.add(command_id)
                authorization = released_preflight_authorizations.get(operation_id or "")
                command_authorization = payload.get("authoritative_build_preflight")
                if authorization is not None:
                    if not isinstance(command_authorization, dict) or (
                        command_authorization.get("authorization_id")
                        != authorization.get("authorization_id")
                    ):
                        producer_inconsistency.add(operation_id or f"event:{event.event_id}")
                        _invalid(operation_id)
                elif (
                    any(item["open"] for item in states.values())
                    and not _typed_digest_identity(operation_id, "operation:")
                    and (
                        str(lineage.get("action_name", "")).startswith("Build_")
                        or str(lineage.get("semantic_action", "")).upper().startswith("BUILD ")
                    )
                ):
                    post_open_unknown.add(f"event:{event.event_id}")
        elif event.event_type == "command_lifecycle" and payload.get("status") == "dispatched":
            command = payload.get("command")
            if isinstance(command, dict):
                operation_id = _string_value(command.get("operation_id"))
                command_id = _string_value(command.get("command_id"))
                parent_operation_id = _string_value(payload.get("operation_id"))
                parent_command_id = _string_value(payload.get("command_id"))
                if (parent_operation_id is not None and parent_operation_id != operation_id) or (
                    parent_command_id is not None and parent_command_id != command_id
                ):
                    identity_inconsistency.add(command_id or f"event:{event.event_id}")
                    missing_identity.add(command_id or f"event:{event.event_id}")
                dispatch_state = states.get(operation_id or "")
                if (
                    dispatch_state is not None
                    and dispatch_state["open"]
                    and (
                        dispatch_state["opener_event_id"] is None
                        or event.event_id > dispatch_state["opener_event_id"]
                    )
                ):
                    if command_id is None:
                        post_open_unknown.add(f"event:{event.event_id}")
                    else:
                        post_open_dispatches.add(command_id)
                authorization = released_preflight_authorizations.get(operation_id or "")
                command_authorization = command.get("authoritative_build_preflight")
                if authorization is not None:
                    exact_command_contract = bool(
                        isinstance(command_authorization, dict)
                        and command_authorization.get("authorization_id")
                        == authorization.get("authorization_id")
                        and command.get("name") == authorization.get("action_name")
                        and command.get("actor") == authorization.get("actor")
                        and command.get("arguments") == authorization.get("requested_arguments")
                    )
                    if not exact_command_contract:
                        producer_inconsistency.add(operation_id or f"event:{event.event_id}")
                        _invalid(operation_id)
                elif (
                    any(item["open"] for item in states.values())
                    and not _typed_digest_identity(operation_id, "operation:")
                    and str(command.get("name", "")).startswith("Build_")
                ):
                    post_open_unknown.add(f"event:{event.event_id}")

        raw_failure = payload.get("failure_code")
        if event.event_type == "execution" and raw_failure in {
            "authoritative_pre_dispatch_circuit_open",
            "operation_no_start_circuit_open",
        }:
            operation_id = _string_value(payload.get("operation_id"))
            command_id = _string_value(payload.get("command_id"))
            raw_state = states.get(operation_id or "")
            raw_circuit_open_violations.add(command_id or f"event:{event.event_id}")
            if not _typed_digest_identity(operation_id, "operation:"):
                missing_identity.add(command_id or f"event:{event.event_id}")
            elif (
                raw_state is not None
                and raw_state["open"]
                and (
                    raw_state["opener_event_id"] is None
                    or event.event_id > raw_state["opener_event_id"]
                )
            ):
                post_open_rejections.add(command_id or f"event:{event.event_id}")
            elif raw_state is None or not raw_state["open"]:
                missing_open.add(operation_id or f"event:{event.event_id}")
                _invalid(operation_id)

        if (
            event.event_type == "execution"
            and payload.get("execution_stage") == "pre_dispatch"
            and raw_failure in _AUTHORITATIVE_BUILD_PRE_DISPATCH_CODES
            and not records_by_event_id.get(event.event_id)
        ):
            operation_id = _string_value(payload.get("operation_id"))
            command_id = _string_value(payload.get("command_id"))
            evidence_state = states.get(operation_id or "")
            if (
                evidence_state is not None
                and evidence_state["open"]
                and (
                    evidence_state["opener_event_id"] is None
                    or event.event_id > evidence_state["opener_event_id"]
                )
            ):
                post_open_rejections.add(command_id or f"event:{event.event_id}")
            else:
                missing_identity.add(command_id or f"event:{event.event_id}")

        if event.event_type == "execution" and _semantic_build_action_name(payload) is not None:
            operation_id = _string_value(payload.get("operation_id"))
            command_id = _string_value(payload.get("command_id"))
            execution_state = states.get(operation_id or "")
            trace = payload.get("primitive_trace")
            if (
                execution_state is not None
                and execution_state["open"]
                and (
                    execution_state["opener_event_id"] is None
                    or event.event_id > execution_state["opener_event_id"]
                )
                and isinstance(trace, list)
            ):
                for index, primitive in enumerate(trace):
                    if not isinstance(primitive, dict) or primitive.get("accepted") is not True:
                        continue
                    key = (event.event_id, index)
                    post_open_primitives.add(key)
                    if primitive.get("function") == "Move_Move_pt":
                        post_open_approach_primitives.add(key)
                if trace and command_id is None:
                    post_open_unknown.add(f"event:{event.event_id}")
            if _typed_digest_identity(operation_id, "operation:"):
                success_state = states.get(operation_id or "")
                if (
                    success_state is not None
                    and success_state["streak"]
                    and payload.get("status") == "succeeded"
                ):
                    # The success must carry the typed reset envelope (normally
                    # nested in the placement transition emitted by Raw).
                    if not records_by_event_id.get(event.event_id) or not any(
                        record["status"] == "reset"
                        for record in records_by_event_id[event.event_id]
                    ):
                        _invalid(operation_id)

    rendered_operations = {
        operation_id: {
            **{
                key: value
                for key, value in operation.items()
                if key not in {"failure_codes", "reset_reasons"}
            },
            "failure_codes": dict(sorted(operation["failure_codes"].items())),
            "reset_reasons": dict(sorted(operation["reset_reasons"].items())),
        }
        for operation_id, operation in sorted(operations.items())
    }
    return {
        "operation_count": len(operations),
        "failure_count": sum(
            operation["authoritative_failure_count"] for operation in operations.values()
        ),
        "max_failure_streak": max(
            (operation["max_failure_streak"] for operation in operations.values()),
            default=0,
        ),
        "circuit_open_count": sum(
            operation["circuit_open_count"] for operation in operations.values()
        ),
        "circuit_reset_count": sum(
            operation["circuit_reset_count"] for operation in operations.values()
        ),
        "post_open_command_count": len(post_open_commands) + len(post_open_unknown),
        "post_open_dispatch_count": len(post_open_dispatches),
        "post_open_rejection_count": len(post_open_rejections),
        "post_open_primitive_count": len(post_open_primitives),
        "post_open_approach_primitive_count": len(post_open_approach_primitives),
        "raw_circuit_open_violation_count": len(raw_circuit_open_violations),
        "missing_identity_count": len(missing_identity),
        "identity_inconsistency_count": len(identity_inconsistency),
        "invalid_transition_count": invalid_transition_count,
        "missing_open_count": len(missing_open),
        "threshold_violation_count": len(threshold_violations),
        "producer_inconsistency_count": len(producer_inconsistency),
        "duplicate_attempt_count": sum(
            operation["duplicate_attempt_count"] for operation in operations.values()
        ),
        "duplicate_envelope_count": sum(
            operation["duplicate_envelope_count"] for operation in operations.values()
        ),
        "cross_revision_retry_count": sum(
            operation["cross_revision_retry_count"] for operation in operations.values()
        ),
        "operation_epoch_change_count": operation_epoch_change_count,
        "failure_codes": dict(sorted(failure_codes.items())),
        "reset_reasons": dict(sorted(reset_reasons.items())),
        "operations": rendered_operations,
    }


_SEMANTIC_BUILD_FAILURE_STATUSES = frozenset({"failed", "unconfirmed", "cancelled"})
_SEMANTIC_BUILD_CLASSIFICATION_KEYS = (
    "failure_classification",
    "semantic_failure_classification",
    "classification",
)
_SEMANTIC_BUILD_OPERATION_KEYS = (
    "operation_id",
    "semantic_operation_id",
    "operation_key",
    "operation",
)
_SEMANTIC_BUILD_REVISION_KEYS = (
    "target_state_revision",
    "placement_revision",
    "observation_revision",
    "revision",
)


def _semantic_build_operation_audit(events: Sequence[StoredEvent]) -> dict[str, Any]:
    """Audit terminal semantic Build attempts across Runtime/Raw event shapes.

    Runtime command lifecycle events carry the operation and attempt identity in a
    nested command, while placement transitions may carry the revision and circuit
    state in a nested ``transition`` object.  Execution reports are authoritative
    when present; terminal lifecycle rows are used only for legacy journals that do
    not contain an execution report.
    """

    ordered = sorted(events, key=lambda event: event.event_id)
    transitions_by_command: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    command_metadata: dict[str, dict[str, Any]] = {}
    terminal_execution_ids: set[str] = set()
    terminal_attempts: dict[str, dict[str, Any]] = {}

    for event in ordered:
        payload = event.payload
        if event.event_type == "placement_ledger_transition":
            transition = _normalized_placement_transition(payload)
            command_id = _string_value(payload.get("command_id")) or _string_value(
                transition.get("command_id")
            )
            if (
                command_id is not None
                and _semantic_build_action_name(payload, transition) is not None
            ):
                transitions_by_command[command_id].append(transition)
            continue

        action_name = _semantic_build_action_name(payload)
        command = payload.get("command")
        command_payload = command if isinstance(command, dict) else {}
        command_id = _string_value(payload.get("command_id")) or _string_value(
            command_payload.get("command_id")
        )
        if action_name is None or command_id is None:
            continue
        metadata = dict(command_payload)
        metadata.update(
            {
                key: value
                for key, value in payload.items()
                if key not in {"command", "effect_evidence"}
            }
        )
        command_metadata[command_id] = metadata
        if event.event_type != "execution":
            continue
        status = _semantic_build_terminal_status(payload)
        if status is None:
            continue
        terminal_execution_ids.add(command_id)
        terminal_attempts[command_id] = {
            "event_id": event.event_id,
            "command_id": command_id,
            "payload": payload,
            "metadata": metadata,
            "status": status,
            "gameplay_attempt": _pysc2_accepted(payload),
        }

    for event in ordered:
        if event.event_type != "command_lifecycle":
            continue
        payload = event.payload
        action_name = _semantic_build_action_name(payload)
        command = payload.get("command")
        command_payload = command if isinstance(command, dict) else {}
        command_id = _string_value(payload.get("command_id")) or _string_value(
            command_payload.get("command_id")
        )
        if action_name is None or command_id is None or command_id in terminal_execution_ids:
            continue
        status = _semantic_build_terminal_status(payload)
        if status is None:
            continue
        metadata = command_metadata.get(command_id, dict(command_payload))
        terminal_attempts[command_id] = {
            "event_id": event.event_id,
            "command_id": command_id,
            "payload": payload,
            "metadata": metadata,
            "status": status,
            "gameplay_attempt": False,
        }

    attempts_by_operation: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    circuit_command_ids: set[str] = set()
    for attempt in sorted(terminal_attempts.values(), key=lambda item: item["event_id"]):
        if not attempt["gameplay_attempt"]:
            continue
        command_id = attempt["command_id"]
        payload = attempt["payload"]
        metadata = attempt["metadata"]
        transitions = transitions_by_command.get(command_id, [])
        operation_id = _semantic_build_operation_id(payload, metadata, transitions, command_id)
        revision = _semantic_build_revision(payload, metadata, transitions)
        failure = attempt["status"] in _SEMANTIC_BUILD_FAILURE_STATUSES
        failure_code = _string_value(payload.get("failure_code"))
        no_start_failure = failure and failure_code == "no_build_start_evidence"
        failure_classification = (
            _semantic_build_failure_classification(payload, metadata, transitions)
            if failure
            else None
        )
        circuit_open = _semantic_build_circuit_open(payload, metadata, transitions)
        if circuit_open:
            circuit_command_ids.add(command_id)
        attempts_by_operation[operation_id].append(
            {
                "event_id": attempt["event_id"],
                "command_id": command_id,
                "status": attempt["status"],
                "failed": failure,
                "no_start_failed": no_start_failure,
                "failure_code": failure_code,
                "revision": revision,
                "failure_classification": failure_classification,
                "circuit_open": circuit_open,
            }
        )

    typed_circuit_command_ids: set[str] = set()
    typed_circuit_event_count = 0
    for event in ordered:
        if event.event_type not in {
            "operation_circuit_open",
            "semantic_build_circuit_open",
            "circuit_breaker_open",
        }:
            continue
        payload = event.payload
        command_id = _string_value(payload.get("command_id"))
        if command_id is None:
            typed_circuit_event_count += 1
        else:
            typed_circuit_command_ids.add(command_id)

    max_failure_streak = 0
    failure_count = 0
    cross_revision_retry_count = 0
    classification_counts: Counter[str] = Counter()
    operation_diagnostics: dict[str, dict[str, Any]] = {}
    for operation_id, attempts in sorted(attempts_by_operation.items()):
        streak = 0
        operation_max_streak = 0
        operation_no_start_failures = 0
        operation_classifications: Counter[str] = Counter()
        for attempt in attempts:
            if attempt["no_start_failed"]:
                streak += 1
                operation_no_start_failures += 1
            else:
                streak = 0
            if attempt["failed"]:
                failure_count += 1
                classification = attempt["failure_classification"] or "unclassified"
                classification_counts[classification] += 1
                operation_classifications[classification] += 1
            max_failure_streak = max(max_failure_streak, streak)
            operation_max_streak = max(operation_max_streak, streak)
        operation_cross_revision_retries = 0
        for previous, current in zip(attempts, attempts[1:], strict=False):
            if (
                previous["failed"]
                and previous["revision"] is not None
                and current["revision"] is not None
                and previous["revision"] != current["revision"]
            ):
                cross_revision_retry_count += 1
                operation_cross_revision_retries += 1
        operation_diagnostics[operation_id] = {
            "accepted_attempt_count": len(attempts),
            "no_start_failure_count": operation_no_start_failures,
            "max_no_start_failure_streak": operation_max_streak,
            "circuit_breaker_count": sum(attempt["circuit_open"] for attempt in attempts),
            "cross_revision_retry_count": operation_cross_revision_retries,
            "failure_classification": dict(sorted(operation_classifications.items())),
        }

    return {
        "operation_count": len(attempts_by_operation),
        "failure_count": failure_count,
        "max_failure_streak": max_failure_streak,
        "circuit_breaker_count": (
            len(circuit_command_ids | typed_circuit_command_ids) + typed_circuit_event_count
        ),
        "cross_revision_retry_count": cross_revision_retry_count,
        "failure_classification": dict(sorted(classification_counts.items())),
        "operations": operation_diagnostics,
    }


def _semantic_build_action_name(
    payload: dict[str, Any],
    transition: dict[str, Any] | None = None,
) -> str | None:
    candidates: list[Any] = [
        payload.get("action_name"),
        payload.get("semantic_action"),
        payload.get("action"),
    ]
    command = payload.get("command")
    if isinstance(command, dict):
        candidates.extend((command.get("name"), command.get("action_name")))
    if transition is not None:
        candidates.extend((transition.get("action_name"), transition.get("semantic_action")))
    for value in candidates:
        if not isinstance(value, str):
            continue
        normalized = value.replace("_", " ").replace("-", " ").strip().upper()
        if value.startswith("Build_") or normalized.startswith("BUILD "):
            return value
    return None


def _semantic_build_terminal_status(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if isinstance(status, str):
        normalized = status.casefold()
        if normalized == "succeeded" or normalized in _SEMANTIC_BUILD_FAILURE_STATUSES:
            return normalized
    if payload.get("success") is False or payload.get("failure_code") is not None:
        return "failed"
    return None


def _semantic_build_operation_id(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    transitions: Sequence[dict[str, Any]],
    command_id: str,
) -> str:
    for source in (payload, metadata, *_effect_evidence_sources(payload), *transitions):
        for key in _SEMANTIC_BUILD_OPERATION_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return f"command:{command_id}"


def _semantic_build_revision(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    transitions: Sequence[dict[str, Any]],
) -> str | None:
    for source in (payload, metadata, *_effect_evidence_sources(payload), *transitions):
        for key in _SEMANTIC_BUILD_REVISION_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _semantic_build_failure_classification(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    transitions: Sequence[dict[str, Any]],
) -> str | None:
    sources = (*_effect_evidence_sources(payload), payload, metadata, *transitions)
    for source in sources:
        for key in _SEMANTIC_BUILD_CLASSIFICATION_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    for source in sources:
        legacy = source.get("failure_class")
        if isinstance(legacy, str) and legacy:
            return legacy
    code = payload.get("failure_code")
    if isinstance(code, str):
        if code in {"builder_unavailable", "actor_not_available", "builder_not_observable"}:
            return "builder_not_ready"
        if code in _NEGATIVE_BUILD_EFFECT_CODES:
            return "gameplay_no_start_unknown"
    return None


def _semantic_build_circuit_open(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    transitions: Sequence[dict[str, Any]],
) -> bool:
    for source in (payload, metadata, *transitions):
        if source.get("operation_circuit_open") is True:
            return True
        placement_no_start = source.get("placement_no_start")
        if isinstance(placement_no_start, dict) and placement_no_start.get("circuit_open") is True:
            return True
    return False


def _effect_evidence_sources(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    evidence = payload.get("effect_evidence")
    return (evidence,) if isinstance(evidence, dict) else ()


def _normalized_placement_transition(payload: dict[str, Any]) -> dict[str, Any]:
    transition = payload.get("transition")
    normalized = dict(transition) if isinstance(transition, dict) else {}
    no_start = payload.get("placement_no_start")
    if not isinstance(no_start, dict) and isinstance(transition, dict):
        no_start = transition.get("placement_no_start")
    if isinstance(no_start, dict):
        normalized["placement_no_start"] = no_start
        for key in (
            "operation_id",
            "failure_classification",
            "classification_basis",
            "target_state_revision",
            "placement_revision",
            "operation_circuit_open",
            "circuit_breaker_open",
            "circuit_open",
            "breaker_open",
            "circuit_breaker_triggered",
            "breaker_tripped",
            "material_legality_identity",
            "target_legality_fingerprint",
            "available_ability_query",
            "placement_query_result",
            "ability_id",
        ):
            if key in no_start:
                normalized[key] = no_start[key]
    for key in (
        "command_id",
        "operation_id",
        "action_name",
        "semantic_action",
        "target_state_revision",
        "placement_revision",
        "failure_class",
        "failure_classification",
        "classification",
        "operation_circuit_open",
        "circuit_breaker_open",
        "circuit_open",
        "breaker_open",
        "circuit_breaker_triggered",
        "breaker_tripped",
        "next_state",
        "previous_state",
        "release_reason",
        "actor_failure",
        "material_legality_identity",
        "target_legality_fingerprint",
        "available_ability_query",
        "placement_query_result",
        "ability_id",
        "primitive_constructed_game_loop",
        "primitive_submitted_game_loop",
        "action_result",
    ):
        if key in payload:
            normalized[key] = payload[key]
    return normalized


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


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
    requested = evidence.get("requested_target_position")
    validated = evidence.get("final_validated_target_position")
    emitted = evidence.get("emitted_target_position")
    verified = evidence.get("verified_target_position")
    return (
        isinstance(requested, (list, tuple))
        and isinstance(validated, (list, tuple))
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
    temporarily_failed_targets: set[tuple[str, frozenset[tuple[int, int]], str]] = set()
    invalid_transitions = 0
    overlaps = 0
    invalid_redispatches = 0
    unchanged_failed_target_redispatches = 0
    nonspatial_quarantines = 0
    transition_count = 0
    allowed = {
        ("unreserved", "reserved"),
        ("unreserved", "permanent_invalid"),
        ("unreserved", "temporary_suppressed"),
        ("unreserved", "released"),
        ("reserved", "occupied"),
        ("reserved", "build_started"),
        ("build_started", "occupied"),
        ("build_started", "permanent_invalid"),
        ("build_started", "temporary_suppressed"),
        ("build_started", "released"),
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
        action_name = str(payload.get("action_name", ""))
        target_state_revision = payload.get("target_state_revision")
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
            if (
                isinstance(target_state_revision, str)
                and (action_name, cells, target_state_revision) in temporarily_failed_targets
            ):
                unchanged_failed_target_redispatches += 1
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
        elif next_state == "temporary_suppressed" and isinstance(
            target_state_revision,
            str,
        ):
            temporarily_failed_targets.add((action_name, cells, target_state_revision))
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
        "unchanged_failed_target_redispatch_count": unchanged_failed_target_redispatches,
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
    invalid_confirmations = 0
    for _, payload in reports:
        evidence = _evidence(payload)
        if evidence.get("effect_kind") != "combat":
            continue
        confirmation_kind = evidence.get("confirmation_kind")
        if payload.get("status") != "succeeded" or confirmation_kind not in {
            "target_damaged",
            "target_removed",
        }:
            continue

        confirmed_game_loop = evidence.get("confirmed_game_loop")
        if confirmed_game_loop is None:
            confirmed_game_loop = evidence.get("confirmed_loop")
        engagement = evidence.get("engagement_id")
        target_tag = evidence.get("target_tag")
        baseline = _finite_number(evidence.get("baseline_target_health"))
        observed = _finite_number(evidence.get("observed_target_health"))
        delta = _finite_number(evidence.get("target_health_delta"))
        if (
            not isinstance(confirmed_game_loop, int)
            or isinstance(confirmed_game_loop, bool)
            or confirmed_game_loop < 0
            or not isinstance(engagement, str)
            or not engagement
            or not isinstance(target_tag, str)
            or not target_tag
            or baseline is None
            or observed is None
            or delta is None
        ):
            # A claimed confirmation without a complete, internally consistent
            # health transition is itself an attribution failure.  Failed and
            # unconfirmed reports were filtered above and remain diagnostics.
            invalid_confirmations += 1
            continue
        if (
            baseline < 0
            or observed < 0
            or delta < 0
            or observed > baseline
            or not math.isclose(delta, baseline - observed, rel_tol=0.0, abs_tol=1e-6)
            or confirmation_kind == "target_damaged"
            and delta <= 0
        ):
            invalid_confirmations += 1
            continue
        key = (
            target_tag,
            confirmed_game_loop,
            baseline,
            observed,
        )
        engagements[key].add(engagement)
    return invalid_confirmations + sum(max(0, len(values) - 1) for values in engagements.values())


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


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
        "current_batch_selected",
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
        current_batch_selected = int(item["current_batch_selected"])
        unobserved_dispatches = max(0, dispatched - constructing - queued)
        expected = (
            completed
            + constructing
            + queued
            + reserved
            + unobserved_dispatches
            + current_batch_selected
        )
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
