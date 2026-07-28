"""Fail-closed engineering acceptance for one natural-terminal live run."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from rtscortex.evaluation.cortex import compute_cortex_observability
from rtscortex.evaluation.metrics import compute_execution_metrics
from rtscortex.memory import StoredEvent

ENGINEERING_GATES_FILENAME = "engineering-gates.json"
REQUIRED_ENGINEERING_GATES = (
    "runtime_crash_zero",
    "friendly_target_zero",
    "lineage_complete",
    "production_confirmation_complete",
    "build_start_coverage",
    "build_confirmation_rate",
    "build_failure_rate",
    "placement_identity_complete",
    "builder_provenance_complete",
    "nonspatial_quarantine_zero",
    "invalid_footprint_redispatch_zero",
    "repeated_retreat_arrival_zero",
    "unchanged_attack_redispatch_zero",
    "health_delta_engagement_attribution_valid",
    "expansion_terminal_immediate_rearm_zero",
    "defense_inventory_within_cap",
    "subscriber_callback_under_durable_lock_zero",
    "checkpoint_tail_recovery_bounded",
    "postgame_semantic_event_coverage",
    "effective_loops_per_second",
    "natural_run_disk_reduction_ratio",
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
_PERMANENT_SPATIAL_CODES = frozenset(
    {
        "blocked",
        "invalid_terrain",
        "invalid_expansion_anchor",
        "not_pathable",
        "placement_occupied",
    }
)
_NONSPATIAL_CODES = frozenset(
    {
        "builder_not_observable",
        "no_build_order_observed",
        "worker_order_replaced",
        "build_started_effect_missing",
        "no_build_start_evidence",
        "pysc2_rejected",
    }
)


def build_engineering_gate_report(
    events: Sequence[StoredEvent],
    *,
    run_dir: Path,
    natural_run_baseline_bytes_per_loop: float | None,
    recovery_tail_limit: int = 4096,
) -> dict[str, Any]:
    """Derive every SCX-PT-039 engineering gate without permissive defaults."""

    execution = compute_execution_metrics(list(events))
    cortex = compute_cortex_observability(events)
    result = next(
        (event.payload for event in reversed(events) if event.event_type == "episode_result"),
        None,
    )
    reports = [
        (event.event_id, event.payload) for event in events if event.event_type == "execution"
    ]
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
    invalid_redispatches, nonspatial_quarantines = _placement_reuse_violations(accepted_builds)
    retreat_repeats = _repeated_retreat_arrivals(events)
    unchanged_attacks = _unchanged_attack_redispatches(reports)
    health_delta_collisions = _health_delta_engagement_collisions(reports)
    expansion_rearms = _expansion_immediate_rearms(events)
    defense_cap_violations = _defense_inventory_cap_violations(events)
    performance = next(
        (
            event.payload
            for event in reversed(events)
            if event.event_type == "event_store_performance"
        ),
        None,
    )
    recovery_events = [
        event.payload for event in events if event.event_type == "runtime_recovery_completed"
    ]
    max_game_loop = max(
        (
            int(event.payload["game_loop"])
            for event in events
            if isinstance(event.payload.get("game_loop"), int | float)
        ),
        default=(int(result.get("steps", 0)) if isinstance(result, dict) else 0),
    )
    elapsed = _elapsed_seconds(events)
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
        event.event_type == "strategic_consequence_attributed" for event in events
    )
    review = next(
        (
            event.payload
            for event in reversed(events)
            if event.event_type == "postgame_review_completed"
        ),
        None,
    )
    postgame_coverage = (
        None
        if review is None
        else (
            1.0
            if int(review.get("strategic_consequence_count", -1)) == attributed_consequences
            else 0.0
        )
    )
    production_accepted = execution.production_funnel.get("pysc2_accepted", 0)
    metrics: dict[str, bool | int | float | None] = {
        "runtime_crash_zero": (
            isinstance(result, dict)
            and result.get("outcome") in {"victory", "defeat", "draw"}
            and not result.get("failure_reason")
        ),
        "friendly_target_zero": execution.friendly_target_attacks == 0,
        "lineage_complete": (
            cortex.observed
            and cortex.command_lineage_coverage == 1.0
            and cortex.role_lineage_coverage == 1.0
            and cortex.lineage_integrity_violations == 0
        ),
        "production_confirmation_complete": (
            production_accepted == 0 or execution.production_effect_confirmed_rate == 1.0
        ),
        "build_start_coverage": _ratio(build_started, len(accepted_builds)),
        "build_confirmation_rate": _ratio(build_confirmed, len(accepted_builds)),
        "build_failure_rate": _ratio(build_failures, len(accepted_builds)),
        "placement_identity_complete": (
            len(accepted_builds) == 0 or placement_complete == len(accepted_builds)
        ),
        "builder_provenance_complete": (
            len(accepted_builds) == 0 or builder_complete == len(accepted_builds)
        ),
        "nonspatial_quarantine_zero": nonspatial_quarantines == 0,
        "invalid_footprint_redispatch_zero": invalid_redispatches == 0,
        "repeated_retreat_arrival_zero": retreat_repeats == 0,
        "unchanged_attack_redispatch_zero": unchanged_attacks == 0,
        "health_delta_engagement_attribution_valid": health_delta_collisions == 0,
        "expansion_terminal_immediate_rearm_zero": expansion_rearms == 0,
        "defense_inventory_within_cap": defense_cap_violations == 0,
        "subscriber_callback_under_durable_lock_zero": (
            None
            if performance is None
            else int(performance.get("subscriber_callbacks_under_durable_lock", -1)) == 0
        ),
        "checkpoint_tail_recovery_bounded": (
            all(
                int(item.get("tail_event_count", recovery_tail_limit + 1))
                <= int(item.get("tail_event_limit", recovery_tail_limit))
                for item in recovery_events
            )
            if recovery_events
            else True
        ),
        "postgame_semantic_event_coverage": postgame_coverage,
        "effective_loops_per_second": loops_per_second,
        "natural_run_disk_reduction_ratio": disk_ratio,
    }
    thresholds: dict[str, tuple[str, bool | int | float]] = {
        name: ("==", True)
        for name in REQUIRED_ENGINEERING_GATES
        if name
        not in {
            "build_start_coverage",
            "build_confirmation_rate",
            "build_failure_rate",
            "postgame_semantic_event_coverage",
            "effective_loops_per_second",
            "natural_run_disk_reduction_ratio",
        }
    }
    thresholds.update(
        {
            "build_start_coverage": (">=", 1.0),
            "build_confirmation_rate": (">=", 0.90),
            "build_failure_rate": ("<=", 0.10),
            "postgame_semantic_event_coverage": (">=", 1.0),
            "effective_loops_per_second": (">=", 2.0),
            "natural_run_disk_reduction_ratio": (">=", 4.0),
        }
    )
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
        "format_version": "1.0",
        "metrics": metrics,
        "diagnostics": {
            "accepted_build_count": len(accepted_builds),
            "build_start_count": build_started,
            "build_confirmed_count": build_confirmed,
            "build_failure_count": build_failures,
            "invalid_footprint_redispatch_count": invalid_redispatches,
            "nonspatial_quarantine_count": nonspatial_quarantines,
            "repeated_retreat_arrival_count": retreat_repeats,
            "unchanged_attack_redispatch_count": unchanged_attacks,
            "health_delta_engagement_collision_count": health_delta_collisions,
            "expansion_terminal_immediate_rearm_count": expansion_rearms,
            "defense_inventory_cap_violation_count": defense_cap_violations,
            "max_game_loop": max_game_loop,
            "durable_bytes": durable_bytes,
            "durable_bytes_per_game_loop": bytes_per_loop,
            "natural_run_baseline_bytes_per_game_loop": (natural_run_baseline_bytes_per_loop),
        },
        "gates": gates,
        "missing_required_metrics": missing,
        "accepted": not missing and all(item["passed"] is True for item in gates.values()),
    }


def _is_build(payload: dict[str, Any]) -> bool:
    return str(payload.get("action_name", "")).startswith("Build_")


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
    evidence = _evidence(payload)
    return bool(evidence.get("builder_tag"))


def _placement_key(payload: dict[str, Any]) -> tuple[str, tuple[float, float]] | None:
    position = _evidence(payload).get("target_position")
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        return None
    return (
        str(payload.get("action_name", "")),
        (round(float(position[0]), 3), round(float(position[1]), 3)),
    )


def _placement_reuse_violations(
    builds: Sequence[tuple[int, dict[str, Any]]],
) -> tuple[int, int]:
    permanent: set[tuple[str, tuple[float, float]]] = set()
    retryable: set[tuple[str, tuple[float, float]]] = set()
    invalid_redispatches = 0
    nonspatial_quarantines = 0
    for _, payload in sorted(builds):
        key = _placement_key(payload)
        if key is None:
            continue
        if key in permanent:
            invalid_redispatches += 1
        if key in retryable and payload.get("failure_code") == "no_legal_placement":
            nonspatial_quarantines += 1
        code = str(payload.get("failure_code") or "")
        if code in _PERMANENT_SPATIAL_CODES:
            permanent.add(key)
        elif code in _NONSPATIAL_CODES:
            retryable.add(key)
    return invalid_redispatches, nonspatial_quarantines


def _repeated_retreat_arrivals(events: Sequence[StoredEvent]) -> int:
    counts: Counter[str] = Counter()
    for event in events:
        if event.event_type != "tactical_actor_state":
            continue
        if event.payload.get("state") != "retreat_arrived":
            continue
        identity = str(
            event.payload.get("commitment_id")
            or f"missing:{event.payload.get('actor')}:{event.event_id}"
        )
        counts[identity] += 1
    return sum(max(0, count - 1) for count in counts.values())


def _unchanged_attack_redispatches(
    reports: Sequence[tuple[int, dict[str, Any]]],
) -> int:
    counts: Counter[str] = Counter()
    for _, payload in reports:
        if payload.get("action_name") != "Attack_Unit":
            continue
        engagement_id = _evidence(payload).get("engagement_id")
        if isinstance(engagement_id, str):
            counts[engagement_id] += 1
    return sum(max(0, count - 1) for count in counts.values())


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


def _defense_inventory_cap_violations(events: Sequence[StoredEvent]) -> int:
    violations = 0
    for event in events:
        if event.event_type != "observation":
            continue
        state = event.payload.get("state")
        if not isinstance(state, dict):
            continue
        structures = state.get("own_structures")
        if not isinstance(structures, list):
            continue
        shield_batteries = sum(
            isinstance(item, dict) and item.get("unit_type") == "ShieldBattery"
            for item in structures
        )
        violations += shield_batteries > 4
    return violations


def _elapsed_seconds(events: Sequence[StoredEvent]) -> float:
    timestamps: list[datetime] = []
    for event in events:
        try:
            timestamps.append(datetime.fromisoformat(event.created_at))
        except ValueError:
            continue
    if len(timestamps) < 2:
        return 0.0
    return max(0.0, (timestamps[-1] - timestamps[0]).total_seconds())


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


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
