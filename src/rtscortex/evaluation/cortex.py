"""Observability projections for the typed Cortex decision pipeline."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from rtscortex.memory import StoredEvent

CORTEX_EVENT_TYPES = frozenset(
    {
        "race_profile_activated",
        "situation_assessed",
        "situation_shadow_assessed",
        "tactical_policy_shadow",
        "macro_plan_accepted",
        "macro_plan_rejected",
        "macro_step_updated",
        "intent_emitted",
        "role_intent_emitted",
        "intent_arbitrated",
        "intent_arbiter_shadow_diff",
        "candidate_set_built",
        "executor_selection",
        "command_lineage",
        "specialist_failed",
        "specialist_ready",
        "specialist_recovered",
        "race_brain_coordinated",
        "macro_proposal_revalidated",
        "playbook_retrieved",
        "playbook_rule_applied",
        "playbook_rule_updated",
        "playbook_case_recorded",
        "playbook_lesson_candidate",
        "playbook_lesson_promoted",
        "postgame_review_completed",
    }
)


@dataclass(frozen=True)
class CortexObservabilityMetrics:
    """Counts and invariants derived without requiring live Cortex model classes."""

    observed: bool = False
    event_counts: dict[str, int] = field(default_factory=dict)
    intent_counts: dict[str, int] = field(default_factory=dict)
    specialist_failure_counts: dict[str, int] = field(default_factory=dict)
    specialist_ready_counts: dict[str, int] = field(default_factory=dict)
    specialist_recovery_counts: dict[str, int] = field(default_factory=dict)
    macro_requests: int = 0
    macro_latency_ms_p50: float = 0.0
    macro_latency_ms_p95: float = 0.0
    executor_counts: dict[str, int] = field(default_factory=dict)
    executor_selections: int = 0
    executor_abstentions: int = 0
    executor_fallbacks: int = 0
    executor_candidate_violations: int = 0
    lineage_integrity_violations: int = 0
    duplicate_lineage_commands: int = 0
    executor_latency_ms_p50: float = 0.0
    executor_latency_ms_p95: float = 0.0
    dispatched_commands: int = 0
    lineage_commands: int = 0
    missing_lineage_commands: int = 0
    orphan_lineage_commands: int = 0
    command_lineage_coverage: float = 0.0
    role_intent_counts: dict[str, int] = field(default_factory=dict)
    intent_decision_counts: dict[str, int] = field(default_factory=dict)
    intent_conflict_counts: dict[str, int] = field(default_factory=dict)
    intent_shadow_diff_count: int = 0
    playbook_application_count: int = 0
    playbook_block_count: int = 0
    playbook_shadow_block_count: int = 0
    playbook_rule_update_count: int = 0
    role_lineage_coverage: float = 0.0
    active_race: str | None = None
    race_macro_contract_ready: bool | None = None
    race_runtime_mapping_ready: bool | None = None
    race_live_worker_ready: bool | None = None
    race_limitations: tuple[str, ...] = ()
    race_brain_selected_members: dict[str, int] = field(default_factory=dict)
    race_brain_selected_by_phase: dict[str, int] = field(default_factory=dict)
    race_brain_degraded_members: int = 0
    race_brain_unique_frontier_contributions: dict[str, int] = field(default_factory=dict)
    race_brain_proposal_diversity: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_cortex_observability(
    events: Sequence[StoredEvent],
) -> CortexObservabilityMetrics:
    """Project Cortex events while remaining tolerant of legacy journals."""

    cortex_events = [event for event in events if event.event_type in CORTEX_EVENT_TYPES]
    if not cortex_events:
        return CortexObservabilityMetrics()

    event_counts = Counter(event.event_type for event in cortex_events)
    intent_counts: Counter[str] = Counter()
    specialist_failures: Counter[str] = Counter()
    specialist_ready: Counter[str] = Counter()
    specialist_recoveries: Counter[str] = Counter()
    executors: Counter[str] = Counter()
    selected_candidates: list[tuple[str, str | None, str]] = []
    candidate_domains: dict[str, set[str]] = {}
    selections_by_id: dict[str, tuple[str, str]] = {}
    emitted_intents: Counter[str] = Counter()
    emitted_intent_roles: dict[str, str] = {}
    accepted_macro_plans: Counter[str] = Counter()
    candidate_set_intents: set[str] = set()
    pipeline_identity_violations = 0
    latencies: list[float] = []
    macro_latencies: list[float] = []
    abstentions = 0
    fallbacks = 0
    role_intents: Counter[str] = Counter()
    role_source_intents: set[str] = set()
    intent_decisions: Counter[str] = Counter()
    intent_conflicts: Counter[str] = Counter()
    shadow_diffs = 0
    playbook_applications = 0
    playbook_blocks = 0
    playbook_shadow_blocks = 0
    race_profile_payload: dict[str, Any] = {}
    current_phase = "unknown"
    race_brain_selected: Counter[str] = Counter()
    race_brain_selected_by_phase: Counter[str] = Counter()
    race_brain_unique_contributions: Counter[str] = Counter()
    race_brain_degraded_members = 0
    race_brain_diversities: list[float] = []

    for event in cortex_events:
        payload = _object(event.payload)
        if event.event_type == "race_profile_activated":
            race_profile_payload = payload
        elif event.event_type == "situation_assessed":
            current_phase = _text(payload, "phase", "game_phase") or current_phase
        elif event.event_type == "race_brain_coordinated":
            selected_member = _text(payload, "selected_member_id")
            if selected_member is not None:
                race_brain_selected[selected_member] += 1
                race_brain_selected_by_phase[f"{current_phase}/{selected_member}"] += 1
            degraded = payload.get("degraded_member_ids")
            if isinstance(degraded, list):
                race_brain_degraded_members += len(degraded)
            members = payload.get("members")
            if isinstance(members, list):
                frontiers = [
                    (
                        _text(_object(member), "member_id") or "unknown",
                        _text(_object(_object(member).get("frontier")), "source_action"),
                    )
                    for member in members
                ]
                actions = [action for _, action in frontiers if action is not None]
                if actions:
                    race_brain_diversities.append(len(set(actions)) / len(actions))
                    counts = Counter(actions)
                    for member_id, action in frontiers:
                        if action is not None and counts[action] == 1:
                            race_brain_unique_contributions[member_id] += 1
        elif event.event_type == "intent_emitted":
            role = _text(payload, "role", "source_role", "intent_kind", "source") or "unknown"
            intent_counts[role] += 1
            intent_id = _text(payload, "intent_id")
            if intent_id is not None:
                emitted_intents[intent_id] += 1
                if intent_id in emitted_intent_roles:
                    pipeline_identity_violations += 1
                else:
                    emitted_intent_roles[intent_id] = role
        elif event.event_type == "role_intent_emitted":
            intent = _object(payload.get("intent"))
            role = _text(intent, "role") or "unknown"
            role_intents[role] += 1
            source_intent_id = _text(intent, "source_intent_id")
            if source_intent_id is not None:
                role_source_intents.add(source_intent_id)
        elif event.event_type == "intent_arbitrated":
            arbitration = _object(payload.get("arbitration"))
            decisions = arbitration.get("decisions")
            if isinstance(decisions, list):
                for raw_decision in decisions:
                    status = _text(_object(raw_decision), "status") or "unknown"
                    intent_decisions[status] += 1
            conflicts = arbitration.get("conflicts")
            if isinstance(conflicts, list):
                for raw_conflict in conflicts:
                    kind = _text(_object(raw_conflict), "kind") or "unknown"
                    intent_conflicts[kind] += 1
        elif event.event_type == "intent_arbiter_shadow_diff":
            if payload.get("only_actual") or payload.get("only_shadow"):
                shadow_diffs += 1
        elif event.event_type == "playbook_rule_applied":
            playbook_applications += 1
            if payload.get("blocked") is True:
                playbook_blocks += 1
            if payload.get("reason") == "shadow_would_block":
                playbook_shadow_blocks += 1
        elif event.event_type == "macro_plan_accepted":
            plan_id = _text(payload, "plan_id") or _text(_object(payload.get("plan")), "plan_id")
            if plan_id is not None:
                accepted_macro_plans[plan_id] += 1
                if accepted_macro_plans[plan_id] > 1:
                    pipeline_identity_violations += 1
        elif event.event_type == "candidate_set_built":
            intent_id = _text(payload, "intent_id")
            if intent_id is not None:
                if intent_id in candidate_set_intents:
                    pipeline_identity_violations += 1
                candidate_set_intents.add(intent_id)
                candidate_domains.setdefault(intent_id, set()).update(_candidate_ids(payload))
        elif event.event_type == "executor_selection":
            executor = _text(payload, "executor_id", "executor", "model") or "unknown"
            executors[executor] += 1
            selected = _text(payload, "selected_candidate_id", "candidate_id")
            selection_status = _text(payload, "status", "selection")
            if selected:
                intent_id = _text(payload, "intent_id")
                selection_id = _text(payload, "selection_id")
                selected_candidates.append((intent_id or "", selection_id, selected))
                if selection_id is not None and intent_id is not None:
                    if selection_id in selections_by_id:
                        pipeline_identity_violations += 1
                    else:
                        selections_by_id[selection_id] = (intent_id, selected)
            elif selection_status in {"abstain", "abstained"} or payload.get("abstain") is True:
                abstentions += 1
            if payload.get("fallback") is True or _text(payload, "fallback_reason"):
                fallbacks += 1
            latency = payload.get("latency_ms")
            if isinstance(latency, int | float) and not isinstance(latency, bool):
                latencies.append(float(latency))
        elif event.event_type in {
            "specialist_failed",
            "specialist_ready",
            "specialist_recovered",
        }:
            role = _text(payload, "role", "specialist", "module") or "unknown"
            target = {
                "specialist_failed": specialist_failures,
                "specialist_ready": specialist_ready,
                "specialist_recovered": specialist_recoveries,
            }[event.event_type]
            target[role] += 1
        if event.event_type in {"macro_plan_accepted", "macro_plan_rejected"}:
            latency = payload.get("latency_ms")
            if isinstance(latency, int | float) and not isinstance(latency, bool):
                macro_latencies.append(float(latency))

    dispatched = _dispatched_command_ids(events)
    lineage_counts: Counter[str] = Counter()
    valid_lineage_counts: Counter[str] = Counter()
    lineage_integrity_violations = pipeline_identity_violations
    for event in cortex_events:
        if event.event_type != "command_lineage":
            continue
        payload = _object(event.payload)
        lineage = _object(payload.get("lineage")) or payload
        command_id = _command_id(payload)
        intent_id = _text(lineage, "intent_id")
        candidate_id = _text(lineage, "candidate_id")
        selection_id = _text(lineage, "selection_id")
        source_role = _text(lineage, "source_role", "role")
        macro_plan_id = _text(lineage, "macro_plan_id")
        if command_id is not None:
            lineage_counts[command_id] += 1
        valid = (
            command_id is not None
            and intent_id is not None
            and candidate_id is not None
            and selection_id is not None
            and source_role is not None
            and emitted_intents[intent_id] == 1
            and emitted_intent_roles.get(intent_id) == source_role
            and candidate_id in candidate_domains.get(intent_id, set())
            and selections_by_id.get(selection_id) == (intent_id, candidate_id)
            and (
                (
                    source_role == "macro"
                    and macro_plan_id is not None
                    and accepted_macro_plans[macro_plan_id] == 1
                )
                or (source_role == "reflex" and macro_plan_id is None)
                or source_role not in {"macro", "reflex"}
            )
        )
        if valid and command_id is not None:
            valid_lineage_counts[command_id] += 1
        else:
            lineage_integrity_violations += 1
    duplicate_lineage_commands = sum(count - 1 for count in lineage_counts.values() if count > 1)
    lineage_integrity_violations += duplicate_lineage_commands
    valid_lineages = {
        command_id
        for command_id, count in lineage_counts.items()
        if count == 1 and valid_lineage_counts[command_id] == 1
    }
    missing = dispatched - valid_lineages
    orphan = valid_lineages - dispatched
    coverage = len(dispatched & valid_lineages) / len(dispatched) if dispatched else 0.0
    role_covered = {
        command_id
        for command_id, count in lineage_counts.items()
        if count == 1
        and any(
            _text(_object(event.payload.get("lineage")), "intent_id") in role_source_intents
            for event in cortex_events
            if event.event_type == "command_lineage"
            and _command_id(_object(event.payload)) == command_id
        )
    }
    role_coverage = len(dispatched & role_covered) / len(dispatched) if dispatched else 0.0
    violations = sum(
        not intent_id or candidate_id not in candidate_domains.get(intent_id, set())
        for intent_id, _, candidate_id in selected_candidates
    )

    return CortexObservabilityMetrics(
        observed=True,
        event_counts=dict(sorted(event_counts.items())),
        intent_counts=dict(sorted(intent_counts.items())),
        specialist_failure_counts=dict(sorted(specialist_failures.items())),
        specialist_ready_counts=dict(sorted(specialist_ready.items())),
        specialist_recovery_counts=dict(sorted(specialist_recoveries.items())),
        macro_requests=len(macro_latencies),
        macro_latency_ms_p50=_percentile(macro_latencies, 0.50),
        macro_latency_ms_p95=_percentile(macro_latencies, 0.95),
        executor_counts=dict(sorted(executors.items())),
        executor_selections=len(selected_candidates),
        executor_abstentions=abstentions,
        executor_fallbacks=fallbacks,
        executor_candidate_violations=violations,
        lineage_integrity_violations=lineage_integrity_violations,
        executor_latency_ms_p50=_percentile(latencies, 0.50),
        executor_latency_ms_p95=_percentile(latencies, 0.95),
        dispatched_commands=len(dispatched),
        lineage_commands=len(lineage_counts),
        missing_lineage_commands=len(missing),
        orphan_lineage_commands=len(orphan),
        command_lineage_coverage=coverage,
        duplicate_lineage_commands=duplicate_lineage_commands,
        role_intent_counts=dict(sorted(role_intents.items())),
        intent_decision_counts=dict(sorted(intent_decisions.items())),
        intent_conflict_counts=dict(sorted(intent_conflicts.items())),
        intent_shadow_diff_count=shadow_diffs,
        playbook_application_count=playbook_applications,
        playbook_block_count=playbook_blocks,
        playbook_shadow_block_count=playbook_shadow_blocks,
        playbook_rule_update_count=event_counts.get("playbook_rule_updated", 0),
        role_lineage_coverage=role_coverage,
        active_race=_text(race_profile_payload, "race"),
        race_macro_contract_ready=_optional_bool(
            race_profile_payload,
            "macro_contract_ready",
        ),
        race_runtime_mapping_ready=_optional_bool(
            race_profile_payload,
            "runtime_mapping_ready",
        ),
        race_live_worker_ready=_optional_bool(
            race_profile_payload,
            "live_worker_ready",
        ),
        race_limitations=tuple(
            item
            for item in race_profile_payload.get("limitations", [])
            if isinstance(item, str)
        ),
        race_brain_selected_members=dict(sorted(race_brain_selected.items())),
        race_brain_selected_by_phase=dict(sorted(race_brain_selected_by_phase.items())),
        race_brain_degraded_members=race_brain_degraded_members,
        race_brain_unique_frontier_contributions=dict(
            sorted(race_brain_unique_contributions.items())
        ),
        race_brain_proposal_diversity=(
            sum(race_brain_diversities) / len(race_brain_diversities)
            if race_brain_diversities
            else 0.0
        ),
    )


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _candidate_ids(payload: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    values = payload.get("candidates")
    if isinstance(values, list):
        for value in values:
            if candidate_id := _text(_object(value), "candidate_id", "id"):
                identifiers.add(candidate_id)
    raw_ids = payload.get("candidate_ids")
    if isinstance(raw_ids, list):
        identifiers.update(value for value in raw_ids if isinstance(value, str) and value)
    return identifiers


def _command_id(payload: dict[str, Any]) -> str | None:
    direct = _text(payload, "command_id")
    if direct:
        return direct
    for key in ("lineage", "command"):
        if nested := _text(_object(payload.get(key)), "command_id", "id"):
            return nested
    return None


def _dispatched_command_ids(events: Sequence[StoredEvent]) -> set[str]:
    identifiers: set[str] = set()
    for event in events:
        if event.event_type != "command_lifecycle":
            continue
        payload = event.payload
        if payload.get("status") != "dispatched":
            continue
        if command_id := _command_id(payload):
            identifiers.add(command_id)
    return identifiers


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
