"""Metrics derived from one episode's immutable runtime events."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rtscortex.contracts import EpisodeResult
from rtscortex.memory import StoredEvent


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


@dataclass(frozen=True)
class MetricSamples:
    planner_latency_ms: tuple[float, ...] = ()
    reflex_latency_ms: tuple[float, ...] = ()
    tick_latency_ms: tuple[float, ...] = ()
    plan_age_game_loops: tuple[float, ...] = ()
    plan_accept_gap_game_loops: tuple[float, ...] = ()
    confirmation_latency_game_loops: tuple[float, ...] = ()


@dataclass(frozen=True)
class _PlannerProposalAudit:
    module_results: int = 0
    parsed_results: int = 0
    audited_results: int = 0
    build_proposals: int = 0
    production_proposals: int = 0
    unsafe_attacks: int = 0
    builder_attacks: int = 0
    friendly_target_attacks: int = 0
    unsafe_rejected_before_dispatch: int = 0
    unsafe_dispatched: int = 0


@dataclass(frozen=True)
class ExecutionMetrics:
    """Semantic execution outcomes, including legacy journal reconstruction."""

    execution_reports: int
    dispatched_commands: int
    known_lifecycle_commands: int
    terminal_commands_reported: int
    missing_terminal_reports: int
    duplicate_terminal_reports: int
    unexpected_terminal_reports: int
    duplicate_dispatches: int
    terminal_report_coverage: float
    failure_reports: int
    explicitly_classified_failures: int
    failure_classification_coverage: float
    legacy_successes: int
    legacy_action_success_rate: float
    control_noops: int
    control_noop_successes: int
    transport_noop_primitives: int
    meaningful_commands: int
    meaningful_successes: int
    meaningful_failures: int
    meaningful_cancelled: int
    meaningful_unconfirmed: int
    meaningful_action_success_rate: float
    completed_meaningful_commands: int
    completed_execution_success_rate: float
    terminal_backlog_rate: float
    status_counts: dict[str, int]
    failure_by_stage: dict[str, int]
    failure_by_code: dict[str, int]
    failure_by_action: dict[str, int]
    failure_by_actor: dict[str, int]
    command_by_action_actor: dict[str, int]
    failure_by_action_stage_code: dict[str, int]
    build_funnel: dict[str, int]
    build_effect_confirmed_rate: float
    build_effect_timeouts: int
    build_effect_timeout_rate: float
    build_pre_dispatch_rejections: int
    build_pre_dispatch_rejection_rate: float
    production_funnel: dict[str, int]
    production_effect_confirmed_rate: float
    production_provenance_coverage: float
    production_effect_timeouts: int
    production_timeout_rate: float
    production_metrics_applicable: bool
    production_by_action: dict[str, int]
    production_by_producer: dict[str, int]
    confirmation_latency_game_loops_p50: float
    confirmation_latency_game_loops_p95: float
    confirmation_latency_game_loops_samples: int
    planner_module_results: int
    planner_proposal_audited_results: int
    planner_proposal_audit_complete: bool
    planner_unsafe_attack_proposals: int
    planner_builder_attack_proposals: int
    planner_friendly_target_attack_proposals: int
    planner_unsafe_attack_rejected_before_dispatch: int
    planner_unsafe_attack_dispatched: int
    builder_attack_commands: int
    friendly_target_attacks: int
    planner_noop_proposals: int
    generic_translation_failures: int
    upstream_placement_rejections: int
    unattributed_primitives: int
    candidate_outside_pysc2_dispatches: int
    orchestration_573_terminal_reports: int
    decision_count: int
    fallback_decisions: int
    planner_pending_decisions: int
    idle_reason_counts: dict[str, int]
    unique_validation_rejected_command_ids: int


@dataclass(frozen=True)
class EpisodeMetrics:
    """Serializable episode-level metrics plus raw latency samples for aggregation."""

    action_attempts: int
    action_successes: int
    action_success_rate: float
    execution: ExecutionMetrics
    illegal_actions: int
    illegal_action_rate: float
    planner_latency_ms_p50: float
    planner_latency_ms_p95: float
    reflex_latency_ms_p50: float
    reflex_latency_ms_p95: float
    tick_latency_ms_p50: float
    tick_latency_ms_p95: float
    model_requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_cost_usd: float
    reflex_preemptions: int
    plans_accepted: int
    plan_revisions: int
    plan_revision_rate: float
    plan_age_game_loops_p50: float
    plan_age_game_loops_p95: float
    plan_accept_gap_game_loops_p50: float
    plan_accept_gap_game_loops_p95: float
    plan_accept_gap_samples: int
    episode_duration_seconds: float
    failure_reason: str | None
    samples: MetricSamples = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "deprecated_fields": [
                "action_success_rate",
                "execution.legacy_action_success_rate",
            ],
            "action_attempts": self.action_attempts,
            "action_successes": self.action_successes,
            "action_success_rate": self.action_success_rate,
            "execution": {
                "execution_reports": self.execution.execution_reports,
                "dispatched_commands": self.execution.dispatched_commands,
                "known_lifecycle_commands": self.execution.known_lifecycle_commands,
                "terminal_commands_reported": (self.execution.terminal_commands_reported),
                "missing_terminal_reports": self.execution.missing_terminal_reports,
                "duplicate_terminal_reports": self.execution.duplicate_terminal_reports,
                "unexpected_terminal_reports": (self.execution.unexpected_terminal_reports),
                "duplicate_dispatches": self.execution.duplicate_dispatches,
                "terminal_report_coverage": self.execution.terminal_report_coverage,
                "failure_reports": self.execution.failure_reports,
                "explicitly_classified_failures": (self.execution.explicitly_classified_failures),
                "failure_classification_coverage": (self.execution.failure_classification_coverage),
                "legacy_successes": self.execution.legacy_successes,
                "legacy_action_success_rate": self.execution.legacy_action_success_rate,
                "control_noops": self.execution.control_noops,
                "control_noop_successes": self.execution.control_noop_successes,
                "transport_noop_primitives": (self.execution.transport_noop_primitives),
                "meaningful_commands": self.execution.meaningful_commands,
                "meaningful_successes": self.execution.meaningful_successes,
                "meaningful_failures": self.execution.meaningful_failures,
                "meaningful_cancelled": self.execution.meaningful_cancelled,
                "meaningful_unconfirmed": self.execution.meaningful_unconfirmed,
                "meaningful_action_success_rate": (self.execution.meaningful_action_success_rate),
                "completed_meaningful_commands": (self.execution.completed_meaningful_commands),
                "completed_execution_success_rate": (
                    self.execution.completed_execution_success_rate
                ),
                "terminal_backlog_rate": self.execution.terminal_backlog_rate,
                "status_counts": self.execution.status_counts,
                "failure_by_stage": self.execution.failure_by_stage,
                "failure_by_code": self.execution.failure_by_code,
                "failure_by_action": self.execution.failure_by_action,
                "failure_by_actor": self.execution.failure_by_actor,
                "command_by_action_actor": self.execution.command_by_action_actor,
                "failure_by_action_stage_code": (self.execution.failure_by_action_stage_code),
                "build_funnel": self.execution.build_funnel,
                "build_effect_confirmed_rate": (self.execution.build_effect_confirmed_rate),
                "build_effect_timeouts": self.execution.build_effect_timeouts,
                "build_effect_timeout_rate": self.execution.build_effect_timeout_rate,
                "build_pre_dispatch_rejections": (self.execution.build_pre_dispatch_rejections),
                "build_pre_dispatch_rejection_rate": (
                    self.execution.build_pre_dispatch_rejection_rate
                ),
                "production_funnel": self.execution.production_funnel,
                "production_effect_confirmed_rate": (
                    self.execution.production_effect_confirmed_rate
                ),
                "production_provenance_coverage": (self.execution.production_provenance_coverage),
                "production_effect_timeouts": self.execution.production_effect_timeouts,
                "production_timeout_rate": self.execution.production_timeout_rate,
                "production_metrics_applicable": (self.execution.production_metrics_applicable),
                "production_by_action": self.execution.production_by_action,
                "production_by_producer": self.execution.production_by_producer,
                "confirmation_latency_game_loops_p50": (
                    self.execution.confirmation_latency_game_loops_p50
                ),
                "confirmation_latency_game_loops_p95": (
                    self.execution.confirmation_latency_game_loops_p95
                ),
                "confirmation_latency_game_loops_samples": (
                    self.execution.confirmation_latency_game_loops_samples
                ),
                "planner_module_results": self.execution.planner_module_results,
                "planner_proposal_audited_results": (
                    self.execution.planner_proposal_audited_results
                ),
                "planner_proposal_audit_complete": (self.execution.planner_proposal_audit_complete),
                "planner_unsafe_attack_proposals": (self.execution.planner_unsafe_attack_proposals),
                "planner_builder_attack_proposals": (
                    self.execution.planner_builder_attack_proposals
                ),
                "planner_friendly_target_attack_proposals": (
                    self.execution.planner_friendly_target_attack_proposals
                ),
                "planner_unsafe_attack_rejected_before_dispatch": (
                    self.execution.planner_unsafe_attack_rejected_before_dispatch
                ),
                "planner_unsafe_attack_dispatched": (
                    self.execution.planner_unsafe_attack_dispatched
                ),
                "builder_attack_commands": self.execution.builder_attack_commands,
                "friendly_target_attacks": self.execution.friendly_target_attacks,
                "planner_noop_proposals": self.execution.planner_noop_proposals,
                "generic_translation_failures": (self.execution.generic_translation_failures),
                "upstream_placement_rejections": (self.execution.upstream_placement_rejections),
                "unattributed_primitives": self.execution.unattributed_primitives,
                "candidate_outside_pysc2_dispatches": (
                    self.execution.candidate_outside_pysc2_dispatches
                ),
                "orchestration_573_terminal_reports": (
                    self.execution.orchestration_573_terminal_reports
                ),
                "decision_count": self.execution.decision_count,
                "fallback_decisions": self.execution.fallback_decisions,
                "planner_pending_decisions": self.execution.planner_pending_decisions,
                "idle_reason_counts": self.execution.idle_reason_counts,
                "unique_validation_rejected_command_ids": (
                    self.execution.unique_validation_rejected_command_ids
                ),
            },
            "illegal_actions": self.illegal_actions,
            "illegal_action_rate": self.illegal_action_rate,
            "planner_latency_ms_p50": self.planner_latency_ms_p50,
            "planner_latency_ms_p95": self.planner_latency_ms_p95,
            "reflex_latency_ms_p50": self.reflex_latency_ms_p50,
            "reflex_latency_ms_p95": self.reflex_latency_ms_p95,
            "tick_latency_ms_p50": self.tick_latency_ms_p50,
            "tick_latency_ms_p95": self.tick_latency_ms_p95,
            "model_requests": self.model_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_cost_usd": self.model_cost_usd,
            "reflex_preemptions": self.reflex_preemptions,
            "plans_accepted": self.plans_accepted,
            "plan_revisions": self.plan_revisions,
            "plan_revision_rate": self.plan_revision_rate,
            "plan_age_game_loops_p50": self.plan_age_game_loops_p50,
            "plan_age_game_loops_p95": self.plan_age_game_loops_p95,
            "plan_accept_gap_game_loops_p50": self.plan_accept_gap_game_loops_p50,
            "plan_accept_gap_game_loops_p95": self.plan_accept_gap_game_loops_p95,
            "plan_accept_gap_samples": self.plan_accept_gap_samples,
            "episode_duration_seconds": self.episode_duration_seconds,
            "failure_reason": self.failure_reason,
        }


def compute_episode_metrics(
    events: list[StoredEvent],
    result: EpisodeResult,
    *,
    prompt_cost_per_million_tokens: float = 0.0,
    completion_cost_per_million_tokens: float = 0.0,
) -> EpisodeMetrics:
    executions = [event for event in events if event.event_type == "execution"]
    action_attempts = len(executions)
    action_successes = sum(bool(event.payload.get("success")) for event in executions)
    execution_metrics = compute_execution_metrics(events)

    decisions = [event for event in events if event.event_type == "decision"]
    illegal_actions = sum(
        len(event.payload.get("batch", {}).get("rejected_commands", [])) for event in decisions
    )
    candidate_actions = action_attempts + illegal_actions

    planner_latencies = [
        float(event.payload["latency_ms"])
        for event in events
        if event.event_type in {"planner_cycle", "macro_plan_accepted", "macro_plan_rejected"}
        and "latency_ms" in event.payload
    ]
    reflex_latencies = [
        float(event.payload["reflex_latency_ms"])
        for event in decisions
        if "reflex_latency_ms" in event.payload
    ]
    tick_latencies = [
        float(event.payload["tick_latency_ms"])
        for event in decisions
        if "tick_latency_ms" in event.payload
    ]

    model_events = [
        event
        for event in events
        if (event.event_type == "module_result" and event.payload.get("model_call") is True)
        or event.event_type in {"macro_plan_accepted", "macro_plan_rejected"}
    ]
    prompt_tokens = sum(_usage_value(event, "prompt_tokens") for event in model_events)
    completion_tokens = sum(_usage_value(event, "completion_tokens") for event in model_events)
    total_tokens = sum(_usage_total(event) for event in model_events)
    model_cost_usd = (
        prompt_tokens * prompt_cost_per_million_tokens
        + completion_tokens * completion_cost_per_million_tokens
    ) / 1_000_000

    reflex_preemptions = sum(len(event.payload.get("preemptions", [])) for event in decisions)
    plan_events = [
        event for event in events if event.event_type in {"plan_accepted", "macro_plan_accepted"}
    ]
    plan_revisions = sum(event.payload.get("is_revision") is True for event in plan_events)
    revision_opportunities = max(0, len(plan_events) - 1)
    plan_ages = [
        float(event.payload["plan_age_game_loops"])
        for event in plan_events
        if isinstance(event.payload.get("plan_age_game_loops"), int | float)
    ]
    accepted_game_loops = [
        float(event.payload["accepted_game_loop"])
        for event in plan_events
        if isinstance(event.payload.get("accepted_game_loop"), int | float)
    ]
    plan_accept_gaps = [
        current - previous
        for previous, current in zip(
            accepted_game_loops,
            accepted_game_loops[1:],
            strict=False,
        )
        if current >= previous
    ]

    return EpisodeMetrics(
        action_attempts=action_attempts,
        action_successes=action_successes,
        action_success_rate=(action_successes / action_attempts if action_attempts else 0.0),
        execution=execution_metrics,
        illegal_actions=illegal_actions,
        illegal_action_rate=(illegal_actions / candidate_actions if candidate_actions else 0.0),
        planner_latency_ms_p50=_percentile(planner_latencies, 0.50),
        planner_latency_ms_p95=_percentile(planner_latencies, 0.95),
        reflex_latency_ms_p50=_percentile(reflex_latencies, 0.50),
        reflex_latency_ms_p95=_percentile(reflex_latencies, 0.95),
        tick_latency_ms_p50=_percentile(tick_latencies, 0.50),
        tick_latency_ms_p95=_percentile(tick_latencies, 0.95),
        model_requests=len(model_events),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model_cost_usd=model_cost_usd,
        reflex_preemptions=reflex_preemptions,
        plans_accepted=len(plan_events),
        plan_revisions=plan_revisions,
        plan_revision_rate=(
            plan_revisions / revision_opportunities if revision_opportunities else 0.0
        ),
        plan_age_game_loops_p50=_percentile(plan_ages, 0.50),
        plan_age_game_loops_p95=_percentile(plan_ages, 0.95),
        plan_accept_gap_game_loops_p50=_percentile(plan_accept_gaps, 0.50),
        plan_accept_gap_game_loops_p95=_percentile(plan_accept_gaps, 0.95),
        plan_accept_gap_samples=len(plan_accept_gaps),
        episode_duration_seconds=_episode_duration(events),
        failure_reason=result.failure_reason,
        samples=MetricSamples(
            planner_latency_ms=tuple(planner_latencies),
            reflex_latency_ms=tuple(reflex_latencies),
            tick_latency_ms=tuple(tick_latencies),
            plan_age_game_loops=tuple(plan_ages),
            plan_accept_gap_game_loops=tuple(plan_accept_gaps),
            confirmation_latency_game_loops=tuple(_production_confirmation_latencies(events)),
        ),
    )


def aggregate_episode_metrics(metrics: list[EpisodeMetrics]) -> dict[str, Any]:
    action_attempts = sum(item.action_attempts for item in metrics)
    action_successes = sum(item.action_successes for item in metrics)
    illegal_actions = sum(item.illegal_actions for item in metrics)
    candidate_actions = action_attempts + illegal_actions
    plans_accepted = sum(item.plans_accepted for item in metrics)
    plan_revisions = sum(item.plan_revisions for item in metrics)
    revision_opportunities = sum(max(0, item.plans_accepted - 1) for item in metrics)
    planner_latencies = [value for item in metrics for value in item.samples.planner_latency_ms]
    reflex_latencies = [value for item in metrics for value in item.samples.reflex_latency_ms]
    tick_latencies = [value for item in metrics for value in item.samples.tick_latency_ms]
    plan_ages = [value for item in metrics for value in item.samples.plan_age_game_loops]
    plan_accept_gaps = [
        value for item in metrics for value in item.samples.plan_accept_gap_game_loops
    ]
    confirmation_latencies = [
        value for item in metrics for value in item.samples.confirmation_latency_game_loops
    ]
    failure_reasons: dict[str, int] = {}
    for item in metrics:
        if item.failure_reason is not None:
            failure_reasons[item.failure_reason] = failure_reasons.get(item.failure_reason, 0) + 1

    execution_reports = sum(item.execution.execution_reports for item in metrics)
    dispatched_commands = sum(item.execution.dispatched_commands for item in metrics)
    known_lifecycle_commands = sum(item.execution.known_lifecycle_commands for item in metrics)
    terminal_commands_reported = sum(item.execution.terminal_commands_reported for item in metrics)
    missing_terminal_reports = sum(item.execution.missing_terminal_reports for item in metrics)
    duplicate_terminal_reports = sum(item.execution.duplicate_terminal_reports for item in metrics)
    unexpected_terminal_reports = sum(
        item.execution.unexpected_terminal_reports for item in metrics
    )
    duplicate_dispatches = sum(item.execution.duplicate_dispatches for item in metrics)
    failure_reports = sum(item.execution.failure_reports for item in metrics)
    explicitly_classified_failures = sum(
        item.execution.explicitly_classified_failures for item in metrics
    )
    legacy_successes = sum(item.execution.legacy_successes for item in metrics)
    control_noops = sum(item.execution.control_noops for item in metrics)
    control_noop_successes = sum(item.execution.control_noop_successes for item in metrics)
    transport_noop_primitives = sum(item.execution.transport_noop_primitives for item in metrics)
    meaningful_commands = sum(item.execution.meaningful_commands for item in metrics)
    meaningful_successes = sum(item.execution.meaningful_successes for item in metrics)
    meaningful_failures = sum(item.execution.meaningful_failures for item in metrics)
    meaningful_cancelled = sum(item.execution.meaningful_cancelled for item in metrics)
    meaningful_unconfirmed = sum(item.execution.meaningful_unconfirmed for item in metrics)
    completed_meaningful = meaningful_successes + meaningful_failures
    terminal_backlog = meaningful_cancelled + meaningful_unconfirmed
    build_funnel = _merge_counts([item.execution.build_funnel for item in metrics])
    build_effect_timeouts = sum(item.execution.build_effect_timeouts for item in metrics)
    build_pre_dispatch_rejections = sum(
        item.execution.build_pre_dispatch_rejections for item in metrics
    )
    production_funnel = _merge_counts([item.execution.production_funnel for item in metrics])
    production_effect_timeouts = sum(item.execution.production_effect_timeouts for item in metrics)
    production_by_action = _merge_counts([item.execution.production_by_action for item in metrics])
    production_by_producer = _merge_counts(
        [item.execution.production_by_producer for item in metrics]
    )
    planner_module_results = sum(item.execution.planner_module_results for item in metrics)
    planner_proposal_audited_results = sum(
        item.execution.planner_proposal_audited_results for item in metrics
    )
    pysc2_accepted_builds = build_funnel.get("pysc2_accepted", 0)
    proposed_builds = build_funnel.get("proposed", 0)
    pysc2_accepted_production = production_funnel.get("pysc2_accepted", 0)
    production_metrics = [
        item.execution
        for item in metrics
        if item.execution.production_funnel.get("proposed", 0) > 0
        or item.execution.production_funnel.get("pysc2_accepted", 0) > 0
    ]
    execution = {
        "execution_reports": execution_reports,
        "dispatched_commands": dispatched_commands,
        "known_lifecycle_commands": known_lifecycle_commands,
        "terminal_commands_reported": terminal_commands_reported,
        "missing_terminal_reports": missing_terminal_reports,
        "duplicate_terminal_reports": duplicate_terminal_reports,
        "unexpected_terminal_reports": unexpected_terminal_reports,
        "duplicate_dispatches": duplicate_dispatches,
        "terminal_report_coverage": (
            terminal_commands_reported / dispatched_commands if dispatched_commands else 1.0
        ),
        "failure_reports": failure_reports,
        "explicitly_classified_failures": explicitly_classified_failures,
        "failure_classification_coverage": (
            explicitly_classified_failures / failure_reports if failure_reports else 1.0
        ),
        "legacy_successes": legacy_successes,
        "legacy_action_success_rate": (
            legacy_successes / execution_reports if execution_reports else 0.0
        ),
        "control_noops": control_noops,
        "control_noop_successes": control_noop_successes,
        "transport_noop_primitives": transport_noop_primitives,
        "meaningful_commands": meaningful_commands,
        "meaningful_successes": meaningful_successes,
        "meaningful_failures": meaningful_failures,
        "meaningful_cancelled": meaningful_cancelled,
        "meaningful_unconfirmed": meaningful_unconfirmed,
        "meaningful_action_success_rate": (
            meaningful_successes / meaningful_commands if meaningful_commands else 0.0
        ),
        "completed_meaningful_commands": completed_meaningful,
        "completed_execution_success_rate": (
            meaningful_successes / completed_meaningful if completed_meaningful else 0.0
        ),
        "terminal_backlog_rate": (
            terminal_backlog / meaningful_commands if meaningful_commands else 0.0
        ),
        "status_counts": _merge_counts([item.execution.status_counts for item in metrics]),
        "failure_by_stage": _merge_counts([item.execution.failure_by_stage for item in metrics]),
        "failure_by_code": _merge_counts([item.execution.failure_by_code for item in metrics]),
        "failure_by_action": _merge_counts([item.execution.failure_by_action for item in metrics]),
        "failure_by_actor": _merge_counts([item.execution.failure_by_actor for item in metrics]),
        "command_by_action_actor": _merge_counts(
            [item.execution.command_by_action_actor for item in metrics]
        ),
        "failure_by_action_stage_code": _merge_counts(
            [item.execution.failure_by_action_stage_code for item in metrics]
        ),
        "build_funnel": build_funnel,
        "build_effect_confirmed_rate": (
            build_funnel.get("effect_confirmed", 0) / pysc2_accepted_builds
            if pysc2_accepted_builds
            else 0.0
        ),
        "build_effect_timeouts": build_effect_timeouts,
        "build_effect_timeout_rate": (
            build_effect_timeouts / pysc2_accepted_builds if pysc2_accepted_builds else 0.0
        ),
        "build_pre_dispatch_rejections": build_pre_dispatch_rejections,
        "build_pre_dispatch_rejection_rate": (
            build_pre_dispatch_rejections / proposed_builds if proposed_builds else 0.0
        ),
        "production_funnel": production_funnel,
        "production_effect_confirmed_rate": (
            production_funnel.get("effect_confirmed", 0) / pysc2_accepted_production
            if pysc2_accepted_production
            else 0.0
        ),
        "production_provenance_coverage": (
            sum(
                item.execution.production_provenance_coverage
                * item.execution.production_funnel.get("pysc2_accepted", 0)
                for item in metrics
            )
            / pysc2_accepted_production
            if pysc2_accepted_production
            else 0.0
        ),
        "production_effect_timeouts": production_effect_timeouts,
        "production_timeout_rate": (
            production_effect_timeouts / pysc2_accepted_production
            if pysc2_accepted_production
            else 0.0
        ),
        "production_metrics_applicable": (
            bool(production_metrics)
            and all(item.production_metrics_applicable for item in production_metrics)
        ),
        "production_by_action": production_by_action,
        "production_by_producer": production_by_producer,
        "confirmation_latency_game_loops_p50": _percentile(confirmation_latencies, 0.50),
        "confirmation_latency_game_loops_p95": _percentile(confirmation_latencies, 0.95),
        "confirmation_latency_game_loops_samples": len(confirmation_latencies),
        "planner_module_results": planner_module_results,
        "planner_proposal_audited_results": planner_proposal_audited_results,
        "planner_proposal_audit_complete": (
            planner_proposal_audited_results == planner_module_results
        ),
        "planner_unsafe_attack_proposals": sum(
            item.execution.planner_unsafe_attack_proposals for item in metrics
        ),
        "planner_builder_attack_proposals": sum(
            item.execution.planner_builder_attack_proposals for item in metrics
        ),
        "planner_friendly_target_attack_proposals": sum(
            item.execution.planner_friendly_target_attack_proposals for item in metrics
        ),
        "planner_unsafe_attack_rejected_before_dispatch": sum(
            item.execution.planner_unsafe_attack_rejected_before_dispatch for item in metrics
        ),
        "planner_unsafe_attack_dispatched": sum(
            item.execution.planner_unsafe_attack_dispatched for item in metrics
        ),
        "builder_attack_commands": sum(item.execution.builder_attack_commands for item in metrics),
        "friendly_target_attacks": sum(item.execution.friendly_target_attacks for item in metrics),
        "planner_noop_proposals": sum(item.execution.planner_noop_proposals for item in metrics),
        "generic_translation_failures": sum(
            item.execution.generic_translation_failures for item in metrics
        ),
        "upstream_placement_rejections": sum(
            item.execution.upstream_placement_rejections for item in metrics
        ),
        "unattributed_primitives": sum(item.execution.unattributed_primitives for item in metrics),
        "candidate_outside_pysc2_dispatches": sum(
            item.execution.candidate_outside_pysc2_dispatches for item in metrics
        ),
        "orchestration_573_terminal_reports": sum(
            item.execution.orchestration_573_terminal_reports for item in metrics
        ),
        "decision_count": sum(item.execution.decision_count for item in metrics),
        "fallback_decisions": sum(item.execution.fallback_decisions for item in metrics),
        "planner_pending_decisions": sum(
            item.execution.planner_pending_decisions for item in metrics
        ),
        "idle_reason_counts": _merge_counts(
            [item.execution.idle_reason_counts for item in metrics]
        ),
        "unique_validation_rejected_command_ids": sum(
            item.execution.unique_validation_rejected_command_ids for item in metrics
        ),
    }

    return {
        "deprecated_fields": [
            "action_success_rate",
            "execution.legacy_action_success_rate",
        ],
        "action_attempts": action_attempts,
        "action_success_rate": (action_successes / action_attempts if action_attempts else 0.0),
        "illegal_actions": illegal_actions,
        "illegal_action_rate": illegal_actions / candidate_actions if candidate_actions else 0.0,
        "planner_latency_ms_p50": _percentile(planner_latencies, 0.50),
        "planner_latency_ms_p95": _percentile(planner_latencies, 0.95),
        "reflex_latency_ms_p50": _percentile(reflex_latencies, 0.50),
        "reflex_latency_ms_p95": _percentile(reflex_latencies, 0.95),
        "tick_latency_ms_p50": _percentile(tick_latencies, 0.50),
        "tick_latency_ms_p95": _percentile(tick_latencies, 0.95),
        "model_requests": sum(item.model_requests for item in metrics),
        "prompt_tokens": sum(item.prompt_tokens for item in metrics),
        "completion_tokens": sum(item.completion_tokens for item in metrics),
        "total_tokens": sum(item.total_tokens for item in metrics),
        "model_cost_usd": sum(item.model_cost_usd for item in metrics),
        "reflex_preemptions": sum(item.reflex_preemptions for item in metrics),
        "plans_accepted": plans_accepted,
        "plan_revisions": plan_revisions,
        "plan_revision_rate": (
            plan_revisions / revision_opportunities if revision_opportunities else 0.0
        ),
        "plan_age_game_loops_p50": _percentile(plan_ages, 0.50),
        "plan_age_game_loops_p95": _percentile(plan_ages, 0.95),
        "plan_accept_gap_game_loops_p50": _percentile(plan_accept_gaps, 0.50),
        "plan_accept_gap_game_loops_p95": _percentile(plan_accept_gaps, 0.95),
        "plan_accept_gap_samples": len(plan_accept_gaps),
        "mean_episode_duration_seconds": (
            sum(item.episode_duration_seconds for item in metrics) / len(metrics)
            if metrics
            else 0.0
        ),
        "failure_reasons": failure_reasons,
        "execution": execution,
    }


def compute_execution_metrics(events: list[StoredEvent]) -> ExecutionMetrics:
    """Classify v1.1 reports and reconstruct command semantics for v1.0 journals."""

    decisions = [event for event in events if event.event_type == "decision"]
    executions = [event for event in events if event.event_type == "execution"]
    commands = _decision_commands(decisions)
    execution_command_ids = [str(event.payload.get("command_id", "")) for event in executions]
    (
        dispatch_event_ids,
        dispatched_ids,
        dispatched_commands,
        lifecycle_command_ids,
        legacy_dispatch_fallback,
    ) = _dispatched_commands(
        events,
        commands=commands,
        execution_command_ids=execution_command_ids,
    )
    own_tags = _own_tags(events)
    builder_attack_command_ids = {
        command_id
        for command_id, command in dispatched_commands.items()
        if command_id in dispatched_ids
        and command.get("name") == "Attack_Unit"
        and _is_builder_actor(str(command.get("actor", "")))
    }
    friendly_target_attack_ids = {
        command_id
        for command_id, command in dispatched_commands.items()
        if command_id in dispatched_ids
        and command.get("name") == "Attack_Unit"
        and _attack_target_is_friendly(command, own_tags)
    }
    proposal_audit = _audit_planner_proposals(events, dispatched_ids=dispatched_ids)
    status_counts: dict[str, int] = {}
    failure_by_stage: dict[str, int] = {}
    failure_by_code: dict[str, int] = {}
    failure_by_action: dict[str, int] = {}
    failure_by_actor: dict[str, int] = {}
    command_by_action_actor: dict[str, int] = {}
    failure_by_action_stage_code: dict[str, int] = {}
    control_noops = 0
    control_noop_successes = 0
    meaningful_successes = 0
    meaningful_failures = 0
    meaningful_cancelled = 0
    meaningful_unconfirmed = 0
    build_translator_accepted_ids: set[str] = set()
    build_pysc2_accepted_ids: set[str] = set()
    build_effect_confirmed_ids: set[str] = set()
    build_effect_timeout_ids: set[str] = set()
    build_pre_dispatch_rejection_ids: set[str] = set()
    production_translator_accepted_ids: set[str] = set()
    production_pysc2_accepted_ids: set[str] = set()
    production_order_confirmed_ids: set[str] = set()
    production_unit_confirmed_ids: set[str] = set()
    production_acceptance_only_ids: set[str] = set()
    production_provenance_complete_ids: set[str] = set()
    production_effect_timeout_ids: set[str] = set()
    production_confirmation_latencies: dict[str, float] = {}
    production_report_protocols: set[str] = set()
    production_action_by_command: dict[str, str] = {}
    production_producer_by_command: dict[str, str] = {}
    failure_reports = 0
    explicitly_classified_failures = 0
    generic_translation_failures = 0
    upstream_placement_rejections = 0
    orchestration_573_terminal_reports = 0

    for event in executions:
        payload = event.payload
        command_id = str(payload.get("command_id", ""))
        command = commands.get(command_id, {})
        action_name = str(payload.get("action_name") or command.get("name") or "unknown")
        actor = str(payload.get("actor") or command.get("actor") or "unknown")
        status = _execution_status(payload)
        _increment(status_counts, status)
        _increment(command_by_action_actor, f"{action_name} / {actor}")
        is_noop = _is_noop(action_name, payload.get("pysc2_function"), command_id)
        if is_noop:
            control_noops += 1
            if status == "succeeded":
                control_noop_successes += 1
        elif status == "succeeded":
            meaningful_successes += 1
        elif status == "failed":
            meaningful_failures += 1
        elif status == "cancelled":
            meaningful_cancelled += 1
        else:
            meaningful_unconfirmed += 1

        stage = str(payload.get("execution_stage") or _infer_execution_stage(payload, status))
        code = str(payload.get("failure_code") or _infer_failure_code(payload, status))
        if status != "succeeded":
            failure_reports += 1
            if payload.get("execution_stage") and payload.get("failure_code"):
                explicitly_classified_failures += 1
            _increment(failure_by_stage, stage)
            _increment(failure_by_code, code)
            _increment(failure_by_action, action_name)
            _increment(failure_by_actor, actor)
            _increment(
                failure_by_action_stage_code,
                f"{action_name} / {stage} / {code}",
            )
            if action_name == "Attack_Unit" and code == "friendly_target":
                friendly_target_attack_ids.add(command_id)
            if stage == "translation" and code == "translator_rejected":
                generic_translation_failures += 1
            if action_name.lower().startswith("build_") and stage == "translation":
                upstream_placement_rejections += 1

        primitive_trace = payload.get("primitive_trace")
        if status in {"succeeded", "failed"} and isinstance(primitive_trace, list):
            final_primitive = primitive_trace[-1] if primitive_trace else None
            if (
                isinstance(final_primitive, dict)
                and final_primitive.get("origin") == "orchestration"
                and (
                    final_primitive.get("requested_function_id") == 573
                    or final_primitive.get("emitted_function_id") == 573
                )
            ):
                orchestration_573_terminal_reports += 1

        if action_name.lower().startswith("build_"):
            if status != "succeeded" and stage == "pre_dispatch":
                build_pre_dispatch_rejection_ids.add(command_id)
            if code in {
                "effect_timeout",
                "no_build_order_observed",
                "worker_order_replaced",
                "target_not_created",
                "builder_not_observable",
            }:
                build_effect_timeout_ids.add(command_id)
            final_translator = _final_translator_primitive(payload)
            if stage in {"pysc2_acceptance", "effect_verification"} or (
                stage == "episode_end" and final_translator is not None
            ):
                build_translator_accepted_ids.add(command_id)
            if (final_translator is not None and final_translator.get("accepted") is True) or (
                final_translator is None and stage == "effect_verification"
            ):
                build_pysc2_accepted_ids.add(command_id)
            evidence = payload.get("effect_evidence")
            confirmed_tag = (
                (evidence.get("new_structure_tag") or evidence.get("observed_structure_tag"))
                if isinstance(evidence, dict)
                else None
            )
            if status == "succeeded" and (stage == "effect_verification" or confirmed_tag):
                build_effect_confirmed_ids.add(command_id)

        if _is_production_action(action_name):
            production_action_by_command[command_id] = action_name
            production_report_protocols.add(str(payload.get("protocol_version") or "legacy"))
            final_translator = _final_translator_primitive(payload)
            if stage in {"pysc2_acceptance", "effect_verification"} or (
                stage == "episode_end" and final_translator is not None
            ):
                production_translator_accepted_ids.add(command_id)
            final_translator_accepted = (
                final_translator is not None and final_translator.get("accepted") is True
            )
            if (
                (
                    stage == "effect_verification"
                    and (final_translator is None or final_translator_accepted)
                )
                or (stage == "episode_end" and final_translator_accepted)
                or (
                    stage == "pysc2_acceptance"
                    and status == "succeeded"
                    and (final_translator is None or final_translator_accepted)
                )
            ):
                production_pysc2_accepted_ids.add(command_id)

            evidence = payload.get("effect_evidence")
            production_evidence = (
                evidence
                if isinstance(evidence, dict) and evidence.get("effect_kind") == "production"
                else None
            )
            producer = _production_producer_label(production_evidence, actor)
            if producer is not None:
                production_producer_by_command[command_id] = producer
            if production_evidence is not None and _production_provenance_complete(
                production_evidence
            ):
                production_provenance_complete_ids.add(command_id)
            confirmation_kind = (
                production_evidence.get("confirmation_kind")
                if production_evidence is not None
                else None
            )
            if (
                command_id in production_pysc2_accepted_ids
                and status == "succeeded"
                and confirmation_kind in {"producer_order", "producer_morph"}
            ):
                production_order_confirmed_ids.add(command_id)
            elif (
                command_id in production_pysc2_accepted_ids
                and status == "succeeded"
                and confirmation_kind == "new_unit"
            ):
                production_unit_confirmed_ids.add(command_id)

            if (
                _is_train_action(action_name)
                and status == "succeeded"
                and stage == "pysc2_acceptance"
                and production_evidence is None
            ):
                production_acceptance_only_ids.add(command_id)

            if code in {
                "producer_not_observable",
                "no_production_order_observed",
                "production_order_replaced",
            } or (
                _is_train_action(action_name) and code in {"effect_timeout", "target_not_created"}
            ):
                production_effect_timeout_ids.add(command_id)

            if (
                production_evidence is not None
                and confirmation_kind in {"producer_order", "producer_morph", "new_unit"}
                and command_id in production_pysc2_accepted_ids
                and status == "succeeded"
            ):
                latency = _confirmation_latency(production_evidence)
                if latency is not None:
                    production_confirmation_latencies[command_id] = latency

    meaningful_commands = (
        meaningful_successes + meaningful_failures + meaningful_cancelled + meaningful_unconfirmed
    )
    completed_meaningful = meaningful_successes + meaningful_failures
    build_selected_ids = _build_selected_ids(commands)
    build_proposed_ids = build_selected_ids | _build_candidate_ids(decisions)
    build_proposed_count = (
        proposal_audit.build_proposals
        if proposal_audit.parsed_results > 0
        else len(build_proposed_ids)
    )
    build_validated_ids = _build_validated_ids(decisions, fallback=build_selected_ids)
    production_selected_ids = _production_selected_ids(commands)
    production_proposed_ids = production_selected_ids | _production_candidate_ids(decisions)
    production_proposed_count = (
        proposal_audit.production_proposals
        if proposal_audit.parsed_results > 0
        else len(production_proposed_ids)
    )
    production_validated_ids = _production_validated_ids(
        decisions,
        fallback=production_selected_ids,
    )
    rejected_ids = _unique_rejected_command_ids(decisions)
    idle_reasons: dict[str, int] = {}
    fallback_decisions = 0
    planner_pending_decisions = 0
    for decision in decisions:
        batch = decision.payload.get("batch")
        if not isinstance(batch, dict):
            continue
        if batch.get("planner_pending") is True:
            planner_pending_decisions += 1
        idle_reason = batch.get("idle_reason")
        if isinstance(idle_reason, str):
            _increment(idle_reasons, idle_reason)
        selected = batch.get("commands")
        if (
            isinstance(selected, list)
            and selected
            and all(
                isinstance(command, dict) and command.get("source") == "fallback"
                for command in selected
            )
        ):
            fallback_decisions += 1

    execution_reports = len(executions)
    reported_ids = set(execution_command_ids)
    known_command_ids = set(lifecycle_command_ids)
    if legacy_dispatch_fallback:
        known_command_ids.update(reported_ids)
    unexpected_terminal_reports = sum(
        command_id not in known_command_ids for command_id in execution_command_ids
    )
    terminal_commands_reported = len(dispatched_ids & reported_ids)
    transport_noop_primitives = _episode_metric(events, "transport_noop_primitives")
    legacy_successes = sum(event.payload.get("success") is True for event in executions)
    terminal_backlog = meaningful_cancelled + meaningful_unconfirmed
    return ExecutionMetrics(
        execution_reports=execution_reports,
        dispatched_commands=len(dispatched_ids),
        known_lifecycle_commands=len(lifecycle_command_ids),
        terminal_commands_reported=terminal_commands_reported,
        missing_terminal_reports=len(dispatched_ids - reported_ids),
        duplicate_terminal_reports=len(execution_command_ids) - len(reported_ids),
        unexpected_terminal_reports=unexpected_terminal_reports,
        duplicate_dispatches=len(dispatch_event_ids) - len(set(dispatch_event_ids)),
        terminal_report_coverage=(
            terminal_commands_reported / len(dispatched_ids) if dispatched_ids else 1.0
        ),
        failure_reports=failure_reports,
        explicitly_classified_failures=explicitly_classified_failures,
        failure_classification_coverage=(
            explicitly_classified_failures / failure_reports if failure_reports else 1.0
        ),
        legacy_successes=legacy_successes,
        legacy_action_success_rate=(
            legacy_successes / execution_reports if execution_reports else 0.0
        ),
        control_noops=control_noops,
        control_noop_successes=control_noop_successes,
        transport_noop_primitives=transport_noop_primitives,
        meaningful_commands=meaningful_commands,
        meaningful_successes=meaningful_successes,
        meaningful_failures=meaningful_failures,
        meaningful_cancelled=meaningful_cancelled,
        meaningful_unconfirmed=meaningful_unconfirmed,
        meaningful_action_success_rate=(
            meaningful_successes / meaningful_commands if meaningful_commands else 0.0
        ),
        completed_meaningful_commands=completed_meaningful,
        completed_execution_success_rate=(
            meaningful_successes / completed_meaningful if completed_meaningful else 0.0
        ),
        terminal_backlog_rate=(
            terminal_backlog / meaningful_commands if meaningful_commands else 0.0
        ),
        status_counts=status_counts,
        failure_by_stage=failure_by_stage,
        failure_by_code=failure_by_code,
        failure_by_action=failure_by_action,
        failure_by_actor=failure_by_actor,
        command_by_action_actor=command_by_action_actor,
        failure_by_action_stage_code=failure_by_action_stage_code,
        build_funnel={
            "proposed": build_proposed_count,
            "candidate_validated": len(build_validated_ids),
            "translator_accepted": len(build_translator_accepted_ids),
            "pysc2_accepted": len(build_pysc2_accepted_ids),
            "effect_confirmed": len(build_effect_confirmed_ids),
        },
        build_effect_confirmed_rate=(
            len(build_effect_confirmed_ids) / len(build_pysc2_accepted_ids)
            if build_pysc2_accepted_ids
            else 0.0
        ),
        build_effect_timeouts=len(build_effect_timeout_ids),
        build_effect_timeout_rate=(
            len(build_effect_timeout_ids) / len(build_pysc2_accepted_ids)
            if build_pysc2_accepted_ids
            else 0.0
        ),
        build_pre_dispatch_rejections=len(build_pre_dispatch_rejection_ids),
        build_pre_dispatch_rejection_rate=(
            len(build_pre_dispatch_rejection_ids) / build_proposed_count
            if build_proposed_count
            else 0.0
        ),
        production_funnel={
            "proposed": production_proposed_count,
            "candidate_validated": len(production_validated_ids),
            "translator_accepted": len(production_translator_accepted_ids),
            "pysc2_accepted": len(production_pysc2_accepted_ids),
            "order_confirmed": len(production_order_confirmed_ids),
            "unit_fallback_confirmed": len(production_unit_confirmed_ids),
            "effect_confirmed": len(production_order_confirmed_ids | production_unit_confirmed_ids),
            "acceptance_only": len(production_acceptance_only_ids),
        },
        production_effect_confirmed_rate=(
            len(production_order_confirmed_ids | production_unit_confirmed_ids)
            / len(production_pysc2_accepted_ids)
            if production_pysc2_accepted_ids
            else 0.0
        ),
        production_provenance_coverage=(
            len(production_provenance_complete_ids & production_pysc2_accepted_ids)
            / len(production_pysc2_accepted_ids)
            if production_pysc2_accepted_ids
            else 0.0
        ),
        production_effect_timeouts=len(
            production_effect_timeout_ids & production_pysc2_accepted_ids
        ),
        production_timeout_rate=(
            len(production_effect_timeout_ids & production_pysc2_accepted_ids)
            / len(production_pysc2_accepted_ids)
            if production_pysc2_accepted_ids
            else 0.0
        ),
        production_metrics_applicable=(production_report_protocols == {"1.1"}),
        production_by_action=_counts_from_values(production_action_by_command.values()),
        production_by_producer=_counts_from_values(production_producer_by_command.values()),
        confirmation_latency_game_loops_p50=_percentile(
            list(production_confirmation_latencies.values()),
            0.50,
        ),
        confirmation_latency_game_loops_p95=_percentile(
            list(production_confirmation_latencies.values()),
            0.95,
        ),
        confirmation_latency_game_loops_samples=len(production_confirmation_latencies),
        planner_module_results=proposal_audit.module_results,
        planner_proposal_audited_results=proposal_audit.audited_results,
        planner_proposal_audit_complete=(
            proposal_audit.module_results == proposal_audit.audited_results
        ),
        planner_unsafe_attack_proposals=proposal_audit.unsafe_attacks,
        planner_builder_attack_proposals=proposal_audit.builder_attacks,
        planner_friendly_target_attack_proposals=proposal_audit.friendly_target_attacks,
        planner_unsafe_attack_rejected_before_dispatch=(
            proposal_audit.unsafe_rejected_before_dispatch
        ),
        planner_unsafe_attack_dispatched=proposal_audit.unsafe_dispatched,
        builder_attack_commands=len(builder_attack_command_ids),
        friendly_target_attacks=len(friendly_target_attack_ids),
        planner_noop_proposals=_planner_noop_proposals(events),
        generic_translation_failures=generic_translation_failures,
        upstream_placement_rejections=upstream_placement_rejections,
        unattributed_primitives=_episode_metric(events, "unattributed_primitives"),
        candidate_outside_pysc2_dispatches=_episode_metric(
            events,
            "candidate_outside_pysc2_dispatches",
        ),
        orchestration_573_terminal_reports=orchestration_573_terminal_reports,
        decision_count=len(decisions),
        fallback_decisions=fallback_decisions,
        planner_pending_decisions=planner_pending_decisions,
        idle_reason_counts=idle_reasons,
        unique_validation_rejected_command_ids=len(rejected_ids),
    )


def _decision_commands(decisions: list[StoredEvent]) -> dict[str, dict[str, Any]]:
    commands: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        batch = decision.payload.get("batch")
        if not isinstance(batch, dict):
            continue
        selected = batch.get("commands")
        if not isinstance(selected, list):
            continue
        for command in selected:
            if not isinstance(command, dict):
                continue
            command_id = command.get("command_id")
            if isinstance(command_id, str):
                commands.setdefault(command_id, command)
    return commands


def _dispatched_commands(
    events: list[StoredEvent],
    *,
    commands: dict[str, dict[str, Any]],
    execution_command_ids: list[str],
) -> tuple[
    list[str],
    set[str],
    dict[str, dict[str, Any]],
    set[str],
    bool,
]:
    command_index = dict(commands)
    dispatch_event_ids: list[str] = []
    lifecycle_command_ids: set[str] = set()
    for event in events:
        if event.event_type == "command_lifecycle":
            command = event.payload.get("command")
            if not isinstance(command, dict):
                continue
            command_id = command.get("command_id")
            if not isinstance(command_id, str) or not command_id:
                continue
            lifecycle_command_ids.add(command_id)
            command_index.setdefault(command_id, command)
            if event.payload.get("status") == "dispatched":
                dispatch_event_ids.append(command_id)
        elif event.event_type == "execution":
            command_id = event.payload.get("command_id")
            if not isinstance(command_id, str) or not command_id:
                continue
            reconstructed = dict(command_index.get(command_id, {}))
            for source_name, target_name in (
                ("action_name", "name"),
                ("actor", "actor"),
                ("source", "source"),
                ("requested_arguments", "arguments"),
            ):
                value = event.payload.get(source_name)
                if value is not None:
                    reconstructed[target_name] = value
            command_index[command_id] = reconstructed

    dispatched_ids = set(dispatch_event_ids)
    execution_events = [event for event in events if event.event_type == "execution"]
    legacy_dispatch_fallback = (
        not lifecycle_command_ids
        and bool(execution_events)
        and all(event.payload.get("protocol_version") == "1.0" for event in execution_events)
    )
    if legacy_dispatch_fallback:
        dispatched_ids = set(execution_command_ids)
    return (
        dispatch_event_ids,
        dispatched_ids,
        command_index,
        lifecycle_command_ids,
        legacy_dispatch_fallback,
    )


def _own_tags(events: list[StoredEvent]) -> set[str]:
    tags: set[str] = set()
    for event in events:
        if event.event_type != "observation":
            continue
        state = event.payload.get("state")
        if not isinstance(state, dict):
            continue
        for field_name in ("own_units", "own_structures"):
            units = state.get(field_name)
            if not isinstance(units, list):
                continue
            for unit in units:
                if not isinstance(unit, dict):
                    continue
                if unit.get("alliance", "self") not in {"self", "ally"}:
                    continue
                unit_id = unit.get("unit_id")
                if unit_id is not None:
                    tags.add(_normalize_tag(unit_id))
    return tags


def _own_tags_by_step(events: list[StoredEvent]) -> dict[int, set[str]]:
    tags_by_step: dict[int, set[str]] = {}
    for event in events:
        if event.event_type != "observation":
            continue
        tags_by_step[event.step_id] = _own_tags([event])
    return tags_by_step


def _attack_target_is_friendly(command: dict[str, Any], own_tags: set[str]) -> bool:
    arguments = command.get("arguments")
    return (
        isinstance(arguments, list) and bool(arguments) and _normalize_tag(arguments[0]) in own_tags
    )


def _audit_planner_proposals(
    events: list[StoredEvent],
    *,
    dispatched_ids: set[str],
) -> _PlannerProposalAudit:
    module_results = 0
    parsed_results = 0
    audited_results = 0
    build_proposals = 0
    production_proposals = 0
    unsafe_attacks = 0
    builder_attacks = 0
    friendly_target_attacks = 0
    unsafe_rejected_before_dispatch = 0
    unsafe_dispatched = 0
    own_tags_by_step = _own_tags_by_step(events)

    for event in events:
        if event.event_type != "module_result" or event.payload.get("module") != "planning":
            continue
        module_results += 1
        proposals = _raw_planner_proposals(event)
        if proposals is None:
            continue
        parsed_results += 1
        result_audited = True
        source_own_tags = own_tags_by_step.get(event.step_id)
        for index, proposal in enumerate(proposals):
            action_name = str(proposal.get("name", ""))
            if action_name.casefold().startswith("build_"):
                build_proposals += 1
            if _is_production_action(action_name):
                production_proposals += 1
            if action_name != "Attack_Unit":
                continue
            is_builder_attack = _is_builder_actor(str(proposal.get("actor", "")))
            if source_own_tags is None:
                result_audited = False
                is_friendly_target = False
            else:
                is_friendly_target = _attack_target_is_friendly(proposal, source_own_tags)
            if is_builder_attack:
                builder_attacks += 1
            if is_friendly_target:
                friendly_target_attacks += 1
            if not (is_builder_attack or is_friendly_target):
                continue
            unsafe_attacks += 1
            command_id = f"{event.run_id}:{event.episode_id}:{event.step_id}:planner:{index}"
            if command_id in dispatched_ids:
                unsafe_dispatched += 1
            else:
                unsafe_rejected_before_dispatch += 1
        if result_audited:
            audited_results += 1

    return _PlannerProposalAudit(
        module_results=module_results,
        parsed_results=parsed_results,
        audited_results=audited_results,
        build_proposals=build_proposals,
        production_proposals=production_proposals,
        unsafe_attacks=unsafe_attacks,
        builder_attacks=builder_attacks,
        friendly_target_attacks=friendly_target_attacks,
        unsafe_rejected_before_dispatch=unsafe_rejected_before_dispatch,
        unsafe_dispatched=unsafe_dispatched,
    )


def _raw_planner_proposals(event: StoredEvent) -> list[dict[str, Any]] | None:
    output = event.payload.get("output")
    if not isinstance(output, dict):
        return None
    plan = output.get("plan", output)
    if not isinstance(plan, dict):
        return None
    proposals = plan.get("proposed_actions")
    if not isinstance(proposals, list) or any(
        not isinstance(proposal, dict) for proposal in proposals
    ):
        return None
    return proposals


def _build_candidate_ids(decisions: list[StoredEvent]) -> set[str]:
    command_ids: set[str] = set()
    for decision in decisions:
        for candidate_field in ("planner_candidates", "reflex_candidates"):
            candidates = decision.payload.get(candidate_field)
            if not isinstance(candidates, list):
                continue
            for command in candidates:
                if not isinstance(command, dict):
                    continue
                command_id = command.get("command_id")
                action_name = command.get("name")
                if isinstance(command_id, str) and str(action_name).lower().startswith("build_"):
                    command_ids.add(command_id)
    return command_ids


def _build_selected_ids(commands: dict[str, dict[str, Any]]) -> set[str]:
    return {
        command_id
        for command_id, command in commands.items()
        if str(command.get("name", "")).lower().startswith("build_")
    }


def _build_validated_ids(
    decisions: list[StoredEvent],
    *,
    fallback: set[str],
) -> set[str]:
    command_ids: set[str] = set()
    has_explicit_validation = False
    for decision in decisions:
        candidates = decision.payload.get("validated_candidates")
        if not isinstance(candidates, list):
            continue
        has_explicit_validation = True
        for command in candidates:
            if not isinstance(command, dict):
                continue
            command_id = command.get("command_id")
            action_name = command.get("name")
            if isinstance(command_id, str) and str(action_name).lower().startswith("build_"):
                command_ids.add(command_id)
    return command_ids if has_explicit_validation else fallback


def _production_candidate_ids(decisions: list[StoredEvent]) -> set[str]:
    command_ids: set[str] = set()
    for decision in decisions:
        for candidate_field in ("planner_candidates", "reflex_candidates"):
            candidates = decision.payload.get(candidate_field)
            if not isinstance(candidates, list):
                continue
            for command in candidates:
                if not isinstance(command, dict):
                    continue
                command_id = command.get("command_id")
                action_name = str(command.get("name", ""))
                if isinstance(command_id, str) and _is_production_action(action_name):
                    command_ids.add(command_id)
    return command_ids


def _production_selected_ids(commands: dict[str, dict[str, Any]]) -> set[str]:
    return {
        command_id
        for command_id, command in commands.items()
        if _is_production_action(str(command.get("name", "")))
    }


def _production_validated_ids(
    decisions: list[StoredEvent],
    *,
    fallback: set[str],
) -> set[str]:
    command_ids: set[str] = set()
    has_explicit_validation = False
    for decision in decisions:
        candidates = decision.payload.get("validated_candidates")
        if not isinstance(candidates, list):
            continue
        has_explicit_validation = True
        for command in candidates:
            if not isinstance(command, dict):
                continue
            command_id = command.get("command_id")
            action_name = str(command.get("name", ""))
            if isinstance(command_id, str) and _is_production_action(action_name):
                command_ids.add(command_id)
    return command_ids if has_explicit_validation else fallback


def _is_production_action(action_name: str) -> bool:
    return action_name.casefold().startswith("train_")


def _is_train_action(action_name: str) -> bool:
    return action_name.casefold().startswith("train_")


def _production_provenance_complete(evidence: dict[str, Any]) -> bool:
    return (
        evidence.get("effect_kind") == "production"
        and isinstance(evidence.get("producer_tag"), str)
        and bool(str(evidence["producer_tag"]).strip())
        and isinstance(evidence.get("producer_type"), str)
        and bool(str(evidence["producer_type"]).strip())
        and isinstance(evidence.get("expected_unit_type"), str)
        and bool(str(evidence["expected_unit_type"]).strip())
        and isinstance(evidence.get("expected_order_id"), int)
        and not isinstance(evidence.get("expected_order_id"), bool)
    )


def _production_producer_label(
    evidence: dict[str, Any] | None,
    actor: str,
) -> str | None:
    producer_type = evidence.get("producer_type") if evidence is not None else None
    producer_tag = evidence.get("producer_tag") if evidence is not None else None
    if (
        isinstance(producer_type, str)
        and producer_type
        and isinstance(producer_tag, str)
        and producer_tag
    ):
        return f"{producer_type} / {producer_tag}"
    if isinstance(producer_type, str) and producer_type:
        return producer_type
    if isinstance(producer_tag, str) and producer_tag:
        return producer_tag
    if actor and actor.casefold() not in {"unknown", "developer/empty"}:
        return actor
    return "unknown"


def _counts_from_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        _increment(counts, value)
    return counts


def _confirmation_latency(evidence: dict[str, Any]) -> float | None:
    accepted = evidence.get("accepted_game_loop", evidence.get("accepted_loop"))
    confirmed = evidence.get("confirmed_game_loop", evidence.get("confirmed_loop"))
    if (
        not isinstance(accepted, int | float)
        or isinstance(accepted, bool)
        or not isinstance(confirmed, int | float)
        or isinstance(confirmed, bool)
        or confirmed < accepted
    ):
        return None
    return float(confirmed - accepted)


def _production_confirmation_latencies(events: list[StoredEvent]) -> list[float]:
    latencies: dict[str, float] = {}
    for event in events:
        if event.event_type != "execution":
            continue
        payload = event.payload
        action_name = str(payload.get("action_name") or "")
        evidence = payload.get("effect_evidence")
        if (
            not _is_production_action(action_name)
            or _execution_status(payload) != "succeeded"
            or not isinstance(evidence, dict)
            or evidence.get("effect_kind") != "production"
            or evidence.get("confirmation_kind")
            not in {"producer_order", "producer_morph", "new_unit"}
        ):
            continue
        latency = _confirmation_latency(evidence)
        command_id = payload.get("command_id")
        if latency is not None and isinstance(command_id, str):
            latencies[command_id] = latency
    return list(latencies.values())


def _execution_status(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if status in {"succeeded", "failed", "cancelled", "unconfirmed"}:
        return str(status)
    reason = str(payload.get("failure_reason") or "").lower()
    if "episode ended before command completion" in reason:
        return "cancelled"
    return "succeeded" if payload.get("success") is True else "failed"


def _final_translator_primitive(payload: dict[str, Any]) -> dict[str, Any] | None:
    trace = payload.get("primitive_trace")
    if not isinstance(trace, list):
        return None
    translator = [
        entry
        for entry in trace
        if isinstance(entry, dict)
        and entry.get("origin") == "translator"
        and (entry.get("function") or entry.get("function_name")) != "pre_dispatch"
    ]
    if not translator:
        return None
    last = translator[-1]
    ordinal = last.get("ordinal")
    total = last.get("total")
    if isinstance(ordinal, int) and isinstance(total, int) and ordinal != total - 1:
        return None
    return last


def _is_noop(action_name: str, pysc2_function: object, command_id: str) -> bool:
    normalized_name = "".join(character for character in action_name.lower() if character.isalnum())
    if normalized_name in {"noop", "nooperation"}:
        return True
    if action_name != "unknown":
        return False
    normalized_function = str(pysc2_function or "").lower().replace("-", "_")
    if normalized_function and normalized_function.split(" -> ")[-1] in {"no_op", "noop"}:
        return True
    return ":fallback:" in command_id


def _is_builder_actor(actor: str) -> bool:
    normalized = actor.casefold()
    return normalized == "builder" or normalized.startswith("builder/")


def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()


def _infer_execution_stage(payload: dict[str, Any], status: str) -> str:
    reason = str(payload.get("failure_reason") or "").lower()
    if status in {"cancelled", "unconfirmed"}:
        return "episode_end"
    if "effect" in reason or "expected state change" in reason:
        return "effect_verification"
    if "translation" in reason or "argument validation" in reason:
        return "translation"
    return "pysc2_acceptance"


def _infer_failure_code(payload: dict[str, Any], status: str) -> str:
    reason = str(payload.get("failure_reason") or "").lower()
    if status == "cancelled":
        return "episode_ended"
    if status == "unconfirmed":
        return "effect_unconfirmed"
    if "effect" in reason or "expected state change" in reason:
        return "effect_timeout"
    if "translation" in reason or "argument validation" in reason:
        return "translator_rejected"
    return "unknown_failure"


def _unique_rejected_command_ids(decisions: list[StoredEvent]) -> set[str]:
    command_ids: set[str] = set()
    for decision in decisions:
        batch = decision.payload.get("batch")
        if not isinstance(batch, dict):
            continue
        rejected = batch.get("rejected_commands")
        if not isinstance(rejected, list):
            continue
        for reason in rejected:
            if not isinstance(reason, str):
                continue
            command_ids.add(reason.rsplit(": ", maxsplit=1)[0])
    return command_ids


def _planner_noop_proposals(events: list[StoredEvent]) -> int:
    count = 0
    for event in events:
        if event.event_type != "module_result" or event.payload.get("module") != "planning":
            continue
        output = event.payload.get("output")
        if not isinstance(output, dict):
            continue
        plan = output.get("plan", output)
        if not isinstance(plan, dict):
            continue
        proposals = plan.get("proposed_actions")
        if not isinstance(proposals, list):
            continue
        count += sum(
            isinstance(proposal, dict)
            and str(proposal.get("name", "")).casefold() in {"noop", "no_op", "no_operation"}
            for proposal in proposals
        )
    return count


def _episode_metric(events: list[StoredEvent], name: str) -> int:
    for event in reversed(events):
        if event.event_type != "episode_result":
            continue
        metrics = event.payload.get("metrics")
        if not isinstance(metrics, dict):
            return 0
        value = metrics.get(name, 0)
        return int(value) if isinstance(value, int | float) else 0
    return 0


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _merge_counts(groups: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for group in groups:
        for key, value in group.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def _usage_value(event: StoredEvent, name: str) -> int:
    usage = event.payload.get("usage")
    if isinstance(usage, dict):
        value = usage.get(name, 0)
        return int(value) if isinstance(value, int | float) else 0
    metadata = event.payload.get("generation_metadata")
    if not isinstance(metadata, dict):
        return 0
    metadata_name = {
        "prompt_tokens": "prompt_token_count",
        "completion_tokens": "completion_token_count",
    }.get(name, name)
    value = metadata.get(metadata_name, 0)
    return int(value) if isinstance(value, int | float) else 0


def _usage_total(event: StoredEvent) -> int:
    total = _usage_value(event, "total_tokens")
    if total:
        return total
    return _usage_value(event, "prompt_tokens") + _usage_value(event, "completion_tokens")


def _episode_duration(events: list[StoredEvent]) -> float:
    observations = [event for event in events if event.event_type == "observation"]
    if not observations:
        return 0.0
    terminal_events = [event for event in events if event.event_type == "episode_result"]
    end_event = terminal_events[-1] if terminal_events else events[-1]
    start = datetime.fromisoformat(observations[0].created_at)
    end = datetime.fromisoformat(end_event.created_at)
    return max(0.0, (end - start).total_seconds())
