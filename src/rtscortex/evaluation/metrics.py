"""Metrics derived from one episode's immutable runtime events."""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class EpisodeMetrics:
    """Serializable episode-level metrics plus raw latency samples for aggregation."""

    action_attempts: int
    action_successes: int
    action_success_rate: float
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
    episode_duration_seconds: float
    failure_reason: str | None
    samples: MetricSamples = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_attempts": self.action_attempts,
            "action_successes": self.action_successes,
            "action_success_rate": self.action_success_rate,
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

    decisions = [event for event in events if event.event_type == "decision"]
    illegal_actions = sum(
        len(event.payload.get("batch", {}).get("rejected_commands", [])) for event in decisions
    )
    candidate_actions = action_attempts + illegal_actions

    planner_latencies = [
        float(event.payload["latency_ms"])
        for event in events
        if event.event_type == "planner_cycle" and "latency_ms" in event.payload
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
        if event.event_type == "module_result" and event.payload.get("model_call") is True
    ]
    prompt_tokens = sum(_usage_value(event, "prompt_tokens") for event in model_events)
    completion_tokens = sum(_usage_value(event, "completion_tokens") for event in model_events)
    total_tokens = sum(_usage_total(event) for event in model_events)
    model_cost_usd = (
        prompt_tokens * prompt_cost_per_million_tokens
        + completion_tokens * completion_cost_per_million_tokens
    ) / 1_000_000

    reflex_preemptions = sum(len(event.payload.get("preemptions", [])) for event in decisions)
    plan_events = [event for event in events if event.event_type == "plan_accepted"]
    plan_revisions = sum(event.payload.get("is_revision") is True for event in plan_events)
    revision_opportunities = max(0, len(plan_events) - 1)

    return EpisodeMetrics(
        action_attempts=action_attempts,
        action_successes=action_successes,
        action_success_rate=(action_successes / action_attempts if action_attempts else 0.0),
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
        episode_duration_seconds=_episode_duration(events),
        failure_reason=result.failure_reason,
        samples=MetricSamples(
            planner_latency_ms=tuple(planner_latencies),
            reflex_latency_ms=tuple(reflex_latencies),
            tick_latency_ms=tuple(tick_latencies),
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
    failure_reasons: dict[str, int] = {}
    for item in metrics:
        if item.failure_reason is not None:
            failure_reasons[item.failure_reason] = failure_reasons.get(item.failure_reason, 0) + 1

    return {
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
        "mean_episode_duration_seconds": (
            sum(item.episode_duration_seconds for item in metrics) / len(metrics)
            if metrics
            else 0.0
        ),
        "failure_reasons": failure_reasons,
    }


def _usage_value(event: StoredEvent, name: str) -> int:
    usage = event.payload.get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get(name, 0)
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
