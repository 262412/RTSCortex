from __future__ import annotations

import pytest

from rtscortex.contracts import EpisodeOutcome, EpisodeResult
from rtscortex.evaluation.metrics import (
    aggregate_episode_metrics,
    compute_episode_metrics,
)
from rtscortex.memory import StoredEvent


def _event(
    event_id: int,
    event_type: str,
    payload: dict[str, object],
    *,
    second: int = 1,
) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="run-1",
        episode_id="episode-1",
        step_id=event_id,
        event_type=event_type,
        created_at=f"2026-01-01T00:00:0{second}+00:00",
        payload=payload,
    )


def test_episode_metrics_cover_runtime_telemetry() -> None:
    events = [
        _event(1, "observation", {}, second=0),
        _event(
            2,
            "decision",
            {
                "batch": {"rejected_commands": ["unknown action"]},
                "reflex_latency_ms": 1.0,
                "tick_latency_ms": 4.0,
                "preemptions": [{"actor": "army"}, {"actor": "worker"}],
            },
        ),
        _event(
            3,
            "decision",
            {
                "batch": {"rejected_commands": []},
                "reflex_latency_ms": 3.0,
                "tick_latency_ms": 8.0,
                "preemptions": [],
            },
        ),
        _event(4, "execution", {"success": True}),
        _event(5, "execution", {"success": False}),
        _event(6, "planner_cycle", {"latency_ms": 10.0}),
        _event(7, "planner_cycle", {"latency_ms": 20.0}),
        _event(
            8,
            "module_result",
            {
                "model_call": True,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        ),
        _event(
            9,
            "module_result",
            {
                "model_call": True,
                "usage": {
                    "prompt_tokens": 300,
                    "completion_tokens": 100,
                    "total_tokens": 400,
                },
            },
        ),
        _event(10, "plan_accepted", {"is_revision": False}),
        _event(11, "plan_accepted", {"is_revision": True}),
        _event(12, "plan_accepted", {"is_revision": False}),
        _event(13, "episode_result", {}, second=5),
    ]
    result = EpisodeResult(
        run_id="run-1",
        episode_id="episode-1",
        scenario="test",
        seed=7,
        outcome=EpisodeOutcome.TRUNCATED,
        steps=2,
        failure_reason="max_steps_reached",
    )

    metrics = compute_episode_metrics(
        events,
        result,
        prompt_cost_per_million_tokens=2.0,
        completion_cost_per_million_tokens=4.0,
    )

    assert metrics.action_attempts == 2
    assert metrics.action_successes == 1
    assert metrics.action_success_rate == 0.5
    assert metrics.illegal_actions == 1
    assert metrics.illegal_action_rate == pytest.approx(1 / 3)
    assert metrics.planner_latency_ms_p50 == 15.0
    assert metrics.planner_latency_ms_p95 == 19.5
    assert metrics.reflex_latency_ms_p50 == 2.0
    assert metrics.reflex_latency_ms_p95 == pytest.approx(2.9)
    assert metrics.tick_latency_ms_p50 == 6.0
    assert metrics.tick_latency_ms_p95 == pytest.approx(7.8)
    assert metrics.model_requests == 2
    assert metrics.prompt_tokens == 400
    assert metrics.completion_tokens == 150
    assert metrics.total_tokens == 550
    assert metrics.model_cost_usd == pytest.approx(0.0014)
    assert metrics.reflex_preemptions == 2
    assert metrics.plans_accepted == 3
    assert metrics.plan_revisions == 1
    assert metrics.plan_revision_rate == 0.5
    assert metrics.episode_duration_seconds == 5.0
    assert metrics.failure_reason == "max_steps_reached"

    aggregate = aggregate_episode_metrics([metrics, metrics])
    assert aggregate["action_attempts"] == 4
    assert aggregate["model_requests"] == 4
    assert aggregate["model_cost_usd"] == pytest.approx(0.0028)
    assert aggregate["failure_reasons"] == {"max_steps_reached": 2}
