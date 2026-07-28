from __future__ import annotations

from pathlib import Path

import pytest

import scripts.analyze_playbook_experiment as analyzer
from rtscortex.memory import StoredEvent
from scripts.analyze_playbook_experiment import RunMetrics, _comparison


def _metrics(
    *,
    mode: str,
    seed: int,
    arm: str,
    before: str,
    after: str,
    repeated_errors: int,
) -> RunMetrics:
    return RunMetrics(
        mode=mode,
        seed=seed,
        arm=arm,
        exit_code=0,
        run_dir=f"/run/{mode}/{arm}/{seed}",
        natural_terminal=True,
        outcome="victory",
        score=1.0,
        terminal_report_count=1,
        missing_terminal_reports=0,
        duplicate_terminal_reports=0,
        duplicate_dispatches=0,
        candidate_outside_dispatch=0,
        playbook_applications=1,
        playbook_nonzero_score_applications=int(arm == "evolving"),
        playbook_blocks=0,
        repeated_eligible_errors=repeated_errors,
        repeated_errors_per_10k_game_loops=float(repeated_errors * 100),
        repeated_errors_per_eligible_operation=float(repeated_errors),
        eligible_operation_count=1,
        strategic_consequences={},
        hard_rule_false_block_count=0,
        hard_rule_shadow_state_count=100,
        hard_rule_false_block_rate=0.0,
        journal_bytes=100,
        artifact_bytes=200,
        event_count=10,
        max_game_loop=100,
        events_per_game_loop=0.1,
        bytes_per_game_loop=1.0,
        effective_game_loops_per_second=2.0,
        writer_queue_peak=1,
        writer_lag_ms_p95=1.0,
        writer_lag_ms_max=2.0,
        blocked_append_count=0,
        dropped_sampled_event_count=0,
        playbook_before_sha256=before,
        playbook_after_sha256=after,
    )


def test_comparison_separates_independent_pairs_from_sequential_learning() -> None:
    baseline = "baseline"
    metrics: list[RunMetrics] = []
    for seed in (0, 1, 2):
        metrics.extend(
            (
                _metrics(
                    mode="independent_paired",
                    seed=seed,
                    arm="frozen",
                    before=baseline,
                    after=baseline,
                    repeated_errors=10,
                ),
                _metrics(
                    mode="independent_paired",
                    seed=seed,
                    arm="evolving",
                    before=baseline,
                    after=f"independent-{seed}",
                    repeated_errors=4,
                ),
            )
        )
        prior = baseline if seed == 0 else f"sequential-{seed - 1}"
        metrics.extend(
            (
                _metrics(
                    mode="sequential_learning",
                    seed=seed,
                    arm="frozen",
                    before=baseline,
                    after=baseline,
                    repeated_errors=10,
                ),
                _metrics(
                    mode="sequential_learning",
                    seed=seed,
                    arm="evolving",
                    before=prior,
                    after=f"sequential-{seed}",
                    repeated_errors=4,
                ),
            )
        )

    comparison = _comparison(metrics, baseline_sha256=baseline)

    assert comparison["aggregate"]["independent_repeated_error_reduction"] == 0.6
    assert comparison["gates"]["complete_unique_run_matrix"] is True
    assert comparison["gates"]["independent_baseline_identity"] is True
    assert comparison["gates"]["sequential_evolving_carry"] is True
    assert comparison["accepted"] is True


def test_false_block_gate_cannot_pass_without_shadow_states() -> None:
    baseline = "baseline"
    metrics = [
        _metrics(
            mode=mode,
            seed=seed,
            arm=arm,
            before=baseline,
            after=baseline if arm == "frozen" else f"{mode}-{seed}",
            repeated_errors=0,
        )
        for mode in ("independent_paired", "sequential_learning")
        for seed in (0, 1, 2)
        for arm in ("frozen", "evolving")
    ]
    metrics = [
        metric.__class__(
            **{
                **metric.__dict__,
                "hard_rule_shadow_state_count": 0,
            }
        )
        for metric in metrics
    ]

    comparison = _comparison(metrics, baseline_sha256=baseline)

    assert comparison["gates"]["hard_false_block_rate_at_most_1_percent"] is False
    assert comparison["accepted"] is False


def test_run_metrics_streams_events_and_normalizes_error_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    events = (
        StoredEvent(
            event_id=1,
            run_id="run",
            episode_id="episode",
            step_id=1,
            event_type="observation",
            created_at="2026-07-28T00:00:00+00:00",
            payload={"game_loop": 100},
        ),
        StoredEvent(
            event_id=2,
            run_id="run",
            episode_id="episode",
            step_id=2,
            event_type="command_lineage",
            created_at="2026-07-28T00:00:01+00:00",
            payload={"lineage": {"operation_id": "operation:one"}},
        ),
        *(
            StoredEvent(
                event_id=index,
                run_id="run",
                episode_id="episode",
                step_id=index,
                event_type="strategic_consequence_attributed",
                created_at=f"2026-07-28T00:00:0{index}+00:00",
                payload={
                    "consequence_type": "threat_unanswered",
                    "role": "defense",
                    "semantic_action": "Attack_Unit",
                    "condition": {"threat": "high"},
                },
            )
            for index in (3, 4)
        ),
        StoredEvent(
            event_id=5,
            run_id="run",
            episode_id="episode",
            step_id=5,
            event_type="event_store_performance",
            created_at="2026-07-28T00:00:05+00:00",
            payload={
                "max_queue_depth": 3,
                "writer_lag_ms_p95": 2.5,
                "writer_lag_ms_max": 4.0,
                "blocked_append_count": 0,
                "dropped_sampled_event_count": 0,
            },
        ),
    )
    monkeypatch.setattr(analyzer, "read_event_log", lambda _: iter(events))

    metrics = analyzer._run_metrics(
        {
            "mode": "independent_paired",
            "seed": "0",
            "arm": "frozen",
            "exit_code": "0",
            "run_dir": str(run_dir),
            "playbook_before_snapshot": str(tmp_path / "before.sqlite3"),
            "playbook_after_snapshot": str(tmp_path / "after.sqlite3"),
            "playbook_before_sha256": "before",
            "playbook_after_sha256": "after",
        }
    )

    assert metrics.event_count == 5
    assert metrics.artifact_bytes >= metrics.journal_bytes
    assert metrics.repeated_eligible_errors == 1
    assert metrics.repeated_errors_per_10k_game_loops == 100.0
    assert metrics.repeated_errors_per_eligible_operation == 1.0
    assert metrics.writer_queue_peak == 3
    assert metrics.writer_lag_ms_p95 == 2.5
