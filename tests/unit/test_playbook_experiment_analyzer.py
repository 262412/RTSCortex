from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

import scripts.analyze_playbook_experiment as analyzer
from rtscortex.memory import StoredEvent
from scripts.analyze_playbook_counterfactual_canary import build_canary_report
from scripts.analyze_playbook_experiment import RunMetrics, _comparison


def _readiness(*, fixture: bool = False) -> dict[str, object]:
    return {
        "baseline_sha256": "baseline",
        "expected_git_sha": "expected",
        "context_applicable_blocking_hard_count": 1,
        "canary_fixture_rule_ids": ["fixture-rule"] if fixture else [],
        "canary_runnable": True,
    }


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
        repeated_errors_per_lineaged_operation=float(repeated_errors),
        lineaged_operation_count=1,
        strategic_consequences={},
        hard_rule_false_block_count=0,
        hard_rule_shadow_state_count=100,
        hard_rule_unresolved_block_count=0,
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
        sampled_drop_supported=False,
        playbook_before_sha256=before,
        playbook_after_sha256=after,
    )


def _strict_matrix(
    *,
    engineering_overrides: dict[str, bool] | None = None,
    shadow_engineering_overrides: dict[str, bool] | None = None,
    shadow_missing_metrics: tuple[str, ...] = (),
) -> list[RunMetrics]:
    baseline = "baseline"
    rows: list[RunMetrics] = []
    for mode in ("independent_paired", "sequential_learning"):
        for seed in (0, 1, 2):
            for arm in ("frozen", "evolving"):
                before = (
                    baseline
                    if mode == "independent_paired" or arm == "frozen" or seed == 0
                    else f"sequential-{seed - 1}"
                )
                after = (
                    baseline
                    if arm == "frozen"
                    else f"sequential-{seed}"
                    if mode == "sequential_learning"
                    else f"independent-{seed}"
                )
                behavior = replace(
                    _metrics(
                        mode=mode,
                        seed=seed,
                        arm=arm,
                        before=before,
                        after=after,
                        repeated_errors=10 if arm == "frozen" else 4,
                    ),
                    experiment_kind="behavior",
                    subject_arm=arm,
                )
                if engineering_overrides:
                    behavior = replace(
                        behavior,
                        engineering_gates={
                            **behavior.engineering_gates,
                            **engineering_overrides,
                        },
                        engineering_accepted=all(
                            {
                                **behavior.engineering_gates,
                                **engineering_overrides,
                            }.values()
                        ),
                    )
                rows.append(behavior)
                shadow = replace(
                    _metrics(
                        mode=mode,
                        seed=seed,
                        arm="shadow",
                        before=before,
                        after=before,
                        repeated_errors=0,
                    ),
                    experiment_kind="calibration",
                    subject_arm=arm,
                )
                if shadow_engineering_overrides:
                    shadow_gates = {
                        **shadow.engineering_gates,
                        **shadow_engineering_overrides,
                    }
                    shadow = replace(
                        shadow,
                        engineering_gates=shadow_gates,
                        engineering_accepted=all(shadow_gates.values()),
                    )
                if shadow_missing_metrics:
                    shadow = replace(
                        shadow,
                        engineering_missing_metrics=shadow_missing_metrics,
                        engineering_accepted=False,
                    )
                rows.append(shadow)
    return rows


def test_current_acceptance_matrix_can_produce_resolved_would_block() -> None:
    comparison = _comparison(_strict_matrix(), baseline_sha256="baseline")

    assert comparison["gates"]["complete_unique_run_matrix"] is True
    assert comparison["aggregate"]["hard_rule_resolved_block_count"] > 0
    assert comparison["gates"]["hard_false_block_rate_at_most_1_percent"] is True


def test_active_hard_block_requires_matched_shadow_evidence() -> None:
    metrics = _strict_matrix()
    metrics[0] = replace(
        metrics[0],
        active_hard_block_keys=("counterfactual:active-only",),
    )

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert comparison["gates"]["active_hard_blocks_have_matched_shadow_evidence"] is False
    assert comparison["accepted"] is False


def test_comparison_rejects_when_any_required_engineering_gate_fails() -> None:
    comparison = _comparison(
        _strict_matrix(engineering_overrides={"build_confirmation_rate": False}),
        baseline_sha256="baseline",
    )

    assert comparison["gates"]["all_engineering_gates_pass"] is False
    assert comparison["aggregate"]["engineering_gates"]["build_confirmation_rate"] is False
    assert comparison["accepted"] is False


def test_missing_required_engineering_metric_fails_closed() -> None:
    metrics = _strict_matrix()
    metrics[0] = replace(
        metrics[0],
        engineering_missing_metrics=("effective_loops_per_second",),
        engineering_accepted=False,
    )

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert comparison["gates"]["required_engineering_metrics_complete"] is False
    assert comparison["accepted"] is False


def test_shadow_engineering_failure_rejects_comparison() -> None:
    comparison = _comparison(
        _strict_matrix(shadow_engineering_overrides={"build_confirmation_rate": False}),
        baseline_sha256="baseline",
    )

    assert comparison["gates"]["all_engineering_gates_pass"] is False
    assert comparison["aggregate"]["engineering_gates"]["build_confirmation_rate"] is False
    assert comparison["accepted"] is False


def test_shadow_missing_engineering_metric_fails_closed() -> None:
    comparison = _comparison(
        _strict_matrix(shadow_missing_metrics=("health_delta_engagement_attribution_valid",)),
        baseline_sha256="baseline",
    )

    assert comparison["gates"]["required_engineering_metrics_complete"] is False
    assert comparison["accepted"] is False


def test_performance_fields_are_thresholded_not_only_reported() -> None:
    comparison = _comparison(
        _strict_matrix(
            engineering_overrides={
                "effective_loops_per_second": False,
                "natural_run_disk_reduction_ratio": False,
            }
        ),
        baseline_sha256="baseline",
    )

    assert comparison["aggregate"]["engineering_gates"]["effective_loops_per_second"] is False
    assert comparison["aggregate"]["engineering_gates"]["natural_run_disk_reduction_ratio"] is False
    assert comparison["accepted"] is False


def test_build_and_tactical_gates_are_aggregated_across_all_runs() -> None:
    metrics = _strict_matrix()
    target = metrics[-2]
    metrics[-2] = replace(
        target,
        engineering_gates={
            **target.engineering_gates,
            "build_start_coverage": False,
            "unchanged_attack_redispatch_zero": False,
        },
        engineering_accepted=False,
    )

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert comparison["aggregate"]["engineering_gates"]["build_start_coverage"] is False
    assert comparison["aggregate"]["engineering_gates"]["unchanged_attack_redispatch_zero"] is False
    assert comparison["accepted"] is False


def test_dirty_or_unexpected_source_revision_rejects_formal_acceptance() -> None:
    metrics = _strict_matrix()
    metrics[0] = replace(
        metrics[0],
        source_tree_clean=False,
        source_commit_matches_expected_sha=False,
    )

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert comparison["gates"]["source_tree_clean"] is False
    assert comparison["gates"]["source_commit_matches_expected_sha"] is False
    assert comparison["accepted"] is False


def test_submodule_source_change_rejects_formal_acceptance() -> None:
    metrics = _strict_matrix()
    metrics[0] = replace(metrics[0], source_attestation_consistent=False)

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert comparison["gates"]["source_attestation_consistent"] is False
    assert comparison["accepted"] is False


def test_runs_from_different_source_attestations_reject_acceptance() -> None:
    metrics = _strict_matrix()
    metrics[0] = replace(
        metrics[0],
        source_attestation_fingerprint="different-source",
    )

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert comparison["gates"]["source_attestation_consistent"] is False
    assert comparison["accepted"] is False


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


def test_analyzer_cli_exits_nonzero_when_report_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_set = tmp_path / "run-set"
    run_set.mkdir()
    (run_set / "experiment-status.tsv").write_text(
        "mode\tseed\tarm\texit_code\trun_dir\tplaybook_before_sha256\tplaybook_after_sha256\n",
        encoding="utf-8",
    )
    engineering_baseline = tmp_path / "engineering-baseline.json"
    engineering_baseline.write_text(
        json.dumps({"natural_run_bytes_per_game_loop": 100.0}),
        encoding="utf-8",
    )
    recovery_evidence = tmp_path / "recovery-evidence.json"
    recovery_evidence.write_text(
        json.dumps({"passed": True, "git_sha": "expected"}),
        encoding="utf-8",
    )
    counterfactual_canary = tmp_path / "counterfactual-canary.json"
    counterfactual_canary.write_text(
        json.dumps(
            {
                "accepted": True,
                "baseline_sha256": "baseline",
                "expected_git_sha": "expected",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_playbook_experiment.py",
            str(run_set),
            "--baseline-sha256",
            "baseline",
            "--expected-git-sha",
            "expected",
            "--engineering-baseline",
            str(engineering_baseline),
            "--recovery-evidence",
            str(recovery_evidence),
            "--counterfactual-canary",
            str(counterfactual_canary),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        analyzer.main()

    assert raised.value.code == 1
    report = json.loads((run_set / "comparison.json").read_text(encoding="utf-8"))
    assert report["accepted"] is False
    assert (run_set / "report.md").is_file()


def test_paired_runner_propagates_failed_acceptance_gate() -> None:
    runner = (
        Path(__file__).parents[2] / "scripts" / "run_protoss_playbook_paired_natural_terminal.sh"
    ).read_text(encoding="utf-8")

    assert "analysis_status=$?" in runner
    assert "if [[ ${analysis_status} -ne 0 ]]; then\n  overall_status=1\nfi" in runner
    assert 'echo "analysis_exit_code=${analysis_status}"' in runner
    assert "--expected-git-sha" in runner
    assert "run_shadow_calibration" in runner
    assert "--engineering-baseline" in runner
    assert "--recovery-evidence" in runner
    assert "--counterfactual-canary" in runner
    assert "--seeds" in runner
    assert "--expected-seed" in runner
    assert "playbook hard-readiness" in runner
    assert runner.index("playbook hard-readiness") < runner.index(
        "run_recovery_acceptance_canary.py"
    )
    assert "submodule_diff_sha256_before" in runner
    assert "capture_source_attestation" in runner
    assert 'exit "${overall_status}"' in runner

    canary_runner = (
        Path(__file__).parents[2] / "scripts" / "run_protoss_playbook_counterfactual_canary.sh"
    ).read_text(encoding="utf-8")
    assert canary_runner.index("playbook hard-readiness") < canary_runner.index(
        "run_recovery_acceptance_canary.py"
    )


def test_active_intent_cannot_be_resolved_by_different_shadow_target() -> None:
    behavior = replace(
        _metrics(
            mode="causal_canary",
            seed=0,
            arm="active",
            before="baseline",
            after="after",
            repeated_errors=0,
        ),
        active_hard_block_keys=("counterfactual:target-a",),
        active_hard_block_records=(("counterfactual:target-a", 100, "a" * 64),),
    )
    shadow = replace(
        _metrics(
            mode="causal_canary",
            seed=0,
            arm="shadow",
            before="baseline",
            after="baseline",
            repeated_errors=0,
        ),
        resolved_counterfactual_keys=("counterfactual:target-b",),
    )

    report = build_canary_report(
        behavior,
        shadow,
        baseline_sha256="baseline",
        expected_git_sha="expected",
        readiness_evidence=_readiness(),
    )

    assert report["matched_counterfactual_count"] == 0
    assert report["unmatched_active_hard_block_count"] == 1
    assert report["accepted"] is False


def test_counterfactual_canary_reports_first_divergence_and_memory() -> None:
    behavior = replace(
        _metrics(
            mode="causal_canary",
            seed=0,
            arm="active",
            before="baseline",
            after="after",
            repeated_errors=0,
        ),
        active_hard_block_keys=("counterfactual:shared",),
        active_hard_block_records=(("counterfactual:shared", 100, "a" * 64),),
        counterfactual_state_records=((100, "s" * 64, "a" * 64),),
        analysis_peak_rss_kib=1234,
        retained_event_count=50,
        rule_evaluation_count=4,
    )
    shadow = replace(
        _metrics(
            mode="causal_canary",
            seed=0,
            arm="shadow",
            before="baseline",
            after="baseline",
            repeated_errors=0,
        ),
        resolved_counterfactual_keys=("counterfactual:shared",),
        counterfactual_state_records=((100, "s" * 64, "b" * 64),),
    )

    report = build_canary_report(
        behavior,
        shadow,
        baseline_sha256="baseline",
        expected_git_sha="expected",
        readiness_evidence=_readiness(),
    )

    assert report["first_state_hash_divergence_game_loop"] == 100
    assert report["analysis_memory"]["behavior_peak_rss_kib"] == 1234
    assert report["gates"]["matched_prestate_identity"] is False
    assert report["accepted"] is False


def test_formal_comparison_accepts_complete_counterfactual_canary() -> None:
    behavior = replace(
        _metrics(
            mode="causal_canary",
            seed=0,
            arm="active",
            before="baseline",
            after="after",
            repeated_errors=0,
        ),
        active_hard_block_keys=("counterfactual:shared",),
        active_hard_block_records=(("counterfactual:shared", 100, "a" * 64),),
        counterfactual_state_records=((100, "s" * 64, "a" * 64),),
    )
    shadow = replace(
        _metrics(
            mode="causal_canary",
            seed=0,
            arm="shadow",
            before="baseline",
            after="baseline",
            repeated_errors=0,
        ),
        resolved_counterfactual_keys=("counterfactual:shared",),
        counterfactual_state_records=((100, "s" * 64, "a" * 64),),
    )
    canary = build_canary_report(
        behavior,
        shadow,
        baseline_sha256="baseline",
        expected_git_sha="expected",
        readiness_evidence=_readiness(),
    )

    comparison = _comparison(
        _strict_matrix(),
        baseline_sha256="baseline",
        expected_git_sha="expected",
        counterfactual_canary=canary,
    )

    assert canary["accepted"] is True
    assert comparison["gates"]["counterfactual_canary_accepted"] is True


def test_analysis_evidence_overflow_rejects_formal_comparison() -> None:
    metrics = _strict_matrix()
    metrics[0] = replace(metrics[0], analysis_evidence_overflow_count=1)

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert comparison["gates"]["analysis_memory_budget_respected"] is False
    assert comparison["accepted"] is False


def test_unique_command_tracking_is_hard_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        StoredEvent(
            event_id=index,
            run_id="run",
            episode_id="episode",
            step_id=index,
            event_type="execution",
            created_at=f"2026-07-28T00:00:0{index}+00:00",
            payload={"command_id": f"command-{index}"},
        )
        for index in range(1, 4)
    ]

    metrics = _metrics_for_events(
        tmp_path,
        monkeypatch,
        events,
        metric_state_limit=2,
    )

    assert metrics.terminal_report_count == 2
    assert metrics.metric_state_retained_key_count == 2
    assert metrics.metric_state_retention_limit == 2
    assert metrics.analysis_evidence_overflow_count == 1


def test_unique_operation_tracking_is_hard_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        StoredEvent(
            event_id=index,
            run_id="run",
            episode_id="episode",
            step_id=index,
            event_type="command_lineage",
            created_at=f"2026-07-28T00:00:0{index}+00:00",
            payload={"lineage": {"operation_id": f"operation-{index}"}},
        )
        for index in range(1, 4)
    ]

    metrics = _metrics_for_events(
        tmp_path,
        monkeypatch,
        events,
        metric_state_limit=2,
    )

    assert metrics.lineaged_operation_count == 2
    assert metrics.metric_state_retained_key_count == 2
    assert metrics.analysis_evidence_overflow_count == 1


def test_metric_state_overflow_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _metrics_for_events(
        tmp_path,
        monkeypatch,
        [
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
                    "condition": {"index": index},
                },
            )
            for index in range(1, 3)
        ],
        metric_state_limit=1,
    )
    metrics = _strict_matrix()
    metrics[0] = replace(
        metrics[0],
        analysis_evidence_overflow_count=observed.analysis_evidence_overflow_count,
    )

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert observed.analysis_evidence_overflow_count > 0
    assert comparison["gates"]["analysis_memory_budget_respected"] is False
    assert comparison["accepted"] is False


def test_analysis_memory_budget_covers_all_retained_state() -> None:
    budget = analyzer._MetricStateBudget(limit=5)
    terminal: Counter[str] = Counter()
    dispatch: Counter[str] = Counter()
    consequences: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    operations: set[str] = set()

    budget.increment(terminal, "command-terminal")
    budget.increment(dispatch, "command-dispatch")
    budget.increment(consequences, "threat_unanswered")
    budget.increment(signatures, "signature")
    budget.add(operations, "operation")
    budget.increment(terminal, "command-overflow")
    budget.add(operations, "operation-overflow")

    assert budget.retained_key_count == 5
    assert budget.overflow_count == 2
    assert "command-overflow" not in terminal
    assert "operation-overflow" not in operations


def test_formal_comparison_rejects_source_mismatched_counterfactual_canary() -> None:
    comparison = _comparison(
        _strict_matrix(),
        baseline_sha256="baseline",
        expected_git_sha="expected",
        counterfactual_canary={
            "accepted": True,
            "baseline_sha256": "different",
            "expected_git_sha": "expected",
        },
    )

    assert comparison["gates"]["counterfactual_canary_accepted"] is False
    assert comparison["accepted"] is False


def test_formal_comparison_rejects_fixture_counterfactual_canary() -> None:
    behavior = replace(
        _metrics(
            mode="fixture",
            seed=0,
            arm="active",
            before="baseline",
            after="baseline",
            repeated_errors=0,
        ),
        active_hard_block_keys=("counterfactual:shared",),
        active_hard_block_records=(("counterfactual:shared", 100, "a" * 64),),
        counterfactual_state_records=((100, "s" * 64, "a" * 64),),
    )
    shadow = replace(
        _metrics(
            mode="fixture",
            seed=0,
            arm="shadow",
            before="baseline",
            after="baseline",
            repeated_errors=0,
        ),
        resolved_counterfactual_keys=("counterfactual:shared",),
        counterfactual_state_records=((100, "s" * 64, "a" * 64),),
    )
    fixture = build_canary_report(
        behavior,
        shadow,
        baseline_sha256="baseline",
        expected_git_sha="expected",
        readiness_evidence=_readiness(fixture=True),
        canary_kind="fixture",
    )

    comparison = _comparison(
        _strict_matrix(),
        baseline_sha256="baseline",
        expected_git_sha="expected",
        counterfactual_canary=fixture,
    )

    assert fixture["accepted"] is True
    assert fixture["canary_fixture"] is True
    assert comparison["gates"]["counterfactual_canary_accepted"] is False
    assert comparison["accepted"] is False


def test_formal_comparison_accepts_an_explicit_held_out_seed_matrix() -> None:
    held_out = [replace(metric, seed=metric.seed + 3) for metric in _strict_matrix()]

    comparison = _comparison(
        held_out,
        baseline_sha256="baseline",
        expected_seeds=(3, 4, 5),
    )

    assert comparison["expected_seed_ids"] == [3, 4, 5]
    assert comparison["gates"]["complete_unique_run_matrix"] is True


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
        StoredEvent(
            event_id=20,
            run_id="run",
            episode_id="episode",
            step_id=2,
            event_type="command_lineage",
            created_at="2026-07-28T00:00:01.500000+00:00",
            payload={"lineage": {"operation_id": "operation:unrelated"}},
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
                "sampled_drop_supported": False,
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

    assert metrics.event_count == 6
    assert metrics.artifact_bytes >= metrics.journal_bytes
    assert metrics.repeated_eligible_errors == 1
    assert metrics.repeated_errors_per_10k_game_loops == 100.0
    assert metrics.repeated_errors_per_lineaged_operation == 0.5
    assert metrics.lineaged_operation_count == 2
    assert metrics.writer_queue_peak == 3
    assert metrics.writer_lag_ms_p95 == 2.5
    assert metrics.sampled_drop_supported is False


def test_false_blocks_are_preserved_when_hard_rule_becomes_suspended(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:hard-before-suspend",
                strength="hard",
                status="active",
                false_block=True,
            )
        ],
    )

    assert metrics.hard_rule_false_block_count == 1
    assert metrics.hard_rule_shadow_state_count == 1
    assert metrics.hard_rule_false_block_rate == 1.0


def test_soft_to_hard_transition_does_not_import_historical_false_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:soft-history",
                strength="soft",
                status="active",
                false_block=True,
            ),
            _rule_evaluation(
                event_id=2,
                evaluation_id="evaluation:hard-current",
                strength="hard",
                status="active",
                false_block=False,
            ),
        ],
    )

    assert metrics.hard_rule_false_block_count == 0
    assert metrics.hard_rule_shadow_state_count == 1


def test_retired_rule_run_delta_uses_event_time_strength(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:active-hard",
                strength="hard",
                status="active",
                false_block=True,
            ),
            _rule_evaluation(
                event_id=2,
                evaluation_id="evaluation:after-retirement",
                strength="hard",
                status="retired",
                false_block=True,
            ),
        ],
    )

    assert metrics.hard_rule_false_block_count == 1
    assert metrics.hard_rule_shadow_state_count == 1


def test_unselected_would_block_is_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_would_block_outcome_is_unresolved("not_selected", tmp_path, monkeypatch)


def test_unselected_prearbitration_candidate_is_not_counterfactual_observable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:prearbitration",
                strength="hard",
                status="active",
                false_block=None,
                actual_outcome="not_selected",
                counterfactual_observable=False,
            )
        ],
    )

    assert metrics.hard_rule_shadow_state_count == 0
    assert metrics.hard_rule_unresolved_block_count == 0


def test_shadow_observable_block_resolves_from_terminal_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:observable-terminal",
                strength="hard",
                status="active",
                false_block=False,
                actual_outcome="failed",
                counterfactual_observable=True,
            )
        ],
    )

    assert metrics.hard_rule_shadow_state_count == 1
    assert metrics.hard_rule_unresolved_block_count == 0
    assert metrics.hard_rule_false_block_rate == 0.0


def test_strategic_rule_success_is_not_execution_false_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:strategy",
                strength="hard",
                status="active",
                false_block=True,
                actual_outcome="succeeded",
                counterfactual_observable=True,
                rule_kind="strategy",
                strategic_regret=False,
            )
        ],
    )

    assert metrics.hard_rule_shadow_state_count == 0
    assert metrics.hard_rule_false_block_count == 0
    assert metrics.strategic_resolved_count == 1
    assert metrics.strategic_regret_count == 0


@pytest.mark.parametrize(
    ("counterfactual_observable", "rule_kind", "metric_index"),
    [
        (None, "execution_guard", 0),
        (True, None, 0),
        (None, "execution_guard", 1),
        (True, None, 1),
    ],
)
def test_missing_counterfactual_schema_field_fails_closed(
    counterfactual_observable: bool | None,
    rule_kind: str | None,
    metric_index: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:missing-schema",
                strength="hard",
                status="active",
                false_block=False,
                counterfactual_observable=counterfactual_observable,
                rule_kind=rule_kind,
            )
        ],
    )
    metrics = _strict_matrix()
    metrics[metric_index] = replace(
        metrics[metric_index],
        invalid_counterfactual_evidence_count=(observed.invalid_counterfactual_evidence_count),
    )

    comparison = _comparison(metrics, baseline_sha256="baseline")

    assert observed.invalid_counterfactual_evidence_count == 1
    assert comparison["gates"]["counterfactual_schema_complete"] is False
    assert comparison["accepted"] is False


def test_cancelled_would_block_is_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_would_block_outcome_is_unresolved("cancelled", tmp_path, monkeypatch)


@pytest.mark.parametrize("actual_outcome", ["blocked", "unconfirmed", "satisfied_by_peer"])
def test_every_other_unobservable_would_block_outcome_is_unresolved(
    actual_outcome: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_would_block_outcome_is_unresolved(actual_outcome, tmp_path, monkeypatch)


def _assert_would_block_outcome_is_unresolved(
    actual_outcome: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id=f"evaluation:{actual_outcome}",
                strength="hard",
                status="active",
                false_block=None,
                actual_outcome=actual_outcome,
            )
        ],
    )

    assert metrics.hard_rule_shadow_state_count == 0
    assert metrics.hard_rule_unresolved_block_count == 1


def test_would_allow_does_not_enter_false_block_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:would-allow",
                strength="hard",
                status="active",
                false_block=False,
                shadow_decision="would_allow",
            ),
            _rule_evaluation(
                event_id=2,
                evaluation_id="evaluation:would-block",
                strength="hard",
                status="active",
                false_block=True,
            ),
        ],
    )

    assert metrics.hard_rule_shadow_state_count == 1
    assert metrics.hard_rule_false_block_count == 1
    assert metrics.hard_rule_false_block_rate == 1.0


def test_pending_would_block_prevents_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _metrics_for_rule_evaluations(
        tmp_path,
        monkeypatch,
        [
            _rule_evaluation(
                event_id=1,
                evaluation_id="evaluation:pending",
                strength="hard",
                status="active",
                false_block=None,
                actual_outcome="pending",
            )
        ],
    )
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
    first = metrics[0]
    metrics[0] = first.__class__(
        **{
            **first.__dict__,
            "hard_rule_shadow_state_count": observed.hard_rule_shadow_state_count,
            "hard_rule_unresolved_block_count": observed.hard_rule_unresolved_block_count,
        }
    )

    comparison = _comparison(metrics, baseline_sha256=baseline)

    assert comparison["gates"]["hard_false_block_rate_at_most_1_percent"] is False
    assert comparison["accepted"] is False


def test_unresolved_active_hard_block_cannot_pass_false_block_gate() -> None:
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
    first = metrics[0]
    metrics[0] = first.__class__(
        **{
            **first.__dict__,
            "hard_rule_unresolved_block_count": 1,
        }
    )

    comparison = _comparison(metrics, baseline_sha256=baseline)

    assert comparison["gates"]["hard_false_block_rate_at_most_1_percent"] is False
    assert comparison["accepted"] is False


def _metrics_for_rule_evaluations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluations: list[StoredEvent],
) -> RunMetrics:
    run_dir = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(analyzer, "read_event_log", lambda _: iter(evaluations))
    return analyzer._run_metrics(
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


def _metrics_for_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: list[StoredEvent],
    *,
    metric_state_limit: int,
) -> RunMetrics:
    run_dir = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(analyzer, "read_event_log", lambda _: iter(events))
    return analyzer._run_metrics(
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
        },
        metric_state_limit=metric_state_limit,
    )


def _rule_evaluation(
    *,
    event_id: int,
    evaluation_id: str,
    strength: str,
    status: str,
    false_block: bool | None,
    shadow_decision: str = "would_block",
    actual_outcome: str | None = None,
    counterfactual_observable: bool | None = True,
    rule_kind: str | None = "execution_guard",
    strategic_regret: bool | None = None,
    include_counterfactual_identity: bool = True,
) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="run",
        episode_id="episode",
        step_id=event_id,
        event_type="playbook_rule_evaluated",
        created_at=f"2026-07-28T00:00:0{event_id}+00:00",
        payload={
            "evaluation_id": evaluation_id,
            "rule_id": "rule:test",
            "strength_at_evaluation": strength,
            "status_at_evaluation": status,
            "shadow_decision": shadow_decision,
            **(
                {
                    "counterfactual_key": "counterfactual:" + "a" * 64,
                    "counterfactual_signature": "b" * 64,
                    "behavior_before_hash": "c" * 64,
                    "decision_epoch": event_id,
                }
                if include_counterfactual_identity
                else {}
            ),
            **(
                {}
                if counterfactual_observable is None
                else {"counterfactual_observable": counterfactual_observable}
            ),
            **({} if rule_kind is None else {"rule_kind": rule_kind}),
            **({} if strategic_regret is None else {"strategic_regret": strategic_regret}),
            "actual_outcome": (
                actual_outcome
                if actual_outcome is not None
                else ("succeeded" if false_block else "failed")
            ),
            "false_block": false_block,
        },
    )
