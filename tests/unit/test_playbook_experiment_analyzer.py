from __future__ import annotations

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
        strategic_consequences={},
        hard_rule_false_block_count=0,
        hard_rule_shadow_state_count=100,
        hard_rule_false_block_rate=0.0,
        journal_bytes=100,
        max_game_loop=100,
        bytes_per_game_loop=1.0,
        effective_game_loops_per_second=2.0,
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
                    repeated_errors=10,
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

    assert comparison["aggregate"]["sequential_repeated_error_reduction"] == 0.6
    assert comparison["gates"]["complete_unique_run_matrix"] is True
    assert comparison["gates"]["independent_baseline_identity"] is True
    assert comparison["gates"]["sequential_evolving_carry"] is True
    assert comparison["accepted"] is True
